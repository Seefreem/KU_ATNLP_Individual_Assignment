import ray.train as ray_train
import random
from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict


def calculate_accuracies(eval_preds):
    """
    Calculate token-wise accuracy and sequence-wise accuracy.
    
    Args:
        predictions (list[list]): List of predicted token sequences.
        targets (list[list]): List of target token sequences.
    
    Returns:
        tuple: (token_wise_accuracy, sequence_wise_accuracy)
    """
    # Ensure predictions and targets are the same length
    logits, labels = eval_preds
    prediction = logits#.argmax(dim=-1).cpu()
    assert len(logits) == len(labels), "Predictions and targets must have the same number of sequences."
    
    targets = []
    predictions = []
    # print(prediction.shape)
    
    for batch_idx in range(len(labels)):
        target = []
        pre = []
        for i in range(len(labels[batch_idx])):
            if labels[batch_idx][i].item() != -100:
                if labels[batch_idx][i].item() == 102: # tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
                    targets.append(target)
                    predictions.append(pre)
                    break
                target.append(labels[batch_idx][i])
                pre.append(prediction[batch_idx][i])     
    
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    
    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure sequences are the same length
        assert len(pred_seq) == len(target_seq), "Each prediction and target sequence must have the same length."
        
        # Token-wise comparison
        total_tokens += len(target_seq)
        correct_tokens += sum(p == t for p, t in zip(pred_seq, target_seq))
        
        # Sequence-wise comparison
        if pred_seq == target_seq:
            correct_sequences += 1
    
    # Calculate accuracies
    token_wise_accuracy = correct_tokens / total_tokens
    sequence_wise_accuracy = correct_sequences / len(targets)
    ray_train.report({"token_wise_accuracy": token_wise_accuracy,
                      "sequence_wise_accuracy": sequence_wise_accuracy})
    return {"token_wise_accuracy": token_wise_accuracy,
            "sequence_wise_accuracy": sequence_wise_accuracy}


def preprocess_logits_for_metrics(logits, labels):
    prediction = logits.argmax(dim=-1).cpu()
    return prediction


# Define a function to process each example
def process_example(example):
    # Split the example into IN and OUT parts, and remove the labels
    text = example['text']
    parts = text.split('OUT:')
    in_part = parts[0].replace('IN:', '').strip()
    out_part = parts[1].strip() if len(parts) > 1 else ''
    return {'input': in_part, 'output': out_part}

def load_scan_dataset(train, test, extend=False, target_size=None):
    datasets = load_dataset(
        'text', 
        data_files={'train': train , # 'data/simple_split/tasks_train_simple.txt'
                    'test': test}) # 'data/simple_split/tasks_test_simple.txt'
    if extend and target_size != None:
        # Assuming your processed dataset is stored in a Hugging Face Dataset object called `processed_dataset`
        # Get the original samples as a list of dictionaries
        original_samples = datasets['train'].to_dict()

        # Calculate how many additional samples are needed
        # total_samples = 100000
        original_count = len(original_samples['text'])
        additional_count = target_size - original_count

        # Randomly sample additional samples with replacement
        additional_samples = {
            key: random.choices(original_samples[key], k=additional_count)
            for key in original_samples
        }

        # Combine the original samples and the additional samples
        datasets['train'] = Dataset.from_dict({
                key: original_samples[key] + additional_samples[key]
                for key in original_samples
            })

        # Verify the structure of the combined DatasetDict
        print(datasets)

    datasets['train'] = datasets['train'].map(process_example)
    datasets['test'] = datasets['test'].map(process_example)
    # Display the processed dataset
    print(datasets, datasets['train'][0])
    return datasets