from datasets import load_dataset
import random
from datasets import Dataset
from datasets import DatasetDict
import torch
gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu else "cpu")
print(device)
MAX_LENGTH = 512

# Define a function to process each example
def process_example(example):
    # Split the example into IN and OUT parts, and remove the labels
    text = example['text']
    parts = text.split('OUT:')
    in_part = parts[0].replace('IN:', '').strip()
    out_part = parts[1].strip() if len(parts) > 1 else ''
    return {'input': in_part, 'output': out_part}

datasets = load_dataset(
    'text', 
    data_files={'train': 'data/length_split/tasks_train_length.txt',
                'test': 'data/length_split/tasks_test_length.txt'})

datasets['train'] = datasets['train'].map(process_example)
datasets['test'] = datasets['test'].map(process_example)
# Display the processed dataset
print(datasets, datasets['train'][0])



from transformers import BertTokenizer, DataCollatorWithPadding

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', force_download=False)
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained("./models/checkpoint-5000", force_download=False)
model = BertForMaskedLM.from_pretrained(
    "./models/checkpoint-5000",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = BertTokenizer.from_pretrained("./models/checkpoint-5000")


def tokenize_function(example):
    inputs = dict()
    input_str = [example["input"][idx] + tokenizer.sep_token + (' ' + tokenizer.mask_token) * MAX_LENGTH for idx in range(len(example["input"]))]
    output_str = [example["input"][idx] + tokenizer.sep_token + example["output"][idx] + (' ' + tokenizer.sep_token) * MAX_LENGTH for idx in range(len(example["input"]))]

    input_tokens = tokenizer(input_str, 
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=MAX_LENGTH)
    output_tokens = tokenizer(output_str, 
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=MAX_LENGTH)
    
    inputs.update(input_tokens)
    inputs['labels'] = output_tokens['input_ids']
    for idx in range(len(inputs['labels'])):
        for i in range(len(inputs['labels'][idx])):
            sep = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            if inputs['labels'][idx][i] != sep:
                inputs['labels'][idx][i] = -100
            else:
                inputs['labels'][idx][i] = -100
                break


    return inputs
tokenized_datasets = dict()
tokenized_datasets['train'] = datasets['train']
tokenized_datasets['test'] = datasets['test'].map(tokenize_function, 
                                                  batched=True, 
                                                  remove_columns=['text', 'input', 'output'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512, padding=True)


from collections import defaultdict

def calculate_accuracies(predictions, targets, input_lengths, label_lengths):
    """
    Calculate token-wise accuracy, sequence-wise accuracy, and additional accuracies:
        1. Token-level accuracy w.r.t. input sequence length
        2. Token-level accuracy w.r.t. label sequence length
        3. Sentence-level accuracy w.r.t. input sequence length
        4. Sentence-level accuracy w.r.t. label sequence length

    Args:
        predictions (list[list]): List of predicted token sequences.
        targets (list[list]): List of target token sequences.
        input_lengths (list[int]): Lengths of input sequences.
        label_lengths (list[int]): Lengths of label sequences.

    Returns:
        dict: A dictionary containing all calculated accuracies.
    """
    # Ensure predictions and targets are the same length
    assert len(predictions) == len(targets), "Predictions and targets must have the same number of sequences."
    assert len(predictions) == len(input_lengths) == len(label_lengths), \
        "Input lengths and label lengths must match the number of sequences."

    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0

    token_accuracy_wrt_input_length = defaultdict(list)
    token_accuracy_wrt_label_length = defaultdict(list)
    sequence_accuracy_wrt_input_length = defaultdict(list)
    sequence_accuracy_wrt_label_length = defaultdict(list)

    for idx, (pred_seq, target_seq, input_len, label_len) in enumerate(zip(predictions, targets, input_lengths, label_lengths)):
        # Ensure sequences are the same length
        assert len(pred_seq) == len(target_seq), "Each prediction and target sequence must have the same length."
        
        # Token-wise comparison
        sequence_correct = True
        token_correct_count = 0
        for pred_token, target_token in zip(pred_seq, target_seq):
            if pred_token == target_token:
                correct_tokens += 1
                token_correct_count += 1
            else:
                sequence_correct = False
        total_tokens += len(target_seq)

        # Sequence-level accuracy
        if sequence_correct:
            correct_sequences += 1

        # Accumulate token-level accuracy w.r.t input and label sequence length
        token_accuracy_wrt_input_length[input_len].append(token_correct_count / label_len)
        token_accuracy_wrt_label_length[label_len].append(token_correct_count / label_len)

        # Accumulate sequence-level accuracy w.r.t input and label sequence length
        sequence_accuracy_wrt_input_length[input_len].append(1 if sequence_correct else 0)
        sequence_accuracy_wrt_label_length[label_len].append(1 if sequence_correct else 0)

    # Aggregate results
    def aggregate_accuracy(data_dict):
        return {key: sum(values) / len(values) for key, values in data_dict.items()}

    aggregated_token_accuracy_input = aggregate_accuracy(token_accuracy_wrt_input_length)
    aggregated_token_accuracy_label = aggregate_accuracy(token_accuracy_wrt_label_length)
    aggregated_sequence_accuracy_input = aggregate_accuracy(sequence_accuracy_wrt_input_length)
    aggregated_sequence_accuracy_label = aggregate_accuracy(sequence_accuracy_wrt_label_length)

    # Calculate overall accuracies
    token_wise_accuracy = correct_tokens / total_tokens
    sequence_wise_accuracy = correct_sequences / len(targets)

    return {
        "token_wise_accuracy": token_wise_accuracy,
        "sequence_wise_accuracy": sequence_wise_accuracy,
        "token_accuracy_wrt_input_length": aggregated_token_accuracy_input,
        "token_accuracy_wrt_label_length": aggregated_token_accuracy_label,
        "sequence_accuracy_wrt_input_length": aggregated_sequence_accuracy_input,
        "sequence_accuracy_wrt_label_length": aggregated_sequence_accuracy_label
    }



def convert_string(source_string):
    # Split the string into tokens based on spaces
    tokens = source_string.split()
    # Initialize an empty list to hold the combined words
    combined_words = []
    while len(tokens) > 0:
        popped_token = tokens.pop(0)
        if popped_token != '_':
            combined_words.append(popped_token)
        elif len(tokens) > 0 and len(combined_words) > 0:
            new_word = combined_words.pop(-1) + popped_token + tokens.pop(0)
            combined_words.append(new_word)
    return combined_words


from torch.utils.data import DataLoader
from pprint import pprint
import torch
from tqdm import tqdm
# Evaluation
def evaluation(dataset, model, batch_size, tokenizer):
    data_loader = DataLoader(dataset, # tokenized_datasets['test'].with_format("torch") 
                             batch_size=batch_size, 
                             shuffle=False)
    targets = []
    predictions = []
    model.to(device)
    input_lengths = []
    outputs_logits = []
    labels = []
    # print(model.device, device)
    with torch.no_grad():
        for data in tqdm(data_loader):
            for key in data:
                data[key] = data[key].to(model.device)
                # print(key, data[key].device)
            # Predict
            outputs = model(**data, return_dict=True)  
            # Greedy sampling
            prediction = outputs.logits.argmax(dim=-1)
            outputs_logits.append(outputs.logits.detach().cpu())
            # Truncation according to the length of labels
            for batch_idx in range(len(data['labels'])):
                target = []
                pre = []
                input_len = len(tokenizer.decode(data['input_ids'][batch_idx].cpu().tolist(), 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=True).split())
                input_lengths.append(input_len) 
                for i in range(len(data['labels'][batch_idx])):
                    if data['labels'][batch_idx][i] != -100:
                        sep = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
                        if data['labels'][batch_idx][i] == sep: # 102
                            # Convert BERT tokens into task specific tokens
                            target_seq = convert_string(tokenizer.decode(target, 
                                                        skip_special_tokens=True, 
                                                        clean_up_tokenization_spaces=True))
                            pre_seq = convert_string(tokenizer.decode(pre, 
                                                    skip_special_tokens=True, 
                                                    clean_up_tokenization_spaces=True))
                            if len(target_seq) > len(pre_seq):
                                pre_seq += [tokenizer.sep_token] * (len(target_seq) - len(pre_seq))
                            elif len(target_seq) < len(pre_seq):
                                pre_seq = pre_seq[:len(target_seq)]
                            assert len(target_seq) == len(pre_seq), \
                                "Each prediction and target sequence must have the same length."
                            targets.append(target_seq)
                            predictions.append(pre_seq)
                            break
                        target.append(data['labels'][batch_idx][i].cpu().item())
                        labels.append(data['labels'][batch_idx].cpu())
                        pre.append(prediction[batch_idx][i].cpu().item())     
                   
        label_lengths = [len(label_seq) for label_seq in targets]
        outputs_logits = torch.cat(outputs_logits, dim=0)
        labels = torch.cat(labels, dim=0)
    return {
        'acc': calculate_accuracies(predictions, targets, input_lengths, label_lengths),
        'outputs_logits': outputs_logits,
        'labels': labels
    }


result_dict = evaluation(tokenized_datasets['test'].with_format("torch"), 
                     model, batch_size=1, tokenizer=tokenizer)


# Save to files
torch.save(result_dict['outputs_logits'], "outputs_logits.pt")
torch.save(result_dict['labels'], "labels.pt")
results = result_dict['acc']
print(results)

import matplotlib.pyplot as plt
def plot_acc(action_sequence_length, accuracy, xlabel, ylabel, title, file_name):
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))  # Set figure size

    # Plot bar chart
    ax1.bar(action_sequence_length, accuracy, width=1.0, edgecolor='black')

    # Customize axes and labels
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color="black")
    ax1.set_title(title)

    # Add gridlines for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot as an image file
    plt.savefig(file_name, dpi=300, format="pdf", bbox_inches='tight')

    # Display the plot
    # plt.show()


accuracy_token_acc_len = list(results['token_accuracy_wrt_label_length'].keys())
accuracy_token_acc_action = [results['token_accuracy_wrt_label_length'][len] for len in accuracy_token_acc_len]
plot_acc(accuracy_token_acc_len, accuracy_token_acc_action, 
    "Ground-Truth Action Sequence Length (in words)", 
    '"Accuracy on New Commands (%)"', 
    "Token-Level Accuracy by Action Sequence Length", 
    'accuracy_token_acc_action')

accuracy_seq_acc_len = list(results['sequence_accuracy_wrt_label_length'].keys())
accuracy_seq_acc_action = [results['sequence_accuracy_wrt_label_length'][len] for len in accuracy_seq_acc_len]
plot_acc(accuracy_seq_acc_len, accuracy_seq_acc_action, 
    "Ground-Truth Action Sequence Length (in words)", 
    '"Accuracy on New Commands (%)"', 
    "Sequence-Level Accuracy by Action Sequence Length", 
    'accuracy_seq_acc_action')

accuracy_token_acc_command_len = list(results['token_accuracy_wrt_input_length'].keys())
accuracy_token_acc_command = [results['token_accuracy_wrt_input_length'][len] for len in accuracy_token_acc_command_len]
plot_acc(accuracy_token_acc_command_len, accuracy_token_acc_command, 
    "Command Length (in words)", 
    '"Accuracy on New Commands (%)"', 
    "Token-Level Accuracy by Action Sequence Length", 
    'accuracy_token_acc_command')

accuracy_seq_acc_command_len = list(results['sequence_accuracy_wrt_input_length'].keys())
accuracy_seq_acc_command = [results['sequence_accuracy_wrt_input_length'][len] for len in accuracy_seq_acc_command_len]
plot_acc(accuracy_seq_acc_command_len, accuracy_seq_acc_command, 
    "Command Length (in words)", 
    '"Accuracy on New Commands (%)"', 
    "Sequence-Level Accuracy by Action Sequence Length", 
    'accuracy_seq_acc_command')