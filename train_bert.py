import torch
import os

from torch.utils.data import DataLoader
from pprint import pprint
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorWithPadding
from transformers import BertForMaskedLM, Trainer, TrainingArguments

from utilities import load_scan_dataset, preprocess_logits_for_metrics, calculate_accuracies
def train(datasets, dataset_idx, oracle=0):
    # Configurations
    gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu else "cpu")
    print(device)
    MAX_LENGTH = 512

    # Tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', force_download=False)
    def tokenize_function(example):
        inputs = dict()
        input_str = []
        output_str = []
        if oracle == 1:
            input_str = [example["input"][id] + " [hint: target sequence length:"+str(len(tokenizer.tokenize(example["output"][id])))+"] "+ 
                         tokenizer.sep_token + (' ' + tokenizer.mask_token) * MAX_LENGTH for id in range(len(example["input"]))]
            output_str = [example["input"][id] +  " [hint: target sequence length:"+str(len(tokenizer.tokenize(example["output"][id])))+"] "+ 
                         tokenizer.sep_token + example["output"][id] + (' ' + tokenizer.sep_token) * MAX_LENGTH for id in range(len(example["input"]))]
        elif oracle == 2:
            for id in range(len(example["input"])):
                mask_tokens = (' ' + tokenizer.mask_token) * len(tokenizer.tokenize(example["output"][id]))
                input_str.append (example["input"][id] + tokenizer.sep_token + mask_tokens + (' ' + tokenizer.sep_token) * MAX_LENGTH)
                output_str.append(example["input"][id] + tokenizer.sep_token + example["output"][id] + (' ' + tokenizer.sep_token) * MAX_LENGTH)
            # print(input_str[0])
            # print(output_str[0])
        else:
            input_str = [example["input"][id] + tokenizer.sep_token + (' ' + tokenizer.mask_token) * MAX_LENGTH for id in range(len(example["input"]))]
            output_str = [example["input"][id] + tokenizer.sep_token + example["output"][id] + (' ' + tokenizer.sep_token) * MAX_LENGTH for id in range(len(example["input"]))]

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
        # print(input_str[0])
        # print(output_str[0])
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
        # print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
        # print(tokenizer.convert_ids_to_tokens(inputs['labels'][0]))

        return inputs
    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=['text', 'input', 'output'])

    # Model
    model = BertForMaskedLM.from_pretrained("bert-base-cased", force_download=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./bert_finetune_{dataset_idx}_oracle",
        eval_strategy="steps",
        learning_rate=7.83075e-05,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=1,
        num_train_epochs=7,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        eval_steps=500,
        report_to="tensorboard",
        # use_legacy_prediction_loop=True,
    )

    # Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=calculate_accuracies,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained(f"./bert_finetuned_{dataset_idx}_oracle")
    tokenizer.save_pretrained(f"./bert_finetuned_{dataset_idx}_oracle")


if __name__ == '__main__':
    train_data = [
        'tasks_train_length.txt'
    ]
    test_data = [
        'tasks_test_length.txt', 
    ]
    dir = './data/length_split/'
    for dataset_idx, (train_data_file, test_data_file) in enumerate(zip(train_data, test_data)):
        # Load dataset
        datasets = load_scan_dataset(
            os.path.join(dir, train_data_file), 
            os.path.join(dir, test_data_file),
            extend=True,
            target_size=16728)
        # oracle=1: using a hint as the oracle
        # oracle=2: Using [mask] as the oracle 
        train(datasets, dataset_idx, oracle=2)
