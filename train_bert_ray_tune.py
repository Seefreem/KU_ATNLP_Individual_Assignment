import torch

from torch.utils.data import DataLoader
from pprint import pprint
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorWithPadding
from transformers import BertForMaskedLM, Trainer, TrainingArguments

import ray
import ray.train as ray_train
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch


from utilities import load_scan_dataset, preprocess_logits_for_metrics, calculate_accuracies

# Configurations
gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu else "cpu")
print(device)
MAX_LENGTH = 512

# Load dataset
datasets = load_scan_dataset('data/simple_split/tasks_train_simple.txt', 
    'data/simple_split/tasks_test_simple.txt')

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', force_download=False)
def tokenize_function(example):
    inputs = dict()
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
tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=['text', 'input', 'output'])


def sweep_func(config=None):
    lr = config['lr']
    bs = int(config['batch_size'])
    warmup_steps = int(config['warming_up'])
    epochs = int(config['epochs'])
    
    # Model
    model = BertForMaskedLM.from_pretrained("bert-base-cased", force_download=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bert_finetune",
        eval_strategy="steps",
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=1,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        eval_steps=500,
        report_to="tensorboard",
        warmup_steps=0,
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
    model.save_pretrained("./bert_finetuned")
    tokenizer.save_pretrained("./bert_finetuned")

search_space = {
    "lr": tune.loguniform(1e-6, 1e-4),
    "warming_up": tune.uniform(1, 2),
    "epochs": tune.uniform(1, 10),
    'batch_size': tune.uniform(10, 20)
}
algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}, 
                        metric='token_wise_accuracy', mode='max')
algo = ConcurrencyLimiter(algo, max_concurrent=1)
result = tune.run(
    sweep_func,
    resources_per_trial={"cpu": 10, "gpu": 1},
    config=search_space,
    num_samples=20,
    search_alg=algo)


# bayesopt = BayesOptSearch(search_space, metric="token_wise_accuracy", mode="max")
# tuner = tune.Tuner(
#     sweep_func,
#     tune_config=tune.TuneConfig(
#         search_alg=bayesopt,
#     ),
# )
# result = tuner.fit()

print('=> The results of hyperparameter tuning')
print(result.dataframe())
print(result.get_best_config('token_wise_accuracy', 'max'))
print()