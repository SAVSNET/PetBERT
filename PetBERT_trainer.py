from datasets import load_dataset, ClassLabel
from transformers import BertForNextSentencePrediction, BertTokenizerFast, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import argparse

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument(
    "--base_model",
    default="bert-base-uncased",
)
parser.add_argument(
    "--train_set",
    help="Pretrained Model",
    default="data/train.csv",
)
parser.add_argument(
    "--eval_set",
    help="Pretrained Model",
    default="data/eval.csv",
)

args = vars(parser.parse_args())

datasets = load_dataset('csv', data_files={'train': args['train_set'],
                                           'eval': args['eval_set']}, )


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


tokenizer = BertTokenizerFast.from_pretrained(
    args['base_model'], use_fast=True)
tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=20, remove_columns=["text"])

model = BertForNextSentencePrediction.from_pretrained(args['base_model'])


training_args = TrainingArguments(
    output_dir="models/PetBERT",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=20,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_strategy="epoch",
    save_total_limit=5
)

# Modify your data collator to include NSP labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.25)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
)

trainer.train()
