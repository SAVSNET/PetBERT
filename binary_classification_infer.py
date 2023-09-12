
import argparse
import numpy as np
import pandas as pd
from transformers import AutoConfig, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, ClassLabel, Features, Value
import glob


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--base_model', help='Pretrained Model',
                    default="bert-base-uncased")
parser.add_argument('--pretrained_model', help='Custom Model',
                    default="save_none_newset/bert-base-uncased-finetuned-Vet/checkpoint-664281")
parser.add_argument('--batch_size', help='Batch Size', type=int, default=16)
parser.add_argument('--epochs', help='Epochs', type=int, default=1)
parser.add_argument('--ICD', help='Epochs', default="Developmental")
parser.add_argument('--experiment_id', help='experiment_id', default=1)
args = vars(parser.parse_args())


df_train = "data/PetBERT_ICD/raw/train.csv"
df_test = "data/PetBERT_ICD/raw/test.csv"

label_list = glob.glob("data/binary/*")

for label in label_list:
    latest_check = glob.glob(label+"/*")[-1]
    print(f'Running {label.split("/")[2]}')
    tokenizer = AutoTokenizer.from_pretrained(latest_check)

    dataset = load_dataset('csv', data_files={'train': df_train, 'eval': df_test},
                           delimiter=',',  usecols=['text', 'label'], keep_in_memory=True, memory_map=True)

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

    dataset = dataset.map(preprocess_function, batched=True, num_proc=10)

    config = AutoConfig.from_pretrained(args['pretrained_model']+"/config.json",
                                        num_labels=dataset.features['label'].num_classes, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args['pretrained_model'], config=config, local_files_only=True)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    train_predictions = trainer.predict(dataset['train'])
    eval_predictions = trainer.predict(dataset['eval'])

    train_pred = train_predictions.predictions.argmax(axis=1)

    df_train = dataset['train'].to_pandas()
    df_train['predicted_label'] = train_predictions.predictions.argmax(axis=1)
    df_train.to_csv(label.split("/")[2] + "multi_label_train.csv", index=False)

    df_test = dataset['eval'].to_pandas()
    df_test['predicted_label'] = eval_predictions.predictions.argmax(axis=1)
    df_test.to_csv(label.split("/")[2] + "multi_label_test.csv", index=False)
