from transformers import AutoConfig, Trainer, TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments
import pandas as pd
from datasets import load_dataset
import numpy as np
import torch.nn.functional as nnf
import torch
import argparse
import glob

argparser = argparse.ArgumentParser(description='Description of your program')
argparser.add_argument(
    '--model_pathway', help='Pretrained Model', default='models/PetBERT-ICD')
argparser.add_argument('--data', help='Pretrained Model',
                       default='data/PetBERT_ICD/raw/full_dataset.csv')
argparser.add_argument('--threshold', help='Threshold', default=0.8)
args = vars(argparser.parse_args())

latest_checkpoint = glob.glob(args['model_pathway']+"/*")[-1]

tokenizer = AutoTokenizer.from_pretrained(
    latest_checkpoint, padding=True, truncation=True, max_length=512, problem_type="multi_label_classification")
model = AutoModelForSequenceClassification.from_pretrained(str(
    latest_checkpoint), local_files_only=True, problem_type="multi_label_classification", num_labels=20)
config = AutoConfig.from_pretrained(latest_checkpoint+'/config.json', local_files_only=True)

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)


datasets = load_dataset(
    'csv', data_files={"Test": args['data']}, delimiter=',', split='Test[:100%]')


def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)


tokenized_dataset = datasets.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="ICD_10_models/inferences/",
    per_device_eval_batch_size=32,
    dataloader_num_workers=10,
    dataloader_pin_memory=True,
    jit_mode_eval=True,
    bf16=True,
)

label_dictionary = {v: k for k, v in config.label2id.items()}


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

predictions = trainer.predict(tokenized_dataset)

# labels = np.argmax(predictions.predictions, axis=-1)
torch_logits = torch.from_numpy(predictions.predictions).to(torch.float32)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch_logits.squeeze().cpu())

datasets = datasets.to_pandas()

results = []
threshold_val = float(args['threshold'])
for narrative in probs.numpy():
    value = np.where(narrative >= threshold_val)[0]
    labels = [label_dictionary[i] for i in value]
    results.append(labels)


predictions_array = np.zeros(predictions.predictions.shape)
predictions_array[np.where(predictions.predictions >= args['threshold'])] = 1
full_data = pd.DataFrame(predictions_array)
full_data = full_data.rename(columns=dict(label_dictionary))
full_data["ICD_11"] = results
full_data = pd.concat([datasets, full_data], axis=1)
full_data.to_csv("hpcia_datasets/test_new_icd.csv")
