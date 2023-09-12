import argparse
import numpy as np
import pandas as pd
from transformers import AutoConfig, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from datasets import load_dataset, ClassLabel, Features, Value
import glob

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--pretrained_model',
                    help='Pretrained Model', default="models/PetBERT")
parser.add_argument('--batch_size', help='Batch Size', type=int, default=16)
parser.add_argument('--epochs', help='Epochs', type=int, default=100)
parser.add_argument('--ICD', help='Epochs', default="Developmental")
parser.add_argument('--experiment_id', help='experiment_id', default=1)
args = vars(parser.parse_args())

label_datasets = glob.glob("data/binary/*")
latest_model = glob.glob(args['pretrained_model']+"/*")[-1]
for label in label_datasets:
    print(f'Running {label.split("/")[2]}')
    try:
        df_train = label + "/train.csv"
        df_test = label + "/test.csv"

        dataset_features = Features({'text': Value('string'), 'label': ClassLabel(
            names=['Control', 'Case'])})
        datasets = load_dataset('csv', data_files={'train': df_train, 'eval': df_test},
                                delimiter=',',  usecols=['text', 'label'], features=dataset_features, keep_in_memory=True, memory_map=True)

        tokenizer = AutoTokenizer.from_pretrained(latest_model)

        def preprocess_function(examples):
            return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

        tokenized_dataset = datasets.map(
            preprocess_function, batched=True, num_proc=10, remove_columns=['text'])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=label,
            num_train_epochs=args['epochs'],
            per_device_train_batch_size=args['batch_size'],
            per_device_eval_batch_size=args['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            seed=42,
            dataloader_num_workers=10,
            dataloader_pin_memory=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model='f1',
        )

        value = datasets['train'].features['label'].num_classes
        config = AutoConfig.from_pretrained(
            latest_model+"/config.json", num_labels=value, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            latest_model, config=config, local_files_only=True)

        def compute_metrics(p):
            pred, labels = p
            pred = np.argmax(pred, axis=1)
            accuracy = accuracy_score(y_true=labels, y_pred=pred)
            recall = recall_score(y_true=labels, y_pred=pred)
            precision = precision_score(y_true=labels, y_pred=pred)
            f1 = f1_score(y_true=labels, y_pred=pred)
            return {"accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        
        print("Evaluating")
        predictions = trainer.predict(tokenized_dataset["eval"])
        preds = np.argmax(predictions.predictions, axis=-1)

        report = classification_report(
            tokenized_dataset["eval"]["label"], preds, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(label + "/results.csv")
    except Exception as e:
        print(e)
        continue
