from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
)
import numpy as np
import argparse
from utils.training import compute_metrics, calculate_class_weights, multi_label_metrics


parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument(
    "--pretrained_model",
    help="Pretrained Model",
    default="SAVSNET/PetBERT",
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


parser.add_argument(
    "--test_set",
    help="Pretrained Model",
    default="data/test.csv",
)

parser.add_argument("--model_name", help="Pretrained Model", default="pretrain_petBERT")
parser.add_argument("--base_model", help="base_model", default="pretrain_petBERT")
args = vars(parser.parse_args())

model_name = args["pretrained_model"]

ds = load_dataset(
    "csv",
    data_files={
        "train": args["train_set"],
        "eval": args["train_set"],
        "test": args["test_set"],
    },
)


# create labels column
columns = ds["train"].column_names
ds = ds.map(
    lambda x: {
        "labels": [x[c] for c in columns if c not in ["Savsnet_Consult_Id", "Text"]]
    }
)

labels_id = [
    label for label in columns if label not in ["Savsnet_Consult_Id", "Text", "labels"]
]
id2label = {idx: label for idx, label in enumerate(labels_id)}
label2id = {label: idx for idx, label in enumerate(labels_id)}

tokenizer = AutoTokenizer.from_pretrained(
    args["base_model"], problem_type="multi_label_classification"
)


def tokenize_and_encode(examples):
    return tokenizer(
        examples["Text"], truncation=True, padding="max_length", max_length=512
    )


df_enc = ds.map(
    tokenize_and_encode, batched=True, remove_columns=columns[0 : int(len(columns))]
)

df_enc.set_format("torch")
df_enc = df_enc.map(
    lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"]
).rename_column("float_labels", "labels")


model = AutoModelForSequenceClassification.from_pretrained(
    args["pretrained_model"],
    num_labels=len(labels_id),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
).to("cuda")


args = TrainingArguments(
    output_dir=f"models/'args['model_name']",  # output directory
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./m  logs/" + model_name,  # directory for storing logs
    logging_steps=1000, 
    save_strategy="epoch",
    evaluation_strategy="epoch",
    dataloader_num_workers=20,
    dataloader_pin_memory=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(
            weight=calculate_class_weights(ds["train"], labels_id),
            reduction="mean",
        ).to("cuda")
        print(loss_fct.weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=df_enc["train"],
    eval_dataset=df_enc["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

if args['test_set'] != None:
    def predict_labels(trainer, dataset, threshold=0.5):
        predictions = trainer.predict(dataset)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions.predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = df_enc["test"]["labels"].numpy()
        return y_pred, y_true


    y_pred, y_true = predict_labels(trainer, df_enc["test"], threshold=0.5)

    report = classification_report(y_true, y_pred, target_names=labels_id, output_dict=True)


    df = pd.DataFrame(report).transpose()
    df.to_csv("models/" + str(model_name) + "/report.csv")
