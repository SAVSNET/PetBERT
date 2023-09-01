from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback, EvalPrediction
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = preprocessing.LabelEncoder()
model_name ="pretrain_petBERT/checkpoint-156250"

ds = load_dataset('csv', data_files={"test": "ICD_11 experiment_test_new.csv"},)

# create labels column
columns = ds["test"].column_names
cols = columns[0:24]
ds = ds.map(lambda x : {"labels": [x[c] for c in cols if c not in ['savsnet_consult_id', 'consult_record_date', 'item_text',]]})
labels_id = [label for label in ds['test'].features.keys() if label not in ['savsnet_consult_id', 'consult_record_date', 'item_text', 'labels']]
id2label = {idx:label for idx, label in enumerate(labels_id)}
label2id = {label:idx for idx, label in enumerate(labels_id)}
tokenizer = AutoTokenizer.from_pretrained(model_name, problem_type="multi_label_classification")

def tokenize_and_encode(examples):
  return tokenizer(examples["item_text"], truncation=True)

cols = columns[0:38]
ds_enc = ds.map(tokenize_and_encode, batched=True, remove_columns=cols)

# cast label IDs to floats
ds_enc.set_format("torch")
ds_enc = (ds_enc
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))

model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=len(labels_id),
                                                           problem_type="multi_label_classification",
                                                           id2label=id2label,
                                                           label2id=label2id).to('cuda')

args = TrainingArguments(
            output_dir='Datasets/ICD/'+model_name,             # output directory
            num_train_epochs=8,                             # total number of training epochs
            per_device_train_batch_size=32,                 # batch size per device during training
            per_device_eval_batch_size=32,                  # batch size for evaluation
            warmup_steps=500,                               # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                              # strength of weight decay
            logging_dir='./m  logs/'+model_name,            # directory for storing logs
            logging_steps=1000,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            dataloader_num_workers=20,
            dataloader_pin_memory=True,
            save_total_limit=3,
            load_best_model_at_end = True,
            metric_for_best_model = "f1",
        )

def multi_label_metrics(predictions, labels, threshold=0.80):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=ds_enc["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    
)

#trainer.train()

trainer.evaluate()
     
predictions = trainer.predict(ds_enc['test'])
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions.predictions))
y_pred = np.zeros(probs.shape)
y_pred[np.where(probs >= 0.80)] = 1
y_true = np.array(ds_enc['test']['labels'])

report = (classification_report(y_true,y_pred, target_names=labels_id,output_dict=True))


df = pd.DataFrame(report).transpose()
df.to_csv("pretrain_petBERT/new_bert.csv")

# torch_logits = torch.from_numpy(predictions.predictions).to(torch.float32)
# sigmoid = torch.nn.Sigmoid()
# probs = sigmoid(torch_logits.squeeze().cpu())

# datasets = ds['test'].to_pandas()
# #Expand the array to individual columns
# df_results = pd.DataFrame(y_pred, columns=labels_id)
# out = datasets.merge(df_results, left_index=True, right_index=True)
# out.to_csv('results_testset/final/deberta/ep_v5_3+80.csv')