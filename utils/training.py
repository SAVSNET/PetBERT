import torch


from transformers import (
    EvalPrediction,
)
import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)


def multi_label_metrics(predictions, labels, threshold=0.80):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


# iterate over each column and calculate the weight based on max value in column
def calculate_class_weights(train_dataset, label_cols):
    data = train_dataset.to_pandas()
    total_samples = len(data)
    class_weights = []
    for col in label_cols:
        class_frequency = data[col].sum()
        weight = total_samples / (len(label_cols) * class_frequency)
        if weight < 1:
            weight = 1
        class_weights.append(weight)
    return torch.tensor(class_weights, dtype=torch.float32)