"""Evaluation: accuracy, macro-F1, and the confusion matrix."""

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from minilab.data import prepare_batch


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(prepare_batch(x, device, augment=False))
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def evaluate(model, loader, device) -> dict:
    y_true, y_pred = predict(model, loader, device)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion": confusion_matrix(y_true, y_pred),
    }
