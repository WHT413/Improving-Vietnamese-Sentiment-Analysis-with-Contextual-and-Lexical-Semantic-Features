"""Accuracy and macro-F1 computed with plain torch."""

import torch

_NUM_CLASSES = 3


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    if preds.numel() == 0 or labels.numel() == 0:
        raise ValueError("preds and labels must not be empty")

    f1_per_class = []
    for label in range(_NUM_CLASSES):
        predicted = preds == label
        actual = labels == label
        true_positive = (predicted & actual).sum().item()
        false_positive = (predicted & ~actual).sum().item()
        false_negative = (~predicted & actual).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_per_class.append(
            2 * true_positive / denominator if denominator else 0.0
        )

    return {
        "accuracy": (preds == labels).sum().item() / labels.numel(),
        "macro_f1": sum(f1_per_class) / _NUM_CLASSES,
        "f1_per_class": tuple(f1_per_class),
    }
