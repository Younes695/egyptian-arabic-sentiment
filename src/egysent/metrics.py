"""Evaluation helpers shared by the baseline and the fine-tuned models."""

from __future__ import annotations

from collections import Counter

from .data import LABELS


def per_class_f1(gold: list[str], pred: list[str]) -> dict[str, float]:
    scores = {}
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, pred))
        fp = sum(p == label and g != label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def summarize(gold: list[str], pred: list[str]) -> dict:
    """Accuracy, macro-F1 (every class counts equally) and F1 per class."""
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold and pred must be non-empty and the same length")
    f1 = per_class_f1(gold, pred)
    return {
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": sum(f1.values()) / len(f1),
        "f1": f1,
        "support": dict(Counter(gold)),
    }


def confusion(gold: list[str], pred: list[str]) -> list[list[int]]:
    """Rows are the true label, columns the predicted label, both in LABELS order."""
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, pred):
        matrix[index[g]][index[p]] += 1
    return matrix
