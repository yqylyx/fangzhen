from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    objective: str
    metrics: dict[str, Any]


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float('nan')
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    counts = confusion_counts(y_true, y_pred)

    tp = counts['tp']
    tn = counts['tn']
    fp = counts['fp']
    fn = counts['fn']
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / max(1, y_true.shape[0])
    pred_boundary_count = int(y_pred.sum())
    real_boundary_count = int(y_true.sum())

    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'pr_auc': float(pr_auc_np(y_true, y_prob)),
        'roc_auc': float(roc_auc_np(y_true, y_prob)),
        'real_boundary_count': real_boundary_count,
        'pred_boundary_count': pred_boundary_count,
        'count_error': int(pred_boundary_count - real_boundary_count),
        **counts,
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


def build_threshold_grid(min_value: float = 0.05, max_value: float = 0.95, num_steps: int = 181) -> np.ndarray:
    min_value = float(min_value)
    max_value = float(max_value)
    num_steps = max(2, int(num_steps))
    return np.linspace(min_value, max_value, num_steps, dtype=np.float64)


def select_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = 'f1',
    threshold_grid: np.ndarray | None = None,
) -> ThresholdSelection:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    grid = threshold_grid if threshold_grid is not None else build_threshold_grid()
    objective = str(objective).strip().lower()
    if objective not in {'f1', 'recall'}:
        raise ValueError(f'unsupported threshold objective: {objective}')

    best_metrics: dict[str, Any] | None = None
    best_threshold = float(grid[0])
    best_score = -float('inf')
    best_tiebreak = -float('inf')

    for threshold in grid:
        metrics = compute_binary_metrics(y_true, y_prob, float(threshold))
        score = float(metrics[objective])
        tiebreak = float(metrics['f1'] if objective == 'recall' else metrics['recall'])
        if score > best_score or (np.isclose(score, best_score) and tiebreak > best_tiebreak):
            best_score = score
            best_tiebreak = tiebreak
            best_metrics = metrics
            best_threshold = float(threshold)

    if best_metrics is None:
        best_metrics = compute_binary_metrics(y_true, y_prob, 0.5)
        best_threshold = 0.5
    return ThresholdSelection(threshold=best_threshold, objective=objective, metrics=best_metrics)
