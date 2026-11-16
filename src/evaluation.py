"""Classification metrics: accuracy, macro precision/recall/F1 (primary: macro F1),
per-class precision/recall/F1, and confusion matrices."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ["Decreasing", "Stable", "Increasing"]


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS
) -> dict:
    """All metrics for one set of predictions.

    ``zero_division=0``: a class that is never predicted has precision 0 (and, if it
    never occurs, recall 0) instead of raising. The confusion matrix has true labels
    in rows and predicted labels in columns, ordered as ``labels``.
    """
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i, lab in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "true_counts": {lab: int((y_true == lab).sum()) for lab in labels},
        "predicted_counts": {lab: int((y_pred == lab).sum()) for lab in labels},
    }


def select_best(grid_results: list[dict]) -> dict:
    """Pick the grid entry with the highest validation macro F1.

    Deterministic tie-break: smaller ``C`` first, then ``class_weight=None`` before
    ``"balanced"``. Each entry needs ``macro_f1``, ``C`` and ``class_weight``.
    """
    if not grid_results:
        raise ValueError("Empty grid")
    return sorted(
        grid_results,
        key=lambda e: (-e["macro_f1"], e["C"], 0 if e["class_weight"] is None else 1),
    )[0]


def format_metrics_table(results: dict[str, dict]) -> str:
    """Plain-text table of overall metrics per model (``results`` maps name -> metrics)."""
    header = f"{'model':<22}{'macro F1':>10}{'accuracy':>10}{'macro P':>10}{'macro R':>10}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(
            f"{name:<22}{m['macro_f1']:>10.4f}{m['accuracy']:>10.4f}"
            f"{m['macro_precision']:>10.4f}{m['macro_recall']:>10.4f}"
        )
    return "\n".join(lines)


def _macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def paired_bootstrap_macro_f1(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    labels: Sequence[str] = LABELS,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of ``macro_F1(a) - macro_F1(b)`` over evaluation rows.

    The same resampled rows are used for both models. Returns the point difference, a
    95% percentile interval, and the share of resamples where ``a`` beats ``b``. Use on
    validation predictions to judge whether a macro-F1 gap is distinguishable from noise.
    """
    labels = list(labels)
    k = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    yt = np.array([index[v] for v in y_true])
    ya = np.array([index[v] for v in pred_a])
    yb = np.array([index[v] for v in pred_b])
    n = len(yt)

    def macro(truth, pred):
        cm = np.bincount(truth * k + pred, minlength=k * k).reshape(k, k)
        return _macro_f1_from_confusion(cm)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = macro(yt[idx], ya[idx]) - macro(yt[idx], yb[idx])
    return {
        "point_difference": macro(yt, ya) - macro(yt, yb),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "share_resamples_a_better": float(np.mean(diffs > 0)),
        "n_boot": n_boot,
        "seed": seed,
    }
