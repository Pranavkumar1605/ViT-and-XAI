"""Phase 6: classification metrics computed only from saved predictions, with invariant checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = accuracy_score(y_true, y_pred)
    # Invariants (the base paper's reported accuracies failed these).
    assert cm.sum() == len(y_true), "confusion matrix does not cover all samples"
    assert np.isclose(acc, np.trace(cm) / cm.sum()), "accuracy != trace(CM) / sum(CM)"

    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    true_counts = np.bincount(y_true, minlength=len(labels))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1.mean()),
        "majority_class_accuracy": float(true_counts.max() / true_counts.sum()) if len(y_true) else float("nan"),
        "confusion_matrix": cm,
    }
    for i, name in enumerate(class_names):
        out[f"precision_{name}"] = float(p[i])
        out[f"recall_{name}"] = float(r[i])
        out[f"f1_{name}"] = float(f1[i])
        out[f"support_{name}"] = int(support[i])
        out[f"predicted_{name}"] = int((y_pred == i).sum())
    return out
