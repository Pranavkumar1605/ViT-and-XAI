import numpy as np

from vitxai.eval.metrics import classification_metrics

NAMES = ["Hold", "Buy", "Sell"]


def test_accuracy_equals_confusion_diagonal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 1000)
    p = rng.integers(0, 3, 1000)
    m = classification_metrics(y, p, NAMES)
    cm = m["confusion_matrix"]
    assert cm.sum() == 1000
    assert np.isclose(m["accuracy"], np.trace(cm) / cm.sum())


def test_majority_baseline_exposes_always_hold():
    y = np.array([0] * 80 + [1] * 10 + [2] * 10)
    p = np.zeros_like(y)
    m = classification_metrics(y, p, NAMES)
    assert m["accuracy"] == m["majority_class_accuracy"] == 0.8
    assert m["recall_Buy"] == m["recall_Sell"] == 0.0
    assert np.isclose(m["macro_f1"], (2 * 0.8 / 1.8) / 3)
