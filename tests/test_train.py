import numpy as np
from src.train import _metrics

def test_metrics_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = _metrics(y, p)
    for k in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert k in m

def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = _metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
