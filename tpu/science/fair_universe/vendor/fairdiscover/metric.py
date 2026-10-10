"""Faithful transcription of official Phase 2 score.py scoring formula."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve


def low_fpr_tpr(labels, scores):
    """Higher scores mean *more* anomalous. Returns official score in [0,1]."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be equal-length 1D arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("All anomaly scores must be finite")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Need both InD=0 and OoD=1")
    fpr, tpr, _ = roc_curve(labels, scores)
    grid = np.logspace(np.log10(.001), np.log10(.05), 100)
    curve = np.interp(grid, fpr, tpr)
    return float(curve.mean())


def per_family_scores(ind_scores, by_family):
    """Score family by family with the same InD reference distribution."""
    ind_scores = np.asarray(ind_scores)
    result = {}
    for name, ood in by_family.items():
        ood = np.asarray(ood)
        y = np.concatenate([np.zeros(len(ind_scores), dtype=np.int32),
                            np.ones(len(ood), dtype=np.int32)])
        s = np.concatenate([ind_scores, ood])
        result[name] = low_fpr_tpr(y, s)
    return result
