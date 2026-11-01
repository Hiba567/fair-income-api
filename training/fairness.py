"""Group fairness metrics, written from scratch (no external fairness library).

All functions take binary labels/predictions (0/1) and a group label per sample.
"""

from __future__ import annotations

import numpy as np


def _rate(numerator: int, denominator: int) -> float | None:
    """Safe division: returns None when the denominator is zero."""
    return None if denominator == 0 else numerator / denominator


def group_metrics(y_true, y_pred, groups) -> dict:
    """Per-group confusion-matrix rates.

    Returns {group: {n, selection_rate, tpr, fpr, accuracy}}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    results = {}
    for g in np.unique(groups):
        mask = groups == g
        t, p = y_true[mask], y_pred[mask]
        tp = int(((t == 1) & (p == 1)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        pos = int((t == 1).sum())
        neg = int((t == 0).sum())
        results[str(g)] = {
            "n": int(mask.sum()),
            "selection_rate": _rate(int(p.sum()), int(mask.sum())),
            "tpr": _rate(tp, pos),  # true positive rate (recall)
            "fpr": _rate(fp, neg),  # false positive rate
            "accuracy": _rate(int((t == p).sum()), int(mask.sum())),
        }
    return results


def _max_gap(per_group: dict, key: str) -> float:
    values = [m[key] for m in per_group.values() if m[key] is not None]
    return float(max(values) - min(values)) if len(values) >= 2 else 0.0


def fairness_gaps(y_true, y_pred, groups) -> dict:
    """Demographic parity difference and equalized odds difference.

    - demographic_parity_diff: largest gap in selection rate between groups
    - equalized_odds_diff: max of the largest TPR gap and the largest FPR gap
    """
    per_group = group_metrics(y_true, y_pred, groups)
    tpr_gap = _max_gap(per_group, "tpr")
    fpr_gap = _max_gap(per_group, "fpr")
    return {
        "demographic_parity_diff": _max_gap(per_group, "selection_rate"),
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_diff": max(tpr_gap, fpr_gap),
    }


def bootstrap_ci(y_true, y_pred, groups, metric: str = "equalized_odds_diff",
                 n_boot: int = 200, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap confidence interval for a fairness gap."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(fairness_gaps(y_true[idx], y_pred[idx], groups[idx])[metric])
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "low": float(low), "high": float(high), "n_boot": n_boot}
