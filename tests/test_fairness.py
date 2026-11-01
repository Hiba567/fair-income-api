import numpy as np

from training.fairness import bootstrap_ci, fairness_gaps, group_metrics


def test_group_metrics_known_values():
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    m = group_metrics(y_true, y_pred, groups)
    assert m["A"]["tpr"] == 0.5 and m["A"]["fpr"] == 0.0
    assert m["B"]["tpr"] == 1.0 and m["B"]["fpr"] == 0.5


def test_gaps_are_zero_for_identical_groups():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    groups = ["A", "A", "B", "B"]
    gaps = fairness_gaps(y_true, y_pred, groups)
    assert gaps["equalized_odds_diff"] == 0.0
    assert gaps["demographic_parity_diff"] == 0.0


def test_equalized_odds_is_max_of_tpr_and_fpr_gap():
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
    groups = ["A"] * 4 + ["B"] * 4
    gaps = fairness_gaps(y_true, y_pred, groups)
    assert gaps["tpr_gap"] == 0.5 and gaps["fpr_gap"] == 0.5
    assert gaps["equalized_odds_diff"] == 0.5


def test_bootstrap_ci_is_ordered():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_pred = rng.integers(0, 2, 300)
    groups = rng.choice(["A", "B"], 300)
    ci = bootstrap_ci(y_true, y_pred, groups, n_boot=50)
    assert 0.0 <= ci["low"] <= ci["high"] <= 1.0
