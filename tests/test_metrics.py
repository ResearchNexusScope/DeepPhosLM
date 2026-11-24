"""Fast tests for truthfully named evaluation metrics.

Every array here is SYNTHETIC and constructed inline; no result archive,
spreadsheet or dataset CSV is read.  scikit-learn is the oracle for the
"exact agreement" assertions, so no metric value is hard-coded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (  # noqa: E402
    average_precision_metric,
    bootstrap_metric_ci,
    bootstrap_metric_difference,
    compute_metrics,
    roc_auc_metric,
    summarize_folds,
)

LABELS = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
PROBS = np.array([0.05, 0.90, 0.40, 0.55, 0.80, 0.20, 0.95, 0.60, 0.70, 0.10])
THRESHOLD = 0.5


def test_compute_metrics_matches_sklearn_and_never_calls_auc_accuracy() -> None:
    result = compute_metrics(LABELS, PROBS, threshold=THRESHOLD)

    assert result["n"] == LABELS.size
    assert result["n_positive"] == int(LABELS.sum())
    assert result["n_negative"] == int((LABELS == 0).sum())
    expected_predictions = (PROBS >= THRESHOLD).astype(int)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(LABELS, PROBS), rel=0, abs=1e-12)
    assert result["pr_auc"] == pytest.approx(
        average_precision_score(LABELS, PROBS), rel=0, abs=1e-12
    )
    assert result["accuracy"] == pytest.approx(
        accuracy_score(LABELS, expected_predictions), rel=0, abs=1e-12
    )
    assert result["f1"] == pytest.approx(
        f1_score(LABELS, expected_predictions), rel=0, abs=1e-12
    )
    assert result["mcc"] == pytest.approx(
        matthews_corrcoef(LABELS, expected_predictions), rel=0, abs=1e-12
    )
    assert result["precision"] == pytest.approx(
        precision_score(LABELS, expected_predictions), rel=0, abs=1e-12
    )
    assert result["recall"] == pytest.approx(
        recall_score(LABELS, expected_predictions), rel=0, abs=1e-12
    )
    # The legacy manuscript confused these two differently named quantities.
    assert result["roc_auc"] != pytest.approx(result["accuracy"])
    assert isinstance(result["warnings"], list)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metrics_are_none_with_a_warning(label: int) -> None:
    labels = np.full(6, label)
    probabilities = np.array([0.1, 0.3, 0.45, 0.55, 0.7, 0.9])
    result = compute_metrics(labels, probabilities)

    assert result["n"] == 6
    assert result["n_positive"] == (6 if label == 1 else 0)
    assert result["n_negative"] == (6 if label == 0 else 0)
    for name in ("roc_auc", "pr_auc", "f1", "mcc", "precision", "recall"):
        assert result[name] is None, name
    # Accuracy is defined for a single class and is reported under its own name.
    assert result["accuracy"] == pytest.approx(accuracy_score(labels, probabilities >= THRESHOLD))
    assert result["warnings"], "single-class input must be flagged honestly"


def test_per_site_type_grouping_counts_and_missing_y_positives() -> None:
    # S has both classes, T has both classes, Y has negatives only (as in the
    # real dataset, which contains no Y-centred positives).
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1])
    sites = np.array(["S", "S", "S", "T", "T", "Y", "Y", "Y"])

    result = compute_metrics(labels, probabilities, site_types=sites)
    groups = result["per_site_type"]
    assert set(groups) == {"S", "T", "Y"}

    assert groups["S"]["n"] == 3
    assert groups["S"]["n_positive"] == 2
    assert groups["S"]["n_negative"] == 1
    expected_s = compute_metrics(labels[:3], probabilities[:3])
    for name in ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "precision", "recall"):
        assert groups["S"][name] == pytest.approx(expected_s[name])

    assert groups["T"]["n"] == 2
    assert groups["T"]["n_positive"] == 1
    assert groups["T"]["n_negative"] == 1
    assert groups["T"]["roc_auc"] is not None

    assert groups["Y"]["n"] == 3
    assert groups["Y"]["n_positive"] == 0
    assert groups["Y"]["n_negative"] == 3
    assert groups["Y"]["roc_auc"] is None
    assert groups["Y"]["pr_auc"] is None
    assert groups["Y"]["warnings"]

    # An absent site type is still reported, with n=0 and no invented metrics.
    empty = compute_metrics([1, 0], [0.8, 0.2], site_types=np.array(["S", "T"]))
    assert empty["per_site_type"]["Y"]["n"] == 0
    assert empty["per_site_type"]["Y"]["roc_auc"] is None


def test_summarize_folds_keeps_vectors_and_aggregates() -> None:
    fold_a = compute_metrics(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    )
    fold_b = compute_metrics(
        np.array([0, 1, 0, 1]), np.array([0.4, 0.7, 0.3, 0.6])
    )
    # This fold is single-class on purpose: an undefined AUROC must stay None.
    fold_c = compute_metrics(
        np.array([0, 0, 0, 0]), np.array([0.1, 0.4, 0.6, 0.9])
    )

    summary = summarize_folds([fold_a, fold_b, fold_c])
    assert summary["n_folds"] == 3
    assert summary["n_total"] == 12
    assert summary["n_positive_total"] == 4
    assert summary["n_negative_total"] == 8

    auc = summary["metrics"]["roc_auc"]
    assert auc["values"] == [fold_a["roc_auc"], fold_b["roc_auc"], None]
    defined = [fold_a["roc_auc"], fold_b["roc_auc"]]
    assert auc["n_defined"] == 2
    assert auc["mean"] == pytest.approx(float(np.mean(defined)))
    assert auc["std"] == pytest.approx(float(np.std(defined, ddof=0)))

    accuracy = summary["metrics"]["accuracy"]
    assert accuracy["values"] == [fold["accuracy"] for fold in (fold_a, fold_b, fold_c)]
    assert accuracy["n_defined"] == 3
    assert accuracy["mean"] == pytest.approx(
        float(np.mean([fold["accuracy"] for fold in (fold_a, fold_b, fold_c)]))
    )
    assert any("fold 2" in message for message in summary["warnings"])
    assert any("roc_auc" in message for message in summary["warnings"])

    sample_std = summarize_folds([fold_a, fold_b], ddof=1)
    assert sample_std["metrics"]["roc_auc"]["std"] == pytest.approx(
        float(np.std(defined[:2], ddof=1))
    )


def test_bootstrap_ci_is_seeded_and_contains_point_estimate() -> None:
    labels = np.array([0] * 20 + [1] * 20)
    probabilities = np.concatenate(
        [np.linspace(0.01, 0.20, 20), np.linspace(0.80, 0.99, 20)]
    )

    first = bootstrap_metric_ci(
        labels, probabilities, roc_auc_metric, n_boot=64, seed=666
    )
    second = bootstrap_metric_ci(
        labels, probabilities, roc_auc_metric, n_boot=64, seed=666
    )

    assert first == second
    assert first["point_estimate"] == pytest.approx(roc_auc_score(labels, probabilities))
    assert first["n_valid"] == 64
    assert first["ci_low"] <= first["point_estimate"] <= first["ci_high"]
    assert first["ci_level"] == 0.95
    assert len(first["bootstrap_values"]) == 64


def test_paired_bootstrap_difference_for_auroc_and_auprc() -> None:
    labels = np.array([0] * 15 + [1] * 15)
    better = np.concatenate([np.linspace(0.05, 0.25, 15), np.linspace(0.75, 0.95, 15)])
    worse = np.concatenate([np.linspace(0.30, 0.55, 15), np.linspace(0.45, 0.70, 15)])

    for metric_fn, point_fn in (
        (roc_auc_metric, roc_auc_score),
        (average_precision_metric, average_precision_score),
    ):
        result = bootstrap_metric_difference(
            labels, better, worse, metric_fn=metric_fn, n_boot=64, seed=7
        )
        expected = point_fn(labels, better) - point_fn(labels, worse)
        assert result["point_estimate"] == pytest.approx(expected)
        assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_metrics([0, 1], [0.1])
    with pytest.raises(ValueError, match="binary"):
        compute_metrics([0, 2], [0.1, 0.9])
    with pytest.raises(ValueError, match="S/T/Y"):
        compute_metrics([0, 1], [0.1, 0.9], site_types=["S", "Z"])
