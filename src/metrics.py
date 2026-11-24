"""Truthful, unambiguous evaluation metrics for binary phosphosite prediction.

This module is the single source of truth for reporting numbers derived from
the test-prediction archives written by :mod:`src.train_model`.  Names say what
they mean: ``roc_auc`` is the area under the ROC curve, ``pr_auc`` is average
precision (area under the precision-recall curve), and ``accuracy`` is the
fraction of thresholded predictions that are correct.  The legacy manuscript
called ROC-AUC "accuracy" (its micro-AUC value was stored in a misnamed
variable); that conflation is deliberately not reproduced here.

Honesty rules
-------------
* A metric that is mathematically undefined is ``None`` -- never an imputed
  0.5, 0.0 or 1.0.  ROC-AUC and PR-AUC require both classes; MCC, F1, precision
  and recall are degenerate for a single-class input and are likewise None.
  Each such decision is recorded in the result's ``warnings`` list.
* ``accuracy`` is always reported under that exact name and is never used as a
  stand-in for a discrimination metric.  It is class-prior dominated and the
  caller is warned when the input is single-class.
* Per-site-type breakdowns are keyed by S/T/Y.  A group with only one class
  (for example a Y group with no positives in a small or toy dataset) returns
  ``None`` for its undefined positive-class metrics; nothing is synthesised
  to make those numbers appear.  The real dataset does contain Y-centred
  positives, so its Y metrics are defined.

Only NumPy and scikit-learn are used.  There are no hard-coded metric values.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

SITE_TYPES = ("S", "T", "Y")
DEFAULT_THRESHOLD = 0.5
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 666
CI_LEVEL = 0.95
_CI_PERCENTILES = (100.0 * (1.0 - CI_LEVEL) / 2.0, 100.0 * (1.0 + CI_LEVEL) / 2.0)
_NON_SCALAR_KEYS = frozenset({"warnings", "per_site_type"})


def _as_labels(y_true: Sequence[Any]) -> np.ndarray:
    """Validate a 1-D vector of binary labels and return it as int64."""
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {labels.shape}")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return labels.astype(np.int64)


def _as_probabilities(y_prob: Sequence[float], n: int) -> np.ndarray:
    """Validate a 1-D vector of probabilities aligned with ``y_true``."""
    probabilities = np.asarray(y_prob, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError(f"y_prob must be 1-D, got shape {probabilities.shape}")
    if probabilities.size != n:
        raise ValueError(
            f"y_prob has length {probabilities.size} but y_true has length {n}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_prob must contain only finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("y_prob must contain probabilities in [0, 1]")
    return probabilities


def _as_site_types(site_types: Sequence[Any], n: int) -> np.ndarray:
    """Validate a 1-D vector of S/T/Y centre residues aligned with labels."""
    centres = np.asarray([str(value).strip().upper() for value in site_types])
    if centres.ndim != 1 or centres.size != n:
        raise ValueError(
            f"site_types has length {centres.size} but y_true has length {n}"
        )
    unexpected = set(centres.tolist()) - set(SITE_TYPES)
    if unexpected:
        raise ValueError(f"site_types must contain only S/T/Y, got {sorted(unexpected)}")
    return centres


def _metric_block(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Compute the scalar metrics for one validated group of rows."""
    n = int(y_true.size)
    n_positive = int(np.count_nonzero(y_true == 1))
    n_negative = int(np.count_nonzero(y_true == 0))
    warnings: List[str] = []
    result: Dict[str, Any] = {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "threshold": float(threshold),
        "roc_auc": None,
        "pr_auc": None,
        "accuracy": None,
        "f1": None,
        "mcc": None,
        "precision": None,
        "recall": None,
        "warnings": warnings,
    }

    if n == 0:
        warnings.append("empty input: every metric is undefined")
        return result

    y_pred = (y_prob >= threshold).astype(np.int64)
    has_both_classes = n_positive > 0 and n_negative > 0

    # Accuracy is defined whenever there is at least one row: keep it, but say
    # plainly that it is a thresholded accuracy, not an AUC.
    result["accuracy"] = float(accuracy_score(y_true, y_pred))
    if not has_both_classes:
        warnings.append(
            f"single-class input (n_positive={n_positive}, n_negative={n_negative}): "
            "roc_auc, pr_auc, f1, mcc, precision and recall are undefined; reported "
            "as None rather than imputed"
        )
        warnings.append(
            "accuracy is class-prior dominated on single-class input and must not "
            "be reported as a measure of discrimination"
        )
        return result

    result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    result["pr_auc"] = float(average_precision_score(y_true, y_prob))
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    result["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    if not np.any(y_pred == 1):
        warnings.append(
            "no predicted positives at this threshold: precision and F1 follow "
            "sklearn's zero_division=0 convention and are not informative"
        )
    return result


def compute_metrics(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    site_types: Optional[Sequence[Any]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Compute honestly named binary classification metrics.

    Parameters
    ----------
    y_true:
        Binary labels (0 = non-site, 1 = site), one per row.
    y_prob:
        Positive-class probabilities in ``[0, 1]``, aligned with ``y_true``.
    site_types:
        Optional S/T/Y centre residues, one per row.  When supplied, the
        result also contains ``per_site_type`` with S, T and Y entries (even
        when a group is empty).  Positive-class metrics are ``None`` for a
        single-class group (e.g. a Y group with no positives in a small or toy
        dataset); the real dataset has Y-centred positives, so Y metrics are
        defined there.
    threshold:
        Probability cutoff used only for the thresholded metrics (accuracy,
        F1, MCC, precision, recall).  ROC-AUC and PR-AUC are threshold-free.

    Returns
    -------
    dict
        ``n``, ``n_positive``, ``n_negative``, ``roc_auc``, ``pr_auc``,
        ``accuracy``, ``f1``, ``mcc``, ``precision``, ``recall``,
        ``threshold`` and ``warnings``; plus ``per_site_type`` when requested.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold!r}")
    labels = _as_labels(y_true)
    probabilities = _as_probabilities(y_prob, labels.size)
    result = _metric_block(labels, probabilities, threshold)
    if site_types is not None:
        centres = _as_site_types(site_types, labels.size)
        result["per_site_type"] = {
            site: _metric_block(labels[centres == site], probabilities[centres == site], threshold)
            for site in SITE_TYPES
        }
    return result


def summarize_folds(
    fold_metrics: Iterable[Mapping[str, Any]], *, ddof: int = 0
) -> Dict[str, Any]:
    """Aggregate per-fold metric dicts without discarding their spread.

    Every scalar metric keeps its full ``values`` vector (one entry per fold,
    including ``None`` for folds where the metric was undefined).  ``mean`` and
    ``std`` are computed over the defined entries only, and ``n_defined`` says
    how many there were.  ``std`` uses population normalization by default;
    pass ``ddof=1`` for the sample standard deviation.
    """
    folds = list(fold_metrics)
    if not folds:
        raise ValueError("fold_metrics must contain at least one fold")
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof!r}")
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise TypeError(f"fold_metrics[{index}] must be a mapping, got {type(fold).__name__}")

    scalar_keys: List[str] = []
    for fold in folds:
        for key, value in fold.items():
            if key in _NON_SCALAR_KEYS or key in scalar_keys:
                continue
            if isinstance(value, (bool, np.bool_)):
                continue
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            scalar_keys.append(key)

    warnings: List[str] = []
    metrics: Dict[str, Dict[str, Any]] = {}
    for key in scalar_keys:
        values = [fold.get(key) for fold in folds]
        defined = [float(value) for value in values if value is not None]
        entry: Dict[str, Any] = {
            "values": values,
            "n_defined": len(defined),
            "mean": None,
            "std": None,
        }
        if defined:
            array = np.asarray(defined, dtype=float)
            entry["mean"] = float(array.mean())
            if len(defined) > ddof:
                entry["std"] = float(array.std(ddof=ddof))
            else:
                warnings.append(
                    f"{key}: std undefined with n_defined={len(defined)} and ddof={ddof}"
                )
            if len(defined) != len(values):
                warnings.append(
                    f"{key}: {len(values) - len(defined)} of {len(values)} folds undefined; "
                    "mean/std use only defined values"
                )
        else:
            warnings.append(f"{key}: no defined values across folds; mean/std are None")
        metrics[key] = entry

    def _total(name: str) -> Optional[int]:
        if name not in metrics:
            return None
        values = metrics[name]["values"]
        if any(value is None for value in values):
            warnings.append(f"{name}: at least one fold has no value; total is None")
            return None
        return int(sum(int(value) for value in values))

    for index, fold in enumerate(folds):
        for message in fold.get("warnings", ()):
            warnings.append(f"fold {index}: {message}")

    return {
        "n_folds": len(folds),
        "n_total": _total("n"),
        "n_positive_total": _total("n_positive"),
        "n_negative_total": _total("n_negative"),
        "metrics": metrics,
        "ddof": int(ddof),
        "warnings": warnings,
    }


def roc_auc_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC as a ``metric_fn`` for the bootstrap helpers."""
    return float(roc_auc_score(y_true, y_prob))


def average_precision_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUPRC (average precision) as a ``metric_fn`` for the bootstrap helpers."""
    return float(average_precision_score(y_true, y_prob))


def _resample_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stratified bootstrap indices so each replicate keeps both classes."""
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    if positives.size and negatives.size:
        return np.concatenate(
            (
                rng.choice(positives, size=positives.size, replace=True),
                rng.choice(negatives, size=negatives.size, replace=True),
            )
        )
    return rng.choice(y_true.size, size=y_true.size, replace=True)


def _percentile_ci(values: List[float]) -> tuple:
    """Return the (low, high) 95% percentile interval, or (None, None).

    The percentile bootstrap CI is an approximation: with a finite number of
    resamples its coverage is not exact, and no BCa correction is applied.
    """
    if not values:
        return None, None
    low, high = np.percentile(np.asarray(values, dtype=float), _CI_PERCENTILES)
    return float(low), float(high)


def bootstrap_metric_ci(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Point estimate and 95% percentile bootstrap CI for one metric.

    ``metric_fn(y_true_subset, y_prob_subset) -> float`` is called on the full
    data and on ``n_boot`` stratified with-replacement resamples (class sizes
    preserved, so AUROC/AUPRC replicates stay defined).  The RNG is seeded
    locally with ``numpy.random.default_rng(seed)``; repeated calls with the
    same seed return identical results.  Undefined replicates are excluded and
    counted in ``warnings`` rather than imputed.  The percentile bootstrap CI
    is an approximation: with a finite number of resamples its coverage is not
    exact, and no BCa correction is applied.
    """
    labels = _as_labels(y_true)
    probabilities = _as_probabilities(y_prob, labels.size)
    if labels.size == 0:
        raise ValueError("cannot bootstrap an empty input")
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot!r}")
    try:
        point = float(metric_fn(labels, probabilities))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric_fn is undefined for these inputs: {exc}") from exc
    if not np.isfinite(point):
        raise ValueError("metric_fn must return a finite point estimate")

    rng = np.random.default_rng(seed)
    values: List[float] = []
    skipped = 0
    for _ in range(n_boot):
        indices = _resample_indices(labels, rng)
        try:
            value = float(metric_fn(labels[indices], probabilities[indices]))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if np.isfinite(value):
            values.append(value)
        else:
            skipped += 1

    ci_low, ci_high = _percentile_ci(values)
    warnings: List[str] = []
    if skipped:
        warnings.append(
            f"{skipped} of {n_boot} replicates were undefined and excluded from the CI"
        )
    return {
        "metric_name": getattr(metric_fn, "__name__", "metric_fn"),
        "point_estimate": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": CI_LEVEL,
        "n_boot": int(n_boot),
        "n_valid": len(values),
        "seed": int(seed),
        "bootstrap_values": values,
        "warnings": warnings,
    }


def bootstrap_metric_difference(
    y_true: Sequence[Any],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_metric,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Paired bootstrap CI for ``metric_fn(model_a) - metric_fn(model_b)``.

    The same resampled indices are applied to both models on every replicate,
    which is the paired comparison appropriate for two models scored on the
    same test rows.  Pass ``metric_fn=average_precision_metric`` for an AUPRC
    difference; the default is AUROC.
    """
    labels = _as_labels(y_true)
    first = _as_probabilities(prob_a, labels.size)
    second = _as_probabilities(prob_b, labels.size)
    if labels.size == 0:
        raise ValueError("cannot bootstrap an empty input")
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot!r}")
    try:
        point_a = float(metric_fn(labels, first))
        point_b = float(metric_fn(labels, second))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric_fn is undefined for these inputs: {exc}") from exc
    if not (np.isfinite(point_a) and np.isfinite(point_b)):
        raise ValueError("metric_fn must return finite point estimates")

    rng = np.random.default_rng(seed)
    differences: List[float] = []
    skipped = 0
    for _ in range(n_boot):
        indices = _resample_indices(labels, rng)
        try:
            value_a = float(metric_fn(labels[indices], first[indices]))
            value_b = float(metric_fn(labels[indices], second[indices]))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if np.isfinite(value_a) and np.isfinite(value_b):
            differences.append(value_a - value_b)
        else:
            skipped += 1

    ci_low, ci_high = _percentile_ci(differences)
    warnings: List[str] = []
    if skipped:
        warnings.append(
            f"{skipped} of {n_boot} replicates were undefined and excluded from the CI"
        )
    return {
        "metric_name": getattr(metric_fn, "__name__", "metric_fn"),
        "point_estimate": point_a - point_b,
        "metric_a_point": point_a,
        "metric_b_point": point_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": CI_LEVEL,
        "n_boot": int(n_boot),
        "n_valid": len(differences),
        "seed": int(seed),
        "bootstrap_differences": differences,
        "warnings": warnings,
    }
