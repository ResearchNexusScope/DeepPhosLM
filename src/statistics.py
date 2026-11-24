"""Statistical caveats made explicit: BCa paired intervals and bin-level calibration.

This module adds two inferential/descriptive helpers on top of the truthful
metric implementations in :mod:`src.metrics`.  It contains **no reported result
values**: every number returned here is computed from the arrays passed in by
the caller.

What is implemented
-------------------
* :func:`bca_interval` -- paired bootstrap of
  ``metric_fn(y_true, prob_a) - metric_fn(y_true, prob_b)`` on identical rows,
  with bias-corrected and accelerated (BCa) 95% intervals.  The acceleration is
  the jackknife estimate of skewness and the bias correction ``z0`` is derived
  from the bootstrap distribution's rank of the point estimate.
* :func:`calibration_bins` -- the actual equal-width reliability bin points
  (per-bin count, mean prediction, observed positive fraction and ECE
  contribution) plus the total ECE recomputed *from those bin points*.

Finite-sample caveats (read before quoting any interval)
--------------------------------------------------------
* BCa is **not** universally superior to the percentile interval.  It corrects
  bias and skew only when those corrections are estimable; with small ``n``,
  sparse resamples, extreme statistics, or heavily tied resamples, its
  corrections are themselves noisy and can behave worse than the simpler
  percentile interval.  Neither interval has exact finite-sample coverage.
* A BCa interval requires enough resamples to estimate ``z0`` and the
  acceleration.  This module falls back to the plain percentile interval (with
  ``method="percentile_fallback"`` and a stated reason) when ``n < 10``, fewer
  than 100 bootstrap replicates survive, the jackknife is undefined or
  constant, ``z0`` is at the boundary/non-finite, or the adjusted percentiles
  leave ``(0, 1)``.  The fallback is reported honestly, never hidden.
* Bootstrap replicates are stratified (class sizes preserved) via the same
  helper used by :mod:`src.metrics`, so ROC-AUC/AUPRC replicates remain
  defined.  Undefined or non-finite replicates are skipped and counted in
  ``n_valid``; stratification does not remove Monte Carlo error.
* Seeding with ``numpy.random.default_rng(seed)`` makes a call deterministic,
  not exact: a different seed (or ``n_boot``) legitimately changes the interval.
* Equal-width calibration bins are a biased, edge-sensitive estimator.  Empty
  or sparsely occupied bins are reported (``n = 0``, undefined means) and
  contribute zero, but their instability is *not* converted into an interval.
  Pooled-across-seeds ECE is descriptive only.  A bin-level reliability diagram
  is an empirical, finite-sample estimate; it is not a smooth calibrated curve.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import metrics as _metrics  # noqa: E402

CI_LEVEL = 0.95
CI_ALPHAS: Tuple[float, float] = ((1.0 - CI_LEVEL) / 2.0, (1.0 + CI_LEVEL) / 2.0)
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 666
DEFAULT_N_BINS = 10

# Thresholds below which BCa corrections are not trusted. These are method
# constants, not result values; falling below them triggers the honest
# percentile fallback.
MIN_BCA_N = 10
MIN_BCA_N_VALID = 100
# A jackknife whose variance is at floating-point noise level is degenerate:
# its acceleration is undefined even if the values are not bit-identical.
JACKKNIFE_CONSTANT_RTOL = 1e-12

METHOD_BCA = "bca"
METHOD_PERCENTILE_FALLBACK = "percentile_fallback"
METHOD_DESCRIPTION_BCA = (
    "paired stratified bootstrap with BCa (jackknife acceleration + bias correction z0); "
    "95% interval; percentile fallback when the correction is undefined"
)
METHOD_DESCRIPTION_PERCENTILE_FALLBACK = (
    "paired stratified bootstrap percentile interval (BCa undefined; no bias/skew correction)"
)
CALIBRATION_METHOD = (
    "equal-width histogram over [0, 1] (default 10 bins); "
    "ECE = sum_b (n_b/n) * |mean(pred)_b - observed_fraction_b| computed from bin points"
)
CALIBRATION_CAVEATS: Tuple[str, ...] = (
    "finite-sample bias: empty or small bins make mean prediction and observed "
    "fraction noisy; bins with n=0 are reported with n=0 and contribute zero",
    "fixed equal-width bin edges: ECE and the reliability points depend on n_bins "
    "and on where the edges fall; no smoothing or isotonic fit is applied",
    "no inferential interval is attached to ECE or to individual bin points; "
    "pooled-across-seeds values are descriptive only",
)

__all__ = [
    "bca_interval",
    "calibration_bins",
    "CI_LEVEL",
    "DEFAULT_N_BOOT",
    "DEFAULT_N_BINS",
    "DEFAULT_SEED",
    "METHOD_BCA",
    "METHOD_PERCENTILE_FALLBACK",
    "METHOD_DESCRIPTION_BCA",
    "METHOD_DESCRIPTION_PERCENTILE_FALLBACK",
    "CALIBRATION_METHOD",
    "CALIBRATION_CAVEATS",
]


def _paired_difference(
    labels: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    indices: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Return ``(diff, metric_a, metric_b)`` on a (possibly resampled) row set."""
    if indices is None:
        y, left, right = labels, prob_a, prob_b
    else:
        y, left, right = labels[indices], prob_a[indices], prob_b[indices]
    value_a = float(metric_fn(y, left))
    value_b = float(metric_fn(y, right))
    return value_a - value_b, value_a, value_b


def bca_interval(
    y_true: Sequence[Any],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Paired BCa interval for ``metric_fn(a) - metric_fn(b)``.

    The same stratified with-replacement resamples are applied to both models
    (paired comparison on identical test rows).  ``z0`` is the normal quantile
    of the bootstrap distribution's (midpoint tie-handled) rank of the point
    estimate; acceleration is the standard delete-one jackknife skewness
    estimate.  When either correction is undefined, the function returns the
    percentile interval with ``method="percentile_fallback"`` and an explicit
    ``fallback_reason``.  See the module docstring for the caveats.

    Returns a dictionary with ``point_estimate``, ``ci_low``, ``ci_high``,
    ``ci_level``, ``n_boot``, ``n_valid``, ``seed``, ``method``, bootstrap and
    jackknife values, ``z0``, ``acceleration``, ``fallback_reason``, ``warnings``
    and ``caveats``.
    """
    labels = _metrics._as_labels(y_true)
    first = _metrics._as_probabilities(prob_a, labels.size)
    second = _metrics._as_probabilities(prob_b, labels.size)
    n = int(labels.size)
    if n == 0:
        raise ValueError("cannot bootstrap an empty input")
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot!r}")
    try:
        point, point_a, point_b = _paired_difference(labels, first, second, metric_fn)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric_fn is undefined for these inputs: {exc}") from exc
    if not np.isfinite(point):
        raise ValueError("metric_fn must return a finite point estimate")

    rng = np.random.default_rng(int(seed))
    differences: List[float] = []
    skipped = 0
    for _ in range(int(n_boot)):
        indices = _metrics._resample_indices(labels, rng)
        try:
            value, _left, _right = _paired_difference(
                labels, first, second, metric_fn, indices=indices
            )
        except (TypeError, ValueError):
            skipped += 1
            continue
        if np.isfinite(value):
            differences.append(float(value))
        else:
            skipped += 1

    warnings: List[str] = []
    if skipped:
        warnings.append(
            f"{skipped} of {int(n_boot)} replicates were undefined and excluded from the CI"
        )
    bootstrap = np.asarray(differences, dtype=float)

    z0: Optional[float] = None
    acceleration: Optional[float] = None
    jackknife: List[float] = []
    fallback_reason: Optional[str] = None
    adjusted: Optional[Tuple[float, float]] = None

    if bootstrap.size == 0:
        fallback_reason = "no finite bootstrap replicates; BCa and percentile CI are undefined"
    elif n < MIN_BCA_N:
        fallback_reason = (
            f"n={n} < {MIN_BCA_N}: too few rows for a stable delete-one jackknife"
        )
    elif bootstrap.size < MIN_BCA_N_VALID:
        fallback_reason = (
            f"n_valid={bootstrap.size} < {MIN_BCA_N_VALID}: too few resamples to estimate "
            "the BCa bias correction and acceleration"
        )
    else:
        jackknife_ok = True
        for i in range(n):
            delete_one = np.delete(np.arange(n), i)
            try:
                value, _left, _right = _paired_difference(
                    labels, first, second, metric_fn, indices=delete_one
                )
            except (TypeError, ValueError):
                jackknife_ok = False
                break
            if not np.isfinite(value):
                jackknife_ok = False
                break
            jackknife.append(float(value))
        if not jackknife_ok:
            jackknife = []
            fallback_reason = "jackknife undefined (a delete-one subset gives an undefined metric)"
        else:
            # Efron's delete-one jackknife acceleration is centred on the mean
            # of the jackknife values (theta_dot_(.)), NOT on the full-sample
            # point estimate. Degeneracy is therefore a zero *variance of the
            # jackknife values themselves*, even when they differ from `point`.
            jackknife_array = np.asarray(jackknife, dtype=float)
            jackknife_mean = float(jackknife_array.mean())
            jackknife_variance = float(np.var(jackknife_array, ddof=0))
            constant_tolerance = (JACKKNIFE_CONSTANT_RTOL * max(abs(jackknife_mean), 1.0)) ** 2
            if not np.isfinite(jackknife_variance) or jackknife_variance <= constant_tolerance:
                fallback_reason = (
                    "acceleration undefined: jackknife values are constant across "
                    "delete-one samples"
                )
            else:
                deviations = jackknife_mean - jackknife_array
                denominator = 6.0 * float(np.sum(deviations ** 2)) ** 1.5
                acceleration = float(np.sum(deviations ** 3) / denominator)
                if not np.isfinite(acceleration):
                    fallback_reason = "acceleration is non-finite"
                else:
                    n_less = int(np.count_nonzero(bootstrap < point))
                    n_equal = int(np.count_nonzero(bootstrap == point))
                    proportion = (n_less + 0.5 * n_equal) / bootstrap.size
                    if not 0.0 <= proportion <= 1.0:
                        fallback_reason = (
                            "bias correction z0 undefined: bootstrap rank proportion "
                            "outside [0, 1]"
                        )
                    else:
                        z0 = float(norm.ppf(proportion))
                        if not np.isfinite(z0):
                            fallback_reason = (
                                "bias correction z0 is non-finite (bootstrap distribution "
                                "lies entirely on one side of the point estimate)"
                            )
                        else:
                            values: List[float] = []
                            valid = True
                            for alpha in CI_ALPHAS:
                                numerator = z0 + float(norm.ppf(alpha))
                                denominator_z = 1.0 - acceleration * numerator
                                if denominator_z <= 0.0:
                                    valid = False
                                    break
                                adjusted_z = z0 + numerator / denominator_z
                                adjusted_alpha = float(norm.cdf(adjusted_z))
                                if not 0.0 < adjusted_alpha < 1.0:
                                    valid = False
                                    break
                                values.append(100.0 * adjusted_alpha)
                            if not valid or len(values) != 2 or not values[0] < values[1]:
                                fallback_reason = (
                                    "BCa adjusted percentiles fall outside (0, 1) or are "
                                    "out of order (extreme acceleration)"
                                )
                            else:
                                adjusted = (values[0], values[1])

    if adjusted is None:
        # No fallback result may carry stale or non-finite correction fields.
        z0 = None
        acceleration = None
        if not fallback_reason:
            fallback_reason = "BCa correction unavailable for these inputs"

    if adjusted is not None:
        ci_low, ci_high = (float(value) for value in np.percentile(bootstrap, list(adjusted)))
        method = METHOD_BCA
        method_description = METHOD_DESCRIPTION_BCA
    else:
        if bootstrap.size:
            ci_low, ci_high = (
                float(value) for value in np.percentile(bootstrap, [100.0 * a for a in CI_ALPHAS])
            )
        else:
            ci_low, ci_high = None, None
        method = METHOD_PERCENTILE_FALLBACK
        method_description = METHOD_DESCRIPTION_PERCENTILE_FALLBACK
        if fallback_reason:
            warnings.append("BCa unavailable: " + fallback_reason)

    return {
        "metric_name": getattr(metric_fn, "__name__", "metric_fn"),
        "point_estimate": float(point),
        "metric_a_point": float(point_a),
        "metric_b_point": float(point_b),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": CI_LEVEL,
        "n_boot": int(n_boot),
        "n_valid": int(bootstrap.size),
        "seed": int(seed),
        "method": method,
        "method_description": method_description,
        "bootstrap_differences": [float(value) for value in bootstrap],
        "jackknife_values": jackknife,
        "z0": z0,
        "acceleration": acceleration,
        "adjusted_percentiles": list(adjusted) if adjusted is not None else None,
        "fallback_reason": fallback_reason,
        "warnings": warnings,
        "caveats": [
            "BCa is not universally superior to the percentile interval; "
            "its corrections are noisy in small samples",
            "finite n_boot and finite n give only approximate coverage; the "
            "seed and n_boot must be reported with the interval",
            "undefined replicates are skipped (see n_valid), not imputed",
        ],
    }


def calibration_bins(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    n_bins: int = DEFAULT_N_BINS,
) -> Dict[str, Any]:
    """Equal-width reliability bins and ECE recomputed from the bin points.

    Every bin in ``[0, 1]`` is returned, including empty ones (``n = 0`` with
    ``mean_pred``/``observed_fraction`` set to ``None`` and zero ECE
    contribution).  ``ece`` is the sum of the returned per-bin contributions,
    not a separately saved scalar.  See the module docstring for the
    finite-sample, fixed-edge and no-interval caveats.
    """
    labels = _metrics._as_labels(y_true)
    probabilities = _metrics._as_probabilities(y_prob, labels.size)
    if int(n_bins) < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
    n_bins = int(n_bins)
    n = int(labels.size)

    bin_index = np.floor(probabilities * n_bins).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    bins: List[Dict[str, Any]] = []
    for index in range(n_bins):
        mask = bin_index == index
        count = int(np.count_nonzero(mask))
        mean_pred = float(probabilities[mask].mean()) if count else None
        observed = float(labels[mask].mean()) if count else None
        contribution = (
            (count / n) * abs(mean_pred - observed)
            if count and mean_pred is not None and observed is not None
            else 0.0
        )
        bins.append(
            {
                "bin_index": index,
                "bin_low": index / n_bins,
                "bin_high": (index + 1) / n_bins,
                "n": count,
                "mean_pred": mean_pred,
                "observed_fraction": observed,
                "ece_contribution": float(contribution),
            }
        )

    # ECE is derived exclusively from the bin points returned to the caller.
    total_ece = float(sum(entry["ece_contribution"] for entry in bins)) if n else None
    return {
        "bins": bins,
        "n": n,
        "n_bins": n_bins,
        "ece": total_ece,
        "method": CALIBRATION_METHOD,
        "caveats": list(CALIBRATION_CAVEATS),
    }
