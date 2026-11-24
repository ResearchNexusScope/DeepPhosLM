"""Classical, same-split baselines for binary phosphosite prediction.

Scope
-----
This module provides two deliberately simple, *local* comparators for the
neural model in :mod:`src.model`:

* ``logistic_regression`` -- scikit-learn L2 logistic regression.
* ``random_forest``       -- scikit-learn random forest.

They are fitted on the *same* leakage-free training split and scored on the
*same* held-out test split as the deep model.  They are **not** external
phosphosite predictors (NetPhos, GPS, MusiteDeep, TransPhos, GraphPhos, ...):
none of those tools is available offline in this repository, and this module
neither executes nor claims to reproduce them.  No test-set tuning is
performed: every hyperparameter below is fixed before the test split is read.

Features
--------
Inputs may be either raw 15-residue window strings or an already-computed
feature matrix.  Raw strings are encoded with ``src.model.one_hot_encode``,
the *same* 330-dimensional encoder used by the neural model (22-symbol legacy
alphabet ``XWGSAELQDMPFTRIHVNCY_K``, 15 positions x 22 = 330 columns, unknown
symbols mapped to ``X``).  The encoder is imported rather than reimplemented,
so the two model families can never silently diverge in feature construction.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model import ONE_HOT_WIDTH, one_hot_encode

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
BASELINE_NAMES = (LOGISTIC_REGRESSION, RANDOM_FOREST)

_NAME_ALIASES = {
    "lr": LOGISTIC_REGRESSION,
    "logreg": LOGISTIC_REGRESSION,
    "logistic": LOGISTIC_REGRESSION,
    "logistic_regression": LOGISTIC_REGRESSION,
    "rf": RANDOM_FOREST,
    "random_forest": RANDOM_FOREST,
}

_ENCODING_DESCRIPTION = "src.model.one_hot_encode (15 x 22 = 330 columns)"
_NO_TUNING = "fixed hyperparameters; no validation/test tuning or model selection"


def canonical_name(name: str) -> str:
    """Return the canonical baseline name, rejecting unknown names loudly."""
    key = str(name).strip().lower().replace("-", "_")
    if key not in _NAME_ALIASES:
        raise ValueError(
            f"Unknown baseline {name!r}; expected one of {BASELINE_NAMES}"
        )
    return _NAME_ALIASES[key]


def baseline_config(name: str, seed: int) -> Dict[str, Any]:
    """Return a JSON-serialisable, fully fixed configuration for one baseline.

    The returned hyperparameters are the *only* ones used by
    :func:`fit_predict_baseline`; keeping them in one place prevents the
    reported config from drifting away from the fitted estimator.
    """
    canonical = canonical_name(name)
    config: Dict[str, Any] = {
        "name": canonical,
        "encoding": _ENCODING_DESCRIPTION,
        "n_features": ONE_HOT_WIDTH,
        "seed": int(seed),
        "tuning": _NO_TUNING,
    }
    if canonical == LOGISTIC_REGRESSION:
        config["estimator"] = "sklearn.linear_model.LogisticRegression"
        config["hyperparameters"] = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "fit_intercept": True,
            "class_weight": None,
            "random_state": int(seed),
        }
    else:
        config["estimator"] = "sklearn.ensemble.RandomForestClassifier"
        config["hyperparameters"] = {
            "n_estimators": 200,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": int(seed),
            "n_jobs": 1,
        }
    return config


def _new_estimator(name: str, seed: int) -> Union[LogisticRegression, RandomForestClassifier]:
    """Instantiate an estimator from :func:`baseline_config` (single source of truth)."""
    config = baseline_config(name, seed)
    hyperparameters = dict(config["hyperparameters"])
    if config["name"] == LOGISTIC_REGRESSION:
        return LogisticRegression(**hyperparameters)
    return RandomForestClassifier(**hyperparameters)


def _as_features(X: Union[Sequence[Any], np.ndarray]) -> np.ndarray:
    """Return an ``(n, 330)`` float32 feature matrix.

    String rows are treated as 15-residue windows and encoded with the neural
    model's encoder.  A numeric 2-D array is accepted as already-encoded
    features and checked for the exact expected width.
    """
    if isinstance(X, np.ndarray):
        array = X
    else:
        rows = list(X)
        if rows and all(isinstance(row, (str, np.str_)) for row in rows):
            return one_hot_encode(rows)
        array = np.asarray(rows)

    if array.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {np.shape(array)}")
    if array.shape[1] != ONE_HOT_WIDTH:
        raise ValueError(
            f"features must have {ONE_HOT_WIDTH} columns, got {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"numeric features required, got dtype {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("features must contain only finite values")
    return array


def fit_predict_baseline(
    X_train: Union[Sequence[Any], np.ndarray],
    y_train: Sequence[Any],
    X_test: Union[Sequence[Any], np.ndarray],
    name: str,
    seed: int,
) -> np.ndarray:
    """Fit one classical baseline on train and return test P(label == 1).

    Parameters
    ----------
    X_train, X_test:
        Either iterables of 15-residue window strings or ``(n, 330)`` numeric
        arrays.  Strings are encoded with ``src.model.one_hot_encode``.
    y_train:
        1-D binary training labels (0 = non-site, 1 = site).
    name:
        ``"logistic_regression"`` or ``"random_forest"`` (short aliases
        ``lr``/``rf`` are accepted).
    seed:
        Integer seed passed to the estimator; the same seed and data always
        give byte-identical probabilities.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 array of positive-class probabilities in
        ``[0, 1]``, aligned with ``X_test``.
    """
    canonical = canonical_name(name)
    features_train = _as_features(X_train)
    features_test = _as_features(X_test)
    labels = np.asarray(y_train)

    if labels.ndim != 1:
        raise ValueError(f"y_train must be 1-D, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError("y_train is empty; cannot fit a baseline")
    if labels.size != features_train.shape[0]:
        raise ValueError(
            f"y_train has {labels.size} rows but X_train has {features_train.shape[0]}"
        )
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("y_train must contain only binary labels 0 and 1")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(
            "y_train must contain both classes to fit a comparative baseline"
        )

    estimator = _new_estimator(canonical, seed)
    estimator.fit(features_train, labels.astype(np.int64))

    classes = list(estimator.classes_)
    positive_column = classes.index(1)
    probabilities = np.asarray(
        estimator.predict_proba(features_test)[:, positive_column], dtype=np.float64
    )
    if probabilities.shape != (features_test.shape[0],):
        raise ValueError(
            f"unexpected prediction shape {probabilities.shape} for "
            f"{features_test.shape[0]} test rows"
        )
    # predict_proba is already in [0, 1]; clip guards against float drift so
    # the result always satisfies src.metrics' probability validator.
    return np.clip(probabilities, 0.0, 1.0)


def available_baselines() -> Mapping[str, Dict[str, Any]]:
    """Return configs for every implemented baseline, keyed by canonical name."""
    return {name: baseline_config(name, seed=0) for name in BASELINE_NAMES}
