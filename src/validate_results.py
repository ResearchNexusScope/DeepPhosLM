"""Authoritative "validate every result" gate for the real phosphosite experiment.

This module is the ONLY place where numbers reported in the manuscript may be
recomputed. It re-derives every reported quantity strictly from the saved
artifacts (prediction NPZs, training manifests, split CSVs/manifest) using the
existing truthful implementations in :mod:`src.metrics`, :mod:`src.baselines`
and the documented calibration correction. No numeric result is hardcoded here:
the ballpark values quoted in planning notes are deliberately absent.

Manuscript workflow (mandatory)
-------------------------------
1. Run ``python tools/verify_experiment.py`` (the mother-only CLI). It calls
   :func:`validate_all`, writes ``results/real/validation_report.json``, runs
   the single pytest pass and checks any supplied claims.
2. Obtain the exact dotted claim path with :func:`available_claim_paths` on the
   report. **Never retype a reported number into prose.** Build a claims JSON
   from those paths/values and pass it via ``--claims``; every claim is checked
   by :func:`check_claimed_numbers` against the recomputed report, with an
   explicit, documented tolerance. A claim that fails returns status ``FAIL``
   and makes the verifier exit non-zero.

What is recomputed
------------------
* Per-run test metrics (all NPZs via ``src.metrics.compute_metrics``), with
  mean/std and full per-seed vectors for every scalar metric per ablation.
* Per-site-type S/T/Y breakdowns pooled across seeds and per seed; undefined
  groups stay ``None`` (never imputed).
* Seeded paired bootstrap (AUROC and AUPRC) of seq_embedding vs seq_only on the
  same test rows. ``point_estimate = metric(seq_embedding) - metric(seq_only)``;
  positive values favour seq_embedding. The 95% interval is a percentile
  bootstrap CI (an approximation, no BCa correction).
* Expected calibration error (10 equal-width bins in [0, 1]) per ablation, per
  site type and per seed, from the saved probabilities and labels.
* Same-split classical baselines (logistic regression, random forest) fitted on
  the real TRAIN csv via ``src.baselines`` and scored on the real TEST csv via
  ``src.metrics``. Hyperparameters are fixed by ``src.baselines``; there is no
  test-set tuning.
* Cross-validation summary re-read from ``results/real_cv/training_manifest.json``:
  fold count, mean/std/values consistency with stored fold vectors (recomputed
  with ``src.metrics.summarize_folds``) and no cluster overlap with the test
  split.

Every number-bearing section carries a ``provenance`` block: source file paths,
SHA-256 digests, seeds and configs (where applicable). Claim results inherit
the provenance of the recomputed value they refer to.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import metrics as _metrics  # noqa: E402
from src.real_split import sha256_file  # noqa: E402

# Method constants (NOT result values). These describe *how* validation runs.
SCALAR_METRICS: Tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
)
COUNT_KEYS: Tuple[str, ...] = ("n", "n_positive", "n_negative")
NPZ_KEYS: Tuple[str, ...] = (
    "labels",
    "probabilities",
    "site_type",
    "protein_acc",
    "cluster_id",
    "seed",
    "ablation",
    "chosen_epoch",
    "config",
)
SITE_TYPES: Tuple[str, ...] = tuple(_metrics.SITE_TYPES)
DEFAULT_N_BOOT = 1000
DEFAULT_BOOTSTRAP_SEED = 666
DEFAULT_BASELINE_SEED = 666
DEFAULT_ECE_BINS = 10
DEFAULT_ABS_TOL = 1e-9
DEFAULT_REL_TOL = 1e-6
ECE_METHOD = (
    "equal-width histogram over [0, 1] (default 10 bins); "
    "ECE = sum_b (n_b/n) * |mean(confidence_b) - accuracy_b|; "
    "caveats: finite-sample bias, sensitivity to the fixed equal-width bin edges, "
    "and pooled-across-seeds ECE is descriptive only (no inferential interval); "
    "unlike the paired bootstrap's explicit percentile/no-BCa caveat, no resampling "
    "interval is produced here"
)
BOOTSTRAP_METHOD = "stratified paired percentile bootstrap (no BCa correction)"

__all__ = [
    "DEFAULT_ABS_TOL",
    "DEFAULT_BASELINE_SEED",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_ECE_BINS",
    "DEFAULT_N_BOOT",
    "DEFAULT_REL_TOL",
    "available_claim_paths",
    "check_claimed_numbers",
    "ece_report",
    "expected_calibration_error",
    "flatten_report",
    "paired_bootstrap",
    "validate_all",
    "validate_baselines",
    "validate_cv",
    "validate_per_site_type",
    "validate_runs",
]


# --------------------------------------------------------------------------- #
# Provenance / path helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    """SHA-256 of a file, via the same helper used by the split pipeline."""
    return sha256_file(Path(path))


def _file_provenance(path: Path, role: str) -> Dict[str, str]:
    path = Path(path)
    return {"path": str(path), "sha256": _sha256(path), "role": role}


def _resolve(ref: Any, base: Optional[Path]) -> Path:
    """Resolve a path stored in a manifest (absolute or relative)."""
    path = Path(str(ref))
    if path.is_absolute() and path.exists():
        return path
    if base is not None:
        candidate = base / path
        if candidate.exists():
            return candidate
        candidate = base / path.name
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    raise FileNotFoundError(f"Cannot resolve artifact path {ref!r} (base={base})")


def _check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "status": "PASS" if passed else "FAIL", "detail": detail}


def _first_mismatch(
    actual: Sequence[Any], expected: Sequence[Any], field: str
) -> Optional[Dict[str, Any]]:
    """First row where an NPZ vector differs from the test CSV (order matters)."""
    if len(actual) != len(expected):
        return {
            "field": field,
            "index": None,
            "reason": f"{field} length differs: npz={len(actual)} test.csv={len(expected)}",
        }
    for index, (left, right) in enumerate(zip(actual, expected)):
        left_text, right_text = str(left), str(right)
        if left_text != right_text:
            return {
                "field": field,
                "index": int(index),
                "reason": f"{field}[{index}]: npz={left_text!r} test.csv={right_text!r}",
            }
    return None


def _values_close(
    actual: Any, expected: Any, abs_tol: float, rel_tol: float
) -> bool:
    """Compare two scalars (or None) with a documented numeric tolerance."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return bool(actual) == bool(expected)
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual) == str(expected)
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol
        )
    except (TypeError, ValueError):
        return actual == expected


def _scalar_summary(values: Sequence[Any]) -> Dict[str, Any]:
    """Per-seed vector plus mean/std over its defined entries (population std)."""
    defined = [float(value) for value in values if value is not None]
    array = np.asarray(defined, dtype=float)
    return {
        "values": list(values),
        "n_defined": len(defined),
        "mean": float(array.mean()) if defined else None,
        "std": float(array.std(ddof=0)) if defined else None,
        "std_ddof": 0,
    }


def _metric_keys(metrics: Mapping[str, Any]) -> List[str]:
    keys = [key for key in SCALAR_METRICS if key in metrics]
    keys.extend(key for key in COUNT_KEYS if key in metrics)
    return keys


def _load_manifest(results_dir: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Read the real-run training manifest; return it with its provenance."""
    path = results_dir / "training_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"training manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, _file_provenance(path, "training_manifest")


def _load_runs(results_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load and validate every prediction archive named by the manifest."""
    manifest, manifest_prov = _load_manifest(results_dir)
    runs: List[Dict[str, Any]] = []
    for entry in manifest.get("runs", []):
        npz_path = _resolve(entry["prediction_file"], results_dir)
        with np.load(npz_path, allow_pickle=True) as archive:
            for key in NPZ_KEYS:
                if key not in archive.files:
                    raise ValueError(f"{npz_path} is missing key {key!r}")
            labels = np.asarray(archive["labels"]).astype(np.int64)
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
            site_type = np.asarray([str(v).strip().upper() for v in archive["site_type"]])
            protein_acc = np.asarray([str(v) for v in archive["protein_acc"]])
            cluster_id = np.asarray([str(v) for v in archive["cluster_id"]])
            seed = int(np.asarray(archive["seed"]).item())
            ablation = str(np.asarray(archive["ablation"]).item())
            chosen_epoch = int(np.asarray(archive["chosen_epoch"]).item())
            config = dict(np.asarray(archive["config"], dtype=object).item())
        lengths = {
            "labels": labels.size,
            "probabilities": probabilities.size,
            "site_type": site_type.size,
            "protein_acc": protein_acc.size,
            "cluster_id": cluster_id.size,
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{npz_path}: misaligned array lengths {lengths}")
        seed_in_manifest = int(entry.get("seed", seed))
        if seed != seed_in_manifest:
            raise ValueError(f"{npz_path}: seed {seed} != manifest seed {seed_in_manifest}")
        ablation_in_manifest = str(entry.get("ablation", ablation))
        if ablation != ablation_in_manifest:
            raise ValueError(
                f"{npz_path}: ablation {ablation!r} != manifest {ablation_in_manifest!r}"
            )
        runs.append(
            {
                "seed": seed,
                "ablation": ablation,
                "labels": labels,
                "probabilities": probabilities,
                "site_type": site_type,
                "protein_acc": protein_acc,
                "cluster_id": cluster_id,
                "chosen_epoch": chosen_epoch,
                "config": config,
                "source": str(npz_path),
                "sha256": _sha256(npz_path),
            }
        )
    if not runs:
        raise ValueError(f"no runs found in {results_dir / 'training_manifest.json'}")
    return runs, {"manifest": manifest_prov, "manifest_data": manifest}


def _run_provenance(run: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_file": run["source"],
        "sha256": run["sha256"],
        "seed": run["seed"],
        "ablation": run["ablation"],
        "chosen_epoch": run["chosen_epoch"],
        "config": run["config"],
    }


def _group_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["ablation"]), []).append(run)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["seed"]))
    return grouped


def _check_row_alignment(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Verify all seeds/ablations scored byte-identical test rows."""
    reference = runs[0]
    problems: List[str] = []
    for run in runs[1:]:
        for field in ("labels", "site_type", "protein_acc", "cluster_id"):
            if not np.array_equal(reference[field], run[field]):
                problems.append(f"run {run['ablation']}/{run['seed']}: {field} differs")
    return _check(
        "all_runs_score_identical_test_rows",
        not problems,
        "identical labels/site_type/protein_acc/cluster_id across all runs"
        if not problems
        else "; ".join(problems),
    )


# --------------------------------------------------------------------------- #
# 1. Per-run and aggregate test metrics
# --------------------------------------------------------------------------- #
def validate_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recompute per-run test metrics and aggregate them per ablation."""
    # One compute_metrics call per archive; its result feeds both the per-run
    # block and the per-ablation aggregate (sklearn metric calls are costly).
    computed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    per_run: List[Dict[str, Any]] = []
    for run in runs:
        metrics = _metrics.compute_metrics(
            run["labels"], run["probabilities"], site_types=run["site_type"].tolist()
        )
        computed[(str(run["ablation"]), int(run["seed"]))] = metrics
        per_run.append(
            {
                "seed": int(run["seed"]),
                "ablation": str(run["ablation"]),
                "metrics": metrics,
                "provenance": _run_provenance(run),
            }
        )

    grouped = _group_runs(runs)
    aggregates: Dict[str, Any] = {}
    for ablation, group in grouped.items():
        per_seed = {
            str(run["seed"]): computed[(str(run["ablation"]), int(run["seed"]))]
            for run in group
        }
        metric_summaries = {
            key: _scalar_summary(
                [per_seed[str(run["seed"])][key] for run in group]
            )
            for key in _metric_keys(per_seed[str(group[0]["seed"])])
        }
        aggregates[ablation] = {
            "seeds": [int(run["seed"]) for run in group],
            "per_seed": per_seed,
            "metrics": metric_summaries,
            "mean": {key: value["mean"] for key, value in metric_summaries.items()},
            "std": {key: value["std"] for key, value in metric_summaries.items()},
            "provenance": {
                "runs": [_run_provenance(run) for run in group],
                "metrics_source": "src.metrics.compute_metrics",
                "aggregation": "population mean/std (ddof=0) over defined per-seed values",
            },
        }

    return {
        "per_run": per_run,
        "aggregates": aggregates,
        "computed": computed,
        "provenance": {
            "runs": [_run_provenance(run) for run in runs],
            "metrics_source": "src.metrics.compute_metrics",
        },
    }


# --------------------------------------------------------------------------- #
# 2. Per-site-type breakdown
# --------------------------------------------------------------------------- #
def validate_per_site_type(
    runs: Sequence[Mapping[str, Any]],
    precomputed: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Pool rows across seeds per S/T/Y; keep undefined groups as ``None``.

    ``precomputed`` optionally carries the full per-run metrics keyed by
    ``(ablation, seed)`` so the top-level orchestrator does not call sklearn's
    expensive metric functions twice for the same archive.
    """
    grouped = _group_runs(runs)
    result: Dict[str, Any] = {}
    for ablation, group in grouped.items():
        site_block: Dict[str, Any] = {}
        # One compute_metrics call per archive supplies all three per-site-type
        # blocks; pooled aggregates below are the only extra metric calls.
        if precomputed is None:
            full_metrics = {
                int(run["seed"]): _metrics.compute_metrics(
                    run["labels"], run["probabilities"], site_types=run["site_type"].tolist()
                )
                for run in group
            }
        else:
            full_metrics = {
                int(run["seed"]): precomputed[(str(run["ablation"]), int(run["seed"]))]
                for run in group
            }
        for site in SITE_TYPES:
            pooled_labels: List[np.ndarray] = []
            pooled_probs: List[np.ndarray] = []
            per_seed: Dict[str, Any] = {}
            for run in group:
                mask = run["site_type"] == site
                pooled_labels.append(run["labels"][mask])
                pooled_probs.append(run["probabilities"][mask])
                # compute_metrics itself reports undefined positive-class metrics
                # as None for single-class or empty groups.
                per_seed[str(int(run["seed"]))] = full_metrics[int(run["seed"])][
                    "per_site_type"
                ][site]
            site_block[site] = {
                "pooled": _metrics.compute_metrics(
                    np.concatenate(pooled_labels) if pooled_labels else np.empty(0, dtype=np.int64),
                    np.concatenate(pooled_probs) if pooled_probs else np.empty(0, dtype=np.float64),
                ),
                "per_seed": per_seed,
                "metrics": {
                    key: _scalar_summary(
                        [per_seed[str(run["seed"])][key] for run in group]
                    )
                    for key in _metric_keys(per_seed[str(group[0]["seed"])])
                },
                "provenance": {
                    "runs": [_run_provenance(run) for run in group],
                    "site_type": site,
                    "metrics_source": "src.metrics.compute_metrics",
                    "undefined_policy": "None, never imputed",
                },
            }
        # S, T and Y are direct children so claim paths read
        # "per_site_type.<ablation>.<site>.pooled.<metric>" (see module docstring).
        result[ablation] = {**site_block, "provenance": {"runs": [_run_provenance(run) for run in group]}}
    return {
        "per_site_type": result,
        "provenance": {"runs": [_run_provenance(run) for run in runs]},
    }


# --------------------------------------------------------------------------- #
# 3. Seeded paired bootstrap
# --------------------------------------------------------------------------- #
def paired_bootstrap(
    runs: Sequence[Mapping[str, Any]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Paired bootstrap of seq_embedding minus seq_only on identical test rows.

    Probabilities are first averaged across seeds within each ablation (the row
    alignment is checked by ``validate_all``), then the *same* stratified
    resamples are applied to both ablations via
    ``src.metrics.bootstrap_metric_difference``.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2 (production default {DEFAULT_N_BOOT})")
    grouped = _group_runs(runs)
    if set(grouped) != {"seq_only", "seq_embedding"}:
        raise ValueError(
            f"paired bootstrap needs exactly seq_only and seq_embedding, got {sorted(grouped)}"
        )
    averaged: Dict[str, np.ndarray] = {}
    for ablation, group in grouped.items():
        averaged[ablation] = np.mean(
            np.vstack([np.asarray(run["probabilities"], dtype=np.float64) for run in group]),
            axis=0,
        )
    first = grouped["seq_only"][0]
    labels = first["labels"]
    for ablation, group in grouped.items():
        if averaged[ablation].shape != labels.shape:
            raise ValueError(f"row mismatch for {ablation} in paired bootstrap")

    out: Dict[str, Any] = {}
    for metric_name, metric_fn in (
        ("auroc", _metrics.roc_auc_metric),
        ("auprc", _metrics.average_precision_metric),
    ):
        result = _metrics.bootstrap_metric_difference(
            labels,
            averaged["seq_embedding"],
            averaged["seq_only"],
            metric_fn=metric_fn,
            n_boot=n_boot,
            seed=seed,
        )
        result["model_a"] = "seq_embedding"
        result["model_b"] = "seq_only"
        result["difference_definition"] = "metric(seq_embedding) - metric(seq_only)"
        result["provenance"] = {
            "runs": [_run_provenance(run) for group in grouped.values() for run in group],
            "metric_source": f"src.metrics.{metric_fn.__name__}",
            "bootstrap_source": "src.metrics.bootstrap_metric_difference",
            "n_boot": int(n_boot),
            "seed": int(seed),
            "method": BOOTSTRAP_METHOD,
        }
        out[metric_name] = result

    return {
        "paired_bootstrap": out,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "method": BOOTSTRAP_METHOD,
        "provenance": {"runs": [_run_provenance(run) for run in runs]},
    }


# --------------------------------------------------------------------------- #
# 4. Calibration (ECE)
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    y_true: Sequence[Any],
    y_prob: Sequence[float],
    n_bins: int = DEFAULT_ECE_BINS,
) -> Optional[float]:
    """Equal-width-bin ECE; ``None`` for an empty input.

    Each bin contributes ``count/n * |mean(confidence) - observed accuracy|``.
    A vector is perfectly calibrated here when, in every occupied bin, the mean
    predicted probability equals the fraction of positives.

    Caveats: the histogram estimator has finite-sample bias, its value depends
    on the fixed equal-width bin edges (10 by default), and an ECE pooled
    across seeds is descriptive only -- unlike the paired bootstrap, no
    resampling interval is attached to it.
    """
    labels = np.asarray(y_true).astype(np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("y_true and y_prob must be 1-D and equally long")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
    if labels.size == 0:
        return None
    bin_index = np.floor(probabilities * n_bins).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    total = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        total += (count / labels.size) * abs(confidence - accuracy)
    return float(total)


def ece_report(
    runs: Sequence[Mapping[str, Any]], n_bins: int = DEFAULT_ECE_BINS
) -> Dict[str, Any]:
    """ECE per ablation, per site type and per seed, with provenance.

    The pooled value is descriptive only; see :func:`expected_calibration_error`
    for the finite-sample bias and fixed-bin-edge caveats.
    """
    grouped = _group_runs(runs)
    result: Dict[str, Any] = {}
    for ablation, group in grouped.items():
        pooled_labels = np.concatenate([run["labels"] for run in group])
        pooled_probs = np.concatenate([run["probabilities"] for run in group])
        pooled_sites = np.concatenate([run["site_type"] for run in group])
        per_seed = {
            str(run["seed"]): expected_calibration_error(
                run["labels"], run["probabilities"], n_bins=n_bins
            )
            for run in group
        }
        sites: Dict[str, Any] = {}
        for site in SITE_TYPES:
            mask = pooled_sites == site
            sites[site] = {
                "ece": expected_calibration_error(
                    pooled_labels[mask], pooled_probs[mask], n_bins=n_bins
                ),
                "n": int(np.count_nonzero(mask)),
                "provenance": {
                    "runs": [_run_provenance(run) for run in group],
                    "site_type": site,
                },
            }
        result[ablation] = {
            "pooled": {
                "ece": expected_calibration_error(pooled_labels, pooled_probs, n_bins=n_bins),
                "n": int(pooled_labels.size),
            },
            "per_seed": per_seed,
            "per_seed_summary": _scalar_summary(list(per_seed.values())),
            "sites": sites,
            "provenance": {
                "runs": [_run_provenance(run) for run in group],
                "n_bins": int(n_bins),
                "method": ECE_METHOD,
            },
        }
    return {
        "calibration": result,
        "n_bins": int(n_bins),
        "method": ECE_METHOD,
        "provenance": {"runs": [_run_provenance(run) for run in runs]},
    }


# --------------------------------------------------------------------------- #
# 5. Same-split classical baselines
# --------------------------------------------------------------------------- #
def _load_split_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sequence", "label", "site_type", "protein_acc", "cluster_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} is empty")
    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    for column in ("sequence", "site_type", "protein_acc", "cluster_id"):
        frame[column] = frame[column].astype(str)
    return frame


def validate_baselines(
    split_dir: Path, seed: int = DEFAULT_BASELINE_SEED
) -> Dict[str, Any]:
    """Fit both classical baselines on train; score on test. No test tuning."""
    # Imported lazily: src.baselines -> src.model -> torch is only needed when
    # baselines are actually requested (the CLI can skip them for nested tests).
    from src import baselines as _baselines

    split_dir = Path(split_dir)
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    train = _load_split_frame(train_path)
    test = _load_split_frame(test_path)
    result: Dict[str, Any] = {}
    for name in _baselines.BASELINE_NAMES:
        probabilities = _baselines.fit_predict_baseline(
            train["sequence"].tolist(),
            train["label"].to_numpy(),
            test["sequence"].tolist(),
            name,
            seed,
        )
        metrics = _metrics.compute_metrics(
            test["label"].to_numpy(), probabilities, site_types=test["site_type"].tolist()
        )
        result[str(name)] = {
            "metrics": metrics,
            "provenance": {
                "source_files": [
                    _file_provenance(train_path, "baseline_train"),
                    _file_provenance(test_path, "baseline_test"),
                ],
                "seed": int(seed),
                "baseline_config": _baselines.baseline_config(name, seed),
                "metrics_source": "src.metrics.compute_metrics",
                "fit_source": "src.baselines.fit_predict_baseline",
                "tuning": "none (fixed hyperparameters; test split scored once)",
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
            },
        }
    return {
        "baselines": result,
        "provenance": {
            "source_files": [
                _file_provenance(train_path, "baseline_train"),
                _file_provenance(test_path, "baseline_test"),
            ],
            "seed": int(seed),
        },
    }


# --------------------------------------------------------------------------- #
# 6. Cross-validation summary verification
# --------------------------------------------------------------------------- #
def validate_cv(
    cv_dir: Path, split_dir: Path, abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL
) -> Dict[str, Any]:
    """Re-read CV manifest and verify fold counts, summaries and test disjointness."""
    cv_dir, split_dir = Path(cv_dir), Path(split_dir)
    manifest_path = cv_dir / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CV manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_path = split_dir / "test.csv"
    train_path = split_dir / "train.csv"
    test = _load_split_frame(test_path)
    train = _load_split_frame(train_path)
    test_clusters = set(test["cluster_id"].astype(str))
    train_clusters = set(train["cluster_id"].astype(str))

    checks: List[Dict[str, Any]] = []
    runs_out: List[Dict[str, Any]] = []
    configured_folds = (manifest.get("run_config") or {}).get("folds")
    for run in manifest.get("cv_runs", []):
        folds = run.get("folds", [])
        stored = run.get("summary", {})
        ddof = int(stored.get("ddof", 0))
        recomputed = _metrics.summarize_folds(
            [fold["metrics"] for fold in folds], ddof=ddof
        )
        seed = int(run.get("seed", -1))
        name = f"cv_{run.get('ablation')}_{seed}"

        checks.append(
            _check(
                f"{name}.fold_count",
                int(run.get("n_folds", len(folds))) == len(folds)
                and (configured_folds is None or int(configured_folds) == len(folds)),
                f"run n_folds={run.get('n_folds')}, configured={configured_folds}, stored folds={len(folds)}",
            )
        )

        summary_checks: List[Dict[str, Any]] = []
        for metric, stored_entry in stored.get("metrics", {}).items():
            recomputed_entry = recomputed["metrics"].get(metric, {})
            for field in ("values", "mean", "std", "n_defined"):
                actual = recomputed_entry.get(field)
                expected = stored_entry.get(field)
                if field == "values":
                    ok = isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected)
                    if ok:
                        ok = all(
                            _values_close(a, e, abs_tol, rel_tol)
                            for a, e in zip(actual, expected)
                        )
                    detail = f"n_values actual={len(actual) if isinstance(actual, list) else None} stored={len(expected) if isinstance(expected, list) else None}"
                else:
                    ok = _values_close(actual, expected, abs_tol, rel_tol)
                    detail = f"recomputed={actual!r} stored={expected!r}"
                summary_checks.append(
                    _check(f"{name}.summary.{metric}.{field}", ok, detail)
                )
        checks.extend(summary_checks)
        checks.append(
            _check(
                f"{name}.summary.n_folds",
                int(recomputed.get("n_folds", -1)) == int(stored.get("n_folds", -2)),
                f"recomputed={recomputed.get('n_folds')} stored={stored.get('n_folds')}",
            )
        )

        val_sets = [set(fold.get("val_clusters", [])) for fold in folds]
        overlap_pairs: List[str] = []
        for i in range(len(val_sets)):
            for j in range(i + 1, len(val_sets)):
                shared = val_sets[i] & val_sets[j]
                if shared:
                    overlap_pairs.append(f"{i}/{j}:{sorted(shared)[:5]}")
        checks.append(
            _check(
                f"{name}.folds_disjoint",
                not overlap_pairs,
                "whole-cluster folds are pairwise disjoint" if not overlap_pairs else "; ".join(overlap_pairs),
            )
        )

        assignments = (manifest.get("cv_fold_assignments_by_seed") or {}).get(str(seed), {})
        val_to_fold = {cluster: index for index, groups in enumerate(val_sets) for cluster in groups}
        assignment_problems = []
        for cluster, fold_index in assignments.items():
            if val_to_fold.get(cluster) != fold_index:
                assignment_problems.append(f"{cluster}->{fold_index} vs fold {val_to_fold.get(cluster)}")
        for cluster, fold_index in val_to_fold.items():
            if cluster not in assignments:
                assignment_problems.append(f"{cluster} missing from assignments")
        checks.append(
            _check(
                f"{name}.assignments_match_folds",
                not assignment_problems,
                "each CV cluster maps to its stored fold" if not assignment_problems else "; ".join(assignment_problems[:10]),
            )
        )

        cv_clusters = set(assignments) | set().union(*val_sets) if val_sets else set()
        test_overlap = sorted(cv_clusters & test_clusters)
        checks.append(
            _check(
                f"{name}.no_test_cluster_overlap",
                not test_overlap,
                f"{len(cv_clusters)} CV clusters vs {len(test_clusters)} test clusters; overlap={test_overlap[:10]}",
            )
        )
        outside_train = sorted(cv_clusters - train_clusters)
        checks.append(
            _check(
                f"{name}.cv_clusters_within_train",
                not outside_train,
                f"clusters outside train split: {outside_train[:10]}",
            )
        )

        runs_out.append(
            {
                "ablation": str(run.get("ablation")),
                "seed": seed,
                "n_folds": len(folds),
                "fold_val_clusters": [sorted(item) for item in val_sets],
                "summary_stored": stored,
                "summary_recomputed": recomputed,
                "provenance": {
                    "source_file": str(manifest_path),
                    "sha256": _sha256(manifest_path),
                    "train_csv": _file_provenance(train_path, "cv_train"),
                    "test_csv": _file_provenance(test_path, "cv_test_disjointness"),
                    "seed": seed,
                    "ablation": str(run.get("ablation")),
                    "summary_source": manifest.get("cv_summary_source", "unknown"),
                },
            }
        )

    all_assigned: set = set()
    for assignment_map in (manifest.get("cv_fold_assignments_by_seed") or {}).values():
        all_assigned |= set(assignment_map)
    checks.append(
        _check(
            "cv_all_seeds.no_test_cluster_overlap",
            not (all_assigned & test_clusters),
            f"{len(all_assigned)} assigned CV clusters; overlap with test={sorted(all_assigned & test_clusters)[:10]}",
        )
    )

    return {
        "cv": {
            "manifest": str(manifest_path),
            "cv_runs": runs_out,
            "checks": checks,
            "provenance": {
                "source_file": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "source_files": [
                    _file_provenance(train_path, "cv_train"),
                    _file_provenance(test_path, "cv_test_disjointness"),
                ],
                "tolerance": {"abs": abs_tol, "rel": rel_tol},
            },
        }
    }


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
def _default_path(value: Optional[str], relative_to: Path, name: str) -> Path:
    path = Path(value) if value is not None else Path(name)
    if path.exists():
        return path
    candidate = relative_to / path
    if candidate.exists():
        return candidate
    return path


def validate_all(
    results_dir: Optional[str] = None,
    cv_dir: Optional[str] = None,
    split_dir: Optional[str] = None,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    baseline_seed: int = DEFAULT_BASELINE_SEED,
    include_baselines: bool = True,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> Dict[str, Any]:
    """Recompute every reported number and return one provenance-rich report."""
    results_path = _default_path(results_dir, REPO_ROOT / "results", "real")
    cv_path = _default_path(cv_dir, REPO_ROOT / "results", "real_cv")
    split_path = _default_path(split_dir, REPO_ROOT, "data/real/split")

    runs, run_prov = _load_runs(results_path)
    alignment = _check_row_alignment(runs)

    # Every archive must score exactly the rows in the configured test split,
    # in the same order. This is an integrity check only: metric recomputation
    # below is unchanged and still reads solely from the NPZs.
    test_path = split_path / "test.csv"
    test_frame = _load_split_frame(test_path)
    test_csv_sha = _sha256(test_path)
    expected_test_columns = {
        "labels": test_frame["label"].to_numpy(),
        "site_type": test_frame["site_type"].to_numpy(),
        "protein_acc": test_frame["protein_acc"].to_numpy(),
        "cluster_id": test_frame["cluster_id"].to_numpy(),
    }
    test_row_checks: List[Dict[str, Any]] = []
    for run in runs:
        mismatches = [
            mismatch
            for field, expected in expected_test_columns.items()
            for mismatch in [_first_mismatch(run[field], expected, field)]
            if mismatch is not None
        ]
        passed = not mismatches
        detail = (
            "NPZ labels/site_type/protein_acc/cluster_id equal test.csv row-for-row (order included)"
            if passed
            else "first mismatch: " + mismatches[0]["reason"]
        )
        test_row_checks.append(
            {
                **_check(
                    f"run_{run['ablation']}_{run['seed']}.matches_test_csv", passed, detail
                ),
                "source_file": run["source"],
                "sha256": run["sha256"],
                "test_csv": str(test_path),
                "test_csv_sha256": test_csv_sha,
                "first_mismatch": mismatches[0] if mismatches else None,
                "n_mismatched_fields": len(mismatches),
            }
        )

    run_block = validate_runs(runs)
    site_block = validate_per_site_type(runs, precomputed=run_block["computed"])
    bootstrap_block = paired_bootstrap(runs, n_boot=n_boot, seed=bootstrap_seed)
    calibration_block = ece_report(runs)
    cv_block = validate_cv(cv_path, split_path, abs_tol=abs_tol, rel_tol=rel_tol)

    baseline_block: Dict[str, Any] = {}
    if include_baselines:
        baseline_block = validate_baselines(split_path, seed=baseline_seed)
    else:
        baseline_block = {"baselines": {}, "provenance": {"skipped": True}}

    manifest_data = run_prov["manifest_data"]
    config = manifest_data.get("run_config", {})
    checks = [alignment, *test_row_checks]
    checks.append(
        _check(
            "run_count_matches_manifest",
            len(runs) == len(manifest_data.get("runs", [])),
            f"{len(runs)} archives loaded, {len(manifest_data.get('runs', []))} manifest entries",
        )
    )
    run_seeds = sorted({int(run["seed"]) for run in runs})
    configured_seeds = sorted({int(value) for value in config.get("seeds", [])})
    checks.append(
        _check(
            "seed_set_matches_manifest",
            run_seeds == configured_seeds,
            f"run seed set={run_seeds}, configured seed set={configured_seeds}",
        )
    )
    checks.append(
        _check(
            "ablation_set_matches_manifest",
            sorted(set(str(run["ablation"]) for run in runs))
            == sorted(str(value) for value in config.get("ablations", [])),
            f"runs={sorted(set(str(run['ablation']) for run in runs))}, configured={config.get('ablations')}",
        )
    )
    checks.extend(cv_block["cv"]["checks"])

    files: List[Dict[str, str]] = []
    seen_paths = set()
    for prov in [run_prov["manifest"], cv_block["cv"]["provenance"]]:
        path_value = prov.get("path") or prov.get("source_file")
        if path_value and path_value not in seen_paths:
            seen_paths.add(path_value)
            files.append(
                {
                    "path": path_value,
                    "sha256": prov["sha256"],
                    "role": prov.get("role", "manifest"),
                }
            )
    for prov in baseline_block.get("provenance", {}).get("source_files", []):
        if prov["path"] not in seen_paths:
            seen_paths.add(prov["path"])
            files.append(prov)
    for split_name in ("train", "val", "test"):
        split_file = split_path / f"{split_name}.csv"
        if split_file.is_file() and str(split_file) not in seen_paths:
            seen_paths.add(str(split_file))
            files.append(_file_provenance(split_file, f"split_{split_name}"))
    split_manifest = split_path / "manifest.json"
    if split_manifest.is_file():
        files.append(_file_provenance(split_manifest, "split_manifest"))
    for run in runs:
        entry = {"path": run["source"], "sha256": run["sha256"], "role": "predictions"}
        if entry["path"] not in seen_paths:
            seen_paths.add(entry["path"])
            files.append(entry)

    return {
        "config": {
            "validation_module": "src.validate_results",
            "metrics_source": "src.metrics.compute_metrics",
            "baselines_source": "src.baselines.fit_predict_baseline",
            "claim_tolerance": {"abs": abs_tol, "rel": rel_tol},
            "n_boot": int(n_boot),
            "bootstrap_seed": int(bootstrap_seed),
            "baseline_seed": int(baseline_seed),
            "ece_bins": int(calibration_block["n_bins"]),
            "baselines_included": bool(include_baselines),
            "no_result_values_hardcoded": True,
        },
        "provenance": {"files": files, "split_dir": str(split_path)},
        "runs": run_block["per_run"],
        "aggregates": run_block["aggregates"],
        "per_site_type": site_block["per_site_type"],
        "paired_bootstrap": bootstrap_block["paired_bootstrap"],
        "calibration": calibration_block["calibration"],
        "baselines": baseline_block["baselines"],
        "cv": cv_block["cv"],
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# 7. Claim checking (manuscript numbers)
# --------------------------------------------------------------------------- #
def flatten_report(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten a report into dotted paths -> {value, provenance}.

    The nearest ancestor ``provenance`` block is inherited by each leaf, so a
    claim result always names the files/hashes that produced its recomputed
    value.
    """

    flat: Dict[str, Dict[str, Any]] = {}

    def walk(node: Any, prefix: str, inherited: Optional[Any]) -> None:
        if isinstance(node, Mapping):
            provenance = node.get("provenance", inherited)
            for key, value in node.items():
                if key == "provenance":
                    continue
                walk(value, f"{prefix}.{key}" if prefix else str(key), provenance)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{prefix}.{index}", inherited)
        else:
            flat[prefix] = {"value": node, "provenance": inherited}

    walk(report, "", None)
    return flat


def available_claim_paths(
    report: Mapping[str, Any], include_arrays: bool = False
) -> Dict[str, Any]:
    """Return every dotted claim path and its recomputed value.

    Use this (not memory or the manuscript) to build the claims JSON. List-valued
    leaves (bootstrap resamples, per-fold cluster lists) are omitted by default.
    """
    flat = flatten_report(report)
    return {
        path: entry["value"]
        for path, entry in flat.items()
        if include_arrays or not isinstance(entry["value"], (list, tuple, dict))
    }


def check_claimed_numbers(claims: dict, report: dict) -> list[dict]:
    """Compare manuscript claims against the recomputed report.

    ``claims`` maps a claim name to either a bare value or a dict with:

    * ``path`` -- dotted path into ``report`` (defaults to the claim name);
    * ``value`` (or ``claimed``) -- the number written in the manuscript;
    * ``tolerance`` -- optional number (absolute) or ``{"abs": ..., "rel": ...}``.

    Default tolerance is documented in ``report["config"]["claim_tolerance"]``
    (abs=1e-9, rel=1e-6). Manuscript values rounded to three decimals should set
    an explicit absolute tolerance (e.g. 5e-4) in the claim. ``None`` is only
    equal to ``None``; it is never treated as a numeric value.

    Returns one dict per claim with keys ``name``, ``path``, ``claimed``,
    ``recomputed``, ``tolerance``, ``status`` (``PASS``/``FAIL``), ``detail`` and
    ``provenance`` (inherited from the recomputed value).
    """
    flat = flatten_report(report)
    configured = (report.get("config") or {}).get("claim_tolerance", {})
    default_abs = float(configured.get("abs", DEFAULT_ABS_TOL))
    default_rel = float(configured.get("rel", DEFAULT_REL_TOL))
    results: List[Dict[str, Any]] = []
    for name, claim in (claims or {}).items():
        path = str(name)
        value: Any = claim
        tolerance: Any = None
        if isinstance(claim, Mapping):
            path = str(claim.get("path", name))
            if "value" in claim:
                value = claim["value"]
            elif "claimed" in claim:
                value = claim["claimed"]
            else:
                value = None
            tolerance = claim.get("tolerance")
        abs_tol, rel_tol = default_abs, default_rel
        if isinstance(tolerance, Mapping):
            abs_tol = float(tolerance.get("abs", abs_tol))
            rel_tol = float(tolerance.get("rel", rel_tol))
        elif tolerance is not None:
            abs_tol = float(tolerance)
        entry = flat.get(path)
        if entry is None:
            results.append(
                {
                    "name": str(name),
                    "path": path,
                    "claimed": value,
                    "recomputed": None,
                    "tolerance": {"abs": abs_tol, "rel": rel_tol},
                    "status": "FAIL",
                    "detail": "claim path not found in recomputed report",
                    "provenance": None,
                }
            )
            continue
        recomputed = entry["value"]
        passed = _values_close(recomputed, value, abs_tol, rel_tol)
        detail = (
            "within tolerance"
            if passed
            else f"claimed={value!r} recomputed={recomputed!r} (abs={abs_tol}, rel={rel_tol})"
        )
        results.append(
            {
                "name": str(name),
                "path": path,
                "claimed": value,
                "recomputed": recomputed,
                "tolerance": {"abs": abs_tol, "rel": rel_tol},
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
                "provenance": entry["provenance"],
            }
        )
    return results
