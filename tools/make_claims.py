#!/usr/bin/env python3
"""Build the manuscript claims JSON from the recomputed validation report.

This is the ONLY supported way to produce the file passed to
``tools/verify_experiment.py --claims results/real/claims.json``. Every claim
value is read from ``validation_report.json`` through
``src.validate_results.available_claim_paths`` and rounded for publication; no
published number is ever typed into this module. The inherited provenance of
each recomputed leaf is copied into the claim together with an explicit,
documented tolerance suitable for a value rounded to ``DECIMALS`` places.

Usage::

    python tools/make_claims.py \
        --report results/real/validation_report.json \
        --out results/real/claims.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validate_results import (  # noqa: E402
    available_claim_paths,
    flatten_report,
)

# Publication policy constants (NOT result values). The manuscript prints
# floating-point results to DECIMALS places, so half a unit in the last place
# plus a small rounding margin is an explicit, sufficient absolute tolerance.
DECIMALS = 4
FLOAT_ABS_TOL = 0.5 * 10 ** (-DECIMALS) + 1e-9
INT_TOL = 0

HELD_OUT_METRICS = ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "precision", "recall")
CV_METRICS = HELD_OUT_METRICS
SITE_METRICS = ("roc_auc", "pr_auc", "f1")
BASELINE_METRICS = HELD_OUT_METRICS
BOOTSTRAP_FIELDS = ("point_estimate", "ci_low", "ci_high")
ECE_SUMMARY_FIELDS = ("mean", "std")


def _claim(
    name: str,
    path: str,
    available: dict[str, Any],
    flat: dict[str, dict[str, Any]],
    decimals: int = DECIMALS,
) -> dict[str, Any]:
    """Create one claim with a derived value, explicit tolerance, provenance."""
    if path not in available:
        raise KeyError(f"claim path missing from report: {path}")
    raw = available[path]
    if raw is None:
        raise ValueError(f"claim path is undefined (None); cannot publish: {path}")
    if isinstance(raw, bool):
        value: Any = bool(raw)
        tolerance: Any = INT_TOL
    elif isinstance(raw, int):
        value = int(raw)
        tolerance = INT_TOL
    elif isinstance(raw, float):
        value = round(float(raw), decimals)
        tolerance = FLOAT_ABS_TOL
    else:
        # Paths of this kind are not manuscript numbers; fail loudly rather
        # than silently stringify a structure into a claim.
        raise TypeError(f"claim path is not a scalar number: {path} ({type(raw).__name__})")
    return {
        "path": path,
        "value": value,
        "tolerance": tolerance,
        "provenance": flat[path]["provenance"],
    }


def build_claims(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive the complete claims object from one recomputed validation report."""
    available = available_claim_paths(report)
    flat = flatten_report(report)
    claims: dict[str, dict[str, Any]] = {}

    def add(name: str, path: str) -> None:
        if name in claims:
            raise ValueError(f"duplicate claim name: {name}")
        claims[name] = _claim(name, path, available, flat)

    aggregates = report.get("aggregates") or {}
    if not aggregates:
        raise ValueError("report has no aggregates block")
    ablations = [str(key) for key in aggregates]
    first = ablations[0]

    # Held-out test metrics: mean and population std over the report's seeds.
    for ablation in ablations:
        for metric in HELD_OUT_METRICS:
            add(f"test.{ablation}.{metric}.mean", f"aggregates.{ablation}.mean.{metric}")
            add(f"test.{ablation}.{metric}.std", f"aggregates.{ablation}.std.{metric}")
        for count_key in ("n", "n_positive", "n_negative"):
            add(f"test.{ablation}.{count_key}", f"aggregates.{ablation}.mean.{count_key}")

    # Same-split classical baselines (one fit each: no seed std exists).
    for name in (report.get("baselines") or {}):
        for metric in BASELINE_METRICS:
            add(f"baseline.{name}.{metric}", f"baselines.{name}.metrics.{metric}")

    # Seeded paired bootstrap: embedding minus seq_only, with 95% CI.
    for metric in ("auroc", "auprc"):
        for field in BOOTSTRAP_FIELDS:
            add(f"paired.{metric}.{field}", f"paired_bootstrap.{metric}.{field}")

    # Pooled per-site-type breakdown (includes the Y cohort counts).
    per_site = report.get("per_site_type") or {}
    for ablation in ablations:
        sites = sorted(key for key in (per_site.get(ablation) or {}) if key != "provenance")
        for site in sites:
            for count_key in ("n", "n_positive", "n_negative"):
                add(f"per_site.{ablation}.{site}.{count_key}",
                    f"per_site_type.{ablation}.{site}.pooled.{count_key}")
            for metric in SITE_METRICS:
                add(f"per_site.{ablation}.{site}.{metric}",
                    f"per_site_type.{ablation}.{site}.pooled.{metric}")

    # Cross-validation summaries, selected by ablation (not by list position).
    cv_runs = ((report.get("cv") or {}).get("cv_runs") or [])
    cv_index_by_ablation: dict[str, int] = {}
    for index, run in enumerate(cv_runs):
        cv_index_by_ablation.setdefault(str(run.get("ablation")), index)
    for ablation in ablations:
        index = cv_index_by_ablation.get(ablation)
        if index is None:
            continue
        prefix = f"cv.cv_runs.{index}"
        add(f"cv.{ablation}.n_folds", f"{prefix}.n_folds")
        for metric in CV_METRICS:
            add(f"cv.{ablation}.{metric}.mean",
                f"{prefix}.summary_recomputed.metrics.{metric}.mean")
            add(f"cv.{ablation}.{metric}.std",
                f"{prefix}.summary_recomputed.metrics.{metric}.std")

    # Calibration (ECE) pooled, per-seed summary and per-residue breakdown.
    calibration = report.get("calibration") or {}
    for ablation in ablations:
        if ablation not in calibration:
            continue
        add(f"ece.{ablation}.pooled", f"calibration.{ablation}.pooled.ece")
        for field in ECE_SUMMARY_FIELDS:
            add(f"ece.{ablation}.{field}", f"calibration.{ablation}.per_seed_summary.{field}")
        for site in sorted(calibration[ablation].get("sites", {})):
            add(f"ece.{ablation}.{site}", f"calibration.{ablation}.sites.{site}.ece")

    # Dataset counts that are actually recomputed in the report (the broader
    # accession-level accounting lives in hash-pinned manifests and is
    # validated separately by tools/validate_manuscript_numbers.py).
    for count_key in ("n", "n_positive", "n_negative"):
        add(f"dataset.test.{count_key}", f"aggregates.{first}.mean.{count_key}")
        add(f"dataset.Y.{count_key}", f"per_site_type.{first}.Y.pooled.{count_key}")
        # One held-out test set's Y cohort (the per-seed view, n=3,583 here).
        add(f"dataset.Y_test_one_seed.{count_key}",
            f"aggregates.{first}.per_seed.{aggregates[first]['seeds'][0]}.per_site_type.Y.{count_key}")

    return claims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "results" / "real" / "validation_report.json",
        help="validation_report.json produced by tools/verify_experiment.py",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "real" / "claims.json",
        help="destination claims JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("validation report must be a JSON object")
    claims = build_claims(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Derived {len(claims)} claims from {args.report} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
