#!/usr/bin/env python3
"""Fit same-split classical baselines and write ``results/baselines.json``.

Usage (from anywhere)::

    python tools/run_experiments.py
    python tools/run_experiments.py --data-dir data/honest --seed 666 --out results/baselines.json

This script reads only the honest train/val/test CSVs produced by F-001.  It
fits the classical comparators from :mod:`src.baselines` on **train** and
scores them on **test** using :func:`src.metrics.compute_metrics`.  The
validation CSV is read solely to record its checksum; no tuning or model
selection is performed on either val or test.

No external phosphosite predictor (NetPhos, GPS, MusiteDeep, TransPhos,
GraphPhos, ...) is installed or executed here.  The output states that scope
explicitly, and every reported number is computed from actual predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import BASELINE_NAMES, baseline_config, fit_predict_baseline  # noqa: E402
from src.data_pipeline import CSV_COLUMNS  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("sequence", "label", "site_type")
COMPARISON_SCOPE = (
    "Same-split classical baselines only. No external SOTA phosphosite "
    "predictor was executed; no claim of superiority over published tools is made."
)
DEFAULT_SEED = 666

# Imported for reuse by callers/tests that want the exact canonical columns.
DATASET_COLUMNS = CSV_COLUMNS


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit classical same-split phosphosite baselines and write a "
            "provenance-carrying baselines.json."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "honest",
        help="Directory holding train.csv, val.csv, test.csv and manifest.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Estimator seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "baselines.json",
        help="Output JSON path (default: results/baselines.json).",
    )
    return parser


def load_split(path: Path) -> pd.DataFrame:
    """Load one honest split CSV, failing loudly if required columns are absent."""
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    return frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    seed = int(args.seed)

    frames: Dict[str, pd.DataFrame] = {}
    input_hashes: Dict[str, str] = {}
    for split in SPLITS:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing honest split CSV: {csv_path}. Run tools/make_dataset.py first."
            )
        frames[split] = load_split(csv_path)
        input_hashes[f"{split}.csv"] = sha256_file(csv_path)

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)

    train = frames["train"]
    test = frames["test"]
    sequences_train = train["sequence"].astype(str).tolist()
    labels_train = train["label"].astype(int).to_numpy()
    sequences_test = test["sequence"].astype(str).tolist()
    labels_test = test["label"].astype(int).to_numpy()
    site_types_test = test["site_type"].astype(str).tolist()

    models: Dict[str, Dict[str, object]] = {}
    for name in BASELINE_NAMES:
        probabilities = fit_predict_baseline(
            sequences_train, labels_train, sequences_test, name, seed
        )
        test_metrics = compute_metrics(
            labels_test, probabilities, site_types=site_types_test
        )
        models[name] = {
            "name": name,
            "config": baseline_config(name, seed),
            "seed": seed,
            "dataset_manifest_sha256": manifest_sha256,
            "input_csv_sha256": dict(input_hashes),
            "test_metrics": test_metrics,
        }

    report: Dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": COMPARISON_SCOPE,
        "seed": seed,
        "dataset_dir": str(data_dir),
        "dataset_manifest_sha256": manifest_sha256,
        "input_csv_sha256": dict(input_hashes),
        "split_counts": {split: int(len(frames[split])) for split in SPLITS},
        "metrics_source": "src.metrics.compute_metrics",
        "models": models,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {out_path} ({len(models)} baselines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
