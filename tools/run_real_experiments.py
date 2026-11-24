#!/usr/bin/env python3
"""Run multi-seed, multi-ablation training on the REAL homology-controlled splits.

Usage (from anywhere)::

    python tools/run_real_experiments.py --seeds 5 --folds 5
    python tools/run_real_experiments.py --seeds 2 --folds 0 --epochs 1 --limit-rows 64

What it does
------------
For every ``seed x ablation`` combination it trains the binary model from
:mod:`src.model` on the real ``train.csv`` / ``val.csv`` rows, selects the best
epoch on validation only, predicts the held-out ``test.csv`` rows, and writes
``predictions_<ablation>_seed<N>.npz`` plus a provenance-carrying
``training_manifest.json``.  Optional k-fold CV (``--folds > 0``) is run on the
train split only, with whole ``cluster_id`` groups as the fold unit, and its
per-fold metrics are summarised with ``src.metrics.summarize_folds`` (mean/std
and the full per-fold vectors).  No hard-coded metric appears anywhere and no
test label is ever used for model selection.

Cost and configuration
----------------------
CPU runtime scales with ``seeds x ablations`` for the held-out runs and with
``seeds x ablations x folds`` when CV is enabled.  The mother session decides
the actual run configuration; this CLI only implements the mechanism.
``--limit-rows`` truncates each split to a small prefix for fast smoke checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_train import (  # noqa: E402
    NPZ_KEYS,
    SPLITS,
    cross_validate,
    input_file_hashes,
    load_split_csv,
    save_predictions,
    train_one_run,
)

DEFAULT_SPLIT_DIR = "data/real/split"
DEFAULT_OUT_DIR = "results/real"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEEDS = 5
DEFAULT_FOLDS = 5
DEFAULT_SEED_BASE = 666
ABLATION_CHOICES = ("seq_only", "seq_embedding", "both")


def _resolve_input(path: Path) -> Path:
    """Resolve a relative input path from cwd first, then the repo root."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def _resolve_output(path: Path, default: str) -> Path:
    """Resolve an output directory; the shipped default lives at the repo root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if str(candidate).replace("\\", "/") == default:
        return REPO_ROOT / candidate
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the binary phosphosite model on the real splits for several "
            "seeds and ablations, saving held-out predictions and a manifest."
        )
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help=f"Directory with train/val/test CSVs and manifest.json (default: {DEFAULT_SPLIT_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for prediction archives and manifest (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help=f"Number of seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help=f"CV folds on train only; 0 disables CV (default: {DEFAULT_FOLDS})",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--ablation",
        choices=list(ABLATION_CHOICES),
        default="both",
        help="Which architecture variant(s) to train (default: both)",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional cap on rows read per split, for smoke checks only.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help=f"First seed is seed-base; seeds are consecutive (default: {DEFAULT_SEED_BASE})",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.seeds < 1:
        raise ValueError(f"--seeds must be >= 1, got {args.seeds}")
    if args.folds < 0 or args.folds == 1:
        raise ValueError(f"--folds must be 0 or >= 2, got {args.folds}")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.limit_rows is not None and args.limit_rows < 1:
        raise ValueError(f"--limit-rows must be >= 1 when given, got {args.limit_rows}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    split_dir = _resolve_input(Path(args.split_dir))
    out_dir = _resolve_output(Path(args.out_dir), DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    hashes = input_file_hashes(split_dir)
    frames = {}
    for split in SPLITS:
        frame = load_split_csv(split_dir / f"{split}.csv")
        if args.limit_rows is not None:
            frame = frame.head(args.limit_rows).reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"{split}.csv is empty after --limit-rows={args.limit_rows}")
        frames[split] = frame

    ablations = (
        ("seq_only", "seq_embedding") if args.ablation == "both" else (args.ablation,)
    )
    seeds = [int(args.seed_base) + offset for offset in range(int(args.seeds))]

    run_config: Dict[str, Any] = {
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "seed_base": int(args.seed_base),
        "seeds": seeds,
        "folds": int(args.folds),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "ablations": list(ablations),
        "limit_rows": args.limit_rows,
    }

    runs: List[Dict[str, Any]] = []
    cv_runs: List[Dict[str, Any]] = []
    cv_fold_assignments: Dict[str, Dict[str, int]] = {}

    for seed in seeds:
        for ablation in ablations:
            result = train_one_run(
                train_rows=frames["train"],
                val_rows=frames["val"],
                predict_rows=frames["test"],
                seed=seed,
                ablation=ablation,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            filename = f"predictions_{ablation}_seed{seed}.npz"
            prediction_path = save_predictions(result, out_dir / filename)
            runs.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "chosen_epoch": result["chosen_epoch"],
                    "selection_metric": result["selection_metric"],
                    "config": result["config"],
                    "prediction_file": str(prediction_path),
                }
            )

            if args.folds > 0:
                cv = cross_validate(
                    train_rows=frames["train"],
                    n_folds=args.folds,
                    seed=seed,
                    ablation=ablation,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
                # Fold assignment depends only on the seed, so store it once per seed.
                cv_fold_assignments.setdefault(str(seed), cv["fold_assignments"])
                # The summary already carries mean, std and full per-fold vectors.
                cv_runs.append(
                    {
                        "ablation": ablation,
                        "seed": seed,
                        "n_folds": cv["n_folds"],
                        "folds": cv["folds"],
                        "summary": cv["summary"],
                    }
                )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "input_csv_sha256": hashes["csv"],
        "split_manifest_sha256": hashes["manifest"],
        "split_manifest_path": str(split_dir / "manifest.json"),
        "split_counts": {split: int(len(frames[split])) for split in SPLITS},
        "npz_keys": list(NPZ_KEYS),
        "metrics_source": "src.metrics.compute_metrics",
        "cv_summary_source": "src.metrics.summarize_folds",
        "runs": runs,
        "cv_runs": cv_runs,
        "cv_fold_assignments_by_seed": cv_fold_assignments,
        "notes": [
            "Model selection uses validation AUROC (validation BCE fallback when "
            "the validation split is single-class) only; test labels are never "
            "used for selection.",
            "CV, when enabled, uses only train rows and whole cluster_id groups "
            "as the fold unit.",
            "CPU cost scales with seeds x ablations (x folds when CV is enabled); "
            "--limit-rows exists for smoke checks.",
        ],
    }
    manifest_path = out_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {manifest_path} ({len(runs)} runs, {len(cv_runs)} CV runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
