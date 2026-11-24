#!/usr/bin/env python3
"""CLI for the honest imbalance fix: positive-weighted BCE, no fake negatives.

Usage (from anywhere)::

    python tools/train_balanced.py --split-dir data/real/split \
        --out-dir results/real_balanced --seeds 5 --epochs 8 \
        --batch-size 1024 --ablation both --seed-base 666

For every ``seed x ablation`` this trains the unmodified binary model from
:mod:`src.model` on the real train/val CSVs with
``BCEWithLogitsLoss(pos_weight = n_train_negative / n_train_positive)``, selects
the best epoch on validation AUROC only, scores the held-out test rows, and
writes ``predictions_<ablation>_seed<N>.npz`` with the E-003 keys plus the
scalar ``pos_weight`` it actually used.  A ``training_manifest.json`` records
exact class counts, the prevalence-only baseline recomputed with
:func:`src.metrics.compute_metrics`, input/output hashes, seeds and configs.

There is no resampling and no synthetic negative anywhere in this pipeline.
Every metric and count in the manifest is computed from the supplied CSVs and
NPZs at run time; none is hardcoded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.imbalance import (  # noqa: E402
    BALANCED_NPZ_KEYS,
    FIX_DESCRIPTION,
    class_counts,
    compute_pos_weight,
    imbalance_report,
    prevalence_baseline,
    save_balanced_predictions,
    train_weighted_run,
)
from src.real_split import sha256_file  # noqa: E402
from src.real_train import (  # noqa: E402
    SPLITS,
    input_file_hashes,
    load_split_csv,
)

DEFAULT_SPLIT_DIR = "data/real/split"
DEFAULT_OUT_DIR = "results/real_balanced"
DEFAULT_SEEDS = 5
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
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
            "Train positive-weighted BCE runs on the real phosphosite splits "
            "and write honest predictions plus a prevalence-aware manifest."
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
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help=f"Number of seeds (default: {DEFAULT_SEEDS})"
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
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help=f"First seed is seed-base; seeds are consecutive (default: {DEFAULT_SEED_BASE})",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.seeds < 1:
        raise ValueError(f"--seeds must be >= 1, got {args.seeds}")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")


def _selected_ablations(raw: str) -> Tuple[str, ...]:
    if raw == "both":
        return ("seq_only", "seq_embedding")
    return (raw,)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    split_dir = _resolve_input(Path(args.split_dir))
    out_dir = _resolve_output(Path(args.out_dir), DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    hashes = input_file_hashes(split_dir)
    frames = {split: load_split_csv(split_dir / f"{split}.csv") for split in SPLITS}

    ablations = _selected_ablations(args.ablation)
    seeds = [int(args.seed_base) + offset for offset in range(int(args.seeds))]

    counts = {split: class_counts(frames[split]["label"]) for split in SPLITS}
    pos_weight = compute_pos_weight(frames["train"]["label"])
    baseline = {
        split: prevalence_baseline(frames[split]["label"]) for split in SPLITS
    }
    balance = imbalance_report(
        frames["train"]["label"], frames["test"]["label"]
    )

    run_config: Dict[str, Any] = {
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "seed_base": int(args.seed_base),
        "seeds": seeds,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "ablations": list(ablations),
        "loss": "BCEWithLogitsLoss with pos_weight = n_train_negative / n_train_positive",
        "resampling": "none (no synthetic negatives, no dropped rows)",
    }

    runs: List[Dict[str, Any]] = []
    output_hashes: Dict[str, str] = {}
    for seed in seeds:
        for ablation in ablations:
            result = train_weighted_run(
                train_rows=frames["train"],
                val_rows=frames["val"],
                predict_rows=frames["test"],
                seed=seed,
                ablation=ablation,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            filename = f"predictions_{ablation}_seed{seed}.npz"
            prediction_path = save_balanced_predictions(result, out_dir / filename)
            output_hashes[filename] = sha256_file(prediction_path)
            runs.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "chosen_epoch": result["chosen_epoch"],
                    "selection_metric": result["selection_metric"],
                    "pos_weight": result["pos_weight"],
                    "train_counts": result["train_counts"],
                    "config": result["config"],
                    "prediction_file": str(prediction_path),
                }
            )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fix": FIX_DESCRIPTION,
        "run_config": run_config,
        "input_csv_sha256": hashes["csv"],
        "split_manifest_sha256": hashes["manifest"],
        "split_manifest_path": str(split_dir / "manifest.json"),
        "prediction_sha256": output_hashes,
        "split_counts": counts,
        "class_balance": balance,
        "pos_weight": pos_weight,
        "prevalence_baseline": baseline,
        "metrics_source": "src.metrics.compute_metrics (via src.imbalance.prevalence_baseline)",
        "npz_keys": list(BALANCED_NPZ_KEYS),
        "runs": runs,
        "notes": [
            "pos_weight = n_train_negative / n_train_positive; computed from train.csv only.",
            "ROC-AUC of a constant prevalence score is 0.5 and its AUPRC equals "
            "prevalence; these are derived expectations, recomputed here with "
            "src.metrics.compute_metrics and never hardcoded.",
            "Model selection uses validation AUROC (validation BCE fallback only "
            "when validation is single-class); test labels are never used for selection.",
            "No negative is synthesised from other proteins or from unannotated "
            "residues; every supplied row is kept as labelled.",
        ],
    }
    manifest_path = out_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {manifest_path} ({len(runs)} runs, pos_weight={pos_weight})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
