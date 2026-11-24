"""Thin CLI wrapper around ``src.data_pipeline.run_pipeline``.

Usage (from anywhere)::

    python tools/make_dataset.py
    python tools/make_dataset.py --data-dir /path/to/spreadsheets --seed 42 --ratio 1.0 --identity-threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline import (  # noqa: E402
    DEFAULT_IDENTITY_THRESHOLD,
    DEFAULT_RATIO,
    DEFAULT_SEED,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-free kinase phosphosite splits under data/honest/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing kinase_phosphorylation_sites.xlsx and "
            "Modified_TOP_20_FOR_STRUCTURE.xlsx (default: repository root)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 666)."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="Negatives per positive, per kinase (default: 1.0).",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=DEFAULT_IDENTITY_THRESHOLD,
        help="Homology clamp threshold in [0, 1] (default: 0.4).",
    )
    return parser


def print_summary(manifest: dict) -> None:
    """Print a compact, human-readable summary of the generated manifest."""
    totals = manifest["totals"]
    print(f"Seed: {manifest['seed']}  ratio: {manifest['ratio']}  "
          f"threshold: {manifest['threshold']}")
    print(f"Output: {manifest['outputs']['directory']}")
    print(
        f"Total windows: {totals['windows']} "
        f"(positive={totals['positive']}, negative={totals['negative']}); "
        f"clusters={totals['clusters']}, kinases={totals['kinases']}"
    )
    for split in ("train", "val", "test"):
        info = manifest["splits"][split]
        print(
            f"  {split:<5} {info['count']:>5} windows "
            f"(positive={info['positive']}, negative={info['negative']}, "
            f"clusters={len(info['clusters'])}, "
            f"site_types={json.dumps(info['site_type_counts'], sort_keys=True)})"
        )
    if manifest.get("warnings"):
        print(f"Warnings ({len(manifest['warnings'])}):")
        for warning in manifest["warnings"]:
            print(f"  - {warning}")
    print("Manifest: " + str(Path(manifest["outputs"]["directory"]) / "manifest.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_pipeline(
        data_dir=args.data_dir,
        seed=args.seed,
        ratio=args.ratio,
        identity_threshold=args.identity_threshold,
    )
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
