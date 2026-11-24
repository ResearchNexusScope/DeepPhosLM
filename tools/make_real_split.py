"""CLI wrapper that builds the homology-controlled real-data split.

Usage (from anywhere)::

    python tools/make_real_split.py
    python tools/make_real_split.py --min-seq-id 0.4 --seed 666 --threads 4
    python tools/make_real_split.py --data-dir /path/to/data/real \
        --mmseqs /path/to/mmseqs.exe --work-dir /path/to/work --out-dir /path/to/split

Runs :func:`src.real_split.build_real_split` and prints a compact manifest
summary.  No network access happens: sequences come from the pinned FASTA cache
and MMseqs is executed as a local binary.  Importing this module is
side-effect free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_split import (  # noqa: E402
    DEFAULT_MIN_SEQ_ID,
    DEFAULT_SEED,
    DEFAULT_THREADS,
    SPLITS,
    build_real_split,
    default_mmseqs_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster real-data proteins with MMseqs2 and write whole-cluster "
            "train/val/test splits plus a hash-pinned manifest.json."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "real",
        help="Directory holding data.csv and fasta_cache/ (default: data/real).",
    )
    parser.add_argument(
        "--mmseqs",
        type=Path,
        default=default_mmseqs_path(),
        help="Path to the MMseqs binary (default: tools/bin/mmseqs/bin/mmseqs.exe).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch directory for the FASTA, MMseqs DBs and cluster TSV "
        "(default: <data-dir>/split_work).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for split CSVs and manifest.json "
        "(default: <data-dir>/split).",
    )
    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=DEFAULT_MIN_SEQ_ID,
        help=f"MMseqs --min-seq-id clustering threshold (default: {DEFAULT_MIN_SEQ_ID}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the whole-cluster 80/10/10 assignment (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Fixed thread count for every MMseqs command (default: {DEFAULT_THREADS}).",
    )
    return parser


def print_summary(manifest: Mapping[str, Any]) -> None:
    """Print a compact, human-readable summary of the produced manifest."""
    mmseqs = manifest["mmseqs"]
    print(
        f"Method : {manifest['method']}\n"
        f"Seed   : {manifest['seed']}  min_seq_id: {manifest['min_seq_id']}  "
        f"threads: {mmseqs['parameters']['threads']}\n"
        f"MMseqs : {mmseqs['binary']} (version {mmseqs['version']})\n"
        f"Input  : {manifest['inputs']['data_csv']['path']}\n"
        f"Output : {manifest['outputs']['directory']}\n"
        f"Proteins: {manifest['n_proteins']}  "
        f"clusters: {manifest['n_clusters']}  "
        f"proteins absent from TSV (own singleton): {manifest['n_singletons']}  "
        f"size-1 clusters: {manifest['n_singleton_clusters']}"
    )
    for split in SPLITS:
        info = manifest["splits"][split]
        print(
            f"  {split:<5} {info['windows']:>7} windows "
            f"(positive={info['positives']}, negative={info['negatives']}, "
            f"proteins={info['proteins']}, clusters={info['clusters']}, "
            f"site_types={json.dumps(info['site_types'], sort_keys=True)})"
        )
    print(
        "No-overlap assertions: "
        + json.dumps(manifest["no_overlap"], sort_keys=True)
    )
    for command in mmseqs["commands"]:
        print("  mmseqs " + " ".join(command))
    print("Manifest: " + str(Path(manifest["outputs"]["directory"]) / "manifest.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_real_split(
        data_dir=args.data_dir,
        mmseqs=args.mmseqs,
        work_dir=args.work_dir,
        out_dir=args.out_dir,
        min_seq_id=args.min_seq_id,
        seed=args.seed,
        threads=args.threads,
    )
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
