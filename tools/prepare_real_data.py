"""CLI wrapper that builds the real, FASTA-validated phosphosite dataset.

Usage (from anywhere)::

    python tools/prepare_real_data.py
    python tools/prepare_real_data.py --limit-proteins 20 --seed 666
    python tools/prepare_real_data.py --phosphosite /path/to/Phosphosite.xlsx \
        --cache-dir /path/to/cache --out-dir /path/to/output

Network access happens only when this script is invoked (and only for UniProt
FASTA chunks that are not already in the pinned cache).  Importing this module
is side-effect free.
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

from src.real_data import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RATIO,
    DEFAULT_SEED,
    DEFAULT_TIMEOUT,
    PHOSPHOSITE_FILENAME,
    build_real_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real phosphosite dataset: parse Phosphosite.xlsx, fetch "
            "UniProt FASTA, validate annotated positions and write data.csv "
            "plus a hash-pinned manifest.json."
        )
    )
    parser.add_argument(
        "--phosphosite",
        type=Path,
        default=REPO_ROOT / PHOSPHOSITE_FILENAME,
        help=f"Phosphosite spreadsheet (default: <repo>/{PHOSPHOSITE_FILENAME}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "real" / "fasta_cache",
        help="On-disk FASTA cache directory (default: data/real/fasta_cache).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "real",
        help="Output directory for data.csv and manifest.json (default: data/real).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Sampling seed (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"Negatives per validated positive, per protein (default: {DEFAULT_RATIO}).",
    )
    parser.add_argument(
        "--limit-proteins",
        type=int,
        default=None,
        help="Only use the first N substrate accessions (sorted); useful for smoke runs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Accessions per UniProt request (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT:g}).",
    )
    return parser


def print_summary(manifest: Mapping[str, Any]) -> None:
    """Print a compact, human-readable summary of the produced manifest."""
    counts = manifest["counts"]
    discarded = counts["discarded"]
    dedup = counts["deduplicated"]
    validation = manifest["validation"]
    fetch = manifest["fetch"]

    print(
        f"Seed: {manifest['seed']}  ratio: {manifest['ratio']}  "
        f"limit_proteins: {manifest['limit_proteins']}"
    )
    print(f"Input : {manifest['inputs']['phosphosite']['path']}")
    print(f"Output: {manifest['outputs']['directory']}/data.csv")
    print(
        f"Rows: {counts['rows_total']} in sheet -> "
        f"{counts['records_written']} written "
        f"(positive={counts['positives']}, negative={counts['negatives']})"
    )
    print(f"  site types: {json.dumps(counts['site_type_counts'], sort_keys=True)}")
    print(
        f"  validated positives before dedup: "
        f"{counts['validated_positives_before_dedup']}; "
        f"negatives sampled: {counts['negatives_sampled']}"
    )
    print(
        f"  discarded: {json.dumps(discarded, sort_keys=True)} "
        f"(sum={sum(discarded.values())})"
    )
    print(
        f"  deduplicated: identical={dedup['identical_windows_dropped']}, "
        f"conflicting_sequences={dedup['conflicting_sequences_dropped']}, "
        f"conflicting_records={dedup['conflicting_records_dropped']}"
    )
    print(
        f"  validation mismatches/failures: {validation['n_mismatches']} "
        f"(position_mismatch={discarded['position_mismatch']})"
    )
    print(
        f"  fetch: accessions={fetch['n_accessions']}, "
        f"cached chunks={len(fetch['cache_files'])}, "
        f"network calls={fetch['network_calls']}, "
        f"failures={len(fetch['failures'])}"
    )
    if fetch.get("failure_counts"):
        print(f"  failure counts: {json.dumps(fetch['failure_counts'], sort_keys=True)}")
    for failure in fetch["failures"]:
        print(
            f"    ! {failure['accession']} "
            f"[reason={failure.get('reason')}, status={failure.get('status')}]: "
            f"{failure['error']}"
        )
    for warning in manifest["warnings"]:
        print(f"  warning: {warning}")
    print(f"Data SHA-256  : {manifest['outputs']['files']['data.csv']['sha256']}")
    print(f"Input SHA-256 : {manifest['inputs']['phosphosite']['sha256']}")
    print(f"Manifest      : {manifest['outputs']['directory']}/manifest.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_real_dataset(
        phosphosite_path=args.phosphosite,
        cache_dir=args.cache_dir,
        output_dir=args.out_dir,
        seed=args.seed,
        ratio=args.ratio,
        limit_proteins=args.limit_proteins,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
    )
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
