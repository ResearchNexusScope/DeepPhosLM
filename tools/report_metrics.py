#!/usr/bin/env python3
"""Mother-owned glue: turn prediction archives into a provenance-carrying metrics JSON.

Numbers used in the revised manuscript must come from this file, never be retyped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import compute_metrics  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "results" / "metrics.json"))
    ap.add_argument("--dataset-manifest", default=str(REPO / "data" / "honest" / "manifest.json"))
    args = ap.parse_args()

    results = Path(args.results_dir)
    manifest_path = Path(args.dataset_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "dataset_seed": manifest.get("seed") if manifest else None,
        "dataset_split_counts": manifest.get("counts") if manifest else None,
        "models": {},
    }

    for archive in sorted(results.glob("predictions_*.npz")):
        with np.load(archive, allow_pickle=True) as data:
            metrics = compute_metrics(
                data["labels"],
                data["probabilities"],
                site_types=data["site_type"],
            )
            cfg = data["config"].item() if data["config"].shape == () else data["config"]
        report["models"][archive.stem] = {
            "archive_sha256": sha256(archive),
            "archive": str(archive.relative_to(REPO)),
            "config": cfg,
            "test_metrics": metrics,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"WROTE {out} ({len(report['models'])} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
