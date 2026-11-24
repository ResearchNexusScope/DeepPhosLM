#!/usr/bin/env python3
"""Mother-only central verifier for the DeepPhosLM correction swarm.

ONE build/gate per reported fix D. No worker may invoke this. Runs a single pytest
pass over the repo, regenerates the honest dataset, and checks the manifest for
leakage and completeness. Prints a full detailed report; never truncates diagnostics.

Exit 0 = GREEN, 1 = RED.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "honest" / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    failures: list[str] = []
    print(f"VERIFY_CENTRAL repo={REPO}")

    # 1. Single canonical test pass.
    print("\n--- pytest ---")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=REPO, capture_output=True, text=True,
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print("STDERR:\n" + proc.stderr, end="")
    if proc.returncode != 0:
        failures.append(f"pytest failed (exit {proc.returncode})")

    # 2. Regenerate the honest dataset from root-level spreadsheets.
    print("\n--- dataset regeneration ---")
    gen = subprocess.run(
        [sys.executable, "tools/make_dataset.py", "--seed", "666"],
        cwd=REPO, capture_output=True, text=True,
    )
    print(gen.stdout, end="")
    if gen.stderr:
        print("STDERR:\n" + gen.stderr, end="")
    if gen.returncode != 0:
        failures.append(f"make_dataset.py failed (exit {gen.returncode})")

    # 3. Manifest checks: provenance, counts, and no leakage.
    print("\n--- manifest audit ---")
    if not MANIFEST.is_file():
        failures.append("data/honest/manifest.json missing")
    else:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        required = {"seed", "ratio", "identity_threshold", "splits", "counts",
                    "site_type_counts", "inputs", "cluster_membership"}
        for key in sorted(required - set(m)):
            failures.append(f"manifest missing key: {key}")

        # splits: {split_name: {clusters: [...], count, positive, negative, ...}}
        splits = m.get("splits", {})
        sets = {k: set(v.get("clusters", [])) for k, v in splits.items()
                if isinstance(v, dict)}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = sets[a] & sets[b]
                if overlap:
                    failures.append(f"cluster leakage {a}/{b}: {sorted(overlap)[:5]}")

        # A protein must never be assigned to two different splits.
        kinase_split: dict[str, str] = {}
        for cid, info in m.get("cluster_membership", {}).items():
            for kinase in info.get("kinases", []):
                other = kinase_split.get(kinase)
                if other is not None and other != info.get("split"):
                    failures.append(
                        f"protein leakage {kinase}: {cid}->{info.get('split')} also {other}"
                    )
                kinase_split[kinase] = info.get("split")

        # inputs: {key: {path, sha256}}
        for key, info in m.get("inputs", {}).items():
            p = REPO / info.get("path", "")
            if not p.is_file():
                failures.append(f"manifest input missing: {key} -> {p}")
            elif sha256(p) != info.get("sha256"):
                failures.append(f"manifest hash mismatch: {info.get('path')}")

        counts = m.get("counts", {})
        print("manifest counts: " + json.dumps(counts, sort_keys=True))

    if failures:
        print(f"\nVERIFY RED — {len(failures)} failure(s):")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nVERIFY GREEN — pytest pass, dataset regenerated, manifest leakage-free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
