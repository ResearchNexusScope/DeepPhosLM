"""Fast, fully offline tests for the homology-controlled real-data split.

A tiny SYNTHETIC dataset (5 proteins, 9 windows), a synthetic cached FASTA
chunk, and a fake MMseqs TSV are written to ``tmp_path``.  A fake runner is
injected into :func:`build_real_split`, so the real MMseqs binary is never
called.  All assertions are structural/deterministic and run in milliseconds.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_split import (  # noqa: E402
    CLUSTER_ID_COLUMN,
    SPLITS,
    OverlapError,
    assert_no_overlap,
    assign_clusters_to_splits,
    build_real_split,
    cluster_commands,
    parse_cluster_tsv,
    sha256_file,
)

DATA_COLUMNS = ["sequence", "label", "site_type", "protein_acc", "gene", "kinase", "source"]

# (protein, label, site_type) -- 9 windows across 5 proteins.
WINDOWS = [
    ("P00001", 1, "S"), ("P00001", 1, "Y"), ("P00001", 0, "T"),
    ("P00002", 1, "S"), ("P00002", 0, "Y"),
    ("P00003", 1, "T"), ("P00003", 0, "S"),
    ("P00004", 1, "Y"),
    ("P00005", 0, "T"),
]

# Two multi-member clusters (A: P00001+P00002, B: P00004+P00005) and P00003,
# which is deliberately absent -> singleton fallback.
FAKE_TSV = (
    "P00001\tP00001\n"
    "P00001\tP00002\n"
    "P00004\tP00004\n"
    "P00004\tP00005\n"
)


def _write_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sequence": f"M{acc[-1]}AAAAAYASAAATASAAAYA",
            "label": label,
            "site_type": site_type,
            "protein_acc": acc,
            "gene": f"G{acc[-1]}",
            "kinase": "-",
            "source": "synthetic",
        }
        for acc, label, site_type in WINDOWS
    ]
    csv_path = data_dir / "data.csv"
    pd.DataFrame(rows, columns=DATA_COLUMNS).to_csv(csv_path, index=False)

    cache_dir = data_dir / "fasta_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # UniProt-style headers exercise src.real_data.parse_fasta (sp|ACC|NAME ...).
    fasta_lines: List[str] = []
    for acc in sorted({acc for acc, _, _ in WINDOWS}):
        fasta_lines.append(f">sp|{acc}|TEST_HUMAN Test protein OS=Homo sapiens")
        fasta_lines.append("MSAAAAAYASAAATASAAAYAMAAAAAAYASAAATASAAAYA")
    (cache_dir / "chunk_synthetic.fasta").write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    return csv_path


class FakeRunner:
    """Records argv and writes a preset cluster TSV for `createtsv`."""

    def __init__(self, tsv: str = FAKE_TSV) -> None:
        self.calls: List[List[str]] = []
        self.tsv = tsv

    def __call__(self, args: Any) -> str:
        argv = [str(arg) for arg in args]
        self.calls.append(argv)
        command = argv[0]
        if command == "version":
            return "fake-mmseqs-1.0\n"
        if command == "createtsv":
            # The output TSV is the positional arg ending in .tsv (flags follow it).
            tsv_path = next(arg for arg in argv[1:] if arg.endswith(".tsv"))
            Path(tsv_path).write_text(self.tsv, encoding="utf-8")
        return ""


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "real"
    _write_dataset(path)
    return path


def _run(data_dir: Path, runner: FakeRunner, work: str, out: str) -> Dict[str, Any]:
    return build_real_split(
        data_dir=data_dir,
        mmseqs=Path("fake/mmseqs.exe"),
        work_dir=data_dir.parent / work,
        out_dir=data_dir.parent / out,
        min_seq_id=0.3,
        seed=666,
        threads=4,
        runner=runner,
    )


def _read_splits(manifest: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    out_dir = Path(manifest["outputs"]["directory"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    return {split: pd.read_csv(out_dir / f"{split}.csv") for split in SPLITS}


# ---------------------------------------------------------------------------
# 1. Cluster TSV parsing + singleton fallback
# ---------------------------------------------------------------------------


def test_parse_cluster_tsv_and_singleton_fallback(tmp_path: Path) -> None:
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text(FAKE_TSV, encoding="utf-8")
    accessions = {f"P{i:05d}" for i in range(1, 6)}

    result = parse_cluster_tsv(tsv, accessions)

    assert result.protein_to_cluster == {
        "P00001": "P00001",  # representative maps to itself
        "P00002": "P00001",  # same cluster as P00001
        "P00003": "P00003",  # absent from TSV -> own singleton cluster
        "P00004": "P00004",
        "P00005": "P00004",
    }
    assert result.n_clusters == 3
    assert result.n_singletons == 1
    assert set(result.protein_to_cluster) == accessions


def test_parse_cluster_tsv_rejects_unknown_accession(tmp_path: Path) -> None:
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("P00001\tP00999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown accession"):
        parse_cluster_tsv(tsv, {"P00001"})


# ---------------------------------------------------------------------------
# 2. Full pipeline
# ---------------------------------------------------------------------------


def test_build_real_split_full(data_dir: Path) -> None:
    runner = FakeRunner()
    manifest = _run(data_dir, runner, "work1", "out1")
    frames = _read_splits(manifest)

    # Manifest headline counts.
    assert manifest["n_proteins"] == 5
    assert manifest["n_clusters"] == 3
    assert manifest["n_singletons"] == 1
    assert manifest["seed"] == 666
    assert manifest["min_seq_id"] == 0.3
    assert manifest["no_overlap"] == {
        "clusters_disjoint": True,
        "proteins_disjoint": True,
        "all_clusters_assigned": True,
        "all_proteins_assigned": True,
        "cluster_protein_consistency": True,
    }

    # Every row is present exactly once, with all original columns + cluster_id.
    all_rows = pd.concat(frames.values(), ignore_index=True)
    assert len(all_rows) == len(WINDOWS) == 9
    assert set(all_rows.columns) == set(DATA_COLUMNS + [CLUSTER_ID_COLUMN])
    assert set(all_rows["protein_acc"]) == {acc for acc, _, _ in WINDOWS}

    # Whole clusters stay together and no protein/cluster crosses splits.
    protein_cluster = dict(zip(all_rows["protein_acc"], all_rows[CLUSTER_ID_COLUMN]))
    protein_split = {
        acc: split for split, frame in frames.items() for acc in frame["protein_acc"]
    }
    assert protein_cluster["P00002"] == protein_cluster["P00001"]  # cluster A
    assert protein_cluster["P00005"] == protein_cluster["P00004"]  # cluster B
    assert protein_split["P00002"] == protein_split["P00001"]
    assert protein_split["P00005"] == protein_split["P00004"]

    cluster_sets = {split: set(frame[CLUSTER_ID_COLUMN]) for split, frame in frames.items()}
    protein_sets = {split: set(frame["protein_acc"]) for split, frame in frames.items()}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1 :]:
            assert cluster_sets[first].isdisjoint(cluster_sets[second])
            assert protein_sets[first].isdisjoint(protein_sets[second])
    assert set().union(*cluster_sets.values()) == {"P00001", "P00003", "P00004"}
    assert set().union(*protein_sets.values()) == set(protein_cluster)


# ---------------------------------------------------------------------------
# 3. Determinism with a fixed seed
# ---------------------------------------------------------------------------


def test_deterministic_assignment_and_bytes(data_dir: Path) -> None:
    manifest_a = _run(data_dir, FakeRunner(), "work_a", "out_a")
    manifest_b = _run(data_dir, FakeRunner(), "work_b", "out_b")
    frames_a = _read_splits(manifest_a)
    frames_b = _read_splits(manifest_b)

    for split in SPLITS:
        pd.testing.assert_frame_equal(frames_a[split], frames_b[split])
    assert manifest_a["splits"] == manifest_b["splits"]

    cluster_ids_a = assign_clusters_to_splits({"P00001", "P00004", "P00003"}, seed=666)
    cluster_ids_b = assign_clusters_to_splits({"P00003", "P00004", "P00001"}, seed=666)
    assert cluster_ids_a == cluster_ids_b
    assert sorted(cid for ids in cluster_ids_a.values() for cid in ids) == [
        "P00001", "P00003", "P00004"
    ]


# ---------------------------------------------------------------------------
# 4. Explicit overlap rejection
# ---------------------------------------------------------------------------


def test_assert_no_overlap_rejects_deliberate_overlap() -> None:
    valid_cluster_to_split = {"A": "train", "B": "val", "C": "test"}
    valid_protein_to_split = {"p1": "train", "p2": "val", "p3": "test"}
    checks = assert_no_overlap(
        clusters_by_split={"train": ["A"], "val": ["B"], "test": ["C"]},
        proteins_by_split={"train": ["p1"], "val": ["p2"], "test": ["p3"]},
        expected_clusters={"A", "B", "C"},
        expected_proteins={"p1", "p2", "p3"},
    )
    assert all(checks.values())

    # A cluster in two splits must be rejected.
    with pytest.raises(OverlapError, match="cluster 'A' appears in both"):
        assert_no_overlap(
            clusters_by_split={"train": ["A"], "val": ["B"], "test": ["A"]},
            proteins_by_split={"train": ["p1"], "val": ["p2"], "test": ["p3"]},
            expected_clusters={"A", "B"},
            expected_proteins={"p1", "p2", "p3"},
        )

    # A protein in two splits must be rejected too.
    with pytest.raises(OverlapError, match="protein 'p1' appears in both"):
        assert_no_overlap(
            clusters_by_split={"train": ["A"], "val": ["B"], "test": ["C"]},
            proteins_by_split={"train": ["p1"], "val": ["p1"], "test": ["p3"]},
            expected_clusters={"A", "B", "C"},
            expected_proteins={"p1", "p3"},
        )

    # A protein whose cluster lives in another split must be rejected.
    with pytest.raises(OverlapError, match="its cluster"):
        assert_no_overlap(
            clusters_by_split={"train": ["A"], "val": ["B"], "test": ["C"]},
            proteins_by_split={"train": ["p1"], "val": ["p2"], "test": ["p3"]},
            expected_clusters={"A", "B", "C"},
            expected_proteins={"p1", "p2", "p3"},
            protein_to_cluster={"p1": "B", "p2": "B", "p3": "C"},
        )


def test_build_refuses_overlap_and_writes_nothing(data_dir: Path) -> None:
    """Injecting an overlapping split into the pipeline must raise pre-write."""
    import src.real_split as real_split

    real_assign = real_split.assign_clusters_to_splits
    out_dir = data_dir.parent / "out_overlap"

    with patch.object(
        real_split,
        "assign_clusters_to_splits",
        side_effect=lambda cluster_ids, seed: {
            "train": ["P00001", "P00003", "P00004"],
            "val": ["P00004"],  # deliberate overlap: P00004 in train and val
            "test": [],
        },
    ):
        with pytest.raises(OverlapError):
            build_real_split(
                data_dir=data_dir,
                work_dir=data_dir.parent / "work_overlap",
                out_dir=out_dir,
                runner=FakeRunner(),
            )
    assert not out_dir.exists(), "no output directory may be created on overlap"
    assert real_split.assign_clusters_to_splits is real_assign


# ---------------------------------------------------------------------------
# 5. Manifest hashes and documented commands
# ---------------------------------------------------------------------------


def test_manifest_hashes_and_commands_are_truthful(data_dir: Path) -> None:
    runner = FakeRunner()
    manifest = _run(data_dir, runner, "work3", "out3")

    # Input hashes.
    assert manifest["inputs"]["data_csv"]["sha256"] == sha256_file(data_dir / "data.csv")
    combined = manifest["inputs"]["fasta"]["combined_fasta"]["sha256"]
    expected_combined = hashlib.sha256(
        (data_dir.parent / "work3" / "proteins.fasta").read_bytes()
    ).hexdigest()
    assert combined == expected_combined
    assert manifest["mmseqs"]["cluster_tsv"]["sha256"] == sha256_file(
        data_dir.parent / "work3" / "clusters.tsv"
    )

    # Output hashes and per-split counts agree with the written CSVs.
    frames = _read_splits(manifest)
    for split in SPLITS:
        csv_name = f"{split}.csv"
        out_csv = data_dir.parent / "out3" / csv_name
        assert manifest["outputs"]["files"][csv_name]["sha256"] == sha256_file(out_csv)
        info = manifest["splits"][split]
        assert info["windows"] == len(frames[split])
        assert info["positives"] == int((frames[split]["label"] == 1).sum())
        assert info["negatives"] == int((frames[split]["label"] == 0).sum())
        assert info["clusters"] == len(set(frames[split][CLUSTER_ID_COLUMN]))
        assert info["proteins"] == int(frames[split]["protein_acc"].nunique())

    assert sum(manifest["splits"][s]["windows"] for s in SPLITS) == len(WINDOWS)
    assert manifest["mmseqs"]["version"] == "fake-mmseqs-1.0"

    # Exact commands passed to the runner, including the documented flags.
    commands, _ = cluster_commands(
        data_dir.parent / "work3" / "proteins.fasta",
        data_dir.parent / "work3",
        min_seq_id=0.3,
        threads=4,
    )
    assert [call for call in runner.calls if call[0] != "version"] == commands
    cluster_call = next(call for call in runner.calls if call[0] == "cluster")
    assert "--min-seq-id" in cluster_call and "0.3" in cluster_call
    assert "-c" in cluster_call and "0.5" in cluster_call
    assert "--cov-mode" in cluster_call and "0" in cluster_call
    assert "--threads" in cluster_call and "4" in cluster_call
