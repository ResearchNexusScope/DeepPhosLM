"""Unit tests for the leakage-free dataset pipeline.

Only stdlib + pytest + pandas are used.  A tiny synthetic spreadsheet fixture
lives in tmp_path; the real 483-row spreadsheets are never read here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline import (  # noqa: E402
    CSV_COLUMNS,
    FASTA_FILENAME,
    SITES_FILENAME,
    SPLITS,
    WINDOW_LENGTH,
    assign_clusters,
    get_15mer,
    run_pipeline,
    sample_negatives,
    sequence_identity,
)

# Six proteins with two annotated sites each.  Distinct residue patterns keep
# the proteins in separate clusters at the high test threshold, while every
# protein still has plenty of unannotated S/T/Y positions for negatives.
FASTA = {
    "K1": "S" * 24,
    "K2": "T" * 24,
    "K3": "Y" * 24,
    "K4": "ST" * 12,
    "K5": "SY" * 12,
    "K6": "TY" * 12,
}
POSITIONS = (5, 15)
TYPE_BY_KINASE = {
    "K1": "Ser/Thr Kinase",
    "K2": "Ser/Thr Kinase",
    "K3": "Tyrosine Kinase",
    "K4": "Ser/Thr Kinase",
    "K5": "Tyrosine Kinase",  # intentionally typed tyrosine, S centres
    "K6": "Ser/Thr Kinase",
}
THRESHOLD = 0.9  # guarantees singleton clusters for the synthetic proteins


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"KINASE": kinase, "FASTA": seq, "Organ": "lab", "Disease": "none"}
            for kinase, seq in FASTA.items()
        ]
    ).to_excel(data_dir / FASTA_FILENAME, index=False)
    pd.DataFrame(
        [
            {
                "Kinase": kinase,
                "Type": TYPE_BY_KINASE[kinase],
                "Position": position,
                "Motif": "SSSS",
            }
            for kinase in FASTA
            for position in POSITIONS
        ]
    ).to_excel(data_dir / SITES_FILENAME, index=False)
    return data_dir, tmp_path / "out"


def _build(tiny_dataset, output_dir: Path, **kwargs):
    data_dir, _ = tiny_dataset
    return run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        seed=kwargs.pop("seed", 7),
        ratio=kwargs.pop("ratio", 1.0),
        identity_threshold=kwargs.pop("identity_threshold", THRESHOLD),
        **kwargs,
    )


def _read_split(output_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(
        output_dir / f"{split}.csv",
        dtype={"sequence": str, "site_type": str, "kinase": str, "cluster_id": str},
    )


def test_splits_are_leakage_free(tiny_dataset, tmp_path: Path) -> None:
    output_dir = tmp_path / "leakage"
    manifest = _build(tiny_dataset, output_dir)

    frames = {split: _read_split(output_dir, split) for split in SPLITS}
    kinases_by_split = {s: set(df["kinase"]) for s, df in frames.items()}
    clusters_by_split = {s: set(df["cluster_id"]) for s, df in frames.items()}

    for a, b in itertools.combinations(SPLITS, 2):
        assert kinases_by_split[a].isdisjoint(kinases_by_split[b]), (a, b)
        assert clusters_by_split[a].isdisjoint(clusters_by_split[b]), (a, b)

    assert set().union(*kinases_by_split.values()) == set(FASTA)
    assert set().union(*clusters_by_split.values()) == set(manifest["cluster_membership"])
    for split in SPLITS:
        assert clusters_by_split[split] == set(manifest["splits"][split]["clusters"])
        for cluster_id in clusters_by_split[split]:
            assert manifest["cluster_membership"][cluster_id]["split"] == split


def test_ratio_and_counts(tiny_dataset, tmp_path: Path) -> None:
    output_dir = tmp_path / "ratio"
    manifest = _build(tiny_dataset, output_dir)

    frames = {split: _read_split(output_dir, split) for split in SPLITS}
    total_positive = sum(int(df["label"].sum()) for df in frames.values())
    total_negative = sum(len(df) - int(df["label"].sum()) for df in frames.values())

    assert total_positive == len(FASTA) * len(POSITIONS) == 12
    assert total_negative == total_positive  # configured 1:1
    assert manifest["ratio"] == 1.0
    assert manifest["totals"]["positive"] == total_positive
    assert manifest["totals"]["negative"] == total_negative
    for split in SPLITS:
        info = manifest["splits"][split]
        assert info["count"] == len(frames[split]) == manifest["counts"][split]
        assert info["positive"] == int(frames[split]["label"].sum())
        assert info["negative"] == len(frames[split]) - info["positive"]
    assert sum(manifest["site_type_counts"].values()) == total_positive + total_negative


def test_configurable_ratio(tiny_dataset, tmp_path: Path) -> None:
    output_dir = tmp_path / "half"
    manifest = _build(tiny_dataset, output_dir, ratio=0.5)
    frames = [_read_split(output_dir, split) for split in SPLITS]
    assert sum(int(df["label"].sum()) for df in frames) == 12
    assert sum(len(df) - int(df["label"].sum()) for df in frames) == 6
    assert manifest["totals"]["negative"] == 6


def test_windows_are_length_15_with_consistent_centres(tiny_dataset, tmp_path: Path) -> None:
    output_dir = tmp_path / "shape"
    _build(tiny_dataset, output_dir)
    for split in SPLITS:
        df = _read_split(output_dir, split)
        assert tuple(df.columns) == CSV_COLUMNS
        assert df["sequence"].str.len().eq(WINDOW_LENGTH).all()
        assert set(df["label"].unique()) <= {0, 1}
        assert set(df["site_type"].unique()) <= {"S", "T", "Y"}
        # Centre of a 15-mer is index 7; it must equal the recorded site type.
        assert (df["sequence"].str[7] == df["site_type"]).all()


def test_manifest_is_valid_json_with_required_keys(tiny_dataset, tmp_path: Path) -> None:
    output_dir = tmp_path / "manifest"
    manifest = _build(tiny_dataset, output_dir)

    loaded = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded == manifest
    for key in (
        "seed",
        "ratio",
        "threshold",
        "identity_threshold",
        "inputs",
        "counts",
        "totals",
        "site_type_counts",
        "splits",
        "cluster_membership",
    ):
        assert key in loaded
    for split in SPLITS:
        assert {"count", "positive", "negative", "site_type_counts", "clusters"} <= set(
            loaded["splits"][split]
        )
    data_dir, _ = tiny_dataset
    for name, filename in (("sites", SITES_FILENAME), ("fasta", FASTA_FILENAME)):
        expected = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        assert loaded["inputs"][name]["sha256"] == expected


def test_pipeline_is_deterministic_for_fixed_seed(tiny_dataset, tmp_path: Path) -> None:
    first_dir, second_dir = tmp_path / "run1", tmp_path / "run2"
    first = _build(tiny_dataset, first_dir, seed=123)
    second = _build(tiny_dataset, second_dir, seed=123)

    assert first["cluster_membership"] == second["cluster_membership"]
    for split in SPLITS:
        # Compare membership as sets: order is irrelevant to leakage.
        assert set(first["splits"][split]["clusters"]) == set(
            second["splits"][split]["clusters"]
        )
        assert first["counts"][split] == second["counts"][split]
        assert _read_split(first_dir, split).to_dict("records") == _read_split(
            second_dir, split
        ).to_dict("records")


def test_identity_and_greedy_clustering_rules() -> None:
    assert sequence_identity("AAAA", "AAAA") == 1.0
    assert sequence_identity("AAAA", "TTTT") == 0.0
    assert sequence_identity("AAAA", "AA") == 0.5  # 2 matches / max length 4

    clusters = assign_clusters(
        {"A": "SSSSSSSS", "B": "SSSSSSSS", "C": "YYYYYYYY"}, threshold=0.9
    )
    members = list(clusters.values())
    assert sorted(itertools.chain.from_iterable(members)) == ["A", "B", "C"]
    assert any(set(group) == {"A", "B"} for group in members)
    assert any(set(group) == {"C"} for group in members)


def test_missing_inputs_fail_with_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="spreadsheet"):
        run_pipeline(data_dir=tmp_path / "empty", output_dir=tmp_path / "out")


def test_negatives_are_non_site_sty_and_no_y_positive_is_fabricated(
    tiny_dataset, tmp_path: Path
) -> None:
    """Negatives are drawn only from non-site S/T/Y, never from annotated sites.

    The unmodified fixture contains Y-centred positives (K3).  We derive a
    second sites sheet from that same fixture with those rows removed, so the
    "no Y positive" assertion is meaningful rather than trivially true, and Y
    residues remain present in the FASTA throughout.
    """
    data_dir, _ = tiny_dataset
    sites = pd.read_excel(data_dir / SITES_FILENAME)
    fasta = pd.read_excel(data_dir / FASTA_FILENAME)
    seq_by_kinase = {
        str(kinase).strip(): str(sequence).strip()
        for kinase, sequence in zip(fasta["KINASE"], fasta["FASTA"])
    }

    y_centred = [
        seq_by_kinase[str(k).strip()][int(p) - 1] == "Y"
        for k, p in zip(sites["Kinase"], sites["Position"])
    ]
    assert any(y_centred)
    assert any("Y" in sequence for sequence in seq_by_kinase.values())
    no_y_sites = sites.loc[~pd.Series(y_centred, index=sites.index)].copy()
    assert not no_y_sites.empty

    no_y_dir = tmp_path / "no_y_inputs"
    no_y_dir.mkdir()
    no_y_sites.to_excel(no_y_dir / SITES_FILENAME, index=False)
    fasta.to_excel(no_y_dir / FASTA_FILENAME, index=False)

    output_dir = tmp_path / "no_y_out"
    manifest = run_pipeline(
        data_dir=no_y_dir,
        output_dir=output_dir,
        seed=11,
        ratio=1.0,
        identity_threshold=THRESHOLD,
    )
    rows = pd.concat([_read_split(output_dir, s) for s in SPLITS], ignore_index=True)
    positives = rows[rows["label"] == 1]
    negatives = rows[rows["label"] == 0]

    # (c) No Y-centred positive is invented when the sheet has none.
    assert not positives.empty
    assert positives["site_type"].isin(["S", "T"]).all()
    assert not positives["sequence"].str[7].eq("Y").any()
    assert manifest["site_type_counts"]["Y"] == int(
        (negatives["site_type"] == "Y").sum()
    )

    # (a) Every emitted negative centre is S/T/Y and matches its site_type.
    assert len(negatives) == manifest["totals"]["negative"] > 0
    assert negatives["sequence"].str[7].isin(list("STY")).all()
    assert (negatives["sequence"].str[7] == negatives["site_type"]).all()

    # (b) Negative positions exclude annotated sites for each kinase, and the
    # emitted negative windows are exactly those sampled positions.
    remaining_positives = defaultdict(set)
    for kinase, position in zip(no_y_sites["Kinase"], no_y_sites["Position"]):
        remaining_positives[str(kinase).strip()].add(int(position))

    sampled = sample_negatives(seq_by_kinase, remaining_positives, 1.0, 11, [])
    for kinase, positions in sampled.items():
        if remaining_positives.get(kinase):
            assert positions, f"{kinase} has positives but no negatives"
        assert remaining_positives.get(kinase, set()).isdisjoint(positions)
        assert all(seq_by_kinase[kinase][p - 1] in "STY" for p in positions)

    expected = Counter(
        (
            kinase,
            get_15mer(seq_by_kinase[kinase], pos),
            seq_by_kinase[kinase][pos - 1],
        )
        for kinase, positions in sampled.items()
        for pos in positions
    )
    actual = Counter(
        (row.kinase, row.sequence, row.site_type)
        for row in negatives.itertuples(index=False)
    )
    assert actual == expected
