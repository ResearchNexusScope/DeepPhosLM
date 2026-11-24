"""Fast offline tests for real-split multi-seed training and CV.

Every CSV is tiny and SYNTHETIC, written under ``tmp_path``.  The real data
directory is never opened.  One epoch and four rows per split keep the suite
well under ten seconds while still exercising both ablations, two seeds, the
saved npz contract, manifest hashes and whole-cluster CV partitioning.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_train import (  # noqa: E402
    NPZ_KEYS,
    cluster_cv_folds,
    cross_validate,
    load_split_csv,
    train_one_run,
)
from tools import run_real_experiments  # noqa: E402

CSV_COLUMNS = [
    "sequence",
    "label",
    "site_type",
    "protein_acc",
    "gene",
    "kinase",
    "source",
    "cluster_id",
]
TRAIN_CLUSTERS = ["c0", "c1", "c2", "c3"]
VAL_CLUSTERS = ["c4"]
TEST_CLUSTERS = ["c5"]
FLANKS = ("S", "T", "W", "G", "A", "P")


def _window(flank: str, centre: str, neighbour: str) -> str:
    """Build a valid 15-mer with the candidate residue at index 7."""
    window = flank * 7 + centre + neighbour + flank * 6
    assert len(window) == 15 and window[7] == centre
    return window


def _rows_for_clusters(clusters: List[str], salt: int) -> List[Dict[str, Any]]:
    """One positive and one negative S/T-centred window per cluster."""
    rows: List[Dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        for label in (1, 0):
            centre = "S" if label == 1 else "T"
            neighbour = "E" if label == 1 else "G"
            flank = FLANKS[(index + salt) % len(FLANKS)]
            rows.append(
                {
                    "sequence": _window(flank, centre, neighbour),
                    "label": label,
                    "site_type": centre,
                    "protein_acc": f"P{cluster[1:]}",
                    "gene": f"G{cluster[1:]}",
                    "kinase": "-",
                    "source": "synthetic",
                    "cluster_id": cluster,
                }
            )
    return rows


@pytest.fixture
def tiny_split_dir(tmp_path: Path) -> Path:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    for salt, (split, clusters) in enumerate(
        (
            ("train", TRAIN_CLUSTERS),
            ("val", VAL_CLUSTERS),
            ("test", TEST_CLUSTERS),
        )
    ):
        pd.DataFrame(
            _rows_for_clusters(clusters, salt), columns=CSV_COLUMNS
        ).to_csv(split_dir / f"{split}.csv", index=False)
    (split_dir / "manifest.json").write_text(
        json.dumps({"method": "synthetic", "seed": 666}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_writes_truthful_npz_and_manifest(
    tiny_split_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "results"
    assert (
        run_real_experiments.main(
            [
                "--split-dir",
                str(tiny_split_dir),
                "--out-dir",
                str(out_dir),
                "--seeds",
                "2",
                "--folds",
                "0",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--ablation",
                "both",
                "--seed-base",
                "11",
            ]
        )
        == 0
    )

    expected_pairs = {
        (ablation, seed)
        for ablation in ("seq_only", "seq_embedding")
        for seed in (11, 12)
    }
    test_rows = load_split_csv(tiny_split_dir / "test.csv")
    n_test = len(test_rows)

    for ablation, seed in expected_pairs:
        path = out_dir / f"predictions_{ablation}_seed{seed}.npz"
        assert path.is_file()
        with np.load(path, allow_pickle=True) as archive:
            assert set(NPZ_KEYS).issubset(set(archive.files))
            assert archive["labels"].shape == (n_test,)
            assert archive["probabilities"].shape == (n_test,)
            assert np.all(np.isfinite(archive["probabilities"]))
            assert np.all((archive["probabilities"] >= 0.0) & (archive["probabilities"] <= 1.0))
            for key in ("site_type", "protein_acc", "cluster_id"):
                assert archive[key].shape == (n_test,)
            assert int(archive["seed"]) == seed
            assert str(archive["ablation"]) == ablation
            chosen_epoch = int(archive["chosen_epoch"])
            assert 1 <= chosen_epoch <= 1
            config = archive["config"].item()
            assert isinstance(config, dict)
            assert config["ablation"] == ablation
            assert config["seed"] == seed
            assert config["chosen_epoch"] == chosen_epoch
            assert config["epochs_requested"] == 1
            assert config["batch_size"] == 4

    manifest = json.loads((out_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_csv_sha256"] == {
        f"{split}.csv": _sha256(tiny_split_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    assert manifest["split_manifest_sha256"] == _sha256(tiny_split_dir / "manifest.json")
    assert manifest["run_config"]["seeds"] == [11, 12]
    assert manifest["run_config"]["folds"] == 0
    assert manifest["cv_runs"] == []
    assert len(manifest["runs"]) == len(expected_pairs)
    assert {
        (run["ablation"], run["seed"]) for run in manifest["runs"]
    } == expected_pairs
    for run in manifest["runs"]:
        assert run["chosen_epoch"] == run["config"]["chosen_epoch"]
        assert run["seed"] == run["config"]["seed"]
        assert Path(run["prediction_file"]).is_file()


def test_cli_cv_manifest_summarises_whole_cluster_folds(
    tiny_split_dir: Path, tmp_path: Path
) -> None:
    """A CLI run with --folds 2 must record real CV summaries in the manifest."""
    out_dir = tmp_path / "cv_results"
    assert (
        run_real_experiments.main(
            [
                "--split-dir",
                str(tiny_split_dir),
                "--out-dir",
                str(out_dir),
                "--seeds",
                "1",
                "--folds",
                "2",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--ablation",
                "seq_only",
                "--seed-base",
                "21",
            ]
        )
        == 0
    )

    manifest = json.loads((out_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_config"]["folds"] == 2
    assert manifest["run_config"]["seeds"] == [21]
    assert len(manifest["runs"]) == 1
    assert len(manifest["cv_runs"]) == 1

    cv_run = manifest["cv_runs"][0]
    assert cv_run["seed"] == 21
    assert cv_run["ablation"] == "seq_only"
    assert cv_run["n_folds"] == 2
    assert len(cv_run["folds"]) == 2

    summary = cv_run["summary"]
    assert summary["n_folds"] == 2
    roc_auc = summary["metrics"]["roc_auc"]
    assert len(roc_auc["values"]) == 2
    assert roc_auc["n_defined"] == 2
    assert roc_auc["mean"] is not None
    assert roc_auc["std"] is not None
    assert roc_auc["values"] == [fold["metrics"]["roc_auc"] for fold in cv_run["folds"]]

    for fold in cv_run["folds"]:
        assert 1 <= fold["chosen_epoch"] <= 1
        assert fold["metrics"]["roc_auc"] is not None

    assignments = manifest["cv_fold_assignments_by_seed"]["21"]
    assert set(assignments) == set(TRAIN_CLUSTERS)
    assert set(assignments.values()) <= {0, 1}

    val_sets = [set(fold["val_clusters"]) for fold in cv_run["folds"]]
    assert set.intersection(*val_sets) == set()
    assert set.union(*val_sets) == set(TRAIN_CLUSTERS)
    visits: Dict[str, int] = {cluster: 0 for cluster in TRAIN_CLUSTERS}
    for fold in cv_run["folds"]:
        val_clusters = set(fold["val_clusters"])
        train_clusters = set(TRAIN_CLUSTERS) - val_clusters
        assert val_clusters.isdisjoint(train_clusters)
        for cluster in val_clusters:
            assert assignments[cluster] == fold["fold"]
            visits[cluster] += 1
    assert set(visits.values()) == {1}


def test_fixed_seed_reproduces_probabilities(tiny_split_dir: Path) -> None:
    train = load_split_csv(tiny_split_dir / "train.csv")
    val = load_split_csv(tiny_split_dir / "val.csv")
    test = load_split_csv(tiny_split_dir / "test.csv")
    for ablation in ("seq_only", "seq_embedding"):
        first = train_one_run(
            train_rows=train,
            val_rows=val,
            predict_rows=test,
            seed=7,
            ablation=ablation,
            epochs=1,
            batch_size=4,
        )
        second = train_one_run(
            train_rows=train,
            val_rows=val,
            predict_rows=test,
            seed=7,
            ablation=ablation,
            epochs=1,
            batch_size=4,
        )
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
        assert first["chosen_epoch"] == second["chosen_epoch"] == 1
        assert first["config"]["seed"] == second["config"]["seed"] == 7


def test_cross_validation_keeps_whole_clusters_together(tiny_split_dir: Path) -> None:
    train = load_split_csv(tiny_split_dir / "train.csv")
    folds, assignments = cluster_cv_folds(train, n_folds=2, seed=3)
    assert len(folds) == 2
    assert set(assignments) == set(TRAIN_CLUSTERS)

    val_sets = [set(fold["val_clusters"]) for fold in folds]
    assert set.intersection(*val_sets) == set()
    assert set.union(*val_sets) == set(TRAIN_CLUSTERS)

    val_cluster_visits: Dict[str, int] = {cluster: 0 for cluster in TRAIN_CLUSTERS}
    for fold in folds:
        val_clusters = set(fold["val_clusters"])
        train_clusters = set(fold["train_clusters"])
        assert val_clusters.isdisjoint(train_clusters)
        assert val_clusters | train_clusters == set(TRAIN_CLUSTERS)
        assert set(fold["train_rows"]["cluster_id"]) == train_clusters
        assert set(fold["val_rows"]["cluster_id"]) == val_clusters
        for cluster in val_clusters:
            val_cluster_visits[cluster] += 1
        assert all(assignments[cluster] == fold["fold"] for cluster in val_clusters)
    assert set(val_cluster_visits.values()) == {1}

    result = cross_validate(
        train_rows=train,
        n_folds=2,
        seed=3,
        ablation="seq_only",
        epochs=1,
        batch_size=4,
    )
    assert result["n_folds"] == 2
    assert len(result["folds"]) == 2
    assert result["summary"]["n_folds"] == 2
    assert len(result["summary"]["metrics"]["roc_auc"]["values"]) == 2
    assert result["summary"]["metrics"]["roc_auc"]["n_defined"] == 2
    assert result["summary"]["metrics"]["roc_auc"]["mean"] is not None
    assert result["summary"]["metrics"]["roc_auc"]["std"] is not None
    for fold in result["folds"]:
        assert fold["metrics"]["roc_auc"] is not None
        assert 1 <= fold["chosen_epoch"] <= 1


def test_single_class_validation_uses_bce_fallback(tiny_split_dir: Path) -> None:
    train = load_split_csv(tiny_split_dir / "train.csv")
    val = load_split_csv(tiny_split_dir / "val.csv")
    single_class_val = val.loc[val["label"] == 0].reset_index(drop=True)
    assert len(single_class_val) > 0
    result = train_one_run(
        train_rows=train,
        val_rows=single_class_val,
        predict_rows=single_class_val,
        seed=5,
        ablation="seq_only",
        epochs=1,
        batch_size=4,
    )
    assert result["config"]["selection_metric"] == "validation_bce_fallback"
    assert result["selection_metric"] == "validation_bce_fallback"
    assert result["chosen_epoch"] == 1
    assert np.all((result["probabilities"] >= 0.0) & (result["probabilities"] <= 1.0))
