"""Fast offline tests for the statistical imbalance fix (X-003).

All CSVs are tiny and SYNTHETIC, written under ``tmp_path``.  The real split
is never opened.  The fixtures deliberately give train more positives than
negatives so ``pos_weight`` is a non-trivial ratio.  One or two epochs over six
rows keep the whole file well under ten seconds while still exercising both
ablations, fixed-seed determinism, the NPZ+pos_weight contract and the
truthfulness of every manifest count.
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

from src.imbalance import (  # noqa: E402
    BALANCED_NPZ_KEYS,
    class_counts,
    compute_pos_weight,
    imbalance_report,
    prevalence_baseline,
    train_weighted_run,
)
from src.metrics import compute_metrics  # noqa: E402
from src.real_train import load_split_csv  # noqa: E402
from tools import train_balanced  # noqa: E402

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
TRAIN_LABELS = [1, 1, 1, 1, 0, 0]   # 4 positives, 2 negatives -> pos_weight 0.5
VAL_LABELS = [1, 0]
TEST_LABELS = [1, 1, 0]
FLANKS = "SAPTGW"


def _window(index: int, label: int) -> str:
    """Build a valid 15-mer with the candidate S/T residue at index 7."""
    centre = "S" if label == 1 else "T"
    neighbour = "E" if label == 1 else "G"
    flank = FLANKS[index % len(FLANKS)]
    window = flank * 7 + centre + neighbour + flank * 6
    assert len(window) == 15 and window[7] == centre
    return window


def _rows(split: str, labels: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        centre = "S" if label == 1 else "T"
        rows.append(
            {
                "sequence": _window(index, label),
                "label": label,
                "site_type": centre,
                "protein_acc": f"P{split}{index}",
                "gene": f"G{split}{index}",
                "kinase": "-",
                "source": "synthetic",
                "cluster_id": f"{split}c{index}",
            }
        )
    return rows


@pytest.fixture
def tiny_split_dir(tmp_path: Path) -> Path:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    for split, labels in (
        ("train", TRAIN_LABELS),
        ("val", VAL_LABELS),
        ("test", TEST_LABELS),
    ):
        pd.DataFrame(_rows(split, labels), columns=CSV_COLUMNS).to_csv(
            split_dir / f"{split}.csv", index=False
        )
    (split_dir / "manifest.json").write_text(
        json.dumps({"method": "synthetic", "seed": 666}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pos_weight_arithmetic_from_train_labels_only(tiny_split_dir: Path) -> None:
    train = load_split_csv(tiny_split_dir / "train.csv")
    labels = train["label"].to_numpy()
    counts = class_counts(labels)

    assert counts["n"] == 6
    assert counts["n_positive"] == 4
    assert counts["n_negative"] == 2
    assert counts["prevalence"] == pytest.approx(4 / 6)
    assert counts["n_negative_per_positive"] == pytest.approx(2 / 4)
    assert compute_pos_weight(labels) == pytest.approx(2 / 4)
    assert compute_pos_weight([1, 0, 1, 0, 0]) == pytest.approx(3 / 2)

    with pytest.raises(ValueError):
        compute_pos_weight([1, 1, 1])
    with pytest.raises(ValueError):
        compute_pos_weight([0, 0])


def test_prevalence_baseline_matches_src_metrics(tiny_split_dir: Path) -> None:
    labels = load_split_csv(tiny_split_dir / "test.csv")["label"].to_numpy()
    baseline = prevalence_baseline(labels)
    prevalence = baseline["constant_probability"]
    expected = compute_metrics(labels, np.full(len(labels), prevalence))

    assert prevalence == pytest.approx(2 / 3)
    for key in ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "precision", "recall", "n"):
        assert baseline["metrics"][key] == expected[key]
    assert baseline["metrics"]["roc_auc"] == pytest.approx(0.5)
    assert baseline["metrics"]["pr_auc"] == pytest.approx(prevalence)
    assert baseline["metrics"]["accuracy"] == baseline["interpretation"]["expected_accuracy"]
    assert baseline["interpretation"]["expected_roc_auc"] == pytest.approx(0.5)
    assert baseline["interpretation"]["expected_pr_auc"] == pytest.approx(prevalence)

    single = prevalence_baseline([1, 1, 1])
    assert single["metrics"]["roc_auc"] is None
    assert single["metrics"]["pr_auc"] is None
    assert single["interpretation"]["expected_roc_auc"] is None


def test_weighted_training_shape_range_and_fixed_seed_determinism(
    tiny_split_dir: Path,
) -> None:
    train = load_split_csv(tiny_split_dir / "train.csv")
    val = load_split_csv(tiny_split_dir / "val.csv")
    test = load_split_csv(tiny_split_dir / "test.csv")

    common = dict(
        train_rows=train,
        val_rows=val,
        predict_rows=test,
        seed=7,
        epochs=1,
        batch_size=3,
    )
    first = train_weighted_run(ablation="seq_only", **common)
    second = train_weighted_run(ablation="seq_only", **common)

    assert first["probabilities"].shape == (len(test),)
    assert np.all(np.isfinite(first["probabilities"]))
    assert np.all((first["probabilities"] >= 0.0) & (first["probabilities"] <= 1.0))
    np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
    assert first["chosen_epoch"] == second["chosen_epoch"] == 1
    assert first["selection_metric"] == "validation_auroc"
    assert first["pos_weight"] == pytest.approx(0.5)
    assert first["config"]["pos_weight"] == pytest.approx(0.5)
    assert "BCEWithLogitsLoss" in first["config"]["loss"]
    assert first["config"]["ablation"] == "seq_only"
    assert first["config"]["seed"] == 7
    np.testing.assert_array_equal(first["labels"], test["label"].to_numpy())

    # The embedding ablation is exercised by the CLI test below; one direct
    # run here confirms it also uses the weighted loss and returns valid probs.
    embedded = train_weighted_run(ablation="seq_embedding", **common)
    assert embedded["probabilities"].shape == (len(test),)
    assert np.all(
        (embedded["probabilities"] >= 0.0) & (embedded["probabilities"] <= 1.0)
    )
    assert embedded["pos_weight"] == pytest.approx(0.5)


def test_imbalance_report_refuses_to_invent_negatives(tiny_split_dir: Path) -> None:
    train = load_split_csv(tiny_split_dir / "train.csv")
    test = load_split_csv(tiny_split_dir / "test.csv")
    report = imbalance_report(train["label"], test["label"])

    assert report["train"]["n_positive"] == 4
    assert report["train"]["n_negative"] == 2
    assert report["test"]["n_positive"] == 2
    assert report["test"]["n_negative"] == 1
    assert report["pos_weight"] == pytest.approx(0.5)
    assert "2 / 4" in report["exact_rule"]
    assert report["one_to_one_possible_from_same_protein_negatives"] is False
    assert "inventing 2 negatives" in report["why_one_to_one_impossible"]
    assert "discarding 2 real positives" in report["why_one_to_one_impossible"]
    assert "no row is created, duplicated, relabelled or dropped" in report["negative_synthesis"]


def test_cli_manifest_counts_baselines_and_no_synthetic_rows(
    tiny_split_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "results"
    assert (
        train_balanced.main(
            [
                "--split-dir",
                str(tiny_split_dir),
                "--out-dir",
                str(out_dir),
                "--seeds",
                "2",
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

    frames = {
        split: load_split_csv(tiny_split_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    expected_pairs = {
        (ablation, seed)
        for ablation in ("seq_only", "seq_embedding")
        for seed in (11, 12)
    }
    test = frames["test"]

    for ablation, seed in expected_pairs:
        path = out_dir / f"predictions_{ablation}_seed{seed}.npz"
        assert path.is_file()
        with np.load(path, allow_pickle=True) as archive:
            assert set(archive.files) == set(BALANCED_NPZ_KEYS)
            assert archive["labels"].shape == (len(test),)
            assert archive["probabilities"].shape == (len(test),)
            assert np.all((archive["probabilities"] >= 0.0) & (archive["probabilities"] <= 1.0))
            assert float(archive["pos_weight"]) == pytest.approx(0.5)
            np.testing.assert_array_equal(archive["labels"], test["label"].to_numpy())
            np.testing.assert_array_equal(
                archive["cluster_id"], test["cluster_id"].astype(str).to_numpy()
            )
            np.testing.assert_array_equal(
                archive["protein_acc"], test["protein_acc"].astype(str).to_numpy()
            )
            config = archive["config"].item()
            assert config["seed"] == seed
            assert config["ablation"] == ablation
            assert config["pos_weight"] == pytest.approx(0.5)
            assert int(archive["chosen_epoch"]) == config["chosen_epoch"]

    manifest = json.loads((out_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["npz_keys"] == list(BALANCED_NPZ_KEYS)
    assert manifest["run_config"]["seeds"] == [11, 12]
    assert manifest["run_config"]["ablations"] == ["seq_only", "seq_embedding"]
    assert manifest["pos_weight"] == pytest.approx(0.5)
    assert manifest["split_counts"]["train"]["n_positive"] == 4
    assert manifest["split_counts"]["train"]["n_negative"] == 2
    assert manifest["split_counts"]["test"]["n_positive"] == 2
    assert manifest["split_counts"]["test"]["n_negative"] == 1

    for split, frame in frames.items():
        baseline = manifest["prevalence_baseline"][split]
        expected = compute_metrics(
            frame["label"].to_numpy(),
            np.full(len(frame), frame["label"].mean()),
        )
        assert baseline["metrics"]["roc_auc"] == expected["roc_auc"]
        assert baseline["metrics"]["pr_auc"] == pytest.approx(expected["pr_auc"])
        assert baseline["counts"]["n"] == len(frame)
        assert manifest["split_counts"][split]["n"] == len(frame)

    assert manifest["input_csv_sha256"] == {
        f"{split}.csv": _sha256(tiny_split_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    assert manifest["split_manifest_sha256"] == _sha256(tiny_split_dir / "manifest.json")
    assert len(manifest["runs"]) == len(expected_pairs)
    assert {
        (run["ablation"], run["seed"]) for run in manifest["runs"]
    } == expected_pairs
    for run in manifest["runs"]:
        assert run["pos_weight"] == pytest.approx(0.5)
        assert run["config"]["pos_weight"] == pytest.approx(0.5)
        assert Path(run["prediction_file"]).is_file()
        assert run["prediction_file"].endswith(
            f"predictions_{run['ablation']}_seed{run['seed']}.npz"
        )

    # Truthfulness of the negative pool: every scored row comes from test.csv,
    # and no row outside the supplied cluster ids appears anywhere.
    test_clusters = set(test["cluster_id"].astype(str))
    supplied_clusters = {
        cluster
        for frame in frames.values()
        for cluster in frame["cluster_id"].astype(str)
    }
    for filename, digest in manifest["prediction_sha256"].items():
        assert digest == _sha256(out_dir / filename)
        with np.load(out_dir / filename, allow_pickle=True) as archive:
            assert set(archive["cluster_id"]) <= test_clusters
            assert set(archive["cluster_id"]) <= supplied_clusters
            assert int(np.count_nonzero(archive["labels"] == 0)) == 1
            assert int(np.count_nonzero(archive["labels"] == 1)) == 2

    assert manifest["class_balance"]["negative_synthesis"] == (
        "none: no row is created, duplicated, relabelled or dropped"
    )
