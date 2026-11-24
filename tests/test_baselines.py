"""Fast, synthetic-only tests for the classical baselines and experiment runner.

Every sequence, label and CSV here is generated inline under ``tmp_path``: no
spreadsheet, real dataset split or pre-existing result file is read.  The
expected metrics are recomputed with :func:`src.metrics.compute_metrics` from
freshly fitted predictions, so no metric value is hard-coded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import BASELINE_NAMES, baseline_config, fit_predict_baseline  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import ONE_HOT_WIDTH, one_hot_encode  # noqa: E402
from tools.run_experiments import COMPARISON_SCOPE, main as run_experiments_main  # noqa: E402

REQUIRED_SCOPE = (
    "Same-split classical baselines only. No external SOTA phosphosite "
    "predictor was executed; no claim of superiority over published tools is made."
)
SYMBOLS = "WGSAELQDMPFTRIHVNCYK"  # subset of the legacy alphabet in src.model
SPLITS = ("train", "val", "test")
SITE_CYCLE = ("S", "S", "T", "T", "Y", "Y")


def make_rows(n: int, seed: int) -> pd.DataFrame:
    """Build a tiny balanced frame of valid 15-mers with S/T/Y centres."""
    rng = np.random.default_rng(seed)
    labels = np.array([index % 2 for index in range(n)], dtype=int)
    sites = np.array([SITE_CYCLE[index % len(SITE_CYCLE)] for index in range(n)])
    characters = rng.choice(list(SYMBOLS), size=(n, 15))
    for index in range(n):
        # Centre residue matches site_type; the neighbouring position carries
        # a class signal so fitting is meaningful without being tuned.
        characters[index, 7] = sites[index]
        characters[index, 6] = "S" if labels[index] == 1 else "V"
    return pd.DataFrame(
        {
            "sequence": ["".join(row) for row in characters],
            "label": labels,
            "site_type": sites,
            "kinase": ["SYNTHETIC"] * n,
            "cluster_id": ["cluster_syn"] * n,
        }
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_metric_blocks_equal(actual: dict, expected: dict) -> None:
    """Compare nested metric dicts without requiring approx on strings/None."""
    assert set(actual) == set(expected)
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, dict):
            assert isinstance(observed, dict)
            assert_metric_blocks_equal(observed, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            assert observed == pytest.approx(value, rel=0, abs=1e-12), key
        else:
            assert observed == value, key


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_probabilities_shape_range_and_encoder_reuse(name: str) -> None:
    train = make_rows(24, seed=11)
    test = make_rows(10, seed=12)
    arguments = (
        train["sequence"].tolist(),
        train["label"].to_numpy(),
        test["sequence"].tolist(),
        name,
        666,
    )

    probabilities = fit_predict_baseline(*arguments)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(test),)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    # Pre-encoded 330-dim input must give the same result as raw windows: both
    # paths go through src.model.one_hot_encode.
    encoded_train = one_hot_encode(train["sequence"].tolist())
    encoded_test = one_hot_encode(test["sequence"].tolist())
    assert encoded_train.shape == (len(train), ONE_HOT_WIDTH)
    assert encoded_test.shape == (len(test), ONE_HOT_WIDTH)
    from_matrix = fit_predict_baseline(
        encoded_train, train["label"].to_numpy(), encoded_test, name, 666
    )
    np.testing.assert_allclose(probabilities, from_matrix, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_fixed_seed_is_deterministic_and_config_is_fixed(name: str) -> None:
    train = make_rows(18, seed=21)
    test = make_rows(8, seed=22)
    arguments = (
        train["sequence"].tolist(),
        train["label"].to_numpy(),
        test["sequence"].tolist(),
        name,
    )

    first = fit_predict_baseline(*arguments, seed=17)
    second = fit_predict_baseline(*arguments, seed=17)
    assert np.array_equal(first, second)

    config = baseline_config(name, seed=17)
    assert config["name"] == name
    assert config["seed"] == 17
    assert config["n_features"] == ONE_HOT_WIDTH
    assert config["tuning"] == "fixed hyperparameters; no validation/test tuning or model selection"
    assert config["hyperparameters"]["random_state"] == 17


def test_unknown_baseline_and_bad_windows_are_rejected() -> None:
    train = make_rows(8, seed=31)
    with pytest.raises(ValueError, match="Unknown baseline"):
        fit_predict_baseline(
            train["sequence"].tolist(),
            train["label"].to_numpy(),
            train["sequence"].tolist(),
            "not_a_baseline",
            seed=1,
        )
    with pytest.raises(ValueError, match="15-residue window"):
        fit_predict_baseline(
            ["TOO_SHORT"],
            [1],
            ["TOO_SHORT"],
            "logistic_regression",
            seed=1,
        )


@pytest.mark.parametrize("name", BASELINE_NAMES)
@pytest.mark.parametrize("only_label", [0, 1])
def test_single_class_training_labels_are_rejected(name: str, only_label: int) -> None:
    train = make_rows(8, seed=41)
    labels = np.full(len(train), only_label, dtype=int)
    with pytest.raises(ValueError, match="both classes"):
        fit_predict_baseline(
            train["sequence"].tolist(),
            labels,
            train["sequence"].tolist(),
            name,
            seed=5,
        )


def test_runner_writes_provenance_scope_and_computed_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frames = {
        split: make_rows(12, seed=100 + offset)
        for offset, split in enumerate(SPLITS)
    }
    input_hashes: dict[str, str] = {}
    for split in SPLITS:
        csv_path = data_dir / f"{split}.csv"
        frames[split].to_csv(csv_path, index=False)
        input_hashes[f"{split}.csv"] = sha256_of(csv_path)

    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "seed": 666, "notes": ["synthetic"]}),
        encoding="utf-8",
    )
    manifest_hash = sha256_of(manifest_path)

    out_path = tmp_path / "results" / "baselines.json"
    assert run_experiments_main(
        ["--data-dir", str(data_dir), "--seed", "123", "--out", str(out_path)]
    ) == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))

    # Scope statement and top-level provenance.
    assert COMPARISON_SCOPE == REQUIRED_SCOPE
    assert report["comparison_scope"] == REQUIRED_SCOPE
    assert report["seed"] == 123
    assert report["dataset_manifest_sha256"] == manifest_hash
    assert report["input_csv_sha256"] == input_hashes
    assert set(report["models"]) == set(BASELINE_NAMES)

    train = frames["train"]
    test = frames["test"]
    for name in BASELINE_NAMES:
        entry = report["models"][name]
        for key in (
            "name",
            "config",
            "seed",
            "dataset_manifest_sha256",
            "input_csv_sha256",
            "test_metrics",
        ):
            assert key in entry, (name, key)
        assert entry["name"] == name
        assert entry["seed"] == 123
        assert entry["config"]["seed"] == 123
        assert entry["dataset_manifest_sha256"] == manifest_hash
        assert entry["input_csv_sha256"] == input_hashes

        # Recompute from freshly fitted predictions: nothing is hardcoded.
        predictions = fit_predict_baseline(
            train["sequence"].tolist(),
            train["label"].to_numpy(),
            test["sequence"].tolist(),
            name,
            seed=123,
        )
        expected = compute_metrics(
            test["label"].to_numpy(),
            predictions,
            site_types=test["site_type"].tolist(),
        )
        assert entry["test_metrics"]["n"] == len(test)
        assert set(entry["test_metrics"]["per_site_type"]) == {"S", "T", "Y"}
        assert_metric_blocks_equal(entry["test_metrics"], expected)
