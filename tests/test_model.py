"""Fast tests for the binary phosphosite model and trainer.

All inputs here are tiny SYNTHETIC tensors and CSVs written under ``tmp_path``;
the generated data directory and the root spreadsheets are never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import (  # noqa: E402
    ABLATIONS,
    ALPHABET_SIZE,
    AMINO_ACID_ALPHABET,
    ONE_HOT_WIDTH,
    WINDOW_LENGTH,
    BinaryPhosphositeModel,
    ModelConfig,
    one_hot_encode,
    tokenize_sequences,
)
from src.train_model import (  # noqa: E402
    NPZ_KEYS,
    REQUIRED_COLUMNS,
    set_seed,
    train_and_predict,
)

SOURCE_FILES = (
    REPO_ROOT / "src" / "model.py",
    REPO_ROOT / "src" / "train_model.py",
    Path(__file__).resolve(),
)

# Fragmented on purpose: the forbidden terms must not occur literally in the
# files being scanned, including this test module itself.
FORBIDDEN_TERMS = tuple(
    term.lower()
    for term in (
        "co" + "ords",
        "Prot" + "BERT",
        "Sw" + "in",
        "Alpha" + "Fold",
        "P" + "DB",
        "3" + "D",
    )
)


def _window(flank: str, centre: str, neighbour: str) -> str:
    """Build a 15-mer with the candidate centre at index 7."""
    window = flank * 7 + centre + neighbour + flank * 6
    assert len(window) == WINDOW_LENGTH and window[7] == centre
    return window


POSITIVE_WINDOWS = (
    _window("S", "S", "E"),
    _window("T", "T", "D"),
    _window("S", "S", "P"),
    _window("S", "T", "Y"),
)
NEGATIVE_WINDOWS = (
    _window("S", "S", "G"),
    _window("T", "T", "V"),
    _window("S", "S", "A"),
    _window("S", "T", "K"),
)
SEQUENCES = POSITIVE_WINDOWS + NEGATIVE_WINDOWS


@pytest.fixture
def tiny_csv_dir(tmp_path: Path) -> Path:
    """Eight balanced synthetic rows per split, with both labels in every split."""
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    for split in ("train", "val", "test"):
        rows = []
        for index, (sequence, label) in enumerate(
            [(s, 1) for s in POSITIVE_WINDOWS] + [(s, 0) for s in NEGATIVE_WINDOWS]
        ):
            rows.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "site_type": sequence[7],
                    "kinase": f"K{index % 2}",
                    "cluster_id": f"c{index % 2}",
                }
            )
        pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS)).to_csv(
            data_dir / f"{split}.csv", index=False
        )
    return data_dir


def test_forward_shape_range_and_alphabet() -> None:
    assert AMINO_ACID_ALPHABET == "XWGSAELQDMPFTRIHVNCY_K"
    assert ALPHABET_SIZE == 22
    assert ONE_HOT_WIDTH == WINDOW_LENGTH * ALPHABET_SIZE

    one_hot = one_hot_encode(SEQUENCES)
    tokens = tokenize_sequences(SEQUENCES)
    assert one_hot.shape == (len(SEQUENCES), ONE_HOT_WIDTH)
    assert tokens.shape == (len(SEQUENCES), WINDOW_LENGTH)
    assert tokens.min() >= 0 and tokens.max() < ALPHABET_SIZE
    assert np.all(one_hot.sum(axis=1) == WINDOW_LENGTH)

    for ablation in ABLATIONS:
        set_seed(42)
        model = BinaryPhosphositeModel(ModelConfig(ablation=ablation))
        output = model(torch.from_numpy(one_hot), torch.from_numpy(tokens))
        assert output.shape == (len(SEQUENCES),)
        assert bool(torch.all((output >= 0.0) & (output <= 1.0)))


def test_config_rejects_invalid_choices() -> None:
    with pytest.raises(ValueError):
        ModelConfig(ablation="not_an_ablation")
    with pytest.raises(ValueError):
        ModelConfig(ablation="seq_embedding", embedding_dim=101)


def test_one_epoch_run_writes_promised_npz_keys(
    tiny_csv_dir: Path, tmp_path: Path
) -> None:
    for ablation in ABLATIONS:
        out = tmp_path / f"{ablation}.npz"
        train_and_predict(
            data_dir=tiny_csv_dir,
            epochs=1,
            batch_size=4,
            seed=7,
            ablation=ablation,
            out=out,
        )
        with np.load(out, allow_pickle=True) as archive:
            assert set(NPZ_KEYS).issubset(set(archive.files))
            assert archive["labels"].shape == (len(SEQUENCES),)
            assert archive["probabilities"].shape == (len(SEQUENCES),)
            assert np.all((archive["probabilities"] >= 0.0) & (archive["probabilities"] <= 1.0))
            assert archive["site_type"].shape == (len(SEQUENCES),)
            assert archive["kinase"].shape == (len(SEQUENCES),)
            assert archive["cluster_id"].shape == (len(SEQUENCES),)
            assert int(archive["chosen_epoch"]) == 1
            assert int(archive["seed"]) == 7
            saved_config = archive["config"].item()
            assert saved_config["ablation"] == ablation
            assert saved_config["seed"] == 7
        assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8")) == saved_config


def test_fixed_seed_reproduces_probabilities(tiny_csv_dir: Path, tmp_path: Path) -> None:
    first = train_and_predict(
        data_dir=tiny_csv_dir,
        epochs=1,
        batch_size=4,
        seed=123,
        ablation="seq_embedding",
        out=tmp_path / "first.npz",
    )
    second = train_and_predict(
        data_dir=tiny_csv_dir,
        epochs=1,
        batch_size=4,
        seed=123,
        ablation="seq_embedding",
        out=tmp_path / "second.npz",
    )
    np.testing.assert_array_equal(first["probabilities"], second["probabilities"])


def test_owned_sources_have_no_unsupported_input_terms() -> None:
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"{path.name} contains unsupported term {term!r}"
