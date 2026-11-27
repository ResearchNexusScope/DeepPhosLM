"""Tests for the legacy generate_dataset.py CLI using tiny synthetic inputs.

The real spreadsheets are never read; every fixture is written into tmp_path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_dataset as legacy  # noqa: E402

SCRIPT = REPO_ROOT / "generate_dataset.py"
FASTA_FILE = "Modified_TOP_20_FOR_STRUCTURE.xlsx"
SITES_FILE = "kinase_phosphorylation_sites.xlsx"

# Two 24-residue proteins with disjoint residue alphabets and seven annotated
# sites each: 14 rows -> 11 train / 3 test with the legacy 80/20 split.
FASTA = {
    "KIN_A": "S" * 8 + "A" * 8 + "T" * 8,
    "KIN_B": "Y" * 8 + "G" * 8 + "C" * 8,
}
POSITIONS = {
    "KIN_A": (1, 4, 8, 12, 16, 20, 24),
    "KIN_B": (2, 6, 10, 14, 18, 22, 24),
}
N_SITES = sum(len(positions) for positions in POSITIONS.values())
N_TRAIN = int(N_SITES * 0.8)
N_TEST = N_SITES - N_TRAIN
N_FAMS = len(FASTA)


@pytest.fixture
def tiny_inputs(tmp_path: Path) -> Path:
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    pd.DataFrame(
        [{"KINASE": kinase, "FASTA": seq} for kinase, seq in FASTA.items()]
    ).to_excel(data_dir / FASTA_FILE, index=False)
    pd.DataFrame(
        [
            {"Kinase": kinase, "Position": position}
            for kinase, positions in POSITIONS.items()
            for position in positions
        ]
    ).to_excel(data_dir / SITES_FILE, index=False)
    return data_dir


def _column(path: Path) -> list[str]:
    return pd.read_csv(path, header=None, dtype=str)[0].tolist()


def _matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy()


def test_cli_writes_outputs_to_selected_out_dir(tiny_inputs: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "elsewhere" / "generated"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir",
            str(tiny_inputs),
            "--out-dir",
            str(out_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    train_motifs = _column(out_dir / "train_motifs.csv")
    test_motifs = _column(out_dir / "new_test_motifs.csv")
    assert len(train_motifs) == N_TRAIN
    assert len(test_motifs) == N_TEST
    assert all(len(motif) == 15 for motif in train_motifs + test_motifs)

    train_matrix = _matrix(out_dir / "train_motifxFamMatrix.csv")
    test_matrix = _matrix(out_dir / "new_test_motifxFamMatrix.csv")
    assert train_matrix.shape == (N_TRAIN, N_FAMS)
    assert test_matrix.shape == (N_TEST, N_FAMS)
    assert np.all(train_matrix.sum(axis=1) == 1)
    assert np.all(test_matrix.sum(axis=1) == 1)
    assert set(np.unique(train_matrix)) <= {0, 1}
    assert set(np.unique(test_matrix)) <= {0, 1}

    fams = _column(out_dir / "fams.csv")
    assert fams == sorted(FASTA)
    dist_dir = out_dir / "fam_distances_blos62"
    assert _column(dist_dir / "fams.csv") == fams
    dist_matrix = _matrix(dist_dir / "dist_matrix.csv")
    assert dist_matrix.shape == (N_FAMS, N_FAMS)
    assert np.allclose(np.diag(dist_matrix), 0.0)
    assert np.allclose(dist_matrix, dist_matrix.T)
    assert np.allclose(dist_matrix[0, 1], 1.0)  # disjoint residue alphabets

    # The original script wrote relative to the current working directory.
    assert not (tmp_path / "data").exists()


def test_15mer_padding_logic_is_preserved(tiny_inputs: Path, tmp_path: Path) -> None:
    assert legacy.get_15mer(FASTA["KIN_A"], 1) == "_" * 7 + "S" * 8
    assert legacy.get_15mer(FASTA["KIN_A"], 24) == "T" * 8 + "_" * 7
    assert len(legacy.get_15mer(FASTA["KIN_A"], 12)) == 15

    out_dir = tmp_path / "padding"
    legacy.main(["--data-dir", str(tiny_inputs), "--out-dir", str(out_dir)])
    motifs = _column(out_dir / "train_motifs.csv") + _column(
        out_dir / "new_test_motifs.csv"
    )
    assert "_" * 7 + "S" * 8 in motifs
    assert "T" * 8 + "_" * 7 in motifs


def test_missing_spreadsheet_raises_clear_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    out_dir = tmp_path / "should_not_exist"

    with pytest.raises(FileNotFoundError) as excinfo:
        legacy.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)])
    message = str(excinfo.value)
    assert "Missing required spreadsheet" in message
    assert FASTA_FILE in message
    assert SITES_FILE in message
    assert "--data-dir" in message
    assert not out_dir.exists()

    pd.DataFrame({"KINASE": ["KIN_A"], "FASTA": [FASTA["KIN_A"]]}).to_excel(
        data_dir / FASTA_FILE, index=False
    )
    with pytest.raises(FileNotFoundError, match=SITES_FILE):
        legacy.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)])
    assert not out_dir.exists()


def test_no_archive_literal_remains_in_source() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "archive/" not in source
    assert legacy.FASTA_FILENAME == FASTA_FILE
    assert legacy.SITES_FILENAME == SITES_FILE


def test_default_paths_are_repo_root_relative() -> None:
    assert legacy.REPO_ROOT == Path(legacy.__file__).resolve().parent
    assert legacy.DEFAULT_DATA_DIR == legacy.REPO_ROOT
    assert legacy.DEFAULT_OUT_DIR == legacy.REPO_ROOT / "data"
    args = legacy.build_parser().parse_args([])
    assert args.data_dir == legacy.REPO_ROOT
    assert args.out_dir == legacy.REPO_ROOT / "data"
