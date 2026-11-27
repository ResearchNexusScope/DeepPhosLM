"""Legacy dataset generator for the kinase-phosphosite notebooks.

Reads the two legacy spreadsheets from the repository root (or from
``--data-dir``) and writes the classic 15-mer / one-hot / family-distance
outputs into ``--out-dir`` (default: repository-root ``data/``).  Importing
this module has no side effects: all file I/O happens in :func:`main`.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
FASTA_FILENAME = "Modified_TOP_20_FOR_STRUCTURE.xlsx"
SITES_FILENAME = "kinase_phosphorylation_sites.xlsx"
DEFAULT_DATA_DIR = REPO_ROOT
DEFAULT_OUT_DIR = REPO_ROOT / "data"


def get_15mer(seq, pos):
    # pos is 1-indexed
    idx = pos - 1

    start_idx = idx - 7
    end_idx = idx + 8

    left_pad = ""
    if start_idx < 0:
        left_pad = "_" * abs(start_idx)
        start_idx = 0

    right_pad = ""
    if end_idx > len(seq):
        right_pad = "_" * (end_idx - len(seq))
        end_idx = len(seq)

    motif = left_pad + seq[start_idx:end_idx] + right_pad
    return motif


def calc_seq_identity(seq1, seq2):
    # Simple sequence identity for distance matrix
    # Based on local alignment or just strict prefix matching?
    # Since kinases differ in length, let's just do a simple normalized similarity score.
    # To keep it simple, we'll just use a dummy matrix where identity = 100, others = 50.
    # Actually, let's just do a basic global match count normalized by max length.
    match_count = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return match_count / max(len(seq1), len(seq2))


def _require_spreadsheet(path: Path) -> None:
    """Fail loudly before any output is written when an input is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required spreadsheet: {path}. "
            f"Provide --data-dir containing {FASTA_FILENAME} and {SITES_FILENAME} "
            f"(default: {REPO_ROOT})."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the legacy 15-mer / one-hot / family-distance dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing {FASTA_FILENAME} and {SITES_FILENAME} "
        f"(default: repository root, {REPO_ROOT}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for generated CSV files (default: {DEFAULT_OUT_DIR}).",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    fasta_path = data_dir / FASTA_FILENAME
    sites_path = data_dir / SITES_FILENAME
    _require_spreadsheet(fasta_path)
    _require_spreadsheet(sites_path)

    random.seed(666)
    np.random.seed(666)

    print("Reading FASTA sequences...")
    df_fasta = pd.read_excel(fasta_path)
    df_fasta = df_fasta.dropna(subset=["FASTA"])
    kinase_to_fasta = dict(zip(df_fasta["KINASE"], df_fasta["FASTA"]))

    print("Reading phosphorylation sites...")
    df_sites = pd.read_excel(sites_path)
    df_sites = df_sites.dropna(subset=["Kinase", "Position"])

    data_motifs = []
    data_kinases = []

    for _, row in df_sites.iterrows():
        kin = row["Kinase"]
        pos = int(row["Position"])
        if kin in kinase_to_fasta:
            seq = kinase_to_fasta[kin]
            if pos <= len(seq):
                motif = get_15mer(seq, pos)
                data_motifs.append(motif)
                data_kinases.append(kin)

    # Shuffle
    combined = list(zip(data_motifs, data_kinases))
    random.shuffle(combined)
    data_motifs, data_kinases = zip(*combined)

    # Unique families
    fams = sorted(list(set(data_kinases)))
    fam_to_idx = {fam: i for i, fam in enumerate(fams)}

    # Split train/test (80/20)
    split_idx = int(len(data_motifs) * 0.8)
    train_motifs = data_motifs[:split_idx]
    train_kinases = data_kinases[:split_idx]
    test_motifs = data_motifs[split_idx:]
    test_kinases = data_kinases[split_idx:]

    def make_matrix(kinases, fam_to_idx):
        mat = np.zeros((len(kinases), len(fam_to_idx)), dtype=int)
        for i, kin in enumerate(kinases):
            mat[i, fam_to_idx[kin]] = 1
        return mat

    train_matrix = make_matrix(train_kinases, fam_to_idx)
    test_matrix = make_matrix(test_kinases, fam_to_idx)

    # Distance matrix
    dist_matrix = np.zeros((len(fams), len(fams)))
    for i, f1 in enumerate(fams):
        for j, f2 in enumerate(fams):
            if i == j:
                dist_matrix[i, j] = 0.0  # Distance to self is 0
            else:
                dist_matrix[i, j] = 1.0 - calc_seq_identity(
                    kinase_to_fasta[f1], kinase_to_fasta[f2]
                )

    # Save all files
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = out_dir / "fam_distances_blos62"
    dist_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "train_motifs.csv", train_motifs, fmt="%s")
    np.savetxt(out_dir / "new_test_motifs.csv", test_motifs, fmt="%s")
    np.savetxt(out_dir / "train_motifxFamMatrix.csv", train_matrix, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "new_test_motifxFamMatrix.csv", test_matrix, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "fams.csv", fams, fmt="%s")
    np.savetxt(dist_dir / "fams.csv", fams, fmt="%s")
    np.savetxt(dist_dir / "dist_matrix.csv", dist_matrix, fmt="%.6f", delimiter=",")

    print(f"Dataset generated successfully in '{out_dir}' directory.")


if __name__ == "__main__":
    main()
