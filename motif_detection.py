#!/usr/bin/env python3
"""Candidate phosphorylation-motif discovery in kinase sequences.

IMPORTANT: the regular expressions below find *candidate sequence motifs*
only. A regex match is NOT a validated phosphorylation site: there is no
mass-spectrometry evidence, no structural evidence and no kinase-substrate
validation behind a match. Experimental validation is required before any
matched residue may be described as phosphorylated.

The spreadsheet is resolved relative to this repository (or an explicit
``--data-dir`` / ``--spreadsheet``). No machine-specific absolute path is used.

Usage::

    python motif_detection.py --data-dir . --out-dir motif_output
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPREADSHEET_NAMES = (
    "Modified_TOP_20_FOR_STRUCTURE.xlsx",
    "kinase_phosphorylation_sites.xlsx",
)

# Patterns are deliberately permissive short sequence motifs. See the module
# docstring: matches are candidates, NOT validated phosphorylation sites.
MOTIF_PATTERNS = {
    "Ser/Thr kinase (candidate)": r"[ST][^P][^P][RK]",
    "Tyrosine kinase (candidate)": r"[ST]..Y",
}

REQUIRED_COLUMNS = ("KINASE", "FASTA")


def predict_phosphorylation_sites(sequence: str) -> list[int]:
    """Return 0-based positions of candidate phosphorylation motifs.

    The two patterns ``[ST][^P][^P][RK]`` and ``[ST]..Y`` are scanned over
    ``sequence``. A returned position marks a candidate motif *only*; it is
    NOT a validated phosphorylation site. Overlapping matches from the two
    patterns are de-duplicated and the result is sorted.
    """
    positions: set[int] = set()
    for pattern in MOTIF_PATTERNS.values():
        for match in re.finditer(pattern, sequence):
            positions.add(match.start())
    return sorted(positions)


def classify_phosphorylation_sites(sequence: str) -> dict[str, list[tuple[int, str]]]:
    """Group candidate motifs by pattern, returning (1-based position, motif).

    Every entry is a candidate motif, NOT a validated phosphorylation site.
    """
    classified: dict[str, list[tuple[int, str]]] = {key: [] for key in MOTIF_PATTERNS}
    for kinase_type, pattern in MOTIF_PATTERNS.items():
        for match in re.finditer(pattern, sequence):
            classified[kinase_type].append(
                (match.start() + 1, sequence[match.start():match.start() + 4])
            )
    return classified


def resolve_spreadsheet(data_dir: Path | str | None = None,
                        spreadsheet: Path | str | None = None) -> Path:
    """Resolve the input spreadsheet without any machine-specific hard-coding.

    ``spreadsheet`` wins when given. Otherwise look for the known spreadsheet
    names in ``data_dir`` (default: the repository root).
    """
    if spreadsheet:
        path = Path(spreadsheet).expanduser()
        if not path.is_absolute():
            path = (Path(data_dir) if data_dir else REPO_ROOT) / path
        if not path.is_file():
            raise FileNotFoundError(f"Spreadsheet not found: {path}")
        return path.resolve()

    search_root = Path(data_dir).expanduser() if data_dir else REPO_ROOT
    if not search_root.is_absolute():
        search_root = REPO_ROOT / search_root
    for name in DEFAULT_SPREADSHEET_NAMES:
        candidate = search_root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No kinase spreadsheet found in {search_root}; expected one of "
        f"{', '.join(DEFAULT_SPREADSHEET_NAMES)}. Pass --spreadsheet explicitly."
    )


def load_kinase_sequences(path: Path | str):
    """Load the KINASE and FASTA columns from the resolved spreadsheet."""
    import pandas as pd

    frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Spreadsheet {path} is missing columns: {missing}")
    return frame[list(REQUIRED_COLUMNS)].dropna(subset=["FASTA"])


def candidate_site_table(frame) -> "object":
    """Return a long-format table of candidate motif matches (not sites)."""
    import pandas as pd

    records = []
    for _index, row in frame.iterrows():
        kinase = row["KINASE"]
        classified = classify_phosphorylation_sites(str(row["FASTA"]))
        for kinase_type, sites in classified.items():
            for position, motif in sites:
                records.append(
                    {
                        "Kinase": kinase,
                        "Type": kinase_type,
                        "Position": position,
                        "Motif": motif,
                        "Validated": False,
                    }
                )
    return pd.DataFrame(
        records, columns=["Kinase", "Type", "Position", "Motif", "Validated"]
    )


def make_aligner():
    """Create a global pairwise aligner (Biopython required only when used)."""
    try:
        from Bio import Align
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Biopython is required for sequence alignment") from exc
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def conservation_scores(sequences: list[str], aligner) -> list[float]:
    """Pairwise alignment scores for consecutive sequences."""
    return [float(aligner.score(sequences[i], sequences[i + 1]))
            for i in range(len(sequences) - 1)]


def plot_conservation(scores: list[float], out_dir: Path) -> Path | None:
    """Save a conservation line plot under ``out_dir`` (never in the repo)."""
    if not scores:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(scores)), scores, marker="o")
    ax.set_xlabel("Sequence pair index")
    ax.set_ylabel("Alignment score")
    ax.set_title("Pairwise kinase sequence conservation (candidate analysis)")
    fig.tight_layout()
    path = out_dir / "conservation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT,
                        help="directory containing the input spreadsheet")
    parser.add_argument("--spreadsheet", type=Path, default=None,
                        help="explicit spreadsheet path (overrides --data-dir lookup)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="optional output directory for CSV/plot artifacts")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip the conservation plot")
    args = parser.parse_args(argv)

    path = resolve_spreadsheet(args.data_dir, args.spreadsheet)
    frame = load_kinase_sequences(path)
    table = candidate_site_table(frame)
    print(f"Loaded {len(frame)} kinase sequences from {path}")
    print(f"Found {len(table)} candidate motifs (NOT validated phosphorylation sites)")
    print(table.head(20).to_string(index=False))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "candidate_motifs.csv", index=False)
        if not args.no_plots:
            aligner = make_aligner()
            scores = conservation_scores(frame["FASTA"].astype(str).tolist(), aligner)
            plot_conservation(scores, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
