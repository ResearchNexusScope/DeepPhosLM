"""Leakage-free dataset builder for the kinase phosphosite project.

This module replaces the fragile parts of the legacy ``generate_dataset.py``
(which reads ``archive/...`` paths that do not exist on a fresh clone) with a
deterministic, inspectable pipeline:

1. Read the two spreadsheets from the repository root (or ``--data-dir``).
2. Build 15-residue windows around each known phosphorylation site, using the
   exact same ``get_15mer`` rule as the legacy script (1-indexed centre, seven
   residues on each side, ``_`` padding at protein termini).
3. Sample background (negative) windows from S/T/Y residues of the same
   protein that are not annotated as phosphorylation sites.
4. Group *whole proteins* into homology clusters and put entire clusters into
   a single split, so neither a protein nor a homologous cluster can leak
   across train/val/test.

Homology control note
---------------------
``assign_clusters`` implements a **greedy in-repo approximation of CD-HIT**, it
is **not CD-HIT** and does not call any external binary.  The identity measure
is deliberately the simple length-normalised positional identity used by the
legacy code::

    identity(a, b) = number of positions i where a[i] == b[i]
                     / max(len(a), len(b))          (zip stops at shorter end)

Proteins are processed in sorted name order; each protein joins the first
existing cluster that contains at least one member with identity >= threshold
(single-linkage style, first-match wins).  With the real 25-protein FASTA file
and the default 0.4 threshold this metric produces singleton clusters, because
the sequences are only weakly similar positionally; the leakage guard is still
structurally enforced.  See ``manifest["notes"]`` in the generated output.

Negative sampling rule
----------------------
For each kinase, the candidate pool is every position ``i`` in its FASTA whose
residue is S, T or Y and which is *not* in that kinase's set of annotated
positive positions.  We draw ``round(ratio * n_positives_for_that_kinase)``
candidates without replacement, using one ``random.Random(seed)`` instance
iterated over kinases in sorted name order and candidates in ascending position
order.  The draw is clamped to the pool size if a pool is too small (recorded in
``manifest["warnings"]``).  Label 1 = positive, 0 = negative.

Site type rule
--------------
``site_type`` is always the observed centre residue (S/T/Y) for every window,
positive *and* negative.  The spreadsheet's ``Type`` column is consulted only
defensively, through ``site_type_for``'s fallback: ``load_inputs`` already drops
rows whose centre is not S/T/Y, so in practice that fallback is dead code for
positives.  ``Type`` is in fact a *kinase class* ("Ser/Thr Kinase" /
"Tyrosine Kinase"), not a residue; in the supplied data 93 rows labelled
"Tyrosine Kinase" have an S or T centre, so using the literal class as the
residue would corrupt the label.  This is recorded in ``manifest["notes"]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

SITES_FILENAME = "kinase_phosphorylation_sites.xlsx"
FASTA_FILENAME = "Modified_TOP_20_FOR_STRUCTURE.xlsx"

WINDOW_HALF_WIDTH = 7
WINDOW_LENGTH = 2 * WINDOW_HALF_WIDTH + 1
SITE_RESIDUES = frozenset("STY")

DEFAULT_SEED = 666
DEFAULT_RATIO = 1.0
DEFAULT_IDENTITY_THRESHOLD = 0.4

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
CSV_COLUMNS: Tuple[str, ...] = ("sequence", "label", "site_type", "kinase", "cluster_id")

# ``Type`` in the sites spreadsheet is a kinase-class label, not a residue.
# Used only as a fallback for the (unexpected) case of a non-STY centre.
TYPE_CLASS_FALLBACK = {"ser/thr kinase": ("S", "T"), "tyrosine kinase": ("Y",)}


@dataclass(frozen=True)
class Window:
    """A single labelled 15-mer with its provenance."""

    sequence: str
    label: int
    site_type: str
    kinase: str
    position: int

    def as_row(self, cluster_id: str) -> Dict[str, object]:
        return {
            "sequence": self.sequence,
            "label": self.label,
            "site_type": self.site_type,
            "kinase": self.kinase,
            "cluster_id": cluster_id,
        }


# ---------------------------------------------------------------------------
# Paths and file hashing
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def resolve_input_paths(data_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Resolve both spreadsheet paths and fail loudly when either is missing."""
    base = Path(data_dir).resolve() if data_dir is not None else find_repo_root()
    sites = base / SITES_FILENAME
    fasta = base / FASTA_FILENAME
    missing = [str(p) for p in (sites, fasta) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required input spreadsheet(s) not found: "
            + ", ".join(missing)
            + ". Pass --data-dir pointing at the directory containing "
            + f"'{SITES_FILENAME}' and '{FASTA_FILENAME}'."
        )
    return sites, fasta


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Legacy-compatible window construction
# ---------------------------------------------------------------------------


def get_15mer(seq: str, pos: int) -> str:
    """15-residue window centred on 1-indexed ``pos`` (legacy ``get_15mer``).

    ``_`` pads missing residues at either terminus.  The returned string is
    always exactly ``WINDOW_LENGTH`` characters long for a valid position.
    """
    idx = pos - 1
    start_idx = idx - WINDOW_HALF_WIDTH
    end_idx = idx + WINDOW_HALF_WIDTH + 1

    left_pad = ""
    if start_idx < 0:
        left_pad = "_" * abs(start_idx)
        start_idx = 0

    right_pad = ""
    if end_idx > len(seq):
        right_pad = "_" * (end_idx - len(seq))
        end_idx = len(seq)

    return left_pad + seq[start_idx:end_idx] + right_pad


def site_type_for(residue: str, type_label: Optional[str] = None) -> str:
    """Return S/T/Y for a window centre, consulting ``Type`` only as fallback."""
    residue = residue.upper()
    if residue in SITE_RESIDUES:
        return residue
    if type_label:
        allowed = TYPE_CLASS_FALLBACK.get(str(type_label).strip().lower(), ())
        if allowed:
            return allowed[0]
    raise ValueError(f"Cannot determine S/T/Y site type for residue {residue!r}")


# ---------------------------------------------------------------------------
# Homology approximation (NOT CD-HIT)
# ---------------------------------------------------------------------------


def sequence_identity(a: str, b: str) -> float:
    """Length-normalised positional identity: matches / max(len(a), len(b))."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / longest


def assign_clusters(
    kinase_to_fasta: Mapping[str, str], threshold: float = DEFAULT_IDENTITY_THRESHOLD
) -> Dict[str, List[str]]:
    """Greedy, deterministic protein clustering (in-repo CD-HIT approximation).

    Returns a mapping ``{"cluster_000": ["K1", "K4"], ...}``.  Cluster IDs are
    handed out in the order clusters are first created while proteins are
    processed in alphabetical order, so the result is reproducible.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"identity_threshold must be in [0, 1], got {threshold!r}")

    clusters: List[List[str]] = []
    for kinase in sorted(kinase_to_fasta):
        for group in clusters:
            if any(
                sequence_identity(kinase_to_fasta[kinase], kinase_to_fasta[member])
                >= threshold
                for member in group
            ):
                group.append(kinase)
                break
        else:
            clusters.append([kinase])

    return {
        f"cluster_{index:03d}": group for index, group in enumerate(clusters)
    }


def split_cluster_ids(
    cluster_ids: Sequence[str], seed: int = DEFAULT_SEED
) -> Dict[str, List[str]]:
    """Split whole clusters 80/10/10 with a seeded shuffle.

    Small cluster counts are handled explicitly so that, whenever there are at
    least three clusters, every split is non-empty.
    """
    ids = sorted(cluster_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {split: [] for split in SPLITS}
    if n == 1:
        return {"train": ids[:1], "val": [], "test": []}
    if n == 2:
        return {"train": ids[:1], "val": ids[1:], "test": []}

    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_val = 1
        n_train = n - 2
        n_test = 1

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Loading and sampling
# ---------------------------------------------------------------------------


def load_inputs(
    sites_path: Path, fasta_path: Path
) -> Tuple[Dict[str, str], List[Tuple[str, int, str]], List[str]]:
    """Read the spreadsheets.

    Returns ``(fasta_by_kinase, positives, warnings)`` where each positive is
    ``(kinase, 1_indexed_position, Type)``.  Unusable rows are skipped and
    reported in ``warnings`` rather than silently corrupting the dataset.
    """
    warnings: List[str] = []

    fasta_df = pd.read_excel(fasta_path).dropna(subset=["KINASE", "FASTA"])
    fasta_by_kinase: Dict[str, str] = {}
    for kinase, sequence in zip(fasta_df["KINASE"], fasta_df["FASTA"]):
        fasta_by_kinase[str(kinase).strip()] = str(sequence).strip().upper()
    if not fasta_by_kinase:
        raise ValueError(f"No usable FASTA rows found in {fasta_path}")

    sites_df = pd.read_excel(sites_path).dropna(subset=["Kinase", "Position", "Type"])
    positives: List[Tuple[str, int, str]] = []
    seen = set()
    for kinase_raw, position_raw, type_raw in zip(
        sites_df["Kinase"], sites_df["Position"], sites_df["Type"]
    ):
        kinase = str(kinase_raw).strip()
        if kinase not in fasta_by_kinase:
            warnings.append(f"Skipped site for unknown kinase {kinase!r}")
            continue
        try:
            position = int(position_raw)
        except (TypeError, ValueError):
            warnings.append(
                f"Skipped non-integer position {position_raw!r} for {kinase}"
            )
            continue
        sequence = fasta_by_kinase[kinase]
        if not 1 <= position <= len(sequence):
            warnings.append(
                f"Skipped out-of-range position {position} for {kinase} "
                f"(FASTA length {len(sequence)})"
            )
            continue
        if sequence[position - 1] not in SITE_RESIDUES:
            warnings.append(
                f"Skipped position {position} for {kinase}: centre residue "
                f"{sequence[position - 1]!r} is not S/T/Y"
            )
            continue
        key = (kinase, position)
        if key in seen:
            continue
        seen.add(key)
        positives.append((kinase, position, str(type_raw)))

    if not positives:
        raise ValueError(f"No usable phosphorylation sites found in {sites_path}")
    return fasta_by_kinase, positives, warnings


def sample_negatives(
    fasta_by_kinase: Mapping[str, str],
    positive_positions: Mapping[str, set[int]],
    ratio: float,
    seed: int,
    warnings: List[str],
) -> Dict[str, List[int]]:
    """Seeded per-kinase 1:``ratio`` sampling of non-site S/T/Y positions."""
    if ratio < 0:
        raise ValueError(f"ratio must be non-negative, got {ratio!r}")
    rng = random.Random(seed)
    sampled: Dict[str, List[int]] = {}
    for kinase in sorted(fasta_by_kinase):
        sequence = fasta_by_kinase[kinase]
        excluded = positive_positions.get(kinase, set())
        candidates = [
            pos
            for pos, residue in enumerate(sequence, start=1)
            if residue in SITE_RESIDUES and pos not in excluded
        ]
        wanted = int(math.floor(ratio * len(excluded) + 0.5))
        if wanted > len(candidates):
            warnings.append(
                f"Kinase {kinase}: requested {wanted} negatives but only "
                f"{len(candidates)} candidates available; using {len(candidates)}"
            )
            wanted = len(candidates)
        sampled[kinase] = sorted(rng.sample(candidates, wanted))
    return sampled


# ---------------------------------------------------------------------------
# Manifest and output
# ---------------------------------------------------------------------------


def _site_type_counts(windows: Sequence[Window]) -> Dict[str, int]:
    counts = Counter(w.site_type for w in windows)
    return {residue: counts.get(residue, 0) for residue in ("S", "T", "Y")}


def _label_counts(windows: Sequence[Window]) -> Tuple[int, int]:
    positive = sum(1 for w in windows if w.label == 1)
    return positive, len(windows) - positive


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_manifest(
    *,
    output_dir: Path,
    repo_root: Path,
    sites_path: Path,
    fasta_path: Path,
    seed: int,
    ratio: float,
    threshold: float,
    clusters: Mapping[str, Sequence[str]],
    splits: Mapping[str, Sequence[str]],
    windows_by_split: Mapping[str, Sequence[Window]],
    warnings: Sequence[str],
) -> Dict[str, object]:
    """Assemble the JSON-serialisable manifest describing one dataset build."""
    # ``Window`` itself carries no cluster id: derive rows via cluster mapping.
    kinase_to_cluster = {
        kinase: cid for cid, kinases in clusters.items() for kinase in kinases
    }
    rows_by_split = {
        split: [
            w.as_row(kinase_to_cluster[w.kinase])
            for w in windows_by_split[split]
        ]
        for split in SPLITS
    }

    per_split: Dict[str, object] = {}
    for split in SPLITS:
        windows = windows_by_split[split]
        positive, negative = _label_counts(windows)
        per_split[split] = {
            "count": len(windows),
            "positive": positive,
            "negative": negative,
            "site_type_counts": _site_type_counts(windows),
            "clusters": list(splits[split]),
        }

    all_windows = [w for split in SPLITS for w in windows_by_split[split]]
    total_positive, total_negative = _label_counts(all_windows)
    n_singletons = sum(1 for group in clusters.values() if len(group) == 1)
    n_proteins = len({kinase for group in clusters.values() for kinase in group})

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "ratio": ratio,
        "threshold": threshold,
        "identity_threshold": threshold,
        "identity_metric": "matches / max(len(a), len(b)); positional, not alignment-based",
        "window": {
            "half_width": WINDOW_HALF_WIDTH,
            "length": WINDOW_LENGTH,
            "padding": "_",
        },
        "inputs": {
            "sites": {
                "path": _relative_or_absolute(sites_path, repo_root),
                "sha256": sha256_file(sites_path),
            },
            "fasta": {
                "path": _relative_or_absolute(fasta_path, repo_root),
                "sha256": sha256_file(fasta_path),
            },
        },
        "outputs": {
            "directory": _relative_or_absolute(output_dir, repo_root),
            "files": {f"{split}.csv": len(rows_by_split[split]) for split in SPLITS},
        },
        "counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "totals": {
            "windows": len(all_windows),
            "positive": total_positive,
            "negative": total_negative,
            "clusters": len(clusters),
            "kinases": len({w.kinase for w in all_windows}),
        },
        "site_type_counts": _site_type_counts(all_windows),
        "splits": per_split,
        "cluster_membership": {
            cid: {"kinases": list(kinases), "split": next(
                (s for s in SPLITS if cid in splits[s]), None
            )}
            for cid, kinases in clusters.items()
        },
        "warnings": list(warnings),
        "notes": [
            "Clustering is a greedy in-repo approximation of CD-HIT, not CD-HIT: "
            "identity = positional matches / max length; a protein joins the "
            "first cluster with a member at or above the threshold.",
            f"Greedy positional-identity clustering produced {len(clusters)} "
            f"clusters from {n_proteins} proteins ({n_singletons} singletons). "
            "On the supplied 25-kinase FASTA at threshold 0.4 all 25 clusters "
            "are singletons, so homology control is structurally leakage-free "
            "but is not a real CD-HIT-style grouping.",
            "The 80/10/10 split is by CLUSTER count, not by window count, so "
            "sample counts are skewed. This build has train="
            f"{len(rows_by_split['train'])}, val={len(rows_by_split['val'])}, "
            f"test={len(rows_by_split['test'])} windows; for the supplied "
            "spreadsheets at seed 666 the counts are train=860, val=18, "
            "test=88.",
            "Whole clusters are assigned to one split, so no protein or cluster "
            "appears in more than one split.",
            "site_type is the observed centre residue (S/T/Y). The sites 'Type' "
            "column is a kinase class and is used only as a fallback, because "
            "some 'Tyrosine Kinase' rows have S/T centres.",
            "Negatives are S/T/Y residues of the same protein not annotated as "
            "sites; ratio controls negatives per positive, sampled with a "
            "fixed seed.",
        ],
    }
    return manifest


def write_split_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write one split as a header-bearing CSV with the canonical columns."""
    pd.DataFrame(list(rows), columns=list(CSV_COLUMNS)).to_csv(path, index=False)


def run_pipeline(
    *,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    ratio: float = DEFAULT_RATIO,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
) -> Dict[str, object]:
    """Build the honest dataset under ``data/honest`` and return the manifest."""
    repo_root = find_repo_root()
    sites_path, fasta_path = resolve_input_paths(data_dir)

    fasta_by_kinase, positives, warnings = load_inputs(sites_path, fasta_path)
    positive_positions: Dict[str, set[int]] = defaultdict(set)
    for kinase, position, _ in positives:
        positive_positions[kinase].add(position)

    clusters = assign_clusters(fasta_by_kinase, identity_threshold)
    kinase_to_cluster = {
        kinase: cid for cid, kinases in clusters.items() for kinase in kinases
    }
    splits = split_cluster_ids(list(clusters), seed)
    negatives = sample_negatives(
        fasta_by_kinase, positive_positions, ratio, seed, warnings
    )

    windows_by_cluster: Dict[str, List[Window]] = defaultdict(list)
    for kinase, position, type_label in positives:
        sequence = fasta_by_kinase[kinase]
        residue = sequence[position - 1]
        windows_by_cluster[kinase_to_cluster[kinase]].append(
            Window(
                sequence=get_15mer(sequence, position),
                label=1,
                site_type=site_type_for(residue, type_label),
                kinase=kinase,
                position=position,
            )
        )
    for kinase, positions in negatives.items():
        sequence = fasta_by_kinase[kinase]
        for position in positions:
            windows_by_cluster[kinase_to_cluster[kinase]].append(
                Window(
                    sequence=get_15mer(sequence, position),
                    label=0,
                    site_type=site_type_for(sequence[position - 1]),
                    kinase=kinase,
                    position=position,
                )
            )

    # Rows are emitted in (cluster id, kinase, position) order within a split:
    # no randomness here, so identical seeds always produce byte-identical CSVs.
    windows_by_split: Dict[str, List[Window]] = {}
    for split in SPLITS:
        windows: List[Window] = []
        for cid in sorted(splits[split]):
            windows.extend(
                sorted(
                    windows_by_cluster.get(cid, []),
                    key=lambda w: (w.kinase, w.position, w.label),
                )
            )
        windows_by_split[split] = windows

    out_dir = Path(output_dir).resolve() if output_dir is not None else repo_root / "data" / "honest"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_split_csv(
            out_dir / f"{split}.csv",
            [w.as_row(kinase_to_cluster[w.kinase]) for w in windows_by_split[split]],
        )

    manifest = build_manifest(
        output_dir=out_dir,
        repo_root=repo_root,
        sites_path=sites_path,
        fasta_path=fasta_path,
        seed=seed,
        ratio=ratio,
        threshold=identity_threshold,
        clusters=clusters,
        splits=splits,
        windows_by_split=windows_by_split,
        warnings=warnings,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
