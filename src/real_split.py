"""Homology-controlled train/val/test split for the *real* phosphosite dataset.

This module operates on ``data/real/data.csv`` (the FASTA-validated dataset built
by :mod:`src.real_data`) and on the hash-pinned UniProt FASTA chunks in
``data/real/fasta_cache/``.  It clusters whole proteins with MMseqs2 and assigns
*entire clusters* to one split, following published phosphosite benchmark
practice (LitePhospho / DeepPhos / GPS 6.0 cluster at 30-40% identity so no
homologous protein crosses train/test).

Pipeline
--------
1. Collect the unique ``protein_acc`` values from ``data.csv``, read the cached
   FASTA chunks, and write one *clean* combined FASTA (``>ACCESSION`` headers,
   sorted accessions) into the work directory.  Restricting to accessions that
   actually occur in ``data.csv`` makes every MMseqs DB identifier identical to
   a ``protein_acc`` value, so the cluster TSV needs no name translation.
2. Run three explicit MMseqs commands (no network, fixed threads, documented
   flags)::

       mmseqs createdb <proteins.fasta> <seqDB> --dbtype 1 --threads N -v 0
       mmseqs cluster <seqDB> <cluDB> <tmpdir> --min-seq-id 0.3 -c 0.5 \
           --cov-mode 0 --threads N -v 0
       mmseqs createtsv <seqDB> <seqDB> <cluDB> <clusters.tsv> --threads N -v 0

   On Windows MMseqs spawns POSIX shell scripts; the release ships
   ``busybox.exe`` but no ``sh.exe``.  MMseqs resolves that shell relative to
   its own executable, so the default runner idempotently copies
   ``busybox.exe`` to ``sh.exe``/``bash.exe`` **next to the binary** (inside the
   gitignored install dir).  Stale MMseqs DB files in the work dir are removed
   before each run so re-runs are safe.
3. Parse the two-column cluster TSV (``representative \\t member``) into
   ``{protein_acc: cluster_id}`` where ``cluster_id`` is the representative
   accession.  Every protein absent from the TSV becomes its own singleton
   cluster whose id is the protein accession itself.
4. Assign whole clusters to train/val/test 80/10/10 **by cluster count**, with
   a fixed-seed shuffle of the sorted cluster ids.
5. Write ``train.csv``, ``val.csv``, ``test.csv`` (all ``data.csv`` columns plus
   ``cluster_id``) and a hash-pinned ``manifest.json``.  Explicit
   no-overlap assertions run *before* anything is written: no protein and no
   cluster may appear in two splits, and every protein/row must be assigned
   exactly once.  Any violation raises :class:`OverlapError` and no file is
   written.

Honest caveats
--------------
* The split is by **cluster count, not by window count**.  A few large clusters
  can hold many windows, so window counts (and therefore positives/negatives)
  are necessarily skewed.  The **actual** per-split window counts are reported
  in ``manifest['splits'][split]['windows']``; the module docstring deliberately
  does not promise a 80/10/10 split of rows.
* MMseqs clustering is a heuristic (set-cover, default `--cluster-mode 0`) at the
  requested identity threshold; "no homology crosses splits" means "no cluster
  reported by these exact parameters crosses splits".  Distant relationships
  below ``min_seq_id`` are not detected.
* Isoform accessions (``P12345-2``) are treated as independent proteins, exactly
  as they appear in ``data.csv``.
* No network access happens here: sequences come from the pinned cache only.

The pipeline is testable without the binary: :func:`build_real_split` accepts an
injected ``runner`` with the signature ``(args: Sequence[str]) -> str`` (it
receives the argv *after* the binary path and returns stdout).  Tests inject a
fake runner that writes a synthetic cluster TSV.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from src.real_data import CSV_COLUMNS, parse_fasta

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
CLUSTER_ID_COLUMN = "cluster_id"
DEFAULT_MIN_SEQ_ID = 0.3
DEFAULT_SEED = 666
DEFAULT_THREADS = 8
DEFAULT_COVERAGE = 0.5  # -c 0.5
DEFAULT_COV_MODE = 0    # coverage of query and target

# `createtsv` writes this file name in the work directory.
CLUSTER_TSV_NAME = "clusters.tsv"
PROTEINS_FASTA_NAME = "proteins.fasta"

# The runner receives argv *after* the binary path and returns stdout text.
Runner = Callable[[Sequence[str]], str]


class OverlapError(ValueError):
    """Raised when a proposed split lets a protein or cluster cross splits."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def default_mmseqs_path() -> Path:
    """Return the repo-local MMseqs binary path (Windows default)."""
    return find_repo_root() / "tools" / "bin" / "mmseqs" / "bin" / "mmseqs.exe"


def _display_path(path: Path, root: Path) -> str:
    """Prefer a repo-relative path for readable manifests."""
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(Path(path).resolve())


# ---------------------------------------------------------------------------
# Runner / external command execution
# ---------------------------------------------------------------------------


def _ensure_shell_shim(work_dir: Path, binary: Path, env: Dict[str, str]) -> None:
    """Make a real POSIX shell available to MMseqs on Windows.

    MMseqs' clustering modules execute bundled ``*.sh`` scripts.  On Windows the
    release ships ``busybox.exe`` (which provides ``sh``) but no ``sh.exe``, and
    the system ``bash.exe`` is usually the non-functional WSL launcher stub.
    MMseqs resolves the shell via ``argv[0]``'s directory rather than ``PATH``,
    so we copy busybox into the **binary's own directory** (a gitignored install
    dir) under the names ``sh.exe`` and ``bash.exe``.  This is idempotent and is
    the only filesystem side effect outside the work directory.  If no busybox
    is found next to the binary we do nothing (the command will fail loudly with
    MMseqs' own error if a shell was actually required).
    """
    if os.name != "nt":  # pragma: no cover - exercised only on Windows
        return
    del work_dir  # kept in the signature for callers/tests; shims live by the binary
    binary_dir = binary.parent
    busybox = binary_dir / "busybox.exe"
    if not busybox.is_file():
        return
    for name in ("sh.exe", "bash.exe"):
        target = binary_dir / name
        if not target.exists():
            shutil.copy2(busybox, target)
    env["PATH"] = str(binary_dir) + os.pathsep + env.get("PATH", "")


def make_mmseqs_runner(binary: Path, work_dir: Path) -> Runner:
    """Build the default :data:`Runner` that executes the real MMseqs binary."""
    binary = Path(binary).resolve()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def runner(args: Sequence[str]) -> str:
        env = os.environ.copy()
        _ensure_shell_shim(work_dir, binary, env)
        argv = [str(binary)] + [str(arg) for arg in args]
        proc = subprocess.run(
            argv,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "MMseqs command failed with exit "
                f"{proc.returncode}: {' '.join(argv)}\n"
                f"--- stderr ---\n{proc.stderr}\n--- stdout ---\n{proc.stdout}"
            )
        return proc.stdout

    return runner


# ---------------------------------------------------------------------------
# Step 1: collect sequences and write a clean FASTA
# ---------------------------------------------------------------------------


@dataclass
class FastaCollection:
    """Outcome of :func:`collect_fasta`."""

    fasta_path: Path
    sequences: Dict[str, str]
    n_csv_accessions: int
    n_with_fasta: int
    missing_accessions: List[str] = field(default_factory=list)
    cache_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def collect_fasta(data_dir: Path, work_dir: Path) -> FastaCollection:
    """Read cached FASTA chunks and write one clean, sorted FASTA.

    Only accessions present in ``data.csv`` are kept.  Headers are rewritten to
    bare ``>ACCESSION`` so MMseqs DB identifiers equal ``protein_acc`` values
    byte-for-byte (including isoform suffixes such as ``-2``).
    """
    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    csv_path = data_dir / "data.csv"
    cache_dir = data_dir / "fasta_cache"
    if not csv_path.is_file():
        raise FileNotFoundError(f"data.csv not found: {csv_path}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"FASTA cache directory not found: {cache_dir}")

    frame = pd.read_csv(csv_path, usecols=["protein_acc"])
    csv_accessions = sorted({str(acc).strip().upper() for acc in frame["protein_acc"] if str(acc).strip()})
    if not csv_accessions:
        raise ValueError(f"No protein_acc values found in {csv_path}")

    merged: Dict[str, str] = {}
    cache_files: Dict[str, Dict[str, Any]] = {}
    for chunk in sorted(cache_dir.glob("*.fasta")):
        text = chunk.read_text(encoding="utf-8")
        parsed = parse_fasta(text)
        cache_files[chunk.name] = {
            "sha256": sha256_file(chunk),
            "n_sequences": len(parsed),
        }
        for accession, sequence in parsed.items():
            merged.setdefault(accession, sequence)

    sequences = {acc: merged[acc] for acc in csv_accessions if acc in merged}
    missing = [acc for acc in csv_accessions if acc not in merged]

    fasta_path = work_dir / PROTEINS_FASTA_NAME
    work_dir.mkdir(parents=True, exist_ok=True)
    # Binary write keeps the digest stable across platforms (no CRLF translation).
    payload = "".join(f">{acc}\n{sequences[acc]}\n" for acc in sorted(sequences))
    fasta_path.write_bytes(payload.encode("utf-8"))

    return FastaCollection(
        fasta_path=fasta_path,
        sequences=sequences,
        n_csv_accessions=len(csv_accessions),
        n_with_fasta=len(sequences),
        missing_accessions=missing,
        cache_files=cache_files,
    )


# ---------------------------------------------------------------------------
# Step 2: cluster with MMseqs
# ---------------------------------------------------------------------------


@dataclass
class Clustering:
    """Result of MMseqs clustering."""

    protein_to_cluster: Dict[str, str]
    n_clusters: int
    n_singletons: int
    tsv_path: Path
    commands: List[List[str]] = field(default_factory=list)


def cluster_commands(
    fasta_path: Path,
    work_dir: Path,
    *,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    threads: int = DEFAULT_THREADS,
) -> Tuple[List[List[str]], Path]:
    """Return the three exact MMseqs argv lists and the cluster TSV path.

    Kept as a pure function so the commands can be recorded in the manifest and
    inspected in tests without executing anything.
    """
    if not 0.0 < min_seq_id <= 1.0:
        raise ValueError(f"min_seq_id must be in (0, 1], got {min_seq_id!r}")
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads!r}")

    seq_db = work_dir / "seqDB"
    clu_db = work_dir / "cluDB"
    tmp_dir = work_dir / "mmseqs_tmp"
    tsv_path = work_dir / CLUSTER_TSV_NAME
    commands: List[List[str]] = [
        ["createdb", str(fasta_path), str(seq_db),
         "--dbtype", "1", "--threads", str(threads), "-v", "0"],
        ["cluster", str(seq_db), str(clu_db), str(tmp_dir),
         "--min-seq-id", str(min_seq_id),
         "-c", str(DEFAULT_COVERAGE),
         "--cov-mode", str(DEFAULT_COV_MODE),
         "--threads", str(threads), "-v", "0"],
        ["createtsv", str(seq_db), str(seq_db), str(clu_db), str(tsv_path),
         "--threads", str(threads), "-v", "0"],
    ]
    return commands, tsv_path


def cleanup_stale_mmseqs_outputs(work_dir: Path) -> None:
    """Remove MMseqs DB/TSV files left by a previous (possibly failed) run.

    ``mmseqs cluster`` refuses to write over an existing ``cluDB.dbtype``, so a
    re-run must start from a clean set of DB paths.  Only files with the known
    prefixes are touched; the input FASTA is left alone.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return
    for pattern in ("seqDB*", "cluDB*"):
        for path in work_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                path.unlink(missing_ok=True)
    tmp_dir = work_dir / "mmseqs_tmp"
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tsv_path = work_dir / CLUSTER_TSV_NAME
    if tsv_path.is_file():
        tsv_path.unlink()


def cluster_proteins(
    fasta_path: Path,
    work_dir: Path,
    runner: Runner,
    *,
    accessions: Collection[str],
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    threads: int = DEFAULT_THREADS,
) -> Clustering:
    """Run MMseqs via ``runner`` and parse the resulting cluster TSV."""
    cleanup_stale_mmseqs_outputs(work_dir)
    commands, tsv_path = cluster_commands(
        fasta_path, work_dir, min_seq_id=min_seq_id, threads=threads
    )
    for command in commands:
        runner(command)
    return parse_cluster_tsv(tsv_path, accessions, commands=commands)


# ---------------------------------------------------------------------------
# Step 3: parse cluster TSV with singleton fallback
# ---------------------------------------------------------------------------


def parse_cluster_tsv(
    tsv_path: Path,
    accessions: Collection[str],
    *,
    commands: Optional[Sequence[Sequence[str]]] = None,
) -> Clustering:
    """Parse ``representative \\t member`` rows into a protein->cluster map.

    Every accession not present as a member becomes a singleton cluster whose
    id is the accession itself.  Unknown accessions in the TSV, or a member
    assigned to two different representatives, are hard errors.
    """
    known = {str(acc).strip().upper() for acc in accessions}
    mapping: Dict[str, str] = {}
    with open(tsv_path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                representative = member = parts[0]
            elif len(parts) >= 2:
                representative, member = parts[0], parts[1]
            else:  # pragma: no cover - defensive, unreachable
                continue
            representative = representative.strip().upper()
            member = member.strip().upper()
            if member not in known or representative not in known:
                raise ValueError(
                    f"{tsv_path}:{line_number}: unknown accession in cluster row "
                    f"{line!r} (expected only data.csv accessions)"
                )
            previous = mapping.get(member)
            if previous is not None and previous != representative:
                raise ValueError(
                    f"{tsv_path}:{line_number}: protein {member} maps to both "
                    f"{previous!r} and {representative!r}"
                )
            mapping[member] = representative

    singletons = [acc for acc in sorted(known) if acc not in mapping]
    for acc in singletons:
        mapping[acc] = acc

    cluster_ids = set(mapping.values())
    return Clustering(
        protein_to_cluster=mapping,
        n_clusters=len(cluster_ids),
        n_singletons=len(singletons),
        tsv_path=tsv_path,
        commands=[list(command) for command in (commands or [])],
    )


# ---------------------------------------------------------------------------
# Step 4: assign whole clusters to splits
# ---------------------------------------------------------------------------


def assign_clusters_to_splits(
    cluster_ids: Collection[str], seed: int = DEFAULT_SEED
) -> Dict[str, List[str]]:
    """Split whole clusters 80/10/10 by cluster count with a seeded shuffle.

    Small cluster counts are handled explicitly so every split is non-empty
    whenever there are at least three clusters.  The split is by *cluster
    count*, so window counts may be heavily skewed (documented in the manifest).
    """
    ids = sorted(str(cid) for cid in cluster_ids)
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
# Step 5: overlap assertions (must pass before any file is written)
# ---------------------------------------------------------------------------


def assert_no_overlap(
    *,
    clusters_by_split: Mapping[str, Sequence[str]],
    proteins_by_split: Mapping[str, Sequence[str]],
    expected_clusters: Collection[str],
    expected_proteins: Collection[str],
    protein_to_cluster: Optional[Mapping[str, str]] = None,
) -> Dict[str, bool]:
    """Assert that no protein or cluster appears in two splits.

    Returns a dict of the individual checks that passed.  Raises
    :class:`OverlapError` on the first violation; callers must not write output
    until this function returns.
    """
    cluster_sets = {split: set(ids) for split, ids in clusters_by_split.items()}
    protein_sets = {split: set(prots) for split, prots in proteins_by_split.items()}

    # Cluster overlap.
    cluster_owners: Dict[str, str] = {}
    for split in SPLITS:
        for cid in cluster_sets.get(split, set()):
            if cid in cluster_owners:
                raise OverlapError(
                    f"cluster {cid!r} appears in both {cluster_owners[cid]!r} and {split!r}"
                )
            cluster_owners[cid] = split

    # Protein overlap.
    protein_owners: Dict[str, str] = {}
    for split in SPLITS:
        for acc in protein_sets.get(split, set()):
            if acc in protein_owners:
                raise OverlapError(
                    f"protein {acc!r} appears in both {protein_owners[acc]!r} and {split!r}"
                )
            protein_owners[acc] = split

    all_clusters = set(expected_clusters)
    all_proteins = set(expected_proteins)
    assigned_clusters = set(cluster_owners)
    assigned_proteins = set(protein_owners)

    missing_clusters = all_clusters - assigned_clusters
    if missing_clusters:
        raise OverlapError(f"{len(missing_clusters)} cluster(s) not assigned to any split")
    extra_clusters = assigned_clusters - all_clusters
    if extra_clusters:
        raise OverlapError(f"{len(extra_clusters)} unknown cluster(s) assigned to a split")
    missing_proteins = all_proteins - assigned_proteins
    if missing_proteins:
        raise OverlapError(f"{len(missing_proteins)} protein(s) not assigned to any split")
    extra_proteins = assigned_proteins - all_proteins
    if extra_proteins:
        raise OverlapError(f"{len(extra_proteins)} unknown protein(s) assigned to a split")

    consistent = True
    if protein_to_cluster is not None:
        for acc, split in protein_owners.items():
            cid = protein_to_cluster.get(acc)
            if cid is None:
                raise OverlapError(f"protein {acc!r} has no cluster_id")
            if cluster_owners.get(cid) != split:
                raise OverlapError(
                    f"protein {acc!r} is in split {split!r} but its cluster "
                    f"{cid!r} is in {cluster_owners.get(cid)!r}"
                )

    return {
        "clusters_disjoint": True,
        "proteins_disjoint": True,
        "all_clusters_assigned": True,
        "all_proteins_assigned": True,
        "cluster_protein_consistency": consistent,
    }


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _site_type_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = Counter(str(value) for value in frame.get("site_type", pd.Series(dtype=str)))
    return {residue: int(counts.get(residue, 0)) for residue in ("S", "T", "Y")}


def _split_stats(frame: pd.DataFrame, cluster_ids: Sequence[str]) -> Dict[str, Any]:
    """Per-split counts.  `n_*` keys are aliases of the literal names."""
    positives = int((frame["label"] == 1).sum())
    negatives = int(len(frame) - positives)
    stats: Dict[str, Any] = {
        "clusters": len(cluster_ids),
        "proteins": int(frame["protein_acc"].nunique()) if len(frame) else 0,
        "windows": int(len(frame)),
        "positives": positives,
        "negatives": negatives,
        "site_types": _site_type_counts(frame),
        # Aliases kept for explicitness / downstream convenience.
        "n_clusters": len(cluster_ids),
        "n_proteins": int(frame["protein_acc"].nunique()) if len(frame) else 0,
        "n_windows": int(len(frame)),
        "n_positives": positives,
        "n_negatives": negatives,
        "site_type_counts": _site_type_counts(frame),
        "cluster_ids": list(cluster_ids),
    }
    return stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_real_split(
    *,
    data_dir: Path,
    mmseqs: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    runner: Optional[Runner] = None,
) -> Dict[str, Any]:
    """Build homology-controlled splits and return the manifest.

    ``runner`` injects MMseqs execution (used by tests).  When omitted, the real
    binary at ``mmseqs`` is executed with a work-dir shell shim on Windows.
    No output file is written unless all no-overlap assertions pass.
    """
    data_dir = Path(data_dir)
    repo_root = find_repo_root()
    binary = Path(mmseqs) if mmseqs is not None else default_mmseqs_path()
    work_dir = Path(work_dir) if work_dir is not None else data_dir / "split_work"
    out_dir = Path(out_dir) if out_dir is not None else data_dir / "split"
    work_dir.mkdir(parents=True, exist_ok=True)

    if runner is None:
        if not Path(binary).is_file():
            raise FileNotFoundError(
                f"MMseqs binary not found: {binary}. "
                "Run tools/fetch_mmseqs.py or pass --mmseqs."
            )
        runner = make_mmseqs_runner(binary, work_dir)

    # Query the version through the same runner (fake runners can supply one).
    version = str(runner(["version"])).strip().splitlines()[-1].strip() or "unknown"

    # Step 1: sequences -> clean FASTA.
    collection = collect_fasta(data_dir, work_dir)
    if collection.missing_accessions:
        raise ValueError(
            f"{len(collection.missing_accessions)} data.csv accession(s) have no cached "
            f"FASTA (e.g. {collection.missing_accessions[:5]}); refusing to cluster an "
            "incomplete protein set."
        )

    # Steps 2-3: cluster and parse.
    clustering = cluster_proteins(
        collection.fasta_path,
        work_dir,
        runner,
        accessions=collection.sequences.keys(),
        min_seq_id=min_seq_id,
        threads=threads,
    )
    protein_to_cluster = clustering.protein_to_cluster
    cluster_sizes = Counter(protein_to_cluster.values())
    n_singleton_clusters = sum(1 for size in cluster_sizes.values() if size == 1)

    # Load the full dataset (all original columns) and attach cluster ids.
    frame = pd.read_csv(data_dir / "data.csv")
    required = {"label", "site_type", "protein_acc"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"data.csv is missing required column(s): {missing_columns}")
    frame["protein_acc"] = frame["protein_acc"].map(lambda value: str(value).strip().upper())
    unknown = sorted(set(frame["protein_acc"]) - set(protein_to_cluster))
    if unknown:
        raise ValueError(f"data.csv has {len(unknown)} protein(s) with no cluster assignment")
    frame[CLUSTER_ID_COLUMN] = frame["protein_acc"].map(protein_to_cluster)

    # Step 4: whole-cluster split.
    split_clusters = assign_clusters_to_splits(
        set(protein_to_cluster.values()), seed=seed
    )
    cluster_to_split = {
        cid: split for split in SPLITS for cid in split_clusters[split]
    }
    proteins_by_split = {
        split: sorted(
            acc for acc, cid in protein_to_cluster.items()
            if cluster_to_split.get(cid) == split
        )
        for split in SPLITS
    }
    frames_by_split = {
        split: frame[frame[CLUSTER_ID_COLUMN].isin(split_clusters[split])].copy()
        for split in SPLITS
    }

    # Additional row-level safeguard: every data.csv row belongs to exactly one
    # split, matching its protein's split.
    protein_split = {
        acc: split for split in SPLITS for acc in proteins_by_split[split]
    }
    row_split = frame["protein_acc"].map(protein_split)
    if row_split.isna().any():
        raise OverlapError("some data.csv rows have no split assignment")
    for split in SPLITS:
        expected = int((row_split == split).sum())
        if expected != len(frames_by_split[split]):
            raise OverlapError(
                f"row/protein split mismatch for {split}: {expected} vs "
                f"{len(frames_by_split[split])}"
            )

    # Step 5: no-overlap assertions BEFORE writing anything.
    checks = assert_no_overlap(
        clusters_by_split=split_clusters,
        proteins_by_split=proteins_by_split,
        expected_clusters=set(protein_to_cluster.values()),
        expected_proteins=set(protein_to_cluster),
        protein_to_cluster=protein_to_cluster,
    )

    # Write split CSVs.  Preserve the original data.csv column order, appending
    # cluster_id at the end.
    columns = [column for column in frame.columns if column != CLUSTER_ID_COLUMN]
    columns.append(CLUSTER_ID_COLUMN)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_files: Dict[str, Any] = {}
    for split in SPLITS:
        csv_path = out_dir / f"{split}.csv"
        frames_by_split[split][columns].to_csv(csv_path, index=False)
        output_files[f"{split}.csv"] = {
            "sha256": sha256_file(csv_path),
            "rows": int(len(frames_by_split[split])),
            "n_proteins": int(frames_by_split[split]["protein_acc"].nunique()),
            "n_clusters": len(split_clusters[split]),
        }

    data_csv_path = data_dir / "data.csv"
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "method": "MMseqs2 whole-cluster homology-controlled split",
        "seed": int(seed),
        "min_seq_id": float(min_seq_id),
        "n_proteins": len(protein_to_cluster),
        "n_clusters": clustering.n_clusters,
        "n_singletons": clustering.n_singletons,
        "n_singleton_clusters": n_singleton_clusters,
        "mmseqs": {
            "binary": _display_path(binary, repo_root),
            "version": version,
            "commands": [
                [binary.name] + list(command) for command in clustering.commands
            ],
            "parameters": {
                "min_seq_id": float(min_seq_id),
                "coverage": DEFAULT_COVERAGE,
                "cov_mode": DEFAULT_COV_MODE,
                "cluster_mode": 0,
                "dbtype": 1,
                "threads": int(threads),
            },
            "cluster_tsv": {
                "path": _display_path(clustering.tsv_path, repo_root),
                "sha256": sha256_file(clustering.tsv_path),
            },
        },
        "inputs": {
            "data_csv": {
                "path": _display_path(data_csv_path, repo_root),
                "sha256": sha256_file(data_csv_path),
                "rows": int(len(frame)),
                "columns": list(frame.columns[: len(frame.columns) - 1]),
            },
            "fasta": {
                "cache_dir": _display_path(data_dir / "fasta_cache", repo_root),
                "cache_files": collection.cache_files,
                "n_csv_accessions": collection.n_csv_accessions,
                "n_accessions_with_fasta": collection.n_with_fasta,
                "missing_accessions": collection.missing_accessions,
                "combined_fasta": {
                    "path": _display_path(collection.fasta_path, repo_root),
                    "sha256": sha256_file(collection.fasta_path),
                },
            },
        },
        "outputs": {
            "directory": _display_path(out_dir, repo_root),
            "files": output_files,
        },
        "splits": {
            split: _split_stats(frames_by_split[split], split_clusters[split])
            for split in SPLITS
        },
        "no_overlap": checks,
        "notes": [
            "Split is by CLUSTER COUNT (80/10/10 with a fixed-seed shuffle of "
            "sorted cluster ids), NOT by window count; per-split sample counts "
            "are therefore skewed. Actual window counts are in "
            "splits.<split>.windows.",
            f"MMseqs2 clustered {collection.n_with_fasta} proteins at "
            f"min_seq_id={min_seq_id}, coverage={DEFAULT_COVERAGE}, "
            f"cov_mode={DEFAULT_COV_MODE}, producing {clustering.n_clusters} "
            f"clusters ({clustering.n_singletons} singletons).",
            "cluster_id is the MMseqs representative accession; proteins absent "
            "from the cluster TSV are singletons whose id is their own accession "
            "(n_singletons). Separately, n_singleton_clusters counts clusters "
            "with exactly one member, including self-hits emitted by MMseqs.",
            "Whole clusters are assigned to exactly one split: no protein and no "
            "cluster appears in two splits. Overlap assertions ran before any "
            "file was written; a violation raises and writes nothing.",
            "No network access: sequences come only from the pinned FASTA cache.",
            "Window-level homology below min_seq_id is not detected; this split "
            "is leakage-free only with respect to the reported MMseqs clusters.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "CLUSTER_ID_COLUMN",
    "DEFAULT_MIN_SEQ_ID",
    "DEFAULT_SEED",
    "DEFAULT_THREADS",
    "SPLITS",
    "Clustering",
    "FastaCollection",
    "OverlapError",
    "assert_no_overlap",
    "assign_clusters_to_splits",
    "build_real_split",
    "cleanup_stale_mmseqs_outputs",
    "cluster_commands",
    "cluster_proteins",
    "collect_fasta",
    "default_mmseqs_path",
    "make_mmseqs_runner",
    "parse_cluster_tsv",
    "sha256_file",
]
