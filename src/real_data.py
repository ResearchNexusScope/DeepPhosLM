"""Real-data acquisition, validation and dataset building for phosphosites.

This module is deliberately separate from :mod:`src.data_pipeline`, which builds
a small *demo* dataset from two legacy spreadsheets.  Here the ground truth is
the public ``Phosphosite.xlsx`` table (135,063 rows / 3,335 substrate accessions
in the shipped repository) and the protein sequences are fetched from UniProt.

Pipeline (all rules are explicit and reported in the manifest)
--------------------------------------------------------------
1. :func:`load_phosphosite` reads ``ACC_ID, MOD_RSD, RES`` (plus optional
   ``GENE, KIN_ACC_ID, KIN_GENE, source``).  A site is accepted only when the
   ``MOD_RSD`` form (``Y508-p``) and the ``RES`` form (``Y508``) agree on a
   S/T/Y residue and a positive 1-based position.  Every rejected row is
   counted under a precise reason; nothing is silently dropped.
2. :func:`fetch_fasta` validates accession syntax first (RefSeq/EMBL-style
   identifiers are excluded and recorded as ``invalid_accession`` failures),
   then downloads UniProt FASTA in chunks, caching each chunk on disk with a
   SHA-256 sidecar.  A cached chunk whose digest no longer matches is
   re-downloaded.  A 4xx response bisects the chunk down to single accessions
   so one bad value cannot abort the run; only network/5xx errors are retried
   (boundedly) and then raised.  Accessions UniProt does not return are
   recorded as per-accession fetch failures.
3. Every annotated position is **validated against the fetched sequence**: the
   residue at that 1-based index must equal the annotated residue.  Real data
   frequently fails this check (LitePhospho reported ~13% coordinate
   mismatches); failures are discarded and listed in ``validation.mismatches``.
   Out-of-range positions and missing sequences are discarded too.
4. Validated sites become positives with 15-residue windows (seven residues on
   each side, ``_`` padding at the termini).
5. Negatives are sampled from *other* S/T/Y residues of the same proteins, at
   a seeded per-protein ratio (default 1:1).  All annotated positions -- even
   ones that failed validation -- are excluded from the negative candidate
   pool, so a disputed site is never silently turned into a negative.
6. Identical windows are deduplicated globally: exact repeats with the same
   label keep one copy; a window seen with conflicting labels drops *all* of
   its copies.  Counts are recorded in ``counts.deduplicated``.

The emitted CSV has exactly the columns ``sequence,label,site_type,protein_acc,
gene,kinase,source``.  ``manifest.json`` pins the SHA-256 of the input sheet,
of every cached FASTA chunk and of the emitted CSV, and records all counts.

No network access happens at import time; :func:`fetch_fasta` is called only by
:func:`build_real_dataset` when no injected sequence mapping is supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests

PHOSPHOSITE_FILENAME = "Phosphosite.xlsx"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

WINDOW_HALF_WIDTH = 7
WINDOW_LENGTH = 2 * WINDOW_HALF_WIDTH + 1
WINDOW_PADDING = "_"
SITE_RESIDUES = frozenset("STY")
SITE_TYPES: Tuple[str, str, str] = ("S", "T", "Y")

DEFAULT_SEED = 666
DEFAULT_RATIO = 1.0
DEFAULT_CHUNK_SIZE = 100
DEFAULT_TIMEOUT = 60.0

CSV_COLUMNS: Tuple[str, ...] = (
    "sequence",
    "label",
    "site_type",
    "protein_acc",
    "gene",
    "kinase",
    "source",
)

REQUIRED_COLUMNS: Tuple[str, ...] = ("ACC_ID", "MOD_RSD", "RES")
OPTIONAL_COLUMNS: Tuple[str, ...] = ("GENE", "KIN_ACC_ID", "KIN_GENE", "source")

# Generic patterns: a non-S/T/Y residue (e.g. ``H358-p``) is a *different*
# discard reason from a syntactically malformed value.
_MOD_PATTERN = re.compile(r"^([A-Z])(\d+)-p$", re.IGNORECASE)
_RES_PATTERN = re.compile(r"^([A-Z])(\d+)$", re.IGNORECASE)

# Strict UniProt KB accession syntax, with an optional isoform suffix.  RefSeq
# (``NP_001184222``) and EMBL (``AAA58698``) identifiers are NOT UniProt
# accessions: UniProt's ``accession:`` query answers HTTP 400 for them, and a
# single such value poisons an entire multi-accession chunk.
_UNIPROT_ACCESSION_PATTERN = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?"
)

DEFAULT_MAX_RETRIES = 2

# Reasons counted in ``manifest['counts']['discarded']``.
DISCARD_REASONS: Tuple[str, ...] = (
    "malformed_site",
    "conflicting_mod_rsd_res",
    "non_sty_annotation",
    "missing_accession",
    "filtered_by_limit",
    "missing_fasta",
    "out_of_range",
    "position_mismatch",
)


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteAnnotation:
    """One parsed row of the phosphosite spreadsheet."""

    protein_acc: str
    gene: str
    residue: str
    position: int
    kinase: str
    source: str
    mod_rsd: str
    res: str


@dataclass
class LoadResult:
    """Outcome of :func:`load_phosphosite`."""

    sites: List[SiteAnnotation] = field(default_factory=list)
    discarded: "Counter[str]" = field(default_factory=Counter)
    total_rows: int = 0
    gene_by_acc: Dict[str, str] = field(default_factory=dict)
    accession_order: List[str] = field(default_factory=list)


@dataclass
class FastaFetchResult:
    """Outcome of :func:`fetch_fasta`."""

    sequences: Dict[str, str] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    cache_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    network_calls: int = 0


@dataclass(frozen=True)
class SiteRecord:
    """A labelled 15-mer with provenance, before CSV serialisation."""

    sequence: str
    label: int
    site_type: str
    protein_acc: str
    gene: str
    kinase: str
    source: str
    position: int

    def as_row(self) -> Dict[str, object]:
        """Return exactly the public CSV columns (no internal position)."""
        return {
            "sequence": self.sequence,
            "label": self.label,
            "site_type": self.site_type,
            "protein_acc": self.protein_acc,
            "gene": self.gene,
            "kinase": self.kinase,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Paths and hashing
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path, root: Path) -> str:
    """Prefer a repo-relative path for readable manifests."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Site parsing and windows
# ---------------------------------------------------------------------------


def _text(value: object) -> str:
    """Normalise a spreadsheet cell to a stripped string (NaN -> '')."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "none", "<na>"} else text


def parse_mod_rsd(value: object) -> Optional[Tuple[str, int]]:
    """Parse ``Y508-p`` into ``('Y', 508)``; return ``None`` if malformed."""
    match = _MOD_PATTERN.match(_text(value))
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))


def parse_res(value: object) -> Optional[Tuple[str, int]]:
    """Parse ``Y508`` into ``('Y', 508)``; return ``None`` if malformed."""
    match = _RES_PATTERN.match(_text(value))
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))


def parse_site(mod_rsd: object, res: object) -> Tuple[Optional[Tuple[str, int]], str]:
    """Cross-check ``MOD_RSD`` and ``RES`` and return ``(site, reason)``.

    ``reason`` is ``'ok'`` on success, otherwise one of
    ``'malformed_site'``, ``'conflicting_mod_rsd_res'`` or
    ``'non_sty_annotation'``.  Position validity against a FASTA is checked
    later, during validation.
    """
    mod = parse_mod_rsd(mod_rsd)
    res_parsed = parse_res(res)
    if mod is None or res_parsed is None:
        return None, "malformed_site"
    if mod != res_parsed:
        return None, "conflicting_mod_rsd_res"
    if mod[0] not in SITE_RESIDUES:
        return None, "non_sty_annotation"
    if mod[1] < 1:
        return None, "malformed_site"
    return mod, "ok"


def get_15mer(seq: str, pos: int) -> str:
    """Return the 15-residue window centred on 1-based ``pos``.

    Missing residues at either terminus are padded with ``_`` so that the
    result is always exactly :data:`WINDOW_LENGTH` characters long.
    """
    if pos < 1:
        raise ValueError(f"position must be >= 1, got {pos}")
    idx = pos - 1
    start = idx - WINDOW_HALF_WIDTH
    end = idx + WINDOW_HALF_WIDTH + 1
    left_pad = WINDOW_PADDING * max(0, -start)
    right_pad = WINDOW_PADDING * max(0, end - len(seq))
    return left_pad + seq[max(0, start) : min(end, len(seq))] + right_pad


# ---------------------------------------------------------------------------
# Spreadsheet loading
# ---------------------------------------------------------------------------


def load_phosphosite(path: Path) -> LoadResult:
    """Read and parse the phosphosite spreadsheet.

    Accepts ``.xlsx`` (and ``.csv`` for lightweight tests).  Missing optional
    columns are filled with empty strings.  Rejected rows are counted in
    ``result.discarded``; accepted sites are returned in spreadsheet order.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Phosphosite spreadsheet not found: {path}")

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    result = LoadResult(total_rows=len(frame))
    seen_accessions: set[str] = set()
    for row in frame.itertuples(index=False):
        acc = _text(getattr(row, "ACC_ID"))
        mod_rsd = _text(getattr(row, "MOD_RSD"))
        res = _text(getattr(row, "RES"))
        gene = _text(getattr(row, "GENE"))
        # KIN_GENE is the *kinase* gene, not the substrate gene, so it is
        # deliberately not used as a fallback here.
        site, reason = parse_site(mod_rsd, res)
        if reason != "ok" or site is None:
            result.discarded[reason] += 1
            continue
        if not acc or acc == "-":
            result.discarded["missing_accession"] += 1
            continue
        if acc not in seen_accessions:
            seen_accessions.add(acc)
            result.accession_order.append(acc)
            if gene:
                result.gene_by_acc[acc] = gene
        result.sites.append(
            SiteAnnotation(
                protein_acc=acc,
                gene=gene or result.gene_by_acc.get(acc, ""),
                residue=site[0],
                position=site[1],
                kinase=_text(getattr(row, "KIN_ACC_ID")),
                source=_text(getattr(row, "source")),
                mod_rsd=mod_rsd,
                res=res,
            )
        )
    result.accession_order.sort()
    return result


def select_proteins(
    sites: Sequence[SiteAnnotation], limit_proteins: Optional[int]
) -> Tuple[List[SiteAnnotation], "Counter[str]", List[str]]:
    """Keep the first ``limit_proteins`` accessions in sorted order.

    Returns ``(kept_sites, discard_counts, selected_accessions)``.  When
    ``limit_proteins`` is ``None`` all sites pass through unchanged.
    """
    if limit_proteins is None:
        return list(sites), Counter(), sorted({s.protein_acc for s in sites})
    if limit_proteins < 1:
        raise ValueError(f"limit_proteins must be >= 1, got {limit_proteins}")
    accessions = sorted({s.protein_acc for s in sites})[:limit_proteins]
    chosen = set(accessions)
    kept = [s for s in sites if s.protein_acc in chosen]
    discarded = Counter(
        {"filtered_by_limit": len(sites) - len(kept)}
    )
    return kept, discarded, accessions


# ---------------------------------------------------------------------------
# UniProt FASTA fetching with a hash-pinned on-disk cache
# ---------------------------------------------------------------------------


def is_valid_uniprot_accession(accession: object) -> bool:
    """Return True only for a syntactically valid UniProt KB accession.

    Accepts the canonical 6/10-character accessions (``P12345``, ``A1X283``)
    and isoform suffixes (``O00429-3``).  Rejects RefSeq/EMBL-style ids such as
    ``NP_001184222`` and ``AAA58698``, which make UniProt answer HTTP 400 even
    when they are part of a larger ``OR`` query.
    """
    text = str(accession).strip().upper()
    return _UNIPROT_ACCESSION_PATTERN.fullmatch(text) is not None


def _fetch_failure(
    accession: str, reason: str, error: str, status: Optional[int] = None
) -> Dict[str, Any]:
    """Build one uniform, manifest-ready fetch failure record."""
    return {"accession": accession, "reason": reason, "status": status, "error": error}


def parse_fasta(text: str) -> Dict[str, str]:
    """Parse FASTA text into ``{accession: uppercase_sequence}``.

    Handles both UniProt headers (``>sp|P12931|SRC_HUMAN ...`` -> ``P12931``)
    and plain headers (``>P12931``).  Malformed/empty records are ignored.
    """
    sequences: Dict[str, str] = {}
    accession: Optional[str] = None
    chunks: List[str] = []

    def flush() -> None:
        if accession is None:
            return
        sequence = "".join(chunks).upper()
        if sequence:
            sequences.setdefault(accession, sequence)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            flush()
            accession = None
            chunks = []
            header = line[1:].strip()
            if not header:
                continue
            token = header.split()[0]
            if "|" in token:
                fields = token.split("|")
                token = fields[1] if len(fields) > 1 and fields[1] else fields[0]
            if token:
                accession = token.upper()
        elif accession is not None and line:
            chunks.append(line)
    flush()
    return sequences


def chunk_cache_key(accessions: Sequence[str]) -> str:
    """Return a stable cache key for an ordered chunk of accessions."""
    return sha256_text("\x00".join(accessions))[:16]


def _read_cache(cache_file: Path, digest_file: Path) -> Optional[Tuple[str, str]]:
    """Return ``(fasta_text, sha256)`` if a cache entry verifies, else None."""
    if not cache_file.is_file() or not digest_file.is_file():
        return None
    try:
        expected = digest_file.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    actual = sha256_file(cache_file)
    if expected != actual:
        return None
    try:
        return cache_file.read_text(encoding="utf-8"), actual
    except OSError:
        return None


def _write_cache(cache_file: Path, digest_file: Path, text: str) -> str:
    """Write a FASTA chunk and pin the SHA-256 of the exact bytes written.

    Binary writes avoid platform newline translation, so the sidecar digest
    always equals :func:`sha256_file` of the cached chunk on every OS.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    cache_file.write_bytes(payload)
    digest_file.write_text(digest + "\n", encoding="utf-8")
    return digest


def fetch_fasta(
    accessions: Sequence[str],
    cache_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
    session: Optional[Any] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> FastaFetchResult:
    """Fetch UniProt FASTA for ``accessions`` with a pinned, resilient cache.

    Accessions are validated and filtered before any request is made.  Invalid
    identifiers (RefSeq/EMBL-style) never reach UniProt and are recorded as
    ``invalid_accession`` failures.  A 4xx response to a chunked query bisects
    the chunk down to single accessions so one bad value cannot abort the run.
    Only network exceptions and 5xx responses are retried, up to
    ``max_retries`` times, then re-raised.

    ``session`` may be any object exposing ``get(url, params=..., timeout=...)``
    returning a response with ``status_code``, ``raise_for_status()`` and
    ``text``; a fake makes this function testable without network access.

    The cache stores one ``<key>.fasta`` file plus one ``<key>.fasta.sha256``
    sidecar per (sub)chunk.  A missing/mismatched sidecar forces a re-fetch and
    the new digest is always pinned.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    unique = sorted(
        set(str(acc).strip().upper() for acc in accessions if str(acc).strip())
    )
    valid = [acc for acc in unique if is_valid_uniprot_accession(acc)]
    invalid = [acc for acc in unique if not is_valid_uniprot_accession(acc)]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    http = session if session is not None else requests
    result = FastaFetchResult()
    for acc in invalid:
        result.failures.append(
            _fetch_failure(
                acc,
                "invalid_accession",
                "not a UniProt accession; excluded before requesting FASTA",
            )
        )
    if invalid:
        result.warnings.append(
            f"Excluded {len(invalid)} invalid accession(s) before requesting FASTA"
        )

    def paths_for(chunk: Sequence[str]) -> Tuple[str, Path, Path]:
        key = chunk_cache_key(chunk)
        return key, cache_dir / f"chunk_{key}.fasta", cache_dir / f"chunk_{key}.fasta.sha256"

    def consume(chunk: Sequence[str], text: str, digest: str, status: int) -> None:
        """Cache metadata + parse a successful FASTA response."""
        key, cache_file, _ = paths_for(chunk)
        result.cache_files[key] = {
            "path": str(cache_file.resolve()),
            "sha256": digest,
            "accessions": list(chunk),
        }
        parsed = parse_fasta(text)
        for acc in chunk:
            sequence = parsed.get(acc)
            if sequence is None:
                result.failures.append(
                    _fetch_failure(
                        acc,
                        "absent_from_response",
                        "accession absent from UniProt FASTA response",
                        status,
                    )
                )
            else:
                result.sequences[acc] = sequence

    def request(chunk: Sequence[str]) -> Tuple[Any, int]:
        """One HTTP call with bounded retries for network/5xx errors only."""
        query = " OR ".join(f"accession:{acc}" for acc in chunk)
        for attempt in range(max_retries + 1):
            try:
                response = http.get(
                    UNIPROT_STREAM_URL,
                    params={"format": "fasta", "query": query},
                    headers={"Accept": "text/plain"},
                    timeout=timeout,
                )
                result.network_calls += 1
                status = int(getattr(response, "status_code", 200))
                if status >= 500:
                    if attempt == max_retries:
                        response.raise_for_status()  # final, bounded failure
                    continue
                if 400 <= status:
                    return response, status  # caller bisects / records it
                response.raise_for_status()
                return response, status
            except requests.RequestException:
                if attempt == max_retries:
                    raise
        raise RuntimeError("unreachable retry state")  # pragma: no cover

    def fetch_chunk(chunk: Sequence[str]) -> None:
        _, cache_file, digest_file = paths_for(chunk)
        cached = _read_cache(cache_file, digest_file)
        if cached is not None:
            text, digest = cached
            consume(chunk, text, digest, 200)
            return
        if cache_file.is_file():
            result.warnings.append(
                f"Cache entry {cache_file.name} missing/invalid sha256 sidecar; re-fetching"
            )

        response, status = request(chunk)
        if 400 <= status < 500:
            # A single malformed value can poison a whole OR query: narrow the
            # request down to individual accessions instead of aborting.
            if len(chunk) == 1:
                reason = str(getattr(response, "reason", "") or "UniProt rejected this query")
                result.failures.append(
                    _fetch_failure(chunk[0], f"http_{status}", reason, status)
                )
                return
            middle = len(chunk) // 2
            fetch_chunk(chunk[:middle])
            fetch_chunk(chunk[middle:])
            return

        text = str(response.text)
        if not text.strip():
            for acc in chunk:
                result.failures.append(
                    _fetch_failure(
                        acc, "empty_fasta_response", "empty FASTA response for chunk", status
                    )
                )
            return
        digest = _write_cache(cache_file, digest_file, text)
        consume(chunk, text, digest, status)

    for start in range(0, len(valid), chunk_size):
        fetch_chunk(valid[start : start + chunk_size])

    result.failure_counts = dict(
        Counter(failure.get("reason", "unknown") for failure in result.failures)
    )
    return result


# ---------------------------------------------------------------------------
# Validation, windows, negatives and deduplication
# ---------------------------------------------------------------------------


def validate_annotations(
    sites: Sequence[SiteAnnotation],
    sequences: Mapping[str, str],
) -> Tuple[List[SiteRecord], List[Dict[str, Any]], "Counter[str]"]:
    """Check every annotated position against its FASTA and build positives.

    Returns ``(positives, mismatches, discarded_counts)``.  A site is kept only
    when the sequence exists, the 1-based position is in range and the observed
    residue equals the annotated S/T/Y residue.  All failures are recorded in
    ``mismatches`` (with a ``reason``) and counted in ``discarded_counts``.
    """
    positives: List[SiteRecord] = []
    mismatches: List[Dict[str, Any]] = []
    discarded: "Counter[str]" = Counter()

    for site in sites:
        sequence = sequences.get(site.protein_acc)
        base = {
            "protein_acc": site.protein_acc,
            "gene": site.gene,
            "mod_rsd": site.mod_rsd,
            "res": site.res,
            "annotated_residue": site.residue,
            "annotated_position": site.position,
        }
        if sequence is None:
            discarded["missing_fasta"] += 1
            mismatches.append({**base, "reason": "missing_fasta", "observed_residue": None})
            continue
        if not 1 <= site.position <= len(sequence):
            discarded["out_of_range"] += 1
            mismatches.append(
                {
                    **base,
                    "reason": "out_of_range",
                    "observed_residue": None,
                    "fasta_length": len(sequence),
                }
            )
            continue
        observed = sequence[site.position - 1].upper()
        if observed != site.residue:
            discarded["position_mismatch"] += 1
            mismatches.append(
                {
                    **base,
                    "reason": "position_mismatch",
                    "observed_residue": observed,
                    "fasta_length": len(sequence),
                }
            )
            continue
        positives.append(
            SiteRecord(
                sequence=get_15mer(sequence, site.position),
                label=1,
                site_type=observed,
                protein_acc=site.protein_acc,
                gene=site.gene,
                kinase=site.kinase,
                source=site.source,
                position=site.position,
            )
        )
    return positives, mismatches, discarded


def sample_negative_sites(
    sequences: Mapping[str, str],
    annotated_positions: Mapping[str, set[int]],
    positives_per_protein: Mapping[str, int],
    ratio: float,
    seed: int,
    warnings: List[str],
) -> Dict[str, List[int]]:
    """Seeded per-protein sampling of unannotated S/T/Y positions.

    ``wanted = round(ratio * n_validated_positives_for_protein)`` (half-up),
    clamped to the candidate pool; clamping is appended to ``warnings``.
    """
    if ratio < 0:
        raise ValueError(f"ratio must be non-negative, got {ratio}")
    rng = random.Random(seed)
    sampled: Dict[str, List[int]] = {}
    for acc in sorted(sequences):
        sequence = sequences[acc]
        excluded = annotated_positions.get(acc, set())
        candidates = [
            pos
            for pos, residue in enumerate(sequence, start=1)
            if residue in SITE_RESIDUES and pos not in excluded
        ]
        wanted = int(math.floor(ratio * positives_per_protein.get(acc, 0) + 0.5))
        if wanted > len(candidates):
            warnings.append(
                f"{acc}: requested {wanted} negatives, only {len(candidates)} "
                f"candidates available; using {len(candidates)}"
            )
            wanted = len(candidates)
        sampled[acc] = sorted(rng.sample(candidates, wanted)) if wanted else []
    return sampled


def deduplicate_records(
    records: Sequence[SiteRecord],
) -> Tuple[List[SiteRecord], Dict[str, int]]:
    """Deduplicate by the 15-mer sequence, globally across labels.

    * identical windows with the same label/site type: keep the first, count
      the rest as ``identical_windows_dropped``;
    * windows appearing with conflicting labels (or site types): drop *all*
      copies and count both the sequences and the records removed.
    """
    groups: Dict[str, List[SiteRecord]] = defaultdict(list)
    for record in records:
        groups[record.sequence].append(record)

    kept: List[SiteRecord] = []
    stats = Counter()
    for sequence in sorted(groups):
        group = groups[sequence]
        if len(group) == 1:
            kept.append(group[0])
            continue
        labels = {r.label for r in group}
        types = {r.site_type for r in group}
        if len(labels) == 1 and len(types) == 1:
            kept.append(group[0])
            stats["identical_windows_dropped"] += len(group) - 1
        else:
            stats["conflicting_sequences_dropped"] += 1
            stats["conflicting_records_dropped"] += len(group)

    kept.sort(key=lambda r: (r.protein_acc, r.position, r.label))
    return kept, dict(stats)


def _site_type_counts(records: Sequence[SiteRecord]) -> Dict[str, int]:
    counts = Counter(r.site_type for r in records)
    return {residue: int(counts.get(residue, 0)) for residue in SITE_TYPES}


# ---------------------------------------------------------------------------
# Orchestration and manifest
# ---------------------------------------------------------------------------


def build_real_dataset(
    *,
    phosphosite_path: Path,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    ratio: float = DEFAULT_RATIO,
    limit_proteins: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
    fasta_by_acc: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build the real dataset under ``output_dir`` and return the manifest.

    ``fasta_by_acc`` injects sequences directly (used by unit tests); when it
    is omitted this function performs UniProt requests through
    :func:`fetch_fasta`.  Injecting does not relax any validation rule.
    """
    if ratio < 0:
        raise ValueError(f"ratio must be non-negative, got {ratio}")
    repo_root = find_repo_root()
    phosphosite_path = Path(phosphosite_path)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else repo_root / "data" / "real" / "fasta_cache"

    loaded = load_phosphosite(phosphosite_path)
    sites, limit_discards, selected = select_proteins(loaded.sites, limit_proteins)

    # All annotated positions (including invalid ones) are withheld from the
    # negative pool: a disputed site must never become a labelled negative.
    annotated_positions: Dict[str, set[int]] = defaultdict(set)
    for site in sites:
        annotated_positions[site.protein_acc].add(site.position)

    if fasta_by_acc is not None:
        sequences = {
            str(acc).strip().upper(): str(seq).strip().upper()
            for acc, seq in fasta_by_acc.items()
        }
        fetch_info: Dict[str, Any] = {
            "n_accessions": len(selected),
            "cache_dir": _display_path(cache_dir, repo_root),
            "chunk_size": chunk_size,
            "failures": [],
            "failure_counts": {},
            "n_invalid_accessions": 0,
            "cache_files": {},
            "network_calls": 0,
            "injected_sequences": True,
        }
        fetch_warnings: List[str] = []
    else:
        fetched = fetch_fasta(
            selected, cache_dir, chunk_size=chunk_size, timeout=timeout
        )
        sequences = fetched.sequences
        fetch_info = {
            "n_accessions": len(selected),
            "cache_dir": _display_path(cache_dir, repo_root),
            "chunk_size": chunk_size,
            "failures": fetched.failures,
            "failure_counts": fetched.failure_counts,
            "n_invalid_accessions": fetched.failure_counts.get("invalid_accession", 0),
            "cache_files": fetched.cache_files,
            "network_calls": fetched.network_calls,
            "injected_sequences": False,
        }
        fetch_warnings = list(fetched.warnings)

    positives, mismatches, validation_discards = validate_annotations(sites, sequences)
    positives_per_protein = Counter(r.protein_acc for r in positives)

    warnings: List[str] = list(fetch_warnings)
    negative_positions = sample_negative_sites(
        sequences, annotated_positions, positives_per_protein, ratio, seed, warnings
    )
    negatives: List[SiteRecord] = []
    for acc in sorted(negative_positions):
        sequence = sequences[acc]
        for position in negative_positions[acc]:
            negatives.append(
                SiteRecord(
                    sequence=get_15mer(sequence, position),
                    label=0,
                    site_type=sequence[position - 1].upper(),
                    protein_acc=acc,
                    gene=loaded.gene_by_acc.get(acc, ""),
                    kinase="-",
                    source="negative_sampling",
                    position=position,
                )
            )

    records_before_dedup = len(positives) + len(negatives)
    records, dedup_counts = deduplicate_records(positives + negatives)

    discarded = Counter()
    for reason in DISCARD_REASONS:
        discarded[reason] = int(
            loaded.discarded.get(reason, 0)
            + limit_discards.get(reason, 0)
            + validation_discards.get(reason, 0)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "data.csv"
    pd.DataFrame([r.as_row() for r in records], columns=list(CSV_COLUMNS)).to_csv(
        csv_path, index=False
    )

    n_positive = sum(1 for r in records if r.label == 1)
    n_negative = len(records) - n_positive
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "ratio": float(ratio),
        "limit_proteins": limit_proteins,
        "chunk_size": int(chunk_size),
        "window": {
            "half_width": WINDOW_HALF_WIDTH,
            "length": WINDOW_LENGTH,
            "padding": WINDOW_PADDING,
        },
        "inputs": {
            "phosphosite": {
                "path": _display_path(phosphosite_path, repo_root),
                "sha256": sha256_file(phosphosite_path),
            }
        },
        "outputs": {
            "directory": _display_path(output_dir, repo_root),
            "manifest": "manifest.json",
            "files": {
                "data.csv": {
                    "rows": len(records),
                    "sha256": sha256_file(csv_path),
                    "columns": list(CSV_COLUMNS),
                }
            },
        },
        "counts": {
            "rows_total": loaded.total_rows,
            "proteins_in_sheet": len(loaded.accession_order),
            "proteins_selected": len(selected),
            "proteins_with_fasta": sum(1 for acc in selected if acc in sequences),
            "validated_positives_before_dedup": len(positives),
            "negatives_sampled": len(negatives),
            "records_before_dedup": records_before_dedup,
            "positives": n_positive,
            "negatives": n_negative,
            "records_written": len(records),
            "site_type_counts": _site_type_counts(records),
            "positive_site_type_counts": _site_type_counts(
                [r for r in records if r.label == 1]
            ),
            "negative_site_type_counts": _site_type_counts(
                [r for r in records if r.label == 0]
            ),
            "discarded": {reason: discarded[reason] for reason in DISCARD_REASONS},
            "deduplicated": {
                "identical_windows_dropped": int(
                    dedup_counts.get("identical_windows_dropped", 0)
                ),
                "conflicting_sequences_dropped": int(
                    dedup_counts.get("conflicting_sequences_dropped", 0)
                ),
                "conflicting_records_dropped": int(
                    dedup_counts.get("conflicting_records_dropped", 0)
                ),
            },
        },
        "validation": {
            "rule": "FASTA residue at the annotated 1-based position must equal the annotated residue",
            "n_mismatches": len(mismatches),
            "mismatches": mismatches,
        },
        "fetch": fetch_info,
        "warnings": warnings,
        "notes": [
            "Sites are parsed from Phosphosite.xlsx MOD_RSD (e.g. Y508-p) and "
            "cross-checked against RES (e.g. Y508); disagreeing or malformed "
            "rows are discarded with an explicit count.",
            "Every annotated position is validated against the UniProt FASTA; "
            "coordinate mismatches (a known problem in LitePhospho) are "
            "discarded and listed in validation.mismatches.",
            "Negatives are unannotated S/T/Y residues of the same proteins, "
            "sampled with one seeded RNG over accessions in sorted order; "
            "all annotated positions, including failed ones, are excluded.",
            "For negatives kinase is '-' and source is 'negative_sampling' "
            "because no kinase/substrate relation is known for them.",
            "Deduplication is by the 15-mer string: repeats with equal labels "
            "keep one copy, conflicting labels drop all copies.",
            "UniProt accession syntax is validated before any request; invalid "
            "identifiers (e.g. RefSeq NP_/EMBL-style ids) are never sent and "
            "appear in fetch.failures with reason 'invalid_accession'. Chunk "
            "4xx responses bisect to single accessions; only 5xx/network errors "
            "are retried and then raised.",
            "The SHA-256 digests of the input spreadsheet, every cached FASTA "
            "chunk and the emitted CSV are pinned in this manifest.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
