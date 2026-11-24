"""Fast, fully offline tests for the real-data acquisition pipeline.

The spreadsheet, FASTA text and sequences here are tiny SYNTHETIC fixtures made
only of S/T/Y/A/M characters; they are not biological claims and no real data
file is read.  All network calls are replaced by an injected sequence mapping
or a fake requests session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_data import (  # noqa: E402
    CSV_COLUMNS,
    SITE_RESIDUES,
    WINDOW_LENGTH,
    SiteRecord,
    build_real_dataset,
    deduplicate_records,
    fetch_fasta,
    get_15mer,
    is_valid_uniprot_accession,
    load_phosphosite,
    parse_fasta,
    parse_site,
    sample_negative_sites,
    sha256_file,
)

# P00001: S at 2, Y at 8, S at 10, T at 14, S at 16, Y at 20.
SEQ1 = "MSAAAAAYASAAATASAAAYA"
# P00002: S at 8, Y at 14, T at 20.
SEQ2 = "MAAAAAASAAAAAYAAAAAT"
SEQUENCES = {"P00001": SEQ1, "P00002": SEQ2}

SHEET_COLUMNS = [
    "ACC_ID",
    "GENE",
    "PROTEIN",
    "MOD_RSD",
    "RES",
    "ACC_ID_RES",
    "Database",
    "KIN_ACC_ID",
    "KIN_GENE",
    "source",
]


def _row(
    acc: Any,
    mod_rsd: Any,
    res: Any,
    *,
    gene: str = "G1",
    kinase: str = "K00001",
    source: str = "PSP",
) -> Dict[str, Any]:
    return {
        "ACC_ID": acc,
        "GENE": gene,
        "PROTEIN": "PROT",
        "MOD_RSD": mod_rsd,
        "RES": res,
        "ACC_ID_RES": f"{acc}_{mod_rsd}",
        "Database": "PSP",
        "KIN_ACC_ID": kinase,
        "KIN_GENE": "KINGENE",
        "source": source,
    }


def _write_sheet(path: Path, rows: List[Dict[str, Any]]) -> Path:
    pd.DataFrame(rows, columns=SHEET_COLUMNS).to_excel(path, index=False)
    return path


@pytest.fixture
def sheet(tmp_path: Path) -> Path:
    """8 rows: 3 valid sites, 1 positional mismatch, 4 malformed/unusable."""
    return _write_sheet(
        tmp_path / "Phosphosite.xlsx",
        [
            _row("P00001", "Y8-p", "Y8", gene="G1"),
            _row("P00001", "T14-p", "T14", gene="G1"),
            _row("P00001", "S3-p", "S3", gene="G1"),  # FASTA has A at 3
            _row("P00002", "S8-p", "S8", gene="G2", kinase="K00002"),
            _row("P00001", "H5-p", "H5", gene="G1"),  # non-STY annotation
            _row("P00001", "not-a-site", "", gene="G1"),  # malformed
            _row("P00001", "S8-p", "Y8", gene="G1"),  # conflicting columns
            _row("", "Y8-p", "Y8", gene="G1"),  # missing accession
        ],
    )


@pytest.fixture
def manifest(sheet: Path, tmp_path: Path) -> Dict[str, Any]:
    return build_real_dataset(
        phosphosite_path=sheet,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        seed=666,
        ratio=1.0,
        fasta_by_acc=SEQUENCES,
    )


def _data(manifest: Dict[str, Any], tmp_path: Path) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "out" / "data.csv")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_site_mod_rsd_format() -> None:
    assert parse_site("Y508-p", "Y508") == (("Y", 508), "ok")
    assert parse_site("S6-p", "S6") == (("S", 6), "ok")
    assert parse_site("t17-P", "T17") == (("T", 17), "ok")


def test_parse_site_rejects_bad_rows_with_reasons() -> None:
    assert parse_site("H358-p", "H358") == (None, "non_sty_annotation")
    assert parse_site("bogus", "S3") == (None, "malformed_site")
    assert parse_site("", "") == (None, "malformed_site")
    assert parse_site("Y508-p", "S508") == (None, "conflicting_mod_rsd_res")


def test_uniprot_accession_validator() -> None:
    for valid in ("P12345", "A1X283", "O00429-3", "Q9Y6K9", "A0A1B2C3D4"):
        assert is_valid_uniprot_accession(valid), valid
    for invalid in ("NP_001184222", "AAA58698", "P1234", "A1X28", "", "P12345_1"):
        assert not is_valid_uniprot_accession(invalid), invalid


def test_load_phosphosite_counts_discards(sheet: Path) -> None:
    loaded = load_phosphosite(sheet)
    assert loaded.total_rows == 8
    assert len(loaded.sites) == 4
    assert loaded.discarded["non_sty_annotation"] == 1
    assert loaded.discarded["malformed_site"] == 1
    assert loaded.discarded["conflicting_mod_rsd_res"] == 1
    assert loaded.discarded["missing_accession"] == 1
    assert sum(loaded.discarded.values()) == 4
    assert loaded.accession_order == ["P00001", "P00002"]
    assert loaded.gene_by_acc == {"P00001": "G1", "P00002": "G2"}


def test_windows_have_padding_and_fixed_length() -> None:
    assert len(get_15mer(SEQ1, 8)) == WINDOW_LENGTH
    assert get_15mer(SEQ1, 8) == SEQ1[:15]
    assert get_15mer(SEQ1, 2).startswith("______")
    assert get_15mer(SEQ1, 21).endswith("______")
    assert get_15mer(SEQ1, 20).endswith("_")


def test_parse_fasta_handles_uniprot_and_plain_headers() -> None:
    text = (
        ">sp|P00001|TEST_HUMAN Test OS=Homo sapiens OX=9606 GN=G1 PE=1 SV=1\n"
        "MSAAAAAYASA\nAATASAAAYA\n"
        ">P00002 other description\nMAAAAAASAAAAAYAAAAAT\n"
    )
    assert parse_fasta(text) == SEQUENCES


# ---------------------------------------------------------------------------
# Validation, negatives, deduplication
# ---------------------------------------------------------------------------


def test_mismatched_annotations_are_discarded_and_counted(
    sheet: Path, tmp_path: Path, manifest: Dict[str, Any]
) -> None:
    mismatches = manifest["validation"]["mismatches"]
    assert manifest["validation"]["n_mismatches"] == 1
    assert manifest["counts"]["discarded"]["position_mismatch"] == 1
    assert mismatches[0]["reason"] == "position_mismatch"
    assert mismatches[0]["protein_acc"] == "P00001"
    assert mismatches[0]["annotated_residue"] == "S"
    assert mismatches[0]["annotated_position"] == 3
    assert mismatches[0]["observed_residue"] == "A"
    assert manifest["counts"]["validated_positives_before_dedup"] == 3

    data = _data(manifest, tmp_path)
    assert len(data[data["label"] == 1]) == 3


def test_negatives_only_sty_centres_and_disjoint_from_positives(
    manifest: Dict[str, Any], tmp_path: Path
) -> None:
    data = _data(manifest, tmp_path)
    positives = data[data["label"] == 1]
    negatives = data[data["label"] == 0]
    assert len(negatives) == manifest["counts"]["negatives"] == 3
    assert set(negatives["site_type"]).issubset(SITE_RESIDUES)
    assert all(sequence[7] in SITE_RESIDUES for sequence in negatives["sequence"])

    # No (protein, window) pair is both a positive and a negative.
    positive_pairs = set(zip(positives["protein_acc"], positives["sequence"]))
    negative_pairs = set(zip(negatives["protein_acc"], negatives["sequence"]))
    assert positive_pairs.isdisjoint(negative_pairs)
    assert set(negatives["source"]) == {"negative_sampling"}
    assert set(negatives["kinase"]) == {"-"}


def test_negative_sampling_excludes_all_annotated_positions() -> None:
    warnings: List[str] = []
    annotated = {"P00001": {8, 14, 3}, "P00002": {8}}
    per_protein = {"P00001": 2, "P00002": 1}
    sampled = sample_negative_sites(
        SEQUENCES, annotated, per_protein, ratio=1.0, seed=7, warnings=warnings
    )
    assert len(sampled["P00001"]) == 2
    assert len(sampled["P00002"]) == 1
    assert annotated["P00001"].isdisjoint(sampled["P00001"])
    assert annotated["P00002"].isdisjoint(sampled["P00002"])
    assert all(SEQ1[p - 1] in SITE_RESIDUES for p in sampled["P00001"])


def test_windows_are_length_15_with_padding(manifest: Dict[str, Any], tmp_path: Path) -> None:
    data = _data(manifest, tmp_path)
    assert data["sequence"].map(len).eq(WINDOW_LENGTH).all()
    assert data["site_type"].tolist() == [sequence[7] for sequence in data["sequence"]]
    # Terminal padding is verified directly (sampling may not choose a
    # terminal residue for a small seeded run).
    assert get_15mer(SEQ1, 2).startswith("_")
    assert get_15mer(SEQ1, 21).endswith("_")


def test_deduplication_of_identical_and_conflicting_windows() -> None:
    def rec(seq: str, label: int, acc: str, pos: int = 8) -> SiteRecord:
        return SiteRecord(
            sequence=seq,
            label=label,
            site_type=seq[7],
            protein_acc=acc,
            gene="G",
            kinase="-",
            source="PSP" if label else "negative_sampling",
            position=pos,
        )

    kept, stats = deduplicate_records(
        [
            rec("AAAAAAASAAAAAAA", 1, "P1"),
            rec("AAAAAAASAAAAAAA", 1, "P2"),  # identical duplicate
            rec("CCCCCCCTCCCCCCC", 0, "P3"),  # unrelated negative
            rec("GGGGGGGYGGGGGGG", 1, "P4"),  # conflicting pair
            rec("GGGGGGGYGGGGGGG", 0, "P5"),
        ]
    )
    assert stats["identical_windows_dropped"] == 1
    assert stats["conflicting_sequences_dropped"] == 1
    assert stats["conflicting_records_dropped"] == 2
    assert [r.sequence for r in kept] == ["AAAAAAASAAAAAAA", "CCCCCCCTCCCCCCC"]


def test_end_to_end_identical_positive_windows_deduplicated(tmp_path: Path) -> None:
    sheet = _write_sheet(
        tmp_path / "dup.xlsx",
        [
            _row("P00001", "Y8-p", "Y8", gene="G1"),
            _row("P00003", "Y8-p", "Y8", gene="G3"),  # same 15-mer as P00001
        ],
    )
    manifest = build_real_dataset(
        phosphosite_path=sheet,
        output_dir=tmp_path / "dup_out",
        fasta_by_acc={"P00001": SEQ1, "P00003": SEQ1 + "S"},
    )
    assert manifest["counts"]["validated_positives_before_dedup"] == 2
    assert manifest["counts"]["deduplicated"]["identical_windows_dropped"] == 1
    assert manifest["counts"]["positives"] == 1


# ---------------------------------------------------------------------------
# Manifest, hashes, determinism, CLI options
# ---------------------------------------------------------------------------


def test_manifest_required_keys_and_truthful_counts(
    sheet: Path, tmp_path: Path, manifest: Dict[str, Any]
) -> None:
    for key in ("seed", "ratio", "counts", "validation", "fetch", "inputs", "outputs"):
        assert key in manifest
    counts = manifest["counts"]
    assert manifest["seed"] == 666
    assert manifest["ratio"] == 1.0
    assert counts["rows_total"] == 8
    assert sum(counts["discarded"].values()) + counts["validated_positives_before_dedup"] == 8
    assert (
        counts["records_before_dedup"]
        - counts["deduplicated"]["identical_windows_dropped"]
        - counts["deduplicated"]["conflicting_records_dropped"]
        == counts["records_written"]
        == len(_data(manifest, tmp_path))
    )
    assert counts["positives"] + counts["negatives"] == counts["records_written"]
    assert sum(counts["site_type_counts"].values()) == counts["records_written"]
    assert sum(counts["positive_site_type_counts"].values()) == counts["positives"]
    assert sum(counts["negative_site_type_counts"].values()) == counts["negatives"]
    assert manifest["inputs"]["phosphosite"]["sha256"] == sha256_file(sheet)
    assert manifest["outputs"]["files"]["data.csv"]["sha256"] == sha256_file(
        tmp_path / "out" / "data.csv"
    )
    assert manifest["outputs"]["files"]["data.csv"]["columns"] == list(CSV_COLUMNS)
    assert list(pd.read_csv(tmp_path / "out" / "data.csv").columns) == list(CSV_COLUMNS)


def test_determinism_with_fixed_seed(sheet: Path, tmp_path: Path) -> None:
    kwargs = dict(
        phosphosite_path=sheet,
        output_dir=tmp_path / "deterministic",
        seed=42,
        ratio=1.0,
        fasta_by_acc=SEQUENCES,
    )
    first = build_real_dataset(**kwargs)
    first_csv = (tmp_path / "deterministic" / "data.csv").read_bytes()
    first_manifest = (tmp_path / "deterministic" / "manifest.json").read_bytes()
    second = build_real_dataset(**kwargs)
    assert first == second
    assert first_csv == (tmp_path / "deterministic" / "data.csv").read_bytes()
    assert first_manifest == (tmp_path / "deterministic" / "manifest.json").read_bytes()


def test_limit_proteins_is_accounted_for(sheet: Path, tmp_path: Path) -> None:
    manifest = build_real_dataset(
        phosphosite_path=sheet,
        output_dir=tmp_path / "limited",
        cache_dir=tmp_path / "cache",
        limit_proteins=1,
        fasta_by_acc=SEQUENCES,
    )
    assert manifest["counts"]["proteins_selected"] == 1
    assert manifest["counts"]["discarded"]["filtered_by_limit"] == 1
    limited = pd.read_csv(tmp_path / "limited" / "data.csv")
    assert set(limited["protein_acc"]) == {"P00001"}


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200, reason: str = "OK") -> None:
        self.text = text
        self.status_code = status_code
        self.reason = reason

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} {self.reason}")


class _FakeSession:
    def __init__(self, text_by_query: Dict[str, str]) -> None:
        self.text_by_query = text_by_query
        self.calls: List[str] = []

    def get(self, url: str, *, params: Dict[str, str], timeout: float, headers: Any = None):
        self.calls.append(params["query"])
        return _FakeResponse(self.text_by_query.get(params["query"], ">UNKNOWN\nAAAAA"))


def test_fetch_fasta_cache_is_hash_pinned_and_reused(tmp_path: Path) -> None:
    query = "accession:P00001 OR accession:P00002"
    fasta_text = (
        f">sp|P00001|TEST_HUMAN Test\n{SEQ1}\n"
        f">sp|P00002|TEST2_HUMAN Test\n{SEQ2}\n"
    )
    session = _FakeSession({query: fasta_text})
    cache = tmp_path / "cache"

    first = fetch_fasta(["P00002", "P00001"], cache, chunk_size=10, session=session)
    assert first.sequences == SEQUENCES
    assert first.network_calls == 1
    assert not first.failures
    assert len(first.cache_files) == 1
    entry = next(iter(first.cache_files.values()))
    cache_file = Path(entry["path"])
    digest_file = cache_file.with_suffix(".fasta.sha256")
    assert digest_file.is_file()
    assert digest_file.read_text(encoding="utf-8").strip() == sha256_file(cache_file) == entry["sha256"]

    session.calls.clear()
    second = fetch_fasta(["P00001", "P00002"], cache, chunk_size=10, session=session)
    assert second.sequences == first.sequences
    assert second.network_calls == 0
    assert session.calls == []

    cache_file.write_text(">corrupted\nAAAAA\n", encoding="utf-8")
    third = fetch_fasta(["P00001", "P00002"], cache, chunk_size=10, session=session)
    assert third.network_calls == 1
    assert third.sequences == SEQUENCES
    assert cache_file.read_text(encoding="utf-8") == fasta_text


def test_fetch_fasta_records_missing_accessions(tmp_path: Path) -> None:
    session = _FakeSession(
        {"accession:P00001 OR accession:P09999": ">sp|P00001|X\nMSAAAAAYA"}
    )
    result = fetch_fasta(["P00001", "P09999"], tmp_path / "cache", chunk_size=5, session=session)
    assert result.sequences == {"P00001": "MSAAAAAYA"}
    assert result.failures == [
        {
            "accession": "P09999",
            "reason": "absent_from_response",
            "status": 200,
            "error": "accession absent from UniProt FASTA response",
        }
    ]
    assert result.failure_counts == {"absent_from_response": 1}


def test_fetch_fasta_filters_invalid_accessions_before_request(tmp_path: Path) -> None:
    session = _FakeSession({"accession:P12345": ">sp|P12345|X\nMSAAAAAYA"})
    result = fetch_fasta(
        ["P12345", "NP_001184222", "AAA58698"],
        tmp_path / "cache",
        chunk_size=100,
        session=session,
    )
    # Invalid values are never sent to UniProt.
    assert session.calls == ["accession:P12345"]
    assert result.sequences == {"P12345": "MSAAAAAYA"}
    assert [failure["accession"] for failure in result.failures] == [
        "AAA58698",
        "NP_001184222",
    ]
    assert all(
        failure["reason"] == "invalid_accession" and failure["status"] is None
        for failure in result.failures
    )
    assert result.failure_counts == {"invalid_accession": 2}
    assert result.network_calls == 1


class _BadAccessionSession:
    """Returns 400 for any query containing ``bad``, 200 + FASTA otherwise."""

    def __init__(self, bad: str, sequences: Dict[str, str]) -> None:
        self.bad = bad
        self.sequences = sequences
        self.calls: List[str] = []

    def get(self, url: str, *, params: Dict[str, str], timeout: float, headers: Any = None):
        query = params["query"]
        self.calls.append(query)
        accessions = [part.split("accession:", 1)[1] for part in query.split(" OR ")]
        if self.bad in accessions:
            return _FakeResponse("", 400, "Bad Request")
        text = "".join(f">{acc}\n{self.sequences[acc]}\n" for acc in accessions)
        return _FakeResponse(text, 200, "OK")


def test_fetch_fasta_bisects_http_400_and_preserves_successes(tmp_path: Path) -> None:
    sequences = {
        "A1X283": "MSAAAAAYA",
        "O00429": "MSAAAATYA",  # the accession that makes the fake API return 400
        "P12345": "MSAAAASYA",
        "Q9Y6K9": "MSAAAAAYT",
    }
    session = _BadAccessionSession("O00429", sequences)
    result = fetch_fasta(
        list(sequences), tmp_path / "cache", chunk_size=10, session=session
    )
    assert result.sequences == {
        acc: sequence for acc, sequence in sequences.items() if acc != "O00429"
    }
    assert len(result.failures) == 1
    assert result.failures[0] == {
        "accession": "O00429",
        "reason": "http_400",
        "status": 400,
        "error": "Bad Request",
    }
    assert result.failure_counts == {"http_400": 1}
    assert "accession:O00429" in session.calls  # narrowed down to a single accession
    assert result.network_calls == len(session.calls) == 5


class _FlakySession:
    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def get(self, url: str, *, params: Dict[str, str], timeout: float, headers: Any = None):
        self.calls += 1
        if self.calls <= self.fail_times:
            return _FakeResponse("", 503, "Service Unavailable")
        return _FakeResponse(">P12345\nMSAAAAAYA\n", 200, "OK")


def test_fetch_fasta_retries_5xx_only_within_bound(tmp_path: Path) -> None:
    transient = _FlakySession(fail_times=2)
    result = fetch_fasta(
        ["P12345"], tmp_path / "cache", session=transient, max_retries=2
    )
    assert result.sequences == {"P12345": "MSAAAAAYA"}
    assert result.network_calls == transient.calls == 3
    assert not result.failures

    persistent = _FlakySession(fail_times=10)
    with pytest.raises(requests.HTTPError):
        fetch_fasta(
            ["P12345"], tmp_path / "cache2", session=persistent, max_retries=1
        )
    assert persistent.calls == 2
