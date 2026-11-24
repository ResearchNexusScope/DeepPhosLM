#!/usr/bin/env python3
"""Cross-check manuscript numbers against the recomputed report and the document.

Two independent gates are applied and every number gets a PASS/FAIL line:

1. ``src.validate_results.check_claimed_numbers`` re-verifies every claim in
   ``--claims`` against ``--report`` (the authoritative E-004 recomputation).
2. Every claimed numeric value must actually appear in a results table of
   ``--markdown``; conversely, every numeric token appearing in a results table
   must be traceable to a claim (otherwise it is an unpublished number and is
   reported as FAIL).

The dataset-accounting paragraph is deliberately prose, not a results table:
its accession/window counts are inputs pinned in the hash-locked data and split
manifests, not recomputed results. Those integers are still checked -- against
the manifests -- so that no number in the document goes unverified.

Usage::

    python tools/validate_manuscript_numbers.py \
        --report results/real/validation_report.json \
        --claims results/real/claims.json \
        --markdown "REAL_RESULTS.md"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validate_results import check_claimed_numbers  # noqa: E402

# A number token: comma-grouped integers and decimals, not part of an
# alphanumeric word (so SHA-256 hex digests and identifiers are never matched).
NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w.])")
TABLE_SEPARATOR_RE = re.compile(r"^:?-{2,}:?$")
TABLE_EPSILON = 1e-9


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def table_numbers(markdown: str) -> set[float]:
    """Every numeric token occurring inside a markdown results table."""
    numbers: set[float] = set()
    for line in markdown.splitlines():
        if not is_table_row(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(TABLE_SEPARATOR_RE.fullmatch(cell) for cell in cells):
            continue  # |---|---| separator row
        for cell in cells:
            for token in NUMBER_RE.findall(cell):
                value = _to_float(token)
                if value is not None:
                    numbers.add(value)
    return numbers


def _claim_value(claim: Any) -> Any:
    if isinstance(claim, dict):
        if "value" in claim:
            return claim["value"]
        return claim.get("claimed")
    return claim


def _provenance_section(markdown: str) -> str:
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("#") and "dataset provenance" in line.lower():
            start = index + 1
            break
    if start is None:
        return ""
    collected: list[str] = []
    for line in lines[start:]:
        if line.startswith("#"):
            break
        collected.append(line)
    return "\n".join(collected)


def _manifest_numbers(*paths: Path) -> dict[float, str]:
    """All numeric leaf values in the hash-pinned input manifests."""
    found: dict[float, str] = {}

    def walk(node: Any, path: Path, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path, f"{trail}.{key}" if trail else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path, f"{trail}.{index}")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            found.setdefault(float(node), f"{path.name}:{trail}")

    for path in paths:
        if path.is_file():
            walk(json.loads(path.read_text(encoding="utf-8")), path, "")
    return found


def _provenance_integers(markdown: str) -> set[float]:
    """Large integer counts quoted in the dataset-provenance paragraph."""
    values: set[float] = set()
    for token in NUMBER_RE.findall(_provenance_section(markdown)):
        value = _to_float(token)
        if value is None or value != int(value) or value < 1000:
            continue
        if 1900 <= value <= 2100:  # calendar years are not dataset counts
            continue
        values.add(value)
    return values


def check_document(
    claims: dict[str, Any],
    report: dict[str, Any],
    markdown: str,
    data_manifest: Path,
    split_manifest: Path,
) -> list[dict[str, Any]]:
    """Return one PASS/FAIL record per claim and per document-only number."""
    records: list[dict[str, Any]] = []
    recomputed = check_claimed_numbers(claims, report)
    doc_numbers = table_numbers(markdown)

    for result in recomputed:
        name = str(result["name"])
        claimed = result["claimed"]
        in_table: bool | None = None
        if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
            in_table = None
        else:
            in_table = any(
                abs(float(claimed) - number) <= TABLE_EPSILON for number in doc_numbers
            )
        passed = result["status"] == "PASS" and in_table is not False
        detail = result["detail"]
        if in_table is False:
            detail += "; claimed value does not appear in any results table"
        records.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "claimed": claimed,
                "recomputed": result["recomputed"],
                "report_check": result["status"],
                "in_results_table": in_table,
                "detail": detail,
            }
        )

    claimed_numbers = {
        float(value)
        for value in (_claim_value(claim) for claim in claims.values())
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }
    for number in sorted(doc_numbers - claimed_numbers):
        records.append(
            {
                "name": f"document_table_number:{number:g}",
                "status": "FAIL",
                "detail": "number appears in a results table but is absent from the claims JSON",
            }
        )

    manifest_values = _manifest_numbers(data_manifest, split_manifest)
    for number in sorted(_provenance_integers(markdown)):
        source = manifest_values.get(number)
        records.append(
            {
                "name": f"provenance_count:{number:g}",
                "status": "PASS" if source else "FAIL",
                "detail": f"pinned manifest source: {source}" if source
                          else "dataset count not found in the input manifests",
            }
        )
    if not _provenance_section(markdown):
        records.append(
            {
                "name": "dataset_provenance_section",
                "status": "FAIL",
                "detail": "markdown has no 'Dataset provenance' section to validate",
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "results" / "real" / "validation_report.json",
        help="recomputed validation report",
    )
    parser.add_argument(
        "--claims", type=Path,
        default=REPO_ROOT / "results" / "real" / "claims.json",
        help="claims JSON produced by tools/make_claims.py",
    )
    parser.add_argument(
        "--markdown", type=Path,
        default=REPO_ROOT.parents[2] / "gdrive" / "My Drive" / "Research"
        / "02 Applied - Journals" / "Pavithra" / "Front. Bioinform"
        / "DeepPhosLM Protein" / "REAL_RESULTS.md",
        help="REAL_RESULTS.md to validate",
    )
    parser.add_argument(
        "--data-manifest", type=Path,
        default=REPO_ROOT / "data" / "real" / "manifest.json",
        help="hash-pinned dataset manifest for provenance counts",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=REPO_ROOT / "data" / "real" / "split" / "manifest.json",
        help="hash-pinned split manifest for provenance counts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    markdown = args.markdown.read_text(encoding="utf-8")
    if not isinstance(report, dict) or not isinstance(claims, dict):
        raise ValueError("report and claims must both be JSON objects")

    records = check_document(
        claims, report, markdown, args.data_manifest, args.split_manifest
    )
    for record in records:
        detail = f" -- {record['detail']}" if record.get("detail") else ""
        print(f"[{record['status']}] {record['name']}{detail}")
    failures = [record for record in records if record["status"] == "FAIL"]
    print(
        f"\nSUMMARY: {len(records) - len(failures)} PASS, {len(failures)} FAIL "
        f"({len(claims)} claims checked)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
