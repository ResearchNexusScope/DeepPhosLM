#!/usr/bin/env python3
"""Mother-only verifier for experiment E-004 ("validate every result").

ONE gate per reported result. Workers must NOT run this script: the mother
session runs it once, after all result artifacts are final. It recomputes every
number with :mod:`src.validate_results`, optionally checks manuscript claims
against the recomputed report, runs the single canonical pytest pass over
``tests/``, writes the full report to ``results/real/validation_report.json``
(or ``--out``), prints that report in full (never truncated), and exits
non-zero if any integrity check, any claim, or pytest fails.

Canonical mother invocation (from the repository root)::

    python tools/verify_experiment.py --claims <claims.json>

The ``--claims`` argument is REQUIRED for a publication-number validation.
Without it the verifier still recomputes artifact-derived numbers and runs
pytest, but NO published/manuscript number is checked; the report is marked
``claims.status = NO_CLAIMS_SUPPLIED`` and a loud WARNING is printed. A run
without claims must never be presented as full validation of the paper.

Claim files are JSON objects mapping a manuscript claim name to either a bare
value or ``{"path": "<dotted report path>", "value": <number>,
"tolerance": <abs number or {"abs": ..., "rel": ...}>}``. Obtain paths with
``src.validate_results.available_claim_paths(report)``; never retype numbers.

``--skip-baselines`` and ``--skip-pytest`` exist only so the nested self-tests in
``tests/test_validate_results.py`` can exercise the CLI without recursion. The
mother gate must leave both off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validate_results import (  # noqa: E402
    DEFAULT_ABS_TOL,
    DEFAULT_BASELINE_SEED,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    DEFAULT_REL_TOL,
    _file_provenance,
    check_claimed_numbers,
    validate_all,
)


def _json_default(value: Any) -> Any:
    """JSON fallback for stray numpy scalars (the report should have none)."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authoritative validation gate for the real experiment (mother only)."
    )
    parser.add_argument(
        "--results-dir",
        default="real",
        help="Held-out results directory (default: results/real). Absolute paths accepted.",
    )
    parser.add_argument(
        "--cv-dir",
        default="real_cv",
        help="Cross-validation results directory (default: results/real_cv).",
    )
    parser.add_argument(
        "--split-dir",
        default="data/real/split",
        help="Directory with train.csv/val.csv/test.csv/manifest.json.",
    )
    parser.add_argument(
        "--claims",
        default=None,
        help=(
            "REQUIRED for full validation: JSON file of manuscript claims to check "
            "against the recomputed report (canonical: --claims <claims.json>). "
            "Without it no published numbers are checked."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Report destination (default: <results-dir>/validation_report.json).",
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--baseline-seed", type=int, default=DEFAULT_BASELINE_SEED)
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="NESTED SELF-TESTS ONLY: do not refit classical baselines.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="NESTED SELF-TESTS ONLY: do not run the pytest suite.",
    )
    return parser


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    def resolve(raw: str, repo_relative: Path) -> Path:
        path = Path(raw)
        if path.is_dir():
            return path
        candidate = repo_relative / path
        if candidate.is_dir():
            return candidate
        return path

    results = resolve(args.results_dir, REPO_ROOT / "results")
    cv = resolve(args.cv_dir, REPO_ROOT / "results")
    split = resolve(args.split_dir, REPO_ROOT)
    return results, cv, split


def run_pytest() -> Dict[str, Any]:
    """Run exactly one pytest pass over tests/ and preserve full output."""
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "command": command,
        "cwd": str(REPO_ROOT),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _failures(report: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    for check in report.get("checks", []):
        if not check.get("passed", False):
            failures.append(f"integrity check {check.get('name')}: {check.get('detail')}")
    for claim in report.get("claims", {}).get("checks", []):
        if claim.get("status") != "PASS":
            failures.append(f"claim {claim.get('name')}: {claim.get('detail')}")
    suite = report.get("test_suite")
    if suite is not None and suite.get("passed") is False:
        failures.append(
            f"pytest failed (exit {suite.get('returncode')}): {suite.get('stderr', '')[-2000:]}"
        )
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir, cv_dir, split_dir = _resolve_dirs(args)
    out_path = Path(args.out) if args.out else results_dir / "validation_report.json"

    print(f"VERIFY_EXPERIMENT repo={REPO_ROOT}")
    print(f"  results_dir={results_dir}")
    print(f"  cv_dir={cv_dir}")
    print(f"  split_dir={split_dir}")
    print(f"  out={out_path}")

    if not args.claims:
        print("\n" + "!" * 78)
        print("WARNING: NO --claims SUPPLIED — no published/manuscript numbers are checked.")
        print("Canonical mother invocation: python tools/verify_experiment.py --claims <claims.json>")
        print("This run validates artifacts only; it is NOT full validation of the paper.")
        print("!" * 78 + "\n")

    report: Dict[str, Any] = validate_all(
        str(results_dir),
        str(cv_dir),
        str(split_dir),
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
        baseline_seed=args.baseline_seed,
        include_baselines=not args.skip_baselines,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )

    if args.claims:
        claims_path = Path(args.claims)
        if not claims_path.is_file():
            raise FileNotFoundError(f"claims file not found: {claims_path}")
        claims = json.loads(claims_path.read_text(encoding="utf-8"))
        if not isinstance(claims, dict):
            raise ValueError("claims JSON must be an object mapping names to values/specs")
        report["claims"] = {
            "source": str(claims_path),
            "provenance": _file_provenance(claims_path, "manuscript_claims"),
            "checks": check_claimed_numbers(claims, report),
        }
    else:
        report["claims"] = {
            "source": None,
            "checks": [],
            "status": "NO_CLAIMS_SUPPLIED",
            "note": "no claims supplied; no published/manuscript numbers were checked",
        }

    if args.skip_pytest:
        report["test_suite"] = {
            "passed": None,
            "skipped": True,
            "note": "nested self-test only; the mother gate always runs pytest",
        }
    else:
        report["test_suite"] = run_pytest()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    # Full, untruncated report. Large bootstrap vectors are intentionally kept.
    print("\n--- FULL VALIDATION REPORT ---")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print("\n--- test suite output (full) ---")
    suite = report["test_suite"]
    print(json.dumps(suite, indent=2, ensure_ascii=False, default=_json_default))

    failures = _failures(report)
    if failures:
        print(f"\nVERIFY RED — {len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nVERIFY GREEN — all checks, claims and pytest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
