#!/usr/bin/env python3
"""Mother-only verifier for the DeepPhosLM review-criticism drift register.

Run by the orchestrator alone (never by a worker). Checks that the authoritative
register in the Drive research folder is complete, evidence-backed and Critic-gated,
and that the repo contains only code/data (no paper markdown, generated dirs ignored).

Exit 0 = GREEN, 1 = RED. Every failure is printed with a precise reason.
"""
from pathlib import Path
import re
import sys

DRIVE = Path(r"C:\Users\sugee\gdrive\My Drive\Research\pavithra\04 DeepPhosLM Protein")
REPO = Path(__file__).resolve().parents[1]
REGISTER = DRIVE / "REVIEW_CRITICISM_REGISTER.md"

SOURCE_IDS = (
    [f"R1-{i}" for i in range(1, 8)]
    + [f"R2-{i}" for i in range(1, 17)]
    + [f"R3-{i}" for i in range(1, 13)]
    + [f"R4-{i}" for i in range(1, 5)]
    + ["EDITOR-1"]
    + [f"R{i}-CHECKLIST" for i in range(1, 5)]
)
D_IDS = [f"D-{i:03d}" for i in range(1, 18)]
VERDICTS = {"UPHELD", "PARTIALLY UPHELD", "NOT UPHELD"}


def main() -> int:
    failures: list[str] = []
    if not REGISTER.is_file():
        print(f"FAIL register missing: {REGISTER}")
        return 1
    text = REGISTER.read_text(encoding="utf-8")

    # 1. Every extracted source criticism must be present as an ID label.
    for sid in SOURCE_IDS:
        if f"**{sid}**" not in text and f"| {sid} " not in text:
            failures.append(f"source item missing: {sid}")

    # 2. Every D must have a register row, a reasoning section and an evidence line.
    for did in D_IDS:
        if not re.search(rf"\|\s*{did}\s*\|", text):
            failures.append(f"{did}: no register row")
            continue
        section = re.search(rf"### {did} — .*?(?=\n### D-|\n---\n|\Z)", text, re.S)
        if not section:
            failures.append(f"{did}: no per-D reasoning section")
            continue
        body = section.group(0)
        if "**Reasoning.**" not in body:
            failures.append(f"{did}: missing reasoning")
        m = re.search(r"\*\*Verdict:\s*([A-Z ]+)\.\*\*", body)
        if not m or m.group(1).strip() not in VERDICTS:
            failures.append(f"{did}: missing/invalid verdict")
        if "**Evidence:**" not in body:
            failures.append(f"{did}: missing evidence")
        if not re.search(r"[A-Za-z_]+\.py:\d+", body):
            failures.append(f"{did}: no file:line evidence in repo")

    # 3. Critic gate: each D must be PASS in section D.
    gate = text.split("## D. Critic gate results", 1)[-1]
    for did in D_IDS:
        row = re.search(rf"\|\s*{did}\s*\|\s*([A-Z]+)\s*\|", gate)
        if not row or row.group(1) != "PASS":
            failures.append(f"{did}: Critic gate not PASS")

    # 4. No placeholders / fabricated-completion markers anywhere.
    for marker in ("TODO", "TBD", "PLACEHOLDER", "PENDING_CRITIC", "| PENDING |"):
        if marker in text:
            failures.append(f"placeholder marker present: {marker}")

    # 5. Repo hygiene: paper markdown lives in Drive, not in the repo.
    for md in ("PAPER.md", "RESEARCH_SOURCES.md", "REVIEW_CRITICISM_REGISTER.md"):
        if (REPO / md).exists():
            failures.append(f"paper markdown must not be in repo: {md}")
    if (REPO / "docs" / "RESEARCH_SOURCES.md").exists():
        failures.append("docs/RESEARCH_SOURCES.md must not be in repo")

    # 6. Generated output folders must be gitignored.
    gi = (REPO / ".gitignore").read_text(encoding="utf-8")
    for folder in ("FIGS_multiclass/", "FIGS_siam/", "MODELS_multiclass/", "MODELS_siam/",
                   "data/", "__pycache__/"):
        if folder.rstrip("/") not in gi:
            failures.append(f"not gitignored: {folder}")

    if failures:
        print(f"VERIFY RED — {len(failures)} failure(s):")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"VERIFY GREEN — {len(SOURCE_IDS)} source items, {len(D_IDS)} D-IDs, "
          f"all Critic gates PASS, repo hygiene OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
