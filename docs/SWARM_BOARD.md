# Swarm board — DeepPhosLM review validation

Format: `**[n | YYYY-MM-DD] KIND → to — author (subject)**`. Authoritative drift register lives in
Drive: `.../DeepPhosLM Protein/REVIEW_CRITICISM_REGISTER.md`.

**[1 | 2026-11-24] RESET → all — yoga/opencode (Paper markdown moved to Drive; code+data retained in repo)**
MoC: PAPER.md, RESEARCH_SOURCES.md now in the Drive `research` shared folder; generated output dirs
(FIGS_*, MODELS_*, data, __pycache__) gitignored. Repo holds code + data only.

**[2 | 2026-11-24] REPORT → all — yoga/opencode (Review forum mined; 39 criticisms extracted)**
Sources: R1 (7), R2 (16), R3 (12), R4 (4), editor note, four checklists. Mapped to 17 D-IDs in the
authoritative register with reasoning and file:line evidence from the code-availability repo.

**[3 | 2026-11-24] REQUEST → critic — yoga/opencode (Per-D Critic gate on D-001..D-017)**
Gate order per D: VERIFY (mother, tools/verify_review_register.py) → CRITIC (read-only) → COMMIT
(one commit per D). No worker builds or commits.

**[4 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-001 task definition / kinase-substrate contradiction - UPHELD)**

**[5 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-002 dataset composition, negatives, class ratio - UPHELD)**

**[6 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-003 window-level split / data leakage - UPHELD)**
