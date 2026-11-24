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

**[7 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-004 no external SOTA baselines - UPHELD)**

**[8 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-005 metrics and class imbalance - PARTIALLY UPHELD)**

**[9 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-006 methods clarity / pipeline reconstruction - UPHELD)**

**[10 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-007 ProtBERT role / PLM isolation claim - UPHELD)**

**[11 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-008 3D structural data never fed to model - UPHELD)**

**[12 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-009 XAI method missing - PARTIALLY UPHELD)**

**[13 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-010 feature-ablation evidence - PARTIALLY UPHELD)**

**[14 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-011 UMAP / logo / network overinterpretation - UPHELD)**

**[15 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-012 Kaplan-Meier survival analysis - UPHELD)**

**[16 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-013 inconsistent numbers / sample counts - UPHELD)**

**[17 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-014 overstated translational claims - UPHELD)**

**[18 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-015 novelty and scientific focus - UPHELD)**

**[19 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-016 writing, figures, terminology, numbering - UPHELD)**

**[20 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (D-017 reproducibility, provenance, generalisation - UPHELD)**

**[21 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (F-001 honest dataset + leakage-free cluster split; VERIFY 9 tests green, CRITIC PASS)**

**[22 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (F-002 honest binary phosphosite model; VERIFY 14 tests green, CRITIC PASS)**

**[23 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (F-003 honest metrics + per-S/T/Y + bootstrap CI; VERIFY 22 tests green, CRITIC PASS)**

**[24 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (F-004 same-split LR/RF baselines with honest no-SOTA scope; VERIFY 32 tests green, CRITIC PASS)**

**[25 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (F-005 figures from results; simulated survival removed; legacy path/duplicate defects fixed; VERIFY 36 tests green, CRITIC PASS)**

**[26 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (F-006 revised manuscript in Drive; all unsupported claims removed, numbers regenerated; CRITIC PASS, final VERIFY 36 tests green)**

**[27 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (E-001 real PhosphoSite+UniProt dataset: 233,329 windows from 3,113 proteins with residue validation; VERIFY 56 tests green, CRITIC PASS)**

**[28 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (E-002 real MMseqs2 30% whole-cluster split: 2,118 clusters, 188k/23.6k/21.3k windows, zero overlap; VERIFY 63 tests green, CRITIC PASS)**

**[29 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (E-003 real 5-seed dual-ablation training + 5-fold whole-cluster CV; VERIFY 68 tests green, CRITIC PASS after artifact re-audit)**

**[30 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (E-004 independent validator: 10 archives recomputed, row integrity vs test.csv, baselines, ECE, bootstrap, CV; VERIFY 78 tests green, CRITIC PASS after tamper-check fix)**

**[31 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (E-005 real figures + REAL_RESULTS.md + 141 derived claims; VERIFY 82 tests green, validation with claims GREEN, 153 PASS/0 FAIL doc check, CRITIC PASS after artifact re-audit)**

**[32 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (X-005 pinned environment + executable reproduction chain; VERIFY 87 tests green, CRITIC PASS after circular-gate fix)**

**[33 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (X-002 split-threshold sensitivity 0.2/0.3/0.4 all OK, stable ROC-AUC; VERIFY 104 tests green, CRITIC PASS)**

**[34 | 2026-11-24] DONE_VERIFIED -> all - yoga/opencode (X-003 pos_weight imbalance handling; 10 real runs, ranking unchanged; VERIFY 104 tests green, CRITIC PASS)**
