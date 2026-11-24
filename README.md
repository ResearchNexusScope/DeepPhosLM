# Kinase Phosphosite 3D Detection and Structure Classifier

This repository implements a deep learning pipeline to predict kinase-specific phosphorylation sites and classify them into 25 target kinase families. It incorporates both raw sequence motifs (15-mers) and 3D structural embeddings extracted using a Siamese Neural Network.

---

## 📂 Repository Structure

*   `finalised_application.ipynb`: The main end-to-end Jupyter Notebook, split into clean, modular blocks with markdown explanations.
*   `visualization_engine.py`: A modular visualization module containing the plotting functions for the advanced dashboards.
*   `multiclassifier.py`: The PyTorch Multi-Layer Perceptron (MLP) architecture containing support for Seq-Coord, Seq-Only, and Coord-Only models.
*   `generate_dataset.py`: The preprocessing script that reads raw datasets and compiles the CSV matrices.
*   `hotfixes/`: Contains utility scripts used for modularization and layout adjustments.
*   `archive/`: Contains duplicates, legacy copies, and backup scripts.
*   `data/` (Gitignored): Holds generated data files ready for modeling.
*   `FIGS_multiclass/` (Gitignored): Directory where output analysis figures and metrics tables are saved.

---

## 🛠️ Prerequisites & Installation

To run this pipeline, install the required libraries:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn umap-learn shap lifelines
```

---

## 🧪 Reproducing the real experiment

The steps below are the authoritative, **hash-pinned real experiment** (real PhosphoSite + UniProt data, whole-cluster homology split). They are distinct from the legacy notebook pipeline described in the rest of this README. Run every command from the repository root, in order, with the exact versions in [`requirements.lock.txt`](requirements.lock.txt) (observed on Python 3.14.7):

```bash
python -m pip install -r requirements.lock.txt
```

### 1. Fetch the pinned MMseqs2 binary

```bash
python tools/fetch_mmseqs.py
```

Downloads the pinned MMseqs2 win64 release (`20-ae60e`, `mmseqs-win64.zip`) and verifies its SHA-256 against `56d79e866cbd0080f17fb51693c750f210ee315f96774b6c1b2eb89924b52540`. The extracted binary lives under `tools/bin/mmseqs/` and is **gitignored — never commit it**. The script is idempotent: it rechecks the hash on every run and only re-downloads on mismatch (`--force` forces a fresh download).

### 2. Build the real dataset

```bash
python tools/prepare_real_data.py --seed 666 --ratio 1.0
```

Parses `Phosphosite.xlsx`, fetches UniProt FASTA sequences, validates each annotated phosphosite against the actual residue in its sequence, and writes the hash-pinned `data/real/data.csv` and `data/real/manifest.json`. `--seed 666` makes negative sampling reproducible; `--ratio 1.0` gives one validated negative per validated positive per protein.

### 3. Make the homology-controlled split

```bash
python tools/make_real_split.py --min-seq-id 0.3 --seed 666 --threads 8
```

Clusters proteins with MMseqs2 at 30% sequence identity and assigns **whole clusters** to train/val/test (80/10/10), so no homologous sequence crosses a split boundary. Writes `data/real/split/{train,val,test}.csv` plus a hash-pinned `manifest.json`. `--threads 8` fixes MMseqs threading for determinism.

### 4. Train: a held-out run and a separate CV run

Held-out evaluation (5 seeds, cross-validation disabled):

```bash
python tools/run_real_experiments.py --seeds 5 --folds 0 --epochs 8 --batch-size 1024 --ablation both
```

Cross-validation on the train split only (1 seed, 5 folds):

```bash
python tools/run_real_experiments.py --seeds 1 --folds 5 --epochs 6 --batch-size 1024 --ablation both --out-dir results/real_cv
```

**Why two runs?** In `tools/run_real_experiments.py` the CV block (`if args.folds > 0: cross_validate(...)`) is nested inside `for seed in seeds: for ablation in ablations:`. The CLI therefore *couples CV to every seed and ablation*: `--folds > 0` runs a full k-fold CV for each one, and cost scales as `seeds × ablations × folds`. Requesting 5 seeds **and** 5 folds at once would compute five largely redundant CV summaries. We split it instead: `--seeds 5 --folds 0` yields five independent held-out seeds in the default `results/real/`, while `--seeds 1 --folds 5` yields one seed x 2 ablations = 2 five-fold CV summaries in `results/real_cv/` (the `--cv-dir` the validator expects), because CV runs once per seed *and* per ablation. Model selection uses validation AUROC only — test labels are never used for selection — and CV uses whole `cluster_id` groups drawn from the train split only.

### 5. Validate (bootstrap, then canonical gate)

`results/` is gitignored, so on a clean checkout `claims.json` does not exist yet — and it cannot be written until a report exists. Break that circular dependency with an explicit **bootstrap** run first.

**5a. Bootstrap — not full validation: no claims checked yet.** This only recomputes the report from the saved artifacts:

```bash
python tools/verify_experiment.py --results-dir results/real --cv-dir results/real_cv --split-dir data/real/split --out results/real/validation_report.json
```

Without `--claims`, the report is tagged `NO_CLAIMS_SUPPLIED` and a loud WARNING is printed: artifact checks run, but no published/manuscript number is checked. Do not treat this run as a pass.

**5b. Derive the claims from that report.** **`claims.json` is DERIVED via `tools/make_claims.py` — it is never hand-written.**

```bash
python tools/make_claims.py --report results/real/validation_report.json --out results/real/claims.json
```

`make_claims.py` reads every value out of `validation_report.json` and rounds it for publication; no published number is ever typed in by hand.

**5c. Canonical validation gate.** Only now rerun verification with the derived claims — this is the single canonical command:

```bash
python tools/verify_experiment.py --claims results/real/claims.json --results-dir results/real --cv-dir results/real_cv --split-dir data/real/split
```

It recomputes every number from the saved artifacts (bootstrap CIs, classical baselines and the pytest suite) and checks it against the claims before reporting GREEN.

### 6. Figures and per-number manuscript check

```bash
python tools/plot_real_results.py
python tools/validate_manuscript_numbers.py --markdown <REAL_RESULTS.md>
```

`plot_real_results.py` renders all figures only from saved artifacts into `figures/real/`. `validate_manuscript_numbers.py` cross-checks every numeric claim in the manuscript markdown against the recomputed report and the hash-pinned data/split manifests — pass the real path to `REAL_RESULTS.md` (see step 7).

### 7. Generated vs. versioned files

All generated data (`data/`), results (`results/`), figures (`figures/`) and the downloaded MMseqs binary (`tools/bin/`) are **gitignored and regenerated** by the commands above. Only source code is versioned in this repo. The paper markdown (`REAL_RESULTS.md`, `PAPER.md`, `RESEARCH_SOURCES.md`) lives in the Drive research folder, not in the repository.

---

## 🚀 How to Run the Entire Pipeline

You can run this pipeline in two ways: **interactively** (recommended for exploring visualizations) or **headlessly** via the command line.

### Step 1: Preprocess the Input Datasets
Before running the modeling code, compile the raw spreadsheets into machine-learning ready formats. Run the data preprocessing script from your terminal:
```bash
python generate_dataset.py
```
*What this does:* It parses the raw excel files (`Modified_TOP_20_FOR_STRUCTURE.xlsx` and `kinase_phosphorylation_sites.xlsx`), extracts 15-mer motifs (7 upstream and 7 downstream residues centered around the phosphosite), creates the stratified train/test folds, and generates the BLOSUM62 evolutionary distance matrices. All outputs are saved to the `data/` folder.

---

### Step 2: Choose Your Execution Method

#### Method A: Run Interactively (Jupyter Notebook) - Recommended
1.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
2.  **Open the application**: Select and open **`finalised_application.ipynb`**.
3.  **Run All Cells**:
    *   Click on **Cell** in the top menu -> **Run All**, or click the **Restart & Run All** (fast-forward) button in the toolbar.
    *   *Note:* The first cell automatically installs `shap` and `lifelines` if they are missing.
4.  **Explore the Dashboards**: Scroll to the bottom of the notebook to view the interactive plots and metrics comparison tables rendered inline.

#### Method B: Run Headlessly (Terminal Script)
If you prefer running the pipeline as a background script or in a headless terminal, execute the python version:
```bash
python finalised_application.py
```
*What this does:* It trains the Siamese Network, maps UMAP coordinates, trains the 5-fold cross-validation variants, and compiles all the performance metrics.

---

### Step 3: Verify the Outputs
Once the pipeline finishes, check the output directories to verify the results:
*   **Trained Weights**: Check the `MODELS_multiclass/experiment_1/` directory for the saved PyTorch model weights (folds 0–4).
*   **Saved Visualizations**: Check the `FIGS_multiclass/` directory for all the exported 300-DPI analysis plots (CD diagrams, radar charts, SHAP positional maps, Kaplan-Meier survival curves, etc.) and the metrics comparison spreadsheet (`metrics_comparison.csv`).

---

## 📊 Advanced Visualization Suite

The pipeline automatically compiles and exports publication-grade analysis dashboards to the `FIGS_multiclass/` directory:

### 1. Model Performance & Comparison (ML Metrics)
*   **Performance Comparison Table (`metrics_comparison.csv`)**: A converged metrics summary sheet showing the mean $\pm$ standard deviation across the 5 validation folds (Accuracy, Macro/Micro Precision, Macro/Micro Recall, Macro/Micro F1, and AUC).
*   **Ablation Study Bar Plot (`ablation_study.png`)**: Compares the metrics of the Combined model vs. the Sequence-Only and Coordinate-Only baselines, quantifying the boost provided by 3D structure.
*   **Radar Chart (`radar_chart.png`)**: Spider plot mapping Accuracy, Precision, Recall, Specificity, and F1 across the top 5 most common kinase families (*AKT1*, *CDK1*, *MAPK1*, *SRC*, *AURKA*).
*   **Critical Difference (CD) Diagram (`critical_difference.png`)**: Ranks the model variants across the folds and performs Nemenyi post-hoc statistical significance testing.

### 2. Biological & Evolutionary Alignment
*   **Evolutionary Correlation Heatmaps (`evolutionary_alignment.png`)**: Scatter plot and side-by-side matrices matching the inter-family BLOSUM62 phylogenetic distance to the model's confusion rates (verifying if prediction errors align with evolutionary similarity).
*   **Grouped Superfamily Confusion Matrix (`grouped_confusion.png`)**: Heatmap grouping specific kinase families into their major superfamilies (AGC, CAMK, CMGC, TK, TKL, STE, Atypical) for high legibility.

### 3. Model Explainability (SHAP Analysis)
*   **Feature Group Summary (`shap_summary.png`)**: Summarizes the total feature contribution (Sequence vs. 3D Coordinates).
*   **Positional SHAP Map (`shap_positional.png`)**: Maps SHAP feature attribution across each of the 15 positions in the motif window, showing which flanking residues drive the classification.

### 4. Downstream Clinical Translation
*   **Kaplan-Meier Curves (`survival_km.png`)**: Simulates a patient cohort and plots patient survival curves comparing High vs. Low predicted phosphorylation activity of key biomarkers (*EGFR*, *AKT1*, *SRC*), with Log-Rank test $p$-values.
*   **Cox Proportional Hazards Forest Plot (`survival_cox.png`)**: Fits a multivariate Cox model showing Hazard Ratios (HR) and their 95% Confidence Intervals for individual predicted kinases.
