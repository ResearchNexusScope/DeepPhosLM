# DeepPhosLM — Paper Details (extracted from the Drive research folder)

> This file is a factual extraction of the manuscript and review materials held in the
> AmazeSwarm `research` shared folder (single Drive home:
> `~/gdrive/My Drive/Research`, exposed locally as `~/Research`).
> The original PDFs stay in Drive; only this text extraction is versioned here.
> See [docs/RESEARCH_SOURCES.md](docs/RESEARCH_SOURCES.md) for the full source index.

## Identification

| Field | Value |
| --- | --- |
| Title | DeepPhosLM: Protein Language Model-Driven Framework for Robust Protein Phosphorylation Site Prediction using Explainable AI |
| Manuscript ID | 1964362 |
| Journal | Frontiers in Bioinformatics — Genomic Analysis |
| Article type | Original Research Article |
| Received | 11 Aug 2026 |
| Authors | Pavithra S; Sugeeth Jayaraj S A; Ronel Jacob Mathew; Patricia G |
| Affiliation | Vellore Institute of Technology (VIT), Chennai, India |
| Corresponding author | Dr. S. Pavithra — pavithra.sekar@vit.ac.in |
| Handling editor | Amit Kumar Yadav |
| Review status at capture | In review. Associate Editor recommended rejection (28 Sep 2026); no final decision recorded in the supplied packet. |

Keywords: deep learning, FASTA sequence, phosphorylation, protein kinase, protein structure prediction.

## Scope statement

Identify phosphorylation sites in kinase protein sequences; analyze protein sequences and predicted 3D
structures; use BiLSTM and Transformer models for accurate prediction; detect Ser/Thr and Tyrosine
kinase-specific phosphorylation motifs; support kinase–substrate interaction and pathway analysis.

## Abstract (condensed verbatim)

Phosphorylation regulates cell function; protein kinases transfer phosphate groups from ATP to target
amino-acid sites. Experimental characterization is laborious and expensive, so the authors propose an
end-to-end deep-learning framework that takes FASTA sequences, derives latent features, predicts 3D
protein structure and discovers motifs. A hybrid BiLSTM + Transformer model combines sequential and
structural properties of kinases. Kinase-specific motif patterns used:

- **Ser/Thr kinase:** `[ST][^P][^P][RK]`
- **Tyrosine kinase:** `[ST]..Y`

The claimed contribution is improved prediction through integration of structural and motif-based
analysis, supporting kinase pathway analysis and computational biology.

## Data and preprocessing

- Kinase sequences in **FASTA** format from **UniProtKB**.
- **483 experimentally verified phosphorylation events** from **PhosphoSitePlus**.
- External Kaggle dataset referenced in Code Availability: `top-20-kinase-structures-and-site-prediction-data`.
- Kinases studied (25 families) span cell cycle (AURKA, CDK1, LATS1, CHEK1), apoptosis/survival
  (AKT1, ATM, ATR, CHK2), signalling (MAPK1, MAPK3, SRC, ABL1), growth factors
  (EGFR, FLT3, PDGFRB, TGFBR1), immune response (PAK1, LYN, PRKCA), metabolism/stress
  (GSK3B, DYRK1, MTOR, CAMKK1), and motility/cell–cell interaction (AXL, HIPK1).
- Cleaning: drop incomplete sequences; keep only valid phosphorylation positions.
- Windows: **15-residue motif** (±7 residues around the site) with `_` terminal padding.
- Split: **80% train / 20% test** at the *window* level; 10% of train used for validation.
- Labels: one-hot encoded kinase families; motifs tokenised with **1-mer** encoding.
- BioPython `PairwiseAligner` global alignment used for conservation analysis/visualisation only —
  explicitly **not** used for modeling.

## Model / methodology

1. **Siamese neural network** — embedding + BiLSTM layers learn **100-dimensional** kinase-sequence
   representations, trained with **contrastive loss** to cluster similar-family motifs and separate
   dissimilar ones.
2. **ProtBERT** residue embeddings — **1024-dimensional** per residue; sequence-level features by
   **mean pooling**, **max pooling**, and **k-mer** embeddings.
3. **Hierarchical Swin Transformer** encoder with shifted windows for contextual/global feature
   aggregation (patch embedding, layer norm, window attention, GELU MLP; convolutional decoder).
4. **CNN** — 1-D convolution + max pooling for local motif feature extraction.
5. **BiLSTM** — forward/backward hidden states concatenated for long-range dependencies.
6. **Transformer encoder** — patch embedding, sinusoidal positional encoding, multi-head
   self-attention, ReLU feed-forward; classification head = flatten → dense (512, ReLU) → dropout
   (0.5) → sigmoid output.
7. Training: **binary cross-entropy** loss, **Adam** optimizer, dropout regularisation,
   **15 epochs** selected (overfitting observed from ~epoch 12).

Reported motif alignments: query-centric 78.6%, full alignment 81.4%, query-centric + reweighting
85.1%, full alignment + reweighting 88.3%.

## Reported results

Table 1 — "Accuracies of all the model" (column in the manuscript is labelled *ROC curve*, while the
prose calls these accuracy values — an inconsistency flagged by reviewers):

| Model | Reported value |
| --- | --- |
| CNN | 65% |
| BiLSTM | 77% |
| Transformer | 71% |
| Combination (CNN + BiLSTM + Transformer) | 91% |

- Micro-averaged **ROC AUC > 0.9**, **PR AUC ≈ 0.76** across kinase families.
- Src near-perfect; Akt lower PR AUC (class imbalance / similarity to negatives); PKA and PI3K moderate.
- Confusion: Src perfect, PKC 854 correct; AKT confused with PKA; MAPK/CDK and PIKK/CDK2 weak separation.
- Other reported numbers: AKT network analysis 87.5%; phosphorylation prediction 83.2%.

Additional analyses in the manuscript: hydropathy/charge plots, latent-space (UMAP) visualisation,
sequence-logo motifs, phylogenetic analysis, kinase interaction networks, 2D/3D structure figures,
and Kaplan–Meier survival curves (log-rank p ≈ 0.057). Reviewers challenged the link between these
analyses and the main prediction task, and the survival-analysis methodology.

## Review record (from the Frontiers review forum capture)

| Reviewer | Submitted | Recommendation | Status | Raw Q3–Q6 ratings |
| --- | --- | --- | --- | --- |
| Reviewer 1 | 01 Sep 2026 | Major revision is required | Active | 2, 2, 2, 3 |
| Reviewer 2 | 31 Aug 2026 | Reject | Closed / rejected | 1, 1, 1, 1 |
| Reviewer 3 | 30 Sep 2026 | Reject | Closed / rejected | 1, 1, 1, 1 |
| Reviewer 4 | 02 Sep 2026 | Reject | Closed / rejected | 1, 1, 1, 1 |

Common rejection reason for Reviewers 2–4: "The manuscript contains fundamental errors that cannot be
rectified through author revisions."

Reviewer 1's main requests: clarify the role of ProtBERT and the actual XAI method; report dataset
composition (number of proteins, S/T/Y counts, negative-site generation, positive/negative ratio,
database version, redundancy removal); split at **protein level** with CD-HIT/MMseqs2 to avoid
window-level leakage; benchmark against external predictors (MusiteDeep, TransPhos, Phosformer,
GraphPhos, KinasePhos 3.0) on the same test set; report AUPRC/F1/MCC and per-S/T-vs-Y performance;
describe structural feature extraction and run sequence-only vs structure-only vs combined ablations;
connect the auxiliary analyses to the main model and justify or remove the Kaplan–Meier section.

Associate Editor Amit Kumar Yadav (28 Sep 2026): recommends **rejection in present state** and suggests
resubmission as a new evaluation rather than revision of the current manuscript. The packet explicitly
states this is *not* a final editorial decision.

## Code and data availability (as printed in the manuscript)

- Code: https://github.com/SugeethJSA/Kinase-phosphite-3d_detection-_and_structure_code_and_dataset
- Data: https://www.kaggle.com/datasets/tim12jo/top-20-kinase-structures-and-site-predictiondata

## Honesty notes

- Reported numbers above are transcribed from the manuscript, not independently reproduced here.
- Several inconsistencies are documented by the reviewers (metric labelling, sample counts,
  inconsistent accuracy/ROC/AUC values, algorithm numbering, figure citations); treat reported
  performance as author-reported and unverified.
