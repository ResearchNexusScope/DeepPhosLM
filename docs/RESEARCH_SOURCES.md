# Research-folder source index for DeepPhosLM

AmazeSwarm declared shared folder `research`:

- Drive home: `~/gdrive/My Drive/Research` (single copy; Google Drive propagates it between nodes)
- Local path on this node: `~/Research`
- Declared in `swarm/config/shared-folders.json` (id `research`, enabled)

The original PDFs, spreadsheets and notebooks remain in Drive. Only the text extraction in
[`../PAPER.md`](../PAPER.md) is versioned in this repo.

## Primary source — this paper (applied)

Location: `02 Applied - Journals/Pavithra/Front. Bioinform/DeepPhosLM Protein/`

| File | Use in the extraction |
| --- | --- |
| `Manuscript - 1964362_Manuscript.PDF` | Title/authors, abstract, dataset, methodology, results, code/data availability |
| `Frontiers review reports - 1964362_Frontiers_Review_Reports.pdf` | Reviewer recommendations and reports |
| `Review reports - extracted - DeepPhosLM_Frontiers_Review_Reports.pdf` | Faithful text extraction of the review forum (used for reviewer details) |
| `Review reports - Frontiers style - DeepPhosLM_Frontiers_Review_Reports_Frontiers~.pdf` | Formatted review reports |
| `Complete review packet - 1964362_Frontiers_Complete_Review_Packet.pdf` (and v2, v3) | Editorial/review packet, editor recommendation, decision status |
| `Raw review page dump - frontiers-raw-page-dump-2026-10-01T03-56-48-3~.json` | Raw HTML capture of the Frontiers review forum |
| `Review search results - SearchResults.csv` | Literature search results attached to the submission |

## "Protein" search — applied at multiple places

Searching the research folder for `protein` (filenames and folder names) found the DeepPhosLM work
plus two related protein/kinase applications, all under `02 Applied - Journals/Pavithra/`:

1. **`Front. Bioinform/DeepPhosLM Protein/`** — this paper (protein language model +
   phosphorylation-site prediction).
2. **`Journal/GENOMe Kinase/`** — kinase-phosphorylation work:
   `kinase_phosphorylation_sites.xlsx`, `Manuscript draft - genome.docx`,
   `PDF (image-only) - genome (1).pdf`, Turnitin reports. This is the likely origin of the
   `kinase_phosphorylation_sites.xlsx` dataset in this repository.
3. **`Journal/PepCNN Protein/`** — separate protein paper with code:
   `Manuscript - scientific report revision.docx`, `Reviewer comments - scientific Reviewer
   comments.docx`, and `Support/Code/` (`protein_pipeline_torch.py`, notebooks,
   `download_uniprot_dataset.py`, `dataset.csv`, GPU setup docs).

These are distinct manuscripts/applications, not additional versions of DeepPhosLM. They are indexed
here so the provenance of shared datasets and code is traceable.

## Repo link

The DeepPhosLM manuscript's "Code Availablity" section names this repository:
`https://github.com/SugeethJSA/Kinase-phosphite-3d_detection-_and_structure_code_and_dataset`
(cloned locally to `C:\Users\sugee\Github\SugeethJSA\Kinase-phosphite-3d_detection-_and_structure_code_and_dataset`).
