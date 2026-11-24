#!/usr/bin/env python3
"""Reproducible figures derived exclusively from result files (metrics.json, baselines.json, predictions_*.npz).

Every number plotted here is read at run time from ``results/metrics.json``,
``results/baselines.json`` and ``results/predictions_*.npz``. ROC and
precision-recall curves are computed from the stored labels and probabilities;
they are never drawn by hand. No clinical cohort, survival time or other
simulated data is generated or consumed.

Usage::

    python tools/plot_figures.py --results-dir results --out-dir figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this is a batch figure generator
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

FIGURE_DPI = 200  # >= 150 dpi as required for publication-quality output

# Metric keys are the ones actually written by src.metrics.compute_metrics.
METRICS: tuple[tuple[str, str], ...] = (
    ("roc_auc", "AUROC"),
    ("pr_auc", "AUPRC"),
    ("f1", "F1"),
    ("mcc", "MCC"),
)

NO_SIMULATED_DATA_STATEMENT = (
    "No simulated clinical data is used. All values come from result files "
    "(metrics.json, baselines.json, predictions_*.npz); "
    "ROC/PR curves are recomputed from stored labels and probabilities."
)

COMPARISON_FIG = "model_vs_baseline.png"
ROC_FIG = "roc_curves.png"
PR_FIG = "precision_recall_curves.png"
SITE_FIG = "per_site_type.png"
SITE_TABLE_FIG = "per_site_type_table.png"
SITE_CSV = "per_site_type.csv"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ModelResult:
    """One model's test metrics as stored in a results JSON file."""

    name: str
    source: str  # "metrics.json" or "baselines.json"
    test_metrics: dict[str, Any]
    archive: Path | None = None


@dataclass(frozen=True)
class ResultsBundle:
    """Everything the figure generator needs, loaded from disk."""

    models: list[ModelResult]
    npz_paths: list[Path]
    inputs: list[Path]
    seed: int | None
    comparison_scope: str | None
    metrics_raw: dict[str, Any]
    baselines_raw: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 (works for large archives)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _archive_path(results_dir: Path, archive: Any, model_name: str) -> Path | None:
    """Resolve an archive entry without assuming the OS that wrote the path."""
    if archive:
        filename = Path(str(archive).replace("\\", "/")).name
        candidate = results_dir / filename
        if candidate.is_file():
            return candidate
    candidate = results_dir / f"{model_name}.npz"
    return candidate if candidate.is_file() else None


def load_results(results_dir: Path) -> ResultsBundle:
    """Load metrics, baselines and prediction archives from ``results_dir``."""
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    baselines_path = results_dir / "baselines.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing required results file: {metrics_path}")

    metrics_raw = _read_json(metrics_path)
    baselines_raw = _read_json(baselines_path) if baselines_path.is_file() else {}

    models: list[ModelResult] = []
    for source, raw in (("metrics.json", metrics_raw), ("baselines.json", baselines_raw)):
        for name, info in (raw.get("models") or {}).items():
            test_metrics = (info or {}).get("test_metrics") or {}
            archive = _archive_path(results_dir, (info or {}).get("archive"), name)
            models.append(
                ModelResult(name=name, source=source, test_metrics=test_metrics, archive=archive)
            )

    npz_paths: list[Path] = []
    for model in models:
        if model.archive is not None and model.archive not in npz_paths:
            npz_paths.append(model.archive)
    for path in sorted(results_dir.glob("predictions_*.npz")):
        if path not in npz_paths:
            npz_paths.append(path)

    inputs: list[Path] = [metrics_path]
    if baselines_path.is_file():
        inputs.append(baselines_path)
    for path in npz_paths:
        if path not in inputs:
            inputs.append(path)

    seed = metrics_raw.get("dataset_seed", metrics_raw.get("seed"))
    if seed is None:
        seed = baselines_raw.get("seed")
    if seed is None and npz_paths:
        with np.load(npz_paths[0], allow_pickle=True) as data:
            if "seed" in data.files:
                seed = int(np.asarray(data["seed"]).item())

    return ResultsBundle(
        models=models,
        npz_paths=npz_paths,
        inputs=inputs,
        seed=seed,
        comparison_scope=baselines_raw.get("comparison_scope"),
        metrics_raw=metrics_raw,
        baselines_raw=baselines_raw,
    )


def build_comparison_rows(bundle: ResultsBundle) -> list[dict[str, Any]]:
    """Return exactly the stored test metrics, with no imputation."""
    rows: list[dict[str, Any]] = []
    for model in bundle.models:
        row: dict[str, Any] = {
            "model": model.name,
            "source": model.source,
            "n": model.test_metrics.get("n"),
        }
        for key, _label in METRICS:
            row[key] = model.test_metrics.get(key)
        rows.append(row)
    return rows


def compute_curves(npz_path: Path) -> dict[str, Any]:
    """Compute ROC/PR curves from the labels and probabilities in one npz."""
    with np.load(npz_path, allow_pickle=True) as data:
        labels = np.asarray(data["labels"]).ravel().astype(int)
        probabilities = np.asarray(data["probabilities"]).ravel().astype(float)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError(f"Label/probability length mismatch in {npz_path}")
    if np.unique(labels).size < 2:
        # A single-class archive cannot define a ROC or PR curve. It is skipped,
        # never replaced with an invented curve.
        return {"name": Path(npz_path).stem, "degenerate": True, "n": int(labels.size)}

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "name": Path(npz_path).stem,
        "degenerate": False,
        "n": int(labels.size),
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(auc(fpr, tpr)),
        "precision": precision,
        "recall": recall,
        "pr_auc": float(auc(recall, precision)),
        "prevalence": float(labels.mean()),
    }


def build_site_type_rows(bundle: ResultsBundle) -> list[dict[str, Any]]:
    """Return per-site-type metrics showing 'undefined' instead of zero.

    Undefined entries (e.g. all-negative Y cohorts) stay ``None`` in the raw
    fields and become the literal string ``"undefined"`` in the ``*_display``
    fields used by the table figure and CSV.
    """
    rows: list[dict[str, Any]] = []
    for model in bundle.models:
        per_site = model.test_metrics.get("per_site_type") or {}
        for site_type in sorted(per_site):
            stored = per_site[site_type] or {}
            row: dict[str, Any] = {
                "model": model.name,
                "source": model.source,
                "site_type": site_type,
                "n": stored.get("n"),
            }
            for key, _label in METRICS:
                value = stored.get(key)
                if value is not None:
                    value = float(value)
                row[key] = value
                row[f"{key}_display"] = "undefined" if value is None else f"{value:.3f}"
            rows.append(row)
    return rows


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison(bundle: ResultsBundle, out_dir: Path) -> Path:
    """Grouped bar chart: neural models vs classical baselines, n annotated."""
    rows = build_comparison_rows(bundle)
    if not rows:
        raise ValueError("No model metrics found; cannot build comparison figure")

    x = np.arange(len(METRICS))
    n_models = len(rows)
    width = 0.8 / n_models
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(11, 6))
    legend_handles = []
    top = 0.0
    for model_idx, row in enumerate(rows):
        positions = x + (model_idx - (n_models - 1) / 2) * width
        values = [row[key] for key, _ in METRICS]
        handle = ax.bar(
            positions,
            [v if v is not None else 0.0 for v in values],
            width,
            label=row["model"],
            color=colors[model_idx % len(colors)],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        legend_handles.append(handle)
        for position, value in zip(positions, values):
            if value is None:
                ax.text(position, 0.01, "undefined", rotation=90, ha="center", va="bottom", fontsize=7)
                continue
            top = max(top, float(value))
            n_text = "n not reported" if row["n"] is None else f"n={row['n']}"
            ax.text(position, float(value), n_text, rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in METRICS])
    ax.set_ylabel("Score on held-out test split")
    ax.set_ylim(0, min(1.05, max(0.1, top * 1.35)))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Model vs baseline performance (all values read from results JSON)")
    note = bundle.comparison_scope or ""
    if note:
        fig.text(0.5, 0.01, f"Scope: {note}", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out_dir, COMPARISON_FIG)


def plot_curves(bundle: ResultsBundle, out_dir: Path) -> list[Path]:
    """Plot ROC and precision-recall curves computed from the npz archives."""
    curves = [compute_curves(path) for path in bundle.npz_paths]
    usable = [curve for curve in curves if not curve["degenerate"]]
    if not usable:
        raise ValueError("No prediction archive with both classes available for curves")

    roc_fig, roc_ax = plt.subplots(figsize=(6.5, 5.5))
    for curve in usable:
        roc_ax.plot(curve["fpr"], curve["tpr"], lw=2,
                    label=f"{curve['name']} (AUC={curve['roc_auc']:.3f})")
    roc_ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("ROC curves recomputed from stored predictions")
    roc_ax.legend(fontsize=8, loc="lower right")
    roc_ax.grid(alpha=0.3)
    roc_fig.tight_layout()
    roc_path = _save(roc_fig, out_dir, ROC_FIG)

    pr_fig, pr_ax = plt.subplots(figsize=(6.5, 5.5))
    for curve in usable:
        pr_ax.plot(curve["recall"], curve["precision"], lw=2,
                   label=f"{curve['name']} (AUPRC={curve['pr_auc']:.3f})")
        pr_ax.axhline(curve["prevalence"], linestyle="--", lw=0.8,
                      color="grey", alpha=0.6)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-recall curves recomputed from stored predictions")
    pr_ax.legend(fontsize=8, loc="lower left")
    pr_ax.grid(alpha=0.3)
    pr_fig.tight_layout()
    pr_path = _save(pr_fig, out_dir, PR_FIG)
    return [roc_path, pr_path]


def plot_site_types(bundle: ResultsBundle, out_dir: Path) -> list[Path]:
    """Bar chart + table of per-site-type metrics (undefined stays undefined)."""
    rows = build_site_type_rows(bundle)
    if not rows:
        raise ValueError("No per-site-type metrics found")

    # One panel per metric: x axis is site type, bars are grouped by model.
    site_types = sorted({row["site_type"] for row in rows})
    model_names = list(dict.fromkeys(row["model"] for row in rows))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 5), squeeze=False)
    x = np.arange(len(site_types))
    width = 0.8 / max(1, len(model_names))
    colors = plt.get_cmap("tab10").colors
    panel_top = 0.0
    for panel, (key, label) in enumerate(METRICS):
        ax = axes[0][panel]
        for model_idx, model_name in enumerate(model_names):
            model_rows = {row["site_type"]: row for row in rows if row["model"] == model_name}
            positions = x + (model_idx - (len(model_names) - 1) / 2) * width
            color = colors[model_idx % len(colors)]
            for site_type, position in zip(site_types, positions):
                raw = model_rows.get(site_type, {}).get(key)
                if raw is None:
                    # Draw no bar at all: a zero-height bar would imply a score.
                    ax.text(position, 0.01, "undefined", rotation=90, ha="center",
                            va="bottom", fontsize=7, color="firebrick")
                    continue
                ax.bar(position, float(raw), width, color=color,
                       edgecolor="black", linewidth=0.5, alpha=0.9)
                panel_top = max(panel_top, float(raw))
        ax.set_xticks(x)
        ax.set_xticklabels(site_types)
        ax.set_ylim(0, min(1.05, max(0.1, panel_top * 1.25)))
        ax.set_title(label)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0][0].set_ylabel("Score (undefined = metric not defined)")
    handles = [Patch(color=colors[idx % len(colors)], label=name)
               for idx, name in enumerate(model_names)]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8)
    fig.suptitle("Per-site-type metrics (undefined is never plotted as zero)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    bar_path = _save(fig, out_dir, SITE_FIG)

    # Table figure: the literal word "undefined" appears in the rendered cell.
    header = ["Model", "Site", "n"] + [label for _key, label in METRICS]
    cells = [
        [row["model"], row["site_type"], str(row["n"]) if row["n"] is not None else "n/a"]
        + [row[f"{key}_display"] for key, _label in METRICS]
        for row in rows
    ]
    table_fig, table_ax = plt.subplots(figsize=(max(10, len(header) * 1.9),
                                               max(2.2, 0.42 * len(cells) + 1.2)))
    table_ax.axis("off")
    # Model names can be long (e.g. "predictions_seq_embedding"); give the
    # first column a larger share so its text does not overflow the cell.
    col_widths = [0.28] + [0.72 / (len(header) - 1)] * (len(header) - 1)
    table = table_ax.table(cellText=cells, colLabels=header, cellLoc="center",
                           loc="center", colWidths=col_widths,
                           colColours=["#dce6f1"] * len(header))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    table_ax.set_title("Per-site-type metrics as stored in results JSON",
                       fontweight="bold", pad=10)
    table_path = _save(table_fig, out_dir, SITE_TABLE_FIG)

    import csv

    csv_path = out_dir / SITE_CSV
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(cells)

    return [bar_path, table_path, csv_path]


def write_manifest(bundle: ResultsBundle, out_dir: Path, outputs: list[Path]) -> Path:
    """Record hashes, timestamp, seed and the no-simulated-data statement."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": bundle.seed,
        "statement": NO_SIMULATED_DATA_STATEMENT,
        "simulated_clinical_data_used": False,
        "dpi": FIGURE_DPI,
        "inputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in bundle.inputs
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in outputs
            if path.is_file()
        },
    }
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_results = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--results-dir", type=Path, default=default_results,
                        help="directory holding metrics.json, baselines.json and predictions_*.npz")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures",
                        help="directory where figures and manifest.json are written")
    args = parser.parse_args(argv)

    bundle = load_results(args.results_dir)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [plot_comparison(bundle, out_dir)]
    outputs.extend(plot_curves(bundle, out_dir))
    outputs.extend(plot_site_types(bundle, out_dir))
    manifest_path = write_manifest(bundle, out_dir, outputs)

    print(f"Wrote {len(outputs)} figures/derived files to {out_dir}")
    for path in outputs:
        print(f"  {path.name}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
