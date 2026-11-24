#!/usr/bin/env python3
"""Figures for the REAL phosphosite experiment, derived only from saved artifacts.

Every plotted number, curve and annotation is read at run time from:

* ``results/real/validation_report.json`` (recomputed metrics, per-site-type
  breakdown, ECE and provenance),
* the ten ``predictions_*.npz`` archives named by that report, and
* the cross-validation manifest (``results/real_cv/training_manifest.json``).

No result value is typed into this file, and no simulated or toy-set data is
ever generated or consumed. The only constants below are structural (file
names, plot styling, grid resolution and the metric labels used by
``src.metrics``). ROC/PR curves are recomputed from the stored labels and
probabilities; mean +/- std bands are computed across the seeds listed in the
report.

Usage (from the repository root)::

    python tools/plot_real_results.py \
        --report results/real/validation_report.json \
        --results-dir results/real \
        --cv-manifest results/real_cv/training_manifest.json \
        --out-dir figures/real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless batch figure generator
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIGURE_DPI = 200
CURVE_GRID_POINTS = 201

NO_SIMULATED_DATA_STATEMENT = (
    "No simulated data is used. Every plotted value and every curve is derived "
    "at run time from results/real/validation_report.json, the prediction NPZ "
    "archives named in that report, and the cross-validation training manifest. "
    "No toy-set or hand-entered number is consumed."
)

# Structural constants only (metric names used by src.metrics; never values).
BAR_METRICS: tuple[tuple[str, str], ...] = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("f1", "F1"),
    ("mcc", "MCC"),
)
SITE_METRICS: tuple[tuple[str, str], ...] = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("f1", "F1"),
)
CV_METRICS: tuple[tuple[str, str], ...] = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
)

ROC_FIG = "test_roc_mean_std.png"
PR_FIG = "test_pr_mean_std.png"
BASELINE_FIG = "model_vs_baseline.png"
SITE_FIG = "per_site_type.png"
CV_FIG = "cv_fold_means.png"
CALIB_FIG = "calibration_ece.png"
CALIB_OMITTED = "calibration_omitted.txt"
MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------------------- #
# Loading and numeric preparation
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _infer_results_dir(report: dict[str, Any], fallback: Path) -> Path:
    """Prefer the prediction paths pinned in the report over the CLI default."""
    for entry in (report.get("provenance") or {}).get("files", []):
        if entry.get("role") == "predictions":
            candidate = Path(str(entry["path"])).parent
            if candidate.is_dir():
                return candidate
    return fallback


def _prediction_path(report: dict[str, Any], results_dir: Path, ablation: str, seed: int) -> Path:
    """Resolve one NPZ from report provenance (never guessed result values)."""
    for run in report.get("runs", []):
        provenance = run.get("provenance") or {}
        if str(run.get("ablation")) == ablation and int(run.get("seed")) == int(seed):
            ref = provenance.get("source_file")
            if ref:
                candidate = Path(str(ref))
                if candidate.is_file():
                    return candidate
                by_name = results_dir / candidate.name
                if by_name.is_file():
                    return by_name
    raise FileNotFoundError(f"no prediction archive in report for {ablation}/seed {seed}")


def _load_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        labels = np.asarray(data["labels"]).ravel().astype(np.int64)
        probabilities = np.asarray(data["probabilities"]).ravel().astype(np.float64)
    if labels.size != probabilities.size:
        raise ValueError(f"label/probability length mismatch in {path}")
    if np.unique(labels).size < 2:
        raise ValueError(f"single-class archive cannot define curves: {path}")
    return labels, probabilities


def _curves(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": np.asarray([auc(fpr, tpr)], dtype=float),
        "recall": recall,
        "precision": precision,
        "pr_auc": np.asarray([auc(recall, precision)], dtype=float),
        "prevalence": np.asarray([labels.mean()], dtype=float),
    }


def _mean_std_curves(
    per_seed: list[dict[str, np.ndarray]], x_key: str, y_key: str, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each seed's curve onto a common grid, then mean +/- std."""
    stacked = []
    for curve in per_seed:
        x = np.asarray(curve[x_key], dtype=float)
        y = np.asarray(curve[y_key], dtype=float)
        if x.size != y.size:
            raise ValueError(f"curve arrays {x_key}/{y_key} are misaligned")
        # precision_recall_curve returns recall in decreasing order; np.interp
        # needs increasing x. roc_curve's fpr is already increasing.
        if x.size > 1 and x[0] > x[-1]:
            x, y = x[::-1], y[::-1]
        stacked.append(np.interp(grid, x, y))
    matrix = np.vstack(stacked)
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=0)


def build_plot_data(
    report: dict[str, Any], results_dir: Path, cv_manifest_path: Path
) -> dict[str, Any]:
    """Prepare every number the figures use, derived only from the three inputs."""
    results_dir = Path(results_dir)
    aggregates = report.get("aggregates") or {}
    if not aggregates:
        raise ValueError("validation report has no 'aggregates' block")
    ablations = [str(key) for key in aggregates]
    seeds = [int(value) for value in aggregates[ablations[0]]["seeds"]]

    # (a) test ROC/PR curves, one per seed, averaged over the report's seeds.
    curves: dict[str, Any] = {}
    curve_inputs: list[Path] = []
    for ablation in ablations:
        per_seed: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            path = _prediction_path(report, results_dir, ablation, seed)
            curve_inputs.append(path)
            per_seed.append(_curves(*_load_archive(path)))
        fpr_grid = np.linspace(0.0, 1.0, CURVE_GRID_POINTS)
        recall_grid = np.linspace(0.0, 1.0, CURVE_GRID_POINTS)
        roc_mean, roc_std = _mean_std_curves(per_seed, "fpr", "tpr", fpr_grid)
        pr_mean, pr_std = _mean_std_curves(per_seed, "recall", "precision", recall_grid)
        curves[ablation] = {
            "seeds": seeds,
            "fpr_grid": fpr_grid,
            "recall_grid": recall_grid,
            "roc_mean": roc_mean,
            "roc_std": roc_std,
            "pr_mean": pr_mean,
            "pr_std": pr_std,
            "prevalence": float(np.mean([float(c["prevalence"][0]) for c in per_seed])),
            "roc_auc_mean": float(aggregates[ablation]["mean"]["roc_auc"]),
            "pr_auc_mean": float(aggregates[ablation]["mean"]["pr_auc"]),
        }

    # (b) neural models (mean +/- std over seeds) vs same-split baselines.
    comparison: list[dict[str, Any]] = []
    for ablation in ablations:
        comparison.append(
            {
                "name": ablation,
                "kind": "seeds",
                "n": aggregates[ablation]["mean"]["n"],
                "metrics": {
                    key: {
                        "mean": aggregates[ablation]["mean"][key],
                        "std": aggregates[ablation]["std"][key],
                    }
                    for key, _label in BAR_METRICS
                },
            }
        )
    for name, baseline in (report.get("baselines") or {}).items():
        metrics = baseline["metrics"]
        comparison.append(
            {
                "name": str(name),
                "kind": "baseline",
                "n": metrics["n"],
                # A baseline is fitted once; no seed vector exists -> no std.
                "metrics": {key: {"mean": metrics[key], "std": None} for key, _label in BAR_METRICS},
            }
        )

    # (c) per-S/T/Y pooled metrics and n, read from the report.
    per_site: dict[str, Any] = {}
    for ablation in ablations:
        site_block = (report.get("per_site_type") or {}).get(ablation) or {}
        per_site[ablation] = {}
        for site in sorted(key for key in site_block if key != "provenance"):
            pooled = site_block[site]["pooled"]
            per_site[ablation][site] = {
                "n": pooled["n"],
                "n_positive": pooled["n_positive"],
                "n_negative": pooled["n_negative"],
                "metrics": {key: pooled[key] for key, _label in SITE_METRICS},
            }

    # (d) CV fold means with spread, recomputed from the CV manifest folds.
    cv_manifest = _read_json(cv_manifest_path)
    cv_data: list[dict[str, Any]] = []
    for run in cv_manifest.get("cv_runs", []):
        folds = run.get("folds", [])
        if not folds:
            continue
        summary: dict[str, Any] = {}
        for key, _label in CV_METRICS:
            values = np.asarray([float(fold["metrics"][key]) for fold in folds], dtype=float)
            summary[key] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        cv_data.append(
            {
                "ablation": str(run.get("ablation")),
                "seed": int(run.get("seed", -1)),
                "n_folds": len(folds),
                "metrics": summary,
            }
        )

    # (e) calibration: ECE summaries stored in the report. Bin-level reliability
    # points are NOT persisted by E-004, so no invented reliability curve is drawn.
    calibration: dict[str, Any] = {}
    for ablation in ablations:
        block = (report.get("calibration") or {}).get(ablation)
        if not block:
            continue
        calibration[ablation] = {
            "pooled_ece": block["pooled"]["ece"],
            "pooled_n": block["pooled"]["n"],
            "per_seed": block["per_seed_summary"],
            "sites": {
                site: {"ece": values["ece"], "n": values["n"]}
                for site, values in block.get("sites", {}).items()
            },
            "n_bins": (report.get("config") or {}).get("ece_bins"),
        }

    return {
        "ablations": ablations,
        "seeds": seeds,
        "curves": curves,
        "comparison": comparison,
        "per_site": per_site,
        "cv": cv_data,
        "calibration": calibration,
        "curve_inputs": curve_inputs,
        "bootstrap_seed": (report.get("config") or {}).get("bootstrap_seed"),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _colors(count: int) -> list[Any]:
    cmap = plt.get_cmap("tab10")
    return [cmap(index % 10) for index in range(count)]


def plot_curves(data: dict[str, Any], out_dir: Path) -> list[Path]:
    ablations = data["ablations"]
    colors = _colors(len(ablations))
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 5.8))
    for ablation, color in zip(ablations, colors):
        block = data["curves"][ablation]
        grid = block["fpr_grid"]
        ax.plot(grid, block["roc_mean"], color=color, lw=2,
                label=f"{ablation} (mean AUROC={block['roc_auc_mean']:.4f})")
        ax.fill_between(grid, block["roc_mean"] - block["roc_std"],
                        block["roc_mean"] + block["roc_std"], color=color, alpha=0.2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC: mean +/- std over the report's seeds")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outputs.append(_save(fig, out_dir, ROC_FIG))

    fig, ax = plt.subplots(figsize=(7, 5.8))
    for ablation, color in zip(ablations, colors):
        block = data["curves"][ablation]
        grid = block["recall_grid"]
        ax.plot(grid, block["pr_mean"], color=color, lw=2,
                label=f"{ablation} (mean PR-AUC={block['pr_auc_mean']:.4f})")
        ax.fill_between(grid, block["pr_mean"] - block["pr_std"],
                        block["pr_mean"] + block["pr_std"], color=color, alpha=0.2)
        ax.axhline(block["prevalence"], color=color, ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-out precision-recall: mean +/- std over seeds")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outputs.append(_save(fig, out_dir, PR_FIG))
    return outputs


def plot_model_vs_baseline(data: dict[str, Any], out_dir: Path) -> Path:
    rows = data["comparison"]
    x = np.arange(len(BAR_METRICS))
    width = 0.8 / max(1, len(rows))
    colors = _colors(len(rows))
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(rows)), 6))
    for index, (row, color) in enumerate(zip(rows, colors)):
        positions = x + (index - (len(rows) - 1) / 2) * width
        means = [row["metrics"][key]["mean"] for key, _label in BAR_METRICS]
        stds = [row["metrics"][key]["std"] for key, _label in BAR_METRICS]
        ax.bar(positions, means, width, color=color, edgecolor="black", linewidth=0.6,
               yerr=[s if s is not None else 0.0 for s in stds], capsize=3,
               label=f"{row['name']} (n={row['n']})")
        for position, mean, std in zip(positions, means, stds):
            suffix = "+/-" + f"{std:.4f}" if std is not None else "single fit"
            ax.text(position, float(mean), f"{float(mean):.4f}\n{suffix}",
                    ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in BAR_METRICS])
    ax.set_ylabel("Score on held-out test split")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Neural models (mean +/- seed std) vs same-split classical baselines")
    fig.tight_layout()
    return _save(fig, out_dir, BASELINE_FIG)


def plot_per_site(data: dict[str, Any], out_dir: Path) -> Path:
    ablations = data["ablations"]
    sites = sorted({site for block in data["per_site"].values() for site in block})
    colors = _colors(len(ablations))
    fig, axes = plt.subplots(1, len(SITE_METRICS), figsize=(4.6 * len(SITE_METRICS), 5.2),
                             squeeze=False)
    x = np.arange(len(sites))
    width = 0.8 / max(1, len(ablations))
    for panel, (key, label) in enumerate(SITE_METRICS):
        ax = axes[0][panel]
        for index, ablation in enumerate(ablations):
            positions = x + (index - (len(ablations) - 1) / 2) * width
            values = [data["per_site"][ablation][site]["metrics"][key] for site in sites]
            ax.bar(positions, values, width, color=colors[index], edgecolor="black",
                   linewidth=0.5, label=ablation)
            for position, site, value in zip(positions, sites, values):
                n = data["per_site"][ablation][site]["n"]
                ax.text(position, float(value) if value is not None else 0.0,
                        f"{'undefined' if value is None else format(float(value), '.4f')}\nn={n}",
                        ha="center", va="bottom", rotation=90, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0][0].set_ylabel("Pooled held-out score")
    handles = [Patch(color=colors[i], label=name) for i, name in enumerate(ablations)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle("Per-site-type results (pooled across seeds, n annotated)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return _save(fig, out_dir, SITE_FIG)


def plot_cv(data: dict[str, Any], out_dir: Path) -> Path:
    runs = data["cv"]
    if not runs:
        raise ValueError("CV manifest contains no fold runs")
    x = np.arange(len(CV_METRICS))
    width = 0.8 / max(1, len(runs))
    colors = _colors(len(runs))
    fig, ax = plt.subplots(figsize=(max(8, 2.3 * len(runs)), 5.8))
    for index, (run, color) in enumerate(zip(runs, colors)):
        positions = x + (index - (len(runs) - 1) / 2) * width
        means = [run["metrics"][key]["mean"] for key, _label in CV_METRICS]
        stds = [run["metrics"][key]["std"] for key, _label in CV_METRICS]
        ax.bar(positions, means, width, color=color, edgecolor="black", linewidth=0.6,
               yerr=stds, capsize=4,
               label=f"{run['ablation']} ({run['n_folds']} folds, seed {run['seed']})")
        for position, key, mean, std in zip(positions, [k for k, _ in CV_METRICS], means, stds):
            values = run["metrics"][key]["values"]
            for fold_value in values:
                ax.plot(position, fold_value, marker="o", ms=3, color="black", alpha=0.6)
            ax.text(position, float(mean), f"{float(mean):.4f}\n+/-{float(std):.4f}",
                    ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _key, label in CV_METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cross-validated score (train split only)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("CV fold means with population spread; dots are individual folds")
    fig.tight_layout()
    return _save(fig, out_dir, CV_FIG)


def plot_calibration(data: dict[str, Any], out_dir: Path) -> list[Path]:
    """Plot report ECE summaries; omit honestly if the report has no ECE block."""
    calibration = data["calibration"]
    if not calibration:
        note = (
            "Calibration figure omitted: the validation report contains no "
            "'calibration' (ECE) block, so no calibration value can be plotted "
            "without inventing one.\n"
        )
        path = out_dir / CALIB_OMITTED
        path.write_text(note, encoding="utf-8")
        return [path]

    ablations = [name for name in data["ablations"] if name in calibration]
    colors = _colors(len(ablations))
    sites = sorted({site for block in calibration.values() for site in block["sites"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for index, (ablation, color) in enumerate(zip(ablations, colors)):
        block = calibration[ablation]
        position = index
        pooled = float(block["pooled_ece"])
        seed_mean = float(block["per_seed"]["mean"])
        seed_std = float(block["per_seed"]["std"])
        ax.bar(position, seed_mean, 0.6, color=color, edgecolor="black",
               yerr=seed_std, capsize=4, label=ablation)
        for seed_value in block["per_seed"]["values"]:
            ax.plot(position, seed_value, "o", ms=4, color="black", alpha=0.6)
        ax.text(position, seed_mean, f"ECE(pooled)={pooled:.4f}\nn={block['pooled_n']}",
                ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(range(len(ablations)))
    ax.set_xticklabels(ablations)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Expected calibration error, per seed and pooled")

    ax = axes[1]
    x = np.arange(len(sites))
    width = 0.8 / max(1, len(ablations))
    for index, (ablation, color) in enumerate(zip(ablations, colors)):
        positions = x + (index - (len(ablations) - 1) / 2) * width
        values = [float(calibration[ablation]["sites"][site]["ece"]) for site in sites]
        ax.bar(positions, values, width, color=color, edgecolor="black", label=ablation)
        for position, site, value in zip(positions, sites, values):
            n = calibration[ablation]["sites"][site]["n"]
            ax.text(position, value, f"{value:.4f}\nn={n}", ha="center", va="bottom",
                    rotation=90, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_title("Pooled ECE by centred residue type")
    n_bins = data["calibration"][ablations[0]]["n_bins"]
    fig.suptitle(f"Calibration summary from report ECE ({n_bins} equal-width bins)")
    fig.text(
        0.5, 0.005,
        "Honest note: E-004 stores ECE summaries, not bin-level confidence/accuracy "
        "points; a classical reliability curve is therefore not drawn.",
        ha="center", fontsize=8,
    )
    fig.legend(loc="lower center", ncol=len(ablations), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return [_save(fig, out_dir, CALIB_FIG)]


def write_manifest(
    data: dict[str, Any], out_dir: Path, report_path: Path, cv_path: Path, outputs: list[Path]
) -> Path:
    inputs: dict[str, dict[str, Any]] = {}
    for path in [report_path, *data["curve_inputs"], cv_path]:
        key = str(path)
        if key not in inputs:
            inputs[key] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": data["bootstrap_seed"],
        "seeds_used_for_curves": data["seeds"],
        "statement": NO_SIMULATED_DATA_STATEMENT,
        "simulated_data_used": False,
        "dpi": FIGURE_DPI,
        "inputs": inputs,
        "outputs": {
            path.name: {"sha256": sha256_file(path)} for path in outputs if path.is_file()
        },
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def generate_all(
    report_path: Path, results_dir: Path, cv_manifest_path: Path, out_dir: Path
) -> dict[str, Any]:
    report = _read_json(report_path)
    results_dir = _infer_results_dir(report, Path(results_dir))
    data = build_plot_data(report, results_dir, Path(cv_manifest_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    outputs.extend(plot_curves(data, out_dir))
    outputs.append(plot_model_vs_baseline(data, out_dir))
    outputs.append(plot_per_site(data, out_dir))
    outputs.append(plot_cv(data, out_dir))
    outputs.extend(plot_calibration(data, out_dir))
    manifest_path = write_manifest(data, out_dir, report_path, cv_manifest_path, outputs)
    return {"data": data, "outputs": outputs, "manifest": manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", type=Path,
        default=REPO_ROOT / "results" / "real" / "validation_report.json",
        help="validation_report.json from tools/verify_experiment.py",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "real",
        help="directory holding the prediction NPZs (inferred from the report when possible)",
    )
    parser.add_argument(
        "--cv-manifest", type=Path, default=REPO_ROOT / "results" / "real_cv" / "training_manifest.json",
        help="cross-validation training manifest with per-fold metrics",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "real",
        help="destination for figures and manifest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_all(args.report, args.results_dir, args.cv_manifest, args.out_dir)
    print(f"Wrote {len(result['outputs'])} derived figures/files to {args.out_dir}")
    for path in result["outputs"]:
        print(f"  {path.name}")
    print(f"Manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
