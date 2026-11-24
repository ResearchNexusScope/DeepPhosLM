"""Fast, self-contained checks for results-derived figures and legacy defects.

All fixtures are written to ``tmp_path``. No real spreadsheet is ever read.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import plot_figures  # noqa: E402  (path prepared above)


EXPECTED_FIGURES = {
    "model_vs_baseline.png",
    "roc_curves.png",
    "precision_recall_curves.png",
    "per_site_type.png",
    "per_site_type_table.png",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixtures(results_dir: Path) -> list[Path]:
    """Create a tiny, fully synthetic results directory under tmp_path."""
    results_dir.mkdir(parents=True, exist_ok=True)

    neural_metrics = {
        "n": 12,
        "n_positive": 6,
        "n_negative": 6,
        "roc_auc": 0.81,
        "pr_auc": 0.73,
        "f1": 0.66,
        "mcc": 0.5,
        "per_site_type": {
            "S": {"n": 6, "roc_auc": 0.91, "pr_auc": 0.88, "f1": 0.8, "mcc": 0.6},
            "Y": {
                "n": 2,
                "roc_auc": None,
                "pr_auc": None,
                "f1": None,
                "mcc": None,
                "warnings": ["single-class input: discrimination metrics undefined"],
            },
        },
    }
    baseline_metrics = {
        "n": 12,
        "roc_auc": 0.6,
        "pr_auc": 0.62,
        "f1": 0.55,
        "mcc": 0.4,
        "per_site_type": {
            "S": {"n": 6, "roc_auc": 0.7, "pr_auc": 0.68, "f1": 0.6, "mcc": 0.4},
            "Y": {"n": 2, "roc_auc": None, "pr_auc": None, "f1": None, "mcc": None},
        },
    }
    metrics = {
        "dataset_seed": 4242,
        "models": {
            "synthetic_neural": {
                "archive": "results\\predictions_synthetic_neural.npz",
                "test_metrics": neural_metrics,
            }
        },
    }
    baselines = {
        "comparison_scope": "Synthetic same-split baseline; no external tool was run.",
        "seed": 4242,
        "models": {"synthetic_LR": {"test_metrics": baseline_metrics}},
    }
    metrics_path = results_dir / "metrics.json"
    baselines_path = results_dir / "baselines.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    baselines_path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")

    npz_path = results_dir / "predictions_synthetic_neural.npz"
    np.savez(
        npz_path,
        labels=np.array([1, 1, 1, 0, 0, 0], dtype=int),
        probabilities=np.array([0.95, 0.8, 0.7, 0.3, 0.15, 0.04], dtype=float),
        site_type=np.array(["S", "S", "S", "T", "T", "Y"]),
        kinase=np.array(["AURKA"] * 6),
        cluster_id=np.array(["cluster_check"] * 6),
        config={"task": "synthetic binary fixture", "ablation": "seq_only"},
        seed=np.array(4242),
    )
    return [metrics_path, baselines_path, npz_path]


def test_cli_produces_expected_figures_and_manifest(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "figures"
    inputs = _write_fixtures(results_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "plot_figures.py"),
            "--results-dir", str(results_dir),
            "--out-dir", str(out_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    for name in EXPECTED_FIGURES:
        path = out_dir / name
        assert path.is_file(), f"missing figure {name}"
        assert path.stat().st_size > 1000, f"figure {name} looks empty"
    csv_path = out_dir / "per_site_type.csv"
    assert csv_path.is_file()

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 4242
    assert manifest["simulated_clinical_data_used"] is False
    assert "no simulated clinical data" in manifest["statement"].lower()
    assert "committed" not in manifest["statement"].lower()
    assert manifest["dpi"] >= 150
    datetime.fromisoformat(manifest["generated_at"])

    assert set(manifest["inputs"]) == {path.name for path in inputs}
    for path in inputs:
        assert manifest["inputs"][path.name]["sha256"] == _sha256(path)
    assert EXPECTED_FIGURES <= set(manifest["outputs"])
    assert "per_site_type.csv" in manifest["outputs"]
    for name, info in manifest["outputs"].items():
        assert info["sha256"] == _sha256(out_dir / name)

    # Undefined metrics must be labelled, never silently written as 0.
    table_text = csv_path.read_text(encoding="utf-8")
    assert "undefined" in table_text
    assert "0.000" not in table_text


def test_figure_values_are_read_from_json_and_npz(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    inputs = _write_fixtures(results_dir)
    stored_metrics = json.loads(inputs[0].read_text(encoding="utf-8"))
    expected = stored_metrics["models"]["synthetic_neural"]["test_metrics"]

    bundle = plot_figures.load_results(results_dir)
    comparison = {row["model"]: row for row in plot_figures.build_comparison_rows(bundle)}
    assert comparison["synthetic_neural"]["n"] == expected["n"]
    for key in ("roc_auc", "pr_auc", "f1", "mcc"):
        assert comparison["synthetic_neural"][key] == expected[key]

    site_rows = plot_figures.build_site_type_rows(bundle)
    y_row = next(row for row in site_rows
                 if row["model"] == "synthetic_neural" and row["site_type"] == "Y")
    for key in ("roc_auc", "pr_auc", "f1", "mcc"):
        assert y_row[key] is None
        assert y_row[f"{key}_display"] == "undefined"
    s_row = next(row for row in site_rows
                 if row["model"] == "synthetic_neural" and row["site_type"] == "S")
    assert s_row["roc_auc_display"] == f"{expected['per_site_type']['S']['roc_auc']:.3f}"

    # Change the stored JSON: the plotted series must change with it (no literals).
    stored_metrics["models"]["synthetic_neural"]["test_metrics"]["roc_auc"] = 0.123
    inputs[0].write_text(json.dumps(stored_metrics), encoding="utf-8")
    changed = plot_figures.load_results(results_dir)
    changed_row = next(row for row in plot_figures.build_comparison_rows(changed)
                       if row["model"] == "synthetic_neural")
    assert changed_row["roc_auc"] == 0.123

    # Curves are computed from the npz arrays, not copied from the JSON.
    npz_path = results_dir / "predictions_synthetic_neural.npz"
    curves = plot_figures.compute_curves(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        labels = np.asarray(data["labels"])
        probabilities = np.asarray(data["probabilities"])
    assert curves["degenerate"] is False
    assert curves["roc_auc"] == pytest.approx(roc_auc_score(labels, probabilities))
    assert len(curves["fpr"]) > 1 and len(curves["precision"]) > 1


def test_visualization_engine_has_no_fabricated_clinical_analysis() -> None:
    source = (REPO / "visualization_engine.py").read_text(encoding="utf-8")
    for forbidden in (
        "generate_survival_simulation_data",
        "KaplanMeier",
        "logrank_test",
        "CoxPHFitter",
        "np.random.exponential",
        "3D Siamese Coordinates",
    ):
        assert forbidden not in source, f"legacy defect still present: {forbidden}"
    assert re.search(r"\bcoords\b", source) is None
    assert "no clinical cohort" in source.lower()

    module = _load_module("ve_test", REPO / "visualization_engine.py")
    assert not hasattr(module, "generate_survival_simulation_data")
    fams = np.array(["EGFR", "AKT1", "SRC"])
    assert module.resolve_kinase_index(fams, "SRC") == 2
    with pytest.raises(KeyError):
        module.resolve_kinase_index(fams, "MTOR")


def test_motif_detection_single_definition_and_portable_paths(tmp_path: Path) -> None:
    source = (REPO / "motif_detection.py").read_text(encoding="utf-8")
    assert "OneDrive" not in source
    assert "C:\\Users\\Tim" not in source
    assert "C:/Users/Tim" not in source

    tree = ast.parse(source)
    definitions = [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert definitions.count("predict_phosphorylation_sites") == 1

    module = _load_module("motif_test", REPO / "motif_detection.py")
    doc = module.predict_phosphorylation_sites.__doc__ or ""
    assert "candidate" in doc.lower()
    assert "NOT" in doc
    assert module.MOTIF_PATTERNS["Ser/Thr kinase (candidate)"] == r"[ST][^P][^P][RK]"
    assert module.MOTIF_PATTERNS["Tyrosine kinase (candidate)"] == r"[ST]..Y"
    # "SAKK" matches the S/T pattern; "STKY" matches the tyrosine pattern.
    assert module.predict_phosphorylation_sites("SAKKGGSTKY") == [0, 6]

    # Path resolution is rooted at the repo / explicit data dir, not a person's home.
    fake = tmp_path / "Modified_TOP_20_FOR_STRUCTURE.xlsx"
    fake.write_bytes(b"synthetic placeholder")
    assert module.resolve_spreadsheet(data_dir=tmp_path) == fake.resolve()
    with pytest.raises(FileNotFoundError):
        module.resolve_spreadsheet(data_dir=tmp_path / "missing")
