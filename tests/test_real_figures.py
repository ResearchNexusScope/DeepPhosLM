"""Fast, self-contained checks for the REAL-results figures, claims and validator.

Every fixture (report, NPZs, CV manifest, claims, markdown) is SYNTHETIC and
written under ``tmp_path``. No real prediction archive, dataset row or manuscript
is read. The whole module targets well under ten seconds: tiny archives, a
reduced figure DPI and no subprocesses.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

plot_real_results = importlib.import_module("plot_real_results")
make_claims = importlib.import_module("make_claims")
validate_manuscript_numbers = importlib.import_module("validate_manuscript_numbers")

ABLATIONS = ("seq_only", "seq_embedding")
SEEDS = (501, 502)
SITES = ("S", "T", "Y")
METRICS = ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "precision", "recall")
COUNTS = ("n", "n_positive", "n_negative")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path, seed: int, base: float, ablation: str) -> None:
    labels = np.asarray([1, 0] * 15, dtype=np.int64)
    probabilities = np.linspace(0.95, 0.05, labels.size)
    probabilities[1::2] = 1.0 - probabilities[1::2] - seed / 10000.0
    probabilities = np.clip(probabilities + (base - 0.5) * 0.02, 0.01, 0.99)
    np.savez(
        path,
        labels=labels,
        probabilities=probabilities,
        site_type=np.asarray([SITES[i % 3] for i in range(labels.size)]),
        protein_acc=np.asarray([f"P{i}" for i in range(labels.size)]),
        cluster_id=np.asarray([f"C{i % 3}" for i in range(labels.size)]),
        seed=np.int64(seed),
        ablation=np.asarray(ablation),
        chosen_epoch=np.int64(3),
        config=np.asarray({"ablation": ablation, "seed": seed}, dtype=object),
    )


def _metrics(base: float, index: int) -> dict[str, Any]:
    result: dict[str, Any] = {"n": 60, "n_positive": 30, "n_negative": 30}
    for offset, key in enumerate(METRICS):
        result[key] = round(base + 0.01 * index + 0.001 * offset, 6)
    result["threshold"] = 0.5
    result["warnings"] = []
    result["per_site_type"] = {
        site: {"n": 20, "n_positive": 10, "n_negative": 10,
               "roc_auc": round(base + 0.01 * index + 0.002 * i, 6),
               "pr_auc": round(base + 0.02 * index + 0.003 * i, 6),
               "f1": round(base + 0.005 * index, 6), "warnings": []}
        for i, site in enumerate(SITES)
    }
    return result


def _write_synthetic(tmp_path: Path, base: float = 0.4321) -> dict[str, Path]:
    """Create a tiny but structurally complete set of E-003/E-004 artifacts."""
    results_dir = tmp_path / "real"
    results_dir.mkdir(parents=True)
    cv_dir = tmp_path / "real_cv"
    cv_dir.mkdir(parents=True)

    npz_paths: list[Path] = []
    runs: list[dict[str, Any]] = []
    for ablation_index, ablation in enumerate(ABLATIONS):
        for seed in SEEDS:
            path = results_dir / f"predictions_{ablation}_seed{seed}.npz"
            _write_npz(path, seed, base, ablation)
            npz_paths.append(path)
            runs.append(
                {
                    "seed": seed,
                    "ablation": ablation,
                    "provenance": {"source_file": str(path), "seed": seed},
                }
            )

    aggregates: dict[str, Any] = {}
    per_site_type: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    cv_runs: list[dict[str, Any]] = []
    for ablation_index, ablation in enumerate(ABLATIONS):
        per_seed = {str(seed): _metrics(base, ablation_index) for seed in SEEDS}
        mean = {key: per_seed[str(SEEDS[0])][key] for key in [*METRICS, *COUNTS]}
        std = {key: 0.0001 + 0.0001 * index for index, key in enumerate(METRICS)}
        std.update({key: 0 for key in COUNTS})
        aggregates[ablation] = {
            "seeds": list(SEEDS), "per_seed": per_seed, "metrics": {},
            "mean": mean, "std": std,
        }
        per_site_type[ablation] = {
            site: {
                "pooled": {"n": 40 + i, "n_positive": 20, "n_negative": 20 + i,
                           "roc_auc": round(base + 0.01 * ablation_index + 0.001 * i, 6),
                           "pr_auc": round(base + 0.02 * ablation_index + 0.001 * i, 6),
                           "f1": round(base + 0.03 * ablation_index + 0.001 * i, 6)}
            }
            for i, site in enumerate(SITES)
        }
        calibration[ablation] = {
            "pooled": {"ece": round(base / 100 + 0.001 * ablation_index, 6), "n": 120},
            "per_seed_summary": {"values": [0.01, 0.02], "mean": 0.015 + 0.001 * ablation_index,
                                  "std": 0.005, "n_defined": 2, "std_ddof": 0},
            "sites": {site: {"ece": round(base / 200 + 0.001 * i, 6), "n": 40}
                      for i, site in enumerate(SITES)},
        }
        cv_runs.append(
            {
                "ablation": ablation,
                "seed": SEEDS[0],
                "n_folds": 2,
                "summary_recomputed": {
                    "metrics": {
                        key: {"mean": round(base + 0.01 * ablation_index + 0.001 * i, 6),
                              "std": 0.002 + 0.001 * i, "n_defined": 2, "values": [0.4, 0.5]}
                        for i, key in enumerate(METRICS)
                    }
                },
            }
        )

    report: dict[str, Any] = {
        "config": {"bootstrap_seed": SEEDS[0], "ece_bins": 10,
                   "claim_tolerance": {"abs": 1e-9, "rel": 1e-6}},
        "provenance": {"files": [
            {"path": str(path), "sha256": _sha256(path), "role": "predictions"}
            for path in npz_paths
        ]},
        "runs": runs,
        "aggregates": aggregates,
        "per_site_type": per_site_type,
        "paired_bootstrap": {
            metric: {"point_estimate": round(base / 10 + i / 100, 6),
                     "ci_low": round(base / 10 - 0.01, 6),
                     "ci_high": round(base / 10 + 0.01, 6)}
            for i, metric in enumerate(("auroc", "auprc"))
        },
        "calibration": calibration,
        "baselines": {
            "synthetic_lr": {
                "metrics": {"n": 60, **{key: round(base - 0.01 * i, 6)
                                        for i, key in enumerate(METRICS)}}
            }
        },
        "cv": {"cv_runs": cv_runs},
    }
    report_path = results_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cv_manifest = {
        "cv_runs": [
            {
                "ablation": ablation,
                "seed": SEEDS[0],
                "n_folds": 2,
                "folds": [
                    {"fold": fold, "metrics": {
                        "roc_auc": round(base + 0.01 * fold + 0.01 * i, 6),
                        "pr_auc": round(base + 0.02 * fold + 0.01 * i, 6),
                    }}
                    for fold in range(2)
                ],
            }
            for i, ablation in enumerate(ABLATIONS)
        ]
    }
    cv_path = cv_dir / "training_manifest.json"
    cv_path.write_text(json.dumps(cv_manifest, indent=2), encoding="utf-8")

    # Tiny pinned manifests so the provenance-number check has a source.
    data_manifest = results_dir / "data_manifest.json"
    data_manifest.write_text(json.dumps({"counts": {"rows": 424242}}), encoding="utf-8")
    split_manifest = results_dir / "split_manifest.json"
    split_manifest.write_text(json.dumps({"seed": 999}), encoding="utf-8")

    return {
        "report": report_path, "cv": cv_path, "results": results_dir,
        "data_manifest": data_manifest, "split_manifest": split_manifest,
    }


def _markdown_from_claims(claims: dict[str, Any]) -> str:
    """Render a results table containing exactly the claim values (no literals)."""
    lines = [
        "# Synthetic results",
        "",
        "## Dataset provenance",
        "",
        "A total of 424,242 synthetic windows were pinned in the input manifest.",
        "",
        "## Results table",
        "",
        "| Claim | Value |",
        "| --- | --- |",
    ]
    for name, claim in claims.items():
        value = claim["value"] if isinstance(claim, dict) else claim
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines) + "\n"


def test_figures_and_manifest_derive_from_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plot_real_results, "FIGURE_DPI", 40)
    paths = _write_synthetic(tmp_path)
    out = tmp_path / "figs"
    result = plot_real_results.generate_all(paths["report"], paths["results"], paths["cv"], out)

    assert result["manifest"].is_file()
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert manifest["inputs"][str(paths["report"])]["sha256"] == _sha256(paths["report"])
    assert manifest["simulated_data_used"] is False
    assert "No simulated data is used" in manifest["statement"]
    assert {path.name for path in result["outputs"]} >= {
        "test_roc_mean_std.png", "test_pr_mean_std.png", "model_vs_baseline.png",
        "per_site_type.png", "cv_fold_means.png", "calibration_ece.png",
    }
    comparison = result["data"]["comparison"][0]
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert comparison["metrics"]["roc_auc"]["mean"] == report["aggregates"][ABLATIONS[0]]["mean"]["roc_auc"]

    before = (out / "model_vs_baseline.png").read_bytes()
    report["aggregates"][ABLATIONS[0]]["mean"]["roc_auc"] += 0.123456
    changed = tmp_path / "changed_report.json"
    changed.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out2 = tmp_path / "figs2"
    out2.mkdir()
    changed_data = plot_real_results.build_plot_data(
        json.loads(changed.read_text(encoding="utf-8")), paths["results"], paths["cv"]
    )
    plot_real_results.plot_model_vs_baseline(changed_data, out2)
    assert (out2 / "model_vs_baseline.png").read_bytes() != before


def test_claims_are_derived_from_report_not_hardcoded(tmp_path: Path) -> None:
    paths = _write_synthetic(tmp_path, base=0.271828)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    claims = make_claims.build_claims(report)
    name = f"test.{ABLATIONS[0]}.roc_auc.mean"
    expected = round(report["aggregates"][ABLATIONS[0]]["mean"]["roc_auc"], 4)
    assert claims[name]["value"] == expected
    assert claims[name]["path"] == f"aggregates.{ABLATIONS[0]}.mean.roc_auc"
    assert claims[name]["tolerance"] > 0
    assert "provenance" in claims[name]
    # A different recomputed value must propagate into a different claim.
    report["aggregates"][ABLATIONS[0]]["mean"]["roc_auc"] += 0.111111
    assert make_claims.build_claims(report)[name]["value"] == round(
        report["aggregates"][ABLATIONS[0]]["mean"]["roc_auc"], 4
    )
    assert expected not in (0.7148, 0.7223)  # no real-report value leaked in


def test_tampered_markdown_number_is_detected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_synthetic(tmp_path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    claims = make_claims.build_claims(report)
    name = f"test.{ABLATIONS[0]}.roc_auc.mean"
    original = claims[name]["value"]
    markdown = _markdown_from_claims(claims).replace(str(original), str(round(original + 0.5, 4)), 1)
    doc_path = tmp_path / "TAMPERED.md"
    doc_path.write_text(markdown, encoding="utf-8")

    # Write the claims where the CLI expects them (inside tmp_path, never the repo).
    claims_path = paths["results"] / "claims.json"
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    code = validate_manuscript_numbers.main([
        "--report", str(paths["report"]), "--claims", str(claims_path),
        "--markdown", str(doc_path), "--data-manifest", str(paths["data_manifest"]),
        "--split-manifest", str(paths["split_manifest"]),
    ])
    output = capsys.readouterr().out
    assert code != 0
    assert "[FAIL]" in output
    assert name in output


def test_validator_exits_nonzero_on_false_claim(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_synthetic(tmp_path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    claims = make_claims.build_claims(report)
    claims_path = paths["results"] / "claims.json"
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    doc_path = tmp_path / "OK.md"
    doc_path.write_text(_markdown_from_claims(claims), encoding="utf-8")
    args = [
        "--report", str(paths["report"]), "--claims", str(claims_path),
        "--markdown", str(doc_path), "--data-manifest", str(paths["data_manifest"]),
        "--split-manifest", str(paths["split_manifest"]),
    ]
    assert validate_manuscript_numbers.main(args) == 0

    name = f"test.{ABLATIONS[0]}.roc_auc.mean"
    claims[name]["value"] = round(claims[name]["value"] + 0.37, 4)
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    assert validate_manuscript_numbers.main(args) != 0
    output = capsys.readouterr().out
    assert "[FAIL]" in output and name in output
