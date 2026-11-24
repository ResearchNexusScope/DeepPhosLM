"""Fast tests for the authoritative validation gate.

Every NPZ, CSV and manifest here is SYNTHETIC and written under ``tmp_path``;
no real result archive or dataset row is read. ``src.metrics`` is the oracle for
recomputation, so no reported value is hardcoded. The full suite targets <15 s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import metrics as sm  # noqa: E402
from src.real_split import sha256_file  # noqa: E402
from src.validate_results import (  # noqa: E402
    DEFAULT_ECE_BINS,
    available_claim_paths,
    check_claimed_numbers,
    ece_report,
    expected_calibration_error,
    paired_bootstrap,
    validate_all,
    validate_cv,
)

SEEDS = (666, 667)
ABLATIONS = ("seq_only", "seq_embedding")


def _sequence(index: int) -> str:
    return "ACDEFGHIKLMNPQRST"[index % 17] * 15


def _make_frame(n_rows: int, n_clusters: int, prefix: str, rng: np.random.Generator):
    labels = np.arange(n_rows) % 2
    site_types = np.array(["S", "T", "Y"] * ((n_rows + 2) // 3))[:n_rows]
    return {
        "sequence": [_sequence(i) for i in range(n_rows)],
        "label": labels.tolist(),
        "site_type": site_types.tolist(),
        "protein_acc": [f"{prefix}P{i}" for i in range(n_rows)],
        "cluster_id": [f"{prefix}C{i % n_clusters}" for i in range(n_rows)],
    }


def _write_npz(path: Path, labels, probs, site_types, proteins, clusters, seed, ablation, epoch):
    config = {"ablation": ablation, "seed": seed, "chosen_epoch": epoch, "batch_size": 4}
    np.savez(
        path,
        labels=np.asarray(labels, dtype=np.int64),
        probabilities=np.asarray(probs, dtype=np.float64),
        site_type=np.asarray(site_types, dtype=str),
        protein_acc=np.asarray(proteins, dtype=str),
        cluster_id=np.asarray(clusters, dtype=str),
        seed=np.int64(seed),
        ablation=np.asarray(ablation, dtype=str),
        chosen_epoch=np.int64(epoch),
        config=np.asarray(config, dtype=object),
    )
    return path


def _build(tmp_path: Path, *, y_single_class: bool = False):
    """Create a tiny synthetic split, prediction archives and both manifests."""
    rng = np.random.default_rng(11)
    split_dir = tmp_path / "split"
    results_dir = tmp_path / "results"
    cv_dir = tmp_path / "cv"
    for directory in (split_dir, results_dir, cv_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train = _make_frame(24, 6, "R", rng)
    test = _make_frame(12, 3, "E", rng)
    if y_single_class:
        # Make every Y row negative in the test split: its positive-class
        # metrics must be reported as None, not imputed.
        for i, site in enumerate(test["site_type"]):
            if site == "Y":
                test["label"][i] = 0
    import pandas as pd

    pd.DataFrame(train).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame(test).to_csv(split_dir / "test.csv", index=False)
    pd.DataFrame(train).to_csv(split_dir / "val.csv", index=False)
    (split_dir / "manifest.json").write_text(
        json.dumps({"no_overlap": {"clusters_disjoint": True}, "seed": 666}), encoding="utf-8"
    )

    labels = np.asarray(test["label"], dtype=np.int64)
    site_types = np.asarray(test["site_type"], dtype=str)
    proteins = np.asarray(test["protein_acc"], dtype=str)
    clusters = np.asarray(test["cluster_id"], dtype=str)
    npzs = []
    for ablation in ABLATIONS:
        for seed in SEEDS:
            local_rng = np.random.default_rng(seed + (7 if ablation == "seq_embedding" else 0))
            noise = local_rng.uniform(-0.15, 0.15, size=labels.size)
            probs = np.clip(0.25 + 0.55 * labels + noise, 0.02, 0.98)
            path = results_dir / f"predictions_{ablation}_seed{seed}.npz"
            _write_npz(path, labels, probs, site_types, proteins, clusters, seed, ablation, 3)
            npzs.append(path)

    (results_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "prediction_file": str(path),
                        "seed": int(path.stem.split("seed")[-1]),
                        "ablation": "seq_embedding" if "seq_embedding" in path.name else "seq_only",
                    }
                    for path in npzs
                ],
                "run_config": {"seeds": list(SEEDS), "ablations": list(ABLATIONS), "folds": 2},
                "metrics_source": "src.metrics.compute_metrics",
            }
        ),
        encoding="utf-8",
    )

    cv_runs = []
    assignments = {"R" + str(i): i % 2 for i in range(6)}
    for ablation in ABLATIONS:
        folds = []
        fold_metrics = []
        for fold in range(2):
            fold_labels = np.asarray([i % 2 for i in range(6)], dtype=np.int64)
            fold_probs = np.clip(0.2 + 0.6 * fold_labels + np.random.default_rng(fold + 1).uniform(-0.05, 0.05, 6), 0, 1)
            fold_sites = np.array(["S", "T", "Y"] * 2)
            metrics = sm.compute_metrics(fold_labels, fold_probs, site_types=fold_sites)
            folds.append(
                {
                    "fold": fold,
                    "val_clusters": [f"RC{i}" for i in range(6) if i % 2 == fold],
                    "metrics": metrics,
                }
            )
            fold_metrics.append(metrics)
        cv_runs.append(
            {
                "ablation": ablation,
                "seed": 666,
                "n_folds": 2,
                "folds": folds,
                "summary": sm.summarize_folds(fold_metrics, ddof=0),
            }
        )
    (cv_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "cv_runs": cv_runs,
                "cv_fold_assignments_by_seed": {"666": {"RC" + str(i): i % 2 for i in range(6)}},
                "run_config": {"folds": 2},
                "cv_summary_source": "src.metrics.summarize_folds",
            }
        ),
        encoding="utf-8",
    )
    return {
        "split_dir": split_dir,
        "results_dir": results_dir,
        "cv_dir": cv_dir,
        "npzs": npzs,
        "labels": labels,
        "test": test,
        "assignments": assignments,
    }


def _report(fixture, *, n_boot: int = 8, include_baselines: bool = False):
    return validate_all(
        str(fixture["results_dir"]),
        str(fixture["cv_dir"]),
        str(fixture["split_dir"]),
        n_boot=n_boot,
        include_baselines=include_baselines,
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """One tiny synthetic experiment shared by the fast tests."""
    return _build(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="module")
def report(synthetic):
    """One recomputation shared by the fast tests.

    Baselines are intentionally excluded here: importing ``src.baselines``
    pulls in torch and would blow the <15 s synthetic-test budget. The mother
    verifier exercises baselines once on the real split.
    """
    return _report(synthetic)


def test_recomputation_matches_src_metrics_exactly(synthetic, report):
    fixture = synthetic
    for entry in report["runs"]:
        path = next(p for p in fixture["npzs"] if f"_seed{entry['seed']}.npz" in str(p).replace("\\", "/"))
        with np.load(path, allow_pickle=True) as archive:
            expected = sm.compute_metrics(
                archive["labels"], archive["probabilities"], site_types=archive["site_type"].tolist()
            )
        for key in ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "precision", "recall"):
            assert entry["metrics"][key] == pytest.approx(expected[key], abs=1e-12)
        assert entry["metrics"]["n"] == expected["n"]


def test_aggregate_vectors_mean_std(report):
    for ablation in ABLATIONS:
        block = report["aggregates"][ablation]
        seeds = block["seeds"]
        assert seeds == list(SEEDS)
        for metric in ("roc_auc", "pr_auc", "accuracy"):
            values = [block["per_seed"][str(seed)][metric] for seed in seeds]
            assert block["metrics"][metric]["values"] == values
            assert block["mean"][metric] == pytest.approx(float(np.mean(values)), abs=1e-12)
            assert block["std"][metric] == pytest.approx(float(np.std(values)), abs=1e-12)
            assert block["metrics"][metric]["n_defined"] == len(seeds)


def test_check_claimed_numbers_detects_tamper(report):
    path = "aggregates.seq_only.metrics.roc_auc.mean"
    value = available_claim_paths(report)[path]
    good = check_claimed_numbers({"auroc": {"path": path, "value": value}}, report)
    assert good[0]["status"] == "PASS"
    assert good[0]["recomputed"] == pytest.approx(value)
    assert good[0]["provenance"]["runs"]
    bad = check_claimed_numbers({"auroc": {"path": path, "value": value + 0.25}}, report)
    assert bad[0]["status"] == "FAIL"
    assert bad[0]["recomputed"] == pytest.approx(value)
    missing = check_claimed_numbers({"x": {"path": "no.such.path", "value": 1}}, report)
    assert missing[0]["status"] == "FAIL"


def test_ece_known_vectors_and_site_report(report):
    labels = np.array([0, 1] * 5)
    perfect = np.array([0.0, 1.0] * 5)
    miscalibrated = np.array([0.9] * 10)
    assert expected_calibration_error(labels, perfect, n_bins=DEFAULT_ECE_BINS) == pytest.approx(0.0)
    assert expected_calibration_error(labels, miscalibrated, n_bins=DEFAULT_ECE_BINS) == pytest.approx(0.4)
    assert expected_calibration_error(np.array([]), np.array([])) is None

    for ablation in ABLATIONS:
        block = report["calibration"][ablation]
        assert 0.0 <= block["pooled"]["ece"] <= 1.0
        assert set(block["sites"]) == {"S", "T", "Y"}
        for site in ("S", "T", "Y"):
            assert block["sites"][site]["provenance"]["site_type"] == site


def test_paired_bootstrap_deterministic(synthetic, report):
    first = report["paired_bootstrap"]["auroc"]
    second = paired_bootstrap(
        _load_runs_for_test(synthetic), n_boot=8, seed=666
    )["paired_bootstrap"]["auroc"]
    assert first["point_estimate"] == pytest.approx(second["point_estimate"], abs=1e-15)
    np.testing.assert_allclose(first["bootstrap_differences"], second["bootstrap_differences"])
    assert first["n_boot"] == 8
    assert first["model_a"] == "seq_embedding" and first["model_b"] == "seq_only"
    assert first["ci_low"] <= first["ci_high"]
    auprc = report["paired_bootstrap"]["auprc"]
    assert set(auprc) >= {"point_estimate", "ci_low", "ci_high", "metric_a_point", "metric_b_point"}


def _load_runs_for_test(fixture):
    # Reuse the validator's loader without reimplementing artifact parsing here.
    from src.validate_results import _load_runs

    return _load_runs(fixture["results_dir"])[0]


def test_per_site_type_grouping_and_undefined_none(tmp_path):
    from src.validate_results import _load_runs, validate_per_site_type

    fixture = _build(tmp_path, y_single_class=True)
    runs = _load_runs(fixture["results_dir"])[0]
    report = validate_per_site_type(runs)
    for ablation in ABLATIONS:
        sites = report["per_site_type"][ablation]
        assert {key for key in sites if key != "provenance"} == {"S", "T", "Y"}
        assert sites["Y"]["pooled"]["n"] > 0
        assert sites["Y"]["pooled"]["roc_auc"] is None
        assert sites["Y"]["pooled"]["pr_auc"] is None
        assert sites["S"]["pooled"]["roc_auc"] is not None
        pooled_total = sum(sites[site]["pooled"]["n"] for site in ("S", "T", "Y"))
        assert pooled_total == len(SEEDS) * len(fixture["labels"])


def test_provenance_hashes_truthful(report):
    for entry in report["provenance"]["files"]:
        assert entry["sha256"] == sha256_file(Path(entry["path"]))
    for run in report["runs"]:
        assert run["provenance"]["sha256"] == sha256_file(Path(run["provenance"]["source_file"]))
        assert run["provenance"]["config"]["seed"] == run["seed"]


def test_cv_summary_and_disjointness(tmp_path, report):
    cv = report["cv"]
    assert len(cv["cv_runs"]) == 2
    assert all(check["passed"] for check in cv["checks"])
    for run in cv["cv_runs"]:
        assert run["n_folds"] == 2
        assert run["summary_recomputed"]["metrics"]["roc_auc"]["mean"] == pytest.approx(
            run["summary_stored"]["metrics"]["roc_auc"]["mean"], abs=1e-12
        )

    # Fresh synthetic copy for the tamper step: the shared report above must
    # stay backed by untampered artifacts for the other tests.
    fixture = _build(tmp_path)
    manifest_path = fixture["cv_dir"] / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = manifest["cv_runs"][0]["summary"]["metrics"]["roc_auc"]["mean"]
    manifest["cv_runs"][0]["summary"]["metrics"]["roc_auc"]["mean"] = original + 0.2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = validate_cv(fixture["cv_dir"], fixture["split_dir"])
    failed = [check for check in tampered["cv"]["checks"] if not check["passed"]]
    assert any(".summary.roc_auc.mean" in check["name"] for check in failed)


def test_verifier_exit_codes_with_claims(tmp_path, synthetic, report):
    fixture = synthetic
    path = "aggregates.seq_embedding.metrics.pr_auc.mean"
    value = available_claim_paths(report)[path]

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import verify_experiment as verifier

    common = [
        "--results-dir", str(fixture["results_dir"]),
        "--cv-dir", str(fixture["cv_dir"]),
        "--split-dir", str(fixture["split_dir"]),
        "--n-boot", "8",
        "--skip-baselines",
        "--skip-pytest",
    ]
    bad_claims = tmp_path / "bad.json"
    bad_claims.write_text(json.dumps({"pr_auc": {"path": path, "value": value + 0.3}}), encoding="utf-8")
    bad_out = tmp_path / "bad_report.json"
    # main()'s return value is exactly what the module's __main__ guard passes
    # to sys.exit, so a non-zero return here is the verifier's failing exit code.
    assert verifier.main(common + ["--claims", str(bad_claims), "--out", str(bad_out)]) == 1
    saved = json.loads(bad_out.read_text(encoding="utf-8"))
    assert saved["claims"]["checks"][0]["status"] == "FAIL"


def test_npz_test_row_integrity_detects_tampering(tmp_path):
    """A corrupted NPZ row identity must FAIL its check and the global gate."""
    fixture = _build(tmp_path)
    target = next(path for path in fixture["npzs"] if "seq_only_seed666" in path.name)
    with np.load(target, allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    corrupted = arrays["cluster_id"].copy()
    corrupted[0] = "BAD"
    arrays["cluster_id"] = corrupted
    np.savez(target, **arrays)

    report = _report(fixture)
    name = "run_seq_only_666.matches_test_csv"
    checks = [check for check in report["checks"] if check["name"] == name]
    assert len(checks) == 1
    check = checks[0]
    assert check["passed"] is False and check["status"] == "FAIL"
    assert check["first_mismatch"]["field"] == "cluster_id"
    assert check["first_mismatch"]["index"] == 0
    assert "BAD" in check["first_mismatch"]["reason"]
    assert check["sha256"] and check["test_csv_sha256"] and check["source_file"]
    other_checks = [
        item
        for item in report["checks"]
        if item["name"].endswith(".matches_test_csv") and item["name"] != name
    ]
    assert len(other_checks) == len(fixture["npzs"]) - 1
    assert all(item["passed"] for item in other_checks)

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from verify_experiment import _failures

    failures = _failures(report)
    assert any("matches_test_csv" in failure for failure in failures)
    assert not all(check["status"] == "PASS" for check in report["checks"])
