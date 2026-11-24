"""Fast, fully offline tests for the homology-threshold sensitivity CLI.

Every manifest and prediction archive below is synthetic and lives under
``tmp_path``.  A fake runner is injected into ``tools.split_sensitivity.main``,
so the real MMseqs binary and the real training loop are never invoked.  The
tool still reads the manifest/npz files from disk and recomputes all metrics
with :func:`src.metrics.compute_metrics`; expected values are recomputed the
same way rather than hard-coded.  One shared sweep (plus one re-run for
ordering) keeps the whole module fast.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import compute_metrics  # noqa: E402
from src.real_split import OverlapError, SPLITS  # noqa: E402
from tools.split_sensitivity import (  # noqa: E402
    CANONICAL_THRESHOLD,
    REPORTED_METRICS,
    ablation_names,
    main,
    sha256_file,
    threshold_tag,
)

ALL_OK = {
    "clusters_disjoint": True,
    "proteins_disjoint": True,
    "all_clusters_assigned": True,
    "all_proteins_assigned": True,
    "cluster_protein_consistency": True,
}
CLEAN = (0.2, 0.3, 0.7)


@dataclass
class Case:
    """One threshold's synthetic ground truth."""

    threshold: float
    n_proteins: int
    n_clusters: int
    n_singletons: int
    n_singleton_clusters: int
    windows: Dict[str, int]
    no_overlap: Dict[str, bool] = field(default_factory=lambda: dict(ALL_OK))
    overlap_error: Optional[str] = None
    skip_ablation: Optional[str] = None

    @property
    def split_counts(self) -> Dict[str, Any]:
        return {
            "train": {"windows": self.windows["train"], "proteins": 6, "clusters": 3},
            "val": {"windows": self.windows["val"], "proteins": 2, "clusters": 1},
            "test": {"windows": self.windows["test"], "proteins": 2, "clusters": 1},
        }


def make_cases() -> Dict[float, Case]:
    """Distinct fixtures; 0.4/0.5/0.6 exercise three different failure modes."""
    cases = {
        0.2: Case(0.2, 30, 12, 9, 10, {"train": 60, "val": 9, "test": 7}),
        0.3: Case(0.3, 40, 16, 11, 12, {"train": 80, "val": 11, "test": 9}),
        0.4: Case(0.4, 50, 20, 14, 15, {"train": 100, "val": 13, "test": 11}),
        0.5: Case(0.5, 60, 24, 17, 18, {"train": 120, "val": 15, "test": 13}),
        0.6: Case(0.6, 70, 28, 21, 22, {"train": 140, "val": 17, "test": 15}),
        0.7: Case(0.7, 80, 32, 25, 26, {"train": 160, "val": 19, "test": 17}),
    }
    cases[0.4].skip_ablation = "seq_embedding"
    cases[0.5].no_overlap["proteins_disjoint"] = False
    cases[0.6].overlap_error = "protein P00002 appears in both train and test"
    return cases


class FakeRunner:
    """Writes synthetic manifests/archives and records the argv it receives."""

    def __init__(self, cases: Mapping[float, Case]) -> None:
        self.cases = dict(cases)
        self.calls: List[Dict[str, Any]] = []

    def build_split(self, argv: List[str], spec: Any) -> None:
        self.calls.append({"stage": "split", "argv": list(argv), "tag": spec.tag})
        case = self.cases[spec.threshold]
        if case.overlap_error is not None:
            raise OverlapError(case.overlap_error)
        spec.split_dir.mkdir(parents=True, exist_ok=True)
        spec.work_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "method": "synthetic fixture",
            "seed": 666,
            "min_seq_id": spec.threshold,
            "n_proteins": case.n_proteins,
            "n_clusters": case.n_clusters,
            "n_singletons": case.n_singletons,
            "n_singleton_clusters": case.n_singleton_clusters,
            "splits": case.split_counts,
            "no_overlap": dict(case.no_overlap),
            "outputs": {"directory": "synthetic/fixture"},
        }
        (spec.split_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def run_training(self, argv: List[str], spec: Any) -> None:
        self.calls.append({"stage": "train", "argv": list(argv), "tag": spec.tag})
        seed = int(argv[argv.index("--seed-base") + 1])
        requested = argv[argv.index("--ablation") + 1]
        case = self.cases[spec.threshold]
        spec.train_dir.mkdir(parents=True, exist_ok=True)
        for ablation in ablation_names(requested):
            if ablation == case.skip_ablation:
                continue
            labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
            # Distinct, deterministic probabilities per threshold/ablation/seed.
            offset = (int(round(spec.threshold * 100)) + seed) % 7
            bump = {"seq_only": 0.0, "seq_embedding": 0.03}[ablation]
            probabilities = np.clip(
                np.array([0.90, 0.12, 0.80, 0.22, 0.73, 0.31, 0.61, 0.18]) + bump - offset / 100.0,
                0.02,
                0.98,
            )
            site_type = np.array(["S", "S", "T", "T", "Y", "Y", "S", "T"])
            np.savez(
                spec.train_dir / f"predictions_{ablation}_seed{seed}.npz",
                labels=labels,
                probabilities=probabilities,
                site_type=site_type,
                protein_acc=np.array([f"P{i:03d}" for i in range(len(labels))]),
                cluster_id=np.array([f"C{i % 3}" for i in range(len(labels))]),
                seed=np.int64(seed),
                ablation=np.asarray(ablation),
                chosen_epoch=np.int64(3),
                config=np.asarray({"ablation": ablation}, dtype=object),
            )


def invoke(tmp_path: Path, runner: FakeRunner, thresholds: List[float]) -> Dict[str, Any]:
    """Call the real CLI with injected runner and return the written report."""
    data_dir = tmp_path / "data" / "real"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "results" / "sensitivity"
    rc = main(
        [
            "--data-dir", str(data_dir),
            "--mmseqs", str(tmp_path / "fake" / "mmseqs.exe"),
            "--thresholds", *(str(value) for value in thresholds),
            "--seeds", "666",
            "--epochs", "6",
            "--batch-size", "1024",
            "--ablation", "both",
            "--out-dir", str(out_dir),
            "--work-root", str(tmp_path / "work"),
        ],
        runner=runner,
    )
    report_path = out_dir / "sensitivity_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {"rc": rc, "report": report, "report_path": report_path, "root": tmp_path, "runner": runner}


def _entry(report: Mapping[str, Any], threshold: float) -> Dict[str, Any]:
    matches = [entry for entry in report["thresholds"] if entry["threshold"] == threshold]
    assert len(matches) == 1, f"expected exactly one entry for {threshold}"
    return matches[0]


# Fields whose values legitimately change with tmp directory / request order.
_VOLATILE_KEYS = {
    "split_dir", "work_dir", "train_dir", "npz_path", "split_manifest_path",
    "split_argv", "train_argv", "flags",
}


def portable(value: Any) -> Any:
    """Strip path/argv fields so two runs in different tmp dirs can be compared."""
    if isinstance(value, dict):
        return {
            k: ([message.split(":", 1)[0] for message in v] if k == "failures" else portable(v))
            for k, v in value.items()
            if k not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def assert_metric_blocks_equal(actual: dict, expected: dict) -> None:
    """Compare nested metric dicts; use approx only for real numbers."""
    assert set(actual) == set(expected)
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, dict):
            assert isinstance(observed, dict)
            assert_metric_blocks_equal(observed, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            assert observed == pytest.approx(value, rel=0, abs=1e-12), key
        else:
            assert observed == value, key


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """One scrambled-order run covering clean and failing thresholds."""
    return invoke(
        tmp_path_factory.mktemp("sweep"),
        FakeRunner(make_cases()),
        [0.6, 0.2, 0.5, 0.4, 0.7, 0.3],
    )


# ---------------------------------------------------------------------------
# 1. Every requested threshold, deterministic order, honest flags
# ---------------------------------------------------------------------------


def test_all_requested_thresholds_processed_in_deterministic_order(
    tmp_path: Path, sweep: Dict[str, Any]
) -> None:
    report = sweep["report"]
    assert sweep["rc"] == 1  # three intentionally failing thresholds
    expected_order = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert report["thresholds_processed"] == expected_order
    assert report["canonical_threshold"] == CANONICAL_THRESHOLD
    assert report["canonical_threshold_processed"] is True
    assert [entry["threshold"] for entry in report["thresholds"]] == expected_order
    assert [entry["tag"] for entry in report["thresholds"]] == [
        threshold_tag(value) for value in expected_order
    ]
    assert len({entry["split_dir"] for entry in report["thresholds"]}) == 6
    assert len({entry["work_dir"] for entry in report["thresholds"]}) == 6
    assert report["summary"] == {"n_thresholds": 6, "n_ok": 3, "n_failed": 3}

    # A differently ordered request produces the same per-threshold report.
    repeated = invoke(tmp_path, FakeRunner(make_cases()), list(reversed(expected_order)))
    assert [entry["threshold"] for entry in repeated["report"]["thresholds"]] == expected_order
    assert [portable(entry) for entry in report["thresholds"]] == [
        portable(entry) for entry in repeated["report"]["thresholds"]
    ]

    # The fake runner saw the exact CLI flags for each stage.
    runner = sweep["runner"]
    splits = [call["argv"] for call in runner.calls if call["stage"] == "split"]
    trains = [call["argv"] for call in runner.calls if call["stage"] == "train"]
    assert len(splits) == 6
    for argv, threshold in zip(splits, expected_order):
        assert argv[argv.index("--min-seq-id") + 1] == str(threshold)
        assert argv[argv.index("--seed") + 1] == "666"
        assert argv[argv.index("--threads") + 1] == "8"
        assert argv[argv.index("--out-dir") + 1].endswith(f"split_{threshold_tag(threshold)}")
        assert argv[argv.index("--work-dir") + 1].endswith(f"work_{threshold_tag(threshold)}")
    assert trains
    for argv in trains:
        assert argv[argv.index("--seeds") + 1] == "1"
        assert argv[argv.index("--seed-base") + 1] == "666"
        assert argv[argv.index("--folds") + 1] == "0"
        assert argv[argv.index("--epochs") + 1] == "6"
        assert argv[argv.index("--batch-size") + 1] == "1024"
        assert argv[argv.index("--ablation") + 1] == "both"


# ---------------------------------------------------------------------------
# 2. Counts / provenance / metrics are truthful, recomputed from npz
# ---------------------------------------------------------------------------


def test_counts_and_metrics_are_truthful_and_recomputed_from_npz(sweep: Dict[str, Any]) -> None:
    cases = make_cases()
    report = sweep["report"]

    for threshold in CLEAN:
        case = cases[threshold]
        entry = _entry(report, threshold)
        assert entry["status"] == "OK"
        assert entry["cluster_counts"] == {
            "n_proteins": case.n_proteins,
            "n_clusters": case.n_clusters,
        }
        assert entry["singleton_counts"] == {
            "n_singletons": case.n_singletons,
            "n_singleton_clusters": case.n_singleton_clusters,
        }
        assert entry["split_window_counts"] == case.windows
        assert entry["split_protein_counts"] == {s: case.split_counts[s]["proteins"] for s in SPLITS}
        assert entry["split_cluster_counts"] == {s: case.split_counts[s]["clusters"] for s in SPLITS}
        assert entry["no_overlap"] == case.no_overlap
        assert entry["overlap_ok"] is True

        manifest_path = Path(entry["provenance"]["split_manifest_path"])
        assert entry["provenance"]["split_manifest_sha256"] == sha256_file(manifest_path)
        assert manifest_path.is_file()
        assert entry["provenance"]["seed"] == 666
        assert entry["provenance"]["epochs"] == 6
        assert entry["provenance"]["batch_size"] == 1024
        assert entry["provenance"]["ablation"] == "both"
        assert entry["provenance"]["flags"]["thresholds"] == list(cases)
        assert len(entry["provenance"]["train_argv"]) == 1

        assert set(entry["ablations"]) == set(ablation_names("both"))
        for ablation, block in entry["ablations"].items():
            assert len(block["runs"]) == 1
            run = block["runs"][0]
            npz_path = Path(run["npz_path"])
            assert npz_path.is_file()
            assert run["npz_sha256"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()
            assert entry["provenance"]["npz_sha256"][f"{ablation}_seed666"] == run["npz_sha256"]
            with np.load(npz_path, allow_pickle=True) as data:
                # The four reported metrics are the top-level ones; recompute
                # them from the archive rather than trusting the report.
                expected = compute_metrics(data["labels"], data["probabilities"])
            top_level = (
                "n", "n_positive", "n_negative", "roc_auc", "pr_auc", "f1",
                "mcc", "precision", "recall", "accuracy", "threshold",
            )
            observed = {name: run["test_metrics"][name] for name in top_level}
            assert_metric_blocks_equal(observed, {name: expected[name] for name in top_level})
            assert run["test"] == {name: expected[name] for name in REPORTED_METRICS}
            assert "per_site_type" in run["test_metrics"]
            assert run["epochs"] == 6 and run["batch_size"] == 1024
            assert run["seed"] == 666


# ---------------------------------------------------------------------------
# 3. Failed overlap assertions are FAIL rows, never silently included
# ---------------------------------------------------------------------------


def test_overlap_error_is_reported_as_fail_not_included(sweep: Dict[str, Any]) -> None:
    failed = _entry(sweep["report"], 0.6)
    assert failed["status"] == "FAIL"
    assert failed["ablations"] == {}
    assert failed["cluster_counts"] is None
    assert failed["split_window_counts"] is None
    assert failed["no_overlap"] is None
    assert failed["overlap_ok"] is None
    assert any("OverlapError" in message for message in failed["failures"])
    assert any("P00002" in message for message in failed["failures"])
    root = sweep["root"]
    assert not (root / "data" / "real" / f"split_{threshold_tag(0.6)}").exists()


def test_false_no_overlap_flag_is_reported_as_fail(sweep: Dict[str, Any]) -> None:
    failed = _entry(sweep["report"], 0.5)
    assert failed["status"] == "FAIL"
    assert failed["no_overlap"] == {**ALL_OK, "proteins_disjoint": False}
    assert failed["overlap_ok"] is False
    assert failed["ablations"] == {}
    assert any("no-overlap assertions" in message for message in failed["failures"])


def test_missing_prediction_archive_marks_threshold_fail(sweep: Dict[str, Any]) -> None:
    entry = _entry(sweep["report"], 0.4)
    assert entry["status"] == "FAIL"
    assert any("missing predictions archive" in message for message in entry["failures"])
    # The other ablation is still reported truthfully rather than discarded.
    assert set(entry["ablations"]) == {"seq_only"}


def test_clean_control_threshold_still_passes(sweep: Dict[str, Any]) -> None:
    assert _entry(sweep["report"], 0.7)["status"] == "OK"
    assert _entry(sweep["report"], 0.7)["overlap_ok"] is True
