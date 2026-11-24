"""Fast synthetic tests for the BCa interval, bin-level calibration and reliability CLI.

Every vector and NPZ below is generated inside ``tmp_path``; no real result
archive, dataset row or manuscript number is read. The suite targets <15 s.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm as real_norm

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from src import statistics as stats  # noqa: E402
from src.statistics import bca_interval, calibration_bins  # noqa: E402

plot_reliability = importlib.import_module("plot_reliability")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _separation_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Fast separation score (mean positive prob minus mean negative prob)."""
    y = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    return float(probabilities[y == 1].mean() - probabilities[y == 0].mean())


# --------------------------------------------------------------------------- #
# BCa paired interval
# --------------------------------------------------------------------------- #
def test_bca_separable_ci_excludes_zero_and_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    n = 60
    labels = np.asarray([0, 1] * (n // 2), dtype=np.int64)
    # prob_a perfectly separates the classes; prob_b is unrelated noise.
    prob_a = labels.astype(float)
    prob_b = np.clip(0.5 + 0.25 * rng.standard_normal(n), 0.01, 0.99)
    kwargs = dict(n_boot=400, seed=666)

    first = bca_interval(labels, prob_a, prob_b, _separation_metric, **kwargs)
    second = bca_interval(labels, prob_a, prob_b, _separation_metric, **kwargs)

    assert first == second, "same seed and inputs must give an identical interval"
    assert first["method"] == "bca"
    assert first["fallback_reason"] is None
    assert first["point_estimate"] > 0.0
    assert first["ci_low"] is not None and first["ci_low"] > 0.0
    assert first["ci_high"] >= first["ci_low"]
    assert first["n_valid"] == 400
    assert first["z0"] is not None and np.isfinite(first["z0"])
    assert first["acceleration"] is not None and np.isfinite(first["acceleration"])
    assert len(first["jackknife_values"]) == n


def _assert_fallback(result: dict, reason_fragment: str) -> None:
    assert result["method"] == "percentile_fallback"
    assert result["fallback_reason"], "every fallback must state a reason"
    assert reason_fragment in result["fallback_reason"].lower()
    assert result["z0"] is None, "fallback must not carry a stale/non-finite z0"
    assert result["acceleration"] is None, "fallback must not carry a stale acceleration"
    assert result["adjusted_percentiles"] is None
    assert "BCa unavailable" in " ".join(result["warnings"])


def test_bca_falls_back_to_percentile_on_constant_jackknife() -> None:
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    probs = np.asarray([0.1, 0.9] * 6, dtype=float)
    # Identical models: every bootstrap and jackknife difference is exactly 0.
    result = bca_interval(
        labels, probs, probs, _separation_metric, n_boot=120, seed=666
    )

    _assert_fallback(result, "constant")
    assert result["point_estimate"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)


def test_bca_falls_back_when_constant_jackknife_differs_from_point() -> None:
    n = 12
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    prob_a = np.full(n, 0.9)
    prob_b = np.full(n, 0.1)

    def _length_scaled_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        values = np.asarray(y_prob, dtype=float)
        # Full samples divide by n-1; delete-one samples divide by their own
        # length. With constant probabilities the jackknife is constant, but
        # its value (0.9 - 0.1) differs from the full-sample point estimate.
        denominator = len(values) - 1 if len(values) == n else len(values)
        return float(values.sum() / denominator)

    result = bca_interval(
        labels, prob_a, prob_b, _length_scaled_metric, n_boot=120, seed=666
    )

    _assert_fallback(result, "jackknife values are constant")
    assert len(result["jackknife_values"]) == n
    assert all(value == pytest.approx(0.8) for value in result["jackknife_values"])
    assert result["point_estimate"] == pytest.approx(12 * 0.8 / 11)
    assert result["point_estimate"] != pytest.approx(0.8)


def test_bca_falls_back_for_small_n() -> None:
    labels = np.asarray([0, 1] * 4, dtype=np.int64)  # n = 8 < MIN_BCA_N
    prob_a = np.asarray([0.9, 0.1] * 4, dtype=float)
    prob_b = np.asarray([0.8, 0.2] * 4, dtype=float)

    result = bca_interval(
        labels, prob_a, prob_b, _separation_metric, n_boot=150, seed=666
    )

    _assert_fallback(result, "too few rows")
    assert result["n_valid"] == 150


def test_bca_falls_back_when_too_few_valid_resamples() -> None:
    n = 20
    labels = np.asarray([1, 0] * 10, dtype=np.int64)
    prob_a = np.full(n, 0.2)
    prob_b = np.full(n, 0.2)
    prob_a[0], prob_b[0] = 0.9, 0.8

    def _first_row_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        values = np.asarray(y_prob, dtype=float)
        return float(values[0]) if values[0] > 0.5 else float("nan")

    result = bca_interval(
        labels, prob_a, prob_b, _first_row_metric, n_boot=200, seed=666
    )

    _assert_fallback(result, "too few resamples")
    assert 0 < result["n_valid"] < 100
    assert result["point_estimate"] == pytest.approx(0.1)


def test_bca_falls_back_when_z0_is_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    prob_a = labels.astype(float)
    prob_b = np.clip(
        0.5 + 0.3 * np.random.default_rng(11).standard_normal(labels.size), 0.01, 0.99
    )

    class _BrokenNorm:
        @staticmethod
        def ppf(q: float) -> float:
            return float("inf")

        @staticmethod
        def cdf(z: float) -> float:
            return float(real_norm.cdf(z))

    monkeypatch.setattr(stats, "norm", _BrokenNorm)
    result = bca_interval(
        labels, prob_a, prob_b, _separation_metric, n_boot=150, seed=666
    )

    _assert_fallback(result, "z0 is non-finite")


def test_bca_falls_back_when_adjusted_alpha_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    prob_a = labels.astype(float)
    prob_b = np.clip(
        0.5 + 0.3 * np.random.default_rng(11).standard_normal(labels.size), 0.01, 0.99
    )

    class _OutOfRangeNorm:
        @staticmethod
        def ppf(q: float) -> float:
            return float(real_norm.ppf(q))

        @staticmethod
        def cdf(z: float) -> float:
            return 0.0

    monkeypatch.setattr(stats, "norm", _OutOfRangeNorm)
    result = bca_interval(
        labels, prob_a, prob_b, _separation_metric, n_boot=150, seed=666
    )

    _assert_fallback(result, "adjusted percentiles")


def test_paired_resampling_shares_indices_and_point_is_recomputed() -> None:
    n = 12
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    row_ids = np.linspace(0.05, 0.95, n)
    prob_a = row_ids
    prob_b = 1.0 - row_ids  # invertible mapping exposes the resampled row IDs
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def _recording_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        values = np.asarray(y_prob, dtype=float)
        calls.append((np.asarray(y_true).copy(), values.copy()))
        return float(values.mean())

    kwargs = dict(n_boot=30, seed=666)
    result = bca_interval(labels, prob_a, prob_b, _recording_metric, **kwargs)

    pairs = [(calls[index], calls[index + 1]) for index in range(0, len(calls) - 1, 2)]
    assert len(pairs) >= kwargs["n_boot"] + 1
    for left, right in pairs:
        np.testing.assert_array_equal(left[0], right[0])
        # prob_b = 1 - prob_a, so identical paired indices recover the same
        # multiset of row IDs on both sides of every bootstrap replicate.
        np.testing.assert_allclose(
            np.sort(left[1]), np.sort(1.0 - right[1]), rtol=0.0, atol=1e-12
        )

    expected_point = float(prob_a.mean()) - float(prob_b.mean())
    assert result["point_estimate"] == pytest.approx(expected_point)
    assert result["metric_a_point"] == pytest.approx(float(prob_a.mean()))
    assert result["metric_b_point"] == pytest.approx(float(prob_b.mean()))

    shifted = np.clip(prob_a + 0.01, 0.0, 1.0)
    shifted_result = bca_interval(labels, shifted, prob_b, _recording_metric, **kwargs)
    assert shifted_result["point_estimate"] != pytest.approx(expected_point)
    assert shifted_result["point_estimate"] == pytest.approx(
        float(shifted.mean()) - float(prob_b.mean())
    )


# --------------------------------------------------------------------------- #
# Bin-level calibration
# --------------------------------------------------------------------------- #
def test_calibration_bins_ece_zero_for_perfectly_calibrated_vector() -> None:
    labels = np.asarray([0, 1] * 10, dtype=np.int64)
    perfect = labels.astype(float)
    result = calibration_bins(labels, perfect, n_bins=10)

    assert result["ece"] == pytest.approx(0.0)
    assert result["n"] == labels.size
    assert sum(entry["n"] for entry in result["bins"]) == labels.size
    assert len(result["bins"]) == 10
    assert all(entry["ece_contribution"] == pytest.approx(0.0) for entry in result["bins"])


def test_calibration_bins_counts_and_known_toy_fractions() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    probs = np.asarray([0.1, 0.8, 0.6, 0.2], dtype=float)
    result = calibration_bins(labels, probs, n_bins=10)

    assert sum(entry["n"] for entry in result["bins"]) == labels.size
    by_low = {entry["bin_low"]: entry for entry in result["bins"] if entry["n"] > 0}
    assert sorted(by_low) == pytest.approx([0.1, 0.2, 0.6, 0.8])
    assert by_low[0.1]["observed_fraction"] == pytest.approx(0.0)
    assert by_low[0.2]["observed_fraction"] == pytest.approx(0.0)
    assert by_low[0.6]["observed_fraction"] == pytest.approx(1.0)
    assert by_low[0.8]["observed_fraction"] == pytest.approx(1.0)
    assert by_low[0.6]["mean_pred"] == pytest.approx(0.6)
    # (1/4) * (|0.1-0| + |0.2-0| + |0.6-1| + |0.8-1|)
    assert result["ece"] == pytest.approx(0.225)

    empty = [entry for entry in result["bins"] if entry["n"] == 0]
    assert empty and all(entry["mean_pred"] is None for entry in empty)
    assert all(entry["ece_contribution"] == 0.0 for entry in empty)


# --------------------------------------------------------------------------- #
# Reliability CLI (synthetic NPZs only)
# --------------------------------------------------------------------------- #
def test_plot_cli_writes_manifest_and_hashes_from_synthetic_npz(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    rng = np.random.default_rng(3)
    paths = []
    for ablation in ("seq_only", "seq_embedding"):
        probs = np.clip(0.1 + 0.8 * labels + 0.05 * rng.standard_normal(labels.size), 0.0, 1.0)
        path = pred_dir / f"predictions_{ablation}_seed666.npz"
        np.savez(
            path,
            labels=labels,
            probabilities=probs,
            ablation=np.asarray(ablation),
            seed=np.int64(666),
        )
        paths.append(path)

    results_dir = tmp_path / "results"
    out_dir = tmp_path / "figures"
    assert (
        plot_reliability.main(
            [
                "--predictions", str(pred_dir),
                "--results-dir", str(results_dir),
                "--out-dir", str(out_dir),
            ]
        )
        == 0
    )

    bins_path = results_dir / "calibration_bins.json"
    figure_path = out_dir / "reliability_diagram.png"
    manifest_path = out_dir / "reliability_manifest.json"
    assert bins_path.is_file() and figure_path.is_file() and manifest_path.is_file()

    payload = json.loads(bins_path.read_text(encoding="utf-8"))
    assert set(payload["groups"]) == {"seq_only", "seq_embedding", "pooled"}
    for block in payload["groups"].values():
        assert sum(entry["n"] for entry in block["bins"]) == block["n"]
        assert all("sha256" in item for item in block["provenance"]["files"])
        assert all("seed" in item and "n" in item for item in block["provenance"]["files"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in paths:
        assert manifest["inputs"][str(path)]["sha256"] == _sha256(path)
        assert manifest["inputs"][str(path)]["seed"] == 666
    assert manifest["inputs"][str(bins_path)]["sha256"] == _sha256(bins_path)
    assert manifest["outputs"]["reliability_diagram.png"]["sha256"] == _sha256(figure_path)
    assert "finite-sample" in manifest["caption"].lower()
    assert "equal-width" in manifest["caption"].lower()
