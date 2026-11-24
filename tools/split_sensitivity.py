#!/usr/bin/env python3
"""Homology-threshold sensitivity check for the real-data split (X-002).

Usage (from anywhere)::

    python tools/split_sensitivity.py
    python tools/split_sensitivity.py --data-dir data/real \
        --mmseqs tools/bin/mmseqs/bin/mmseqs.exe \
        --thresholds 0.2 0.3 0.4 --seeds 666 --epochs 6 --batch-size 1024 \
        --ablation both --out-dir results/real/sensitivity \
        --work-root data/real/sensitivity_work

What it does
------------
For every requested MMseqs ``--min-seq-id`` threshold it

1. calls the existing split CLI (:mod:`tools.make_real_split`) with a *distinct*
   work directory, writing to ``<data-dir>/split_t<ID>``;
2. reads that split's ``manifest.json`` for cluster / singleton / per-split
   window counts and for its explicit no-overlap assertions;
3. calls the existing training CLI (:mod:`tools.run_real_experiments`) with the
   requested seeds and ablations into ``<out-dir>/t<ID>`` (folds disabled);
4. recomputes test metrics with :func:`src.metrics.compute_metrics` directly
   from the saved ``predictions_<ablation>_seed<N>.npz`` archives; and
5. appends one honest, provenance-carrying entry to
   ``<out-dir>/sensitivity_report.json`` and prints a compact table.

The canonical 0.3 threshold is treated like every other threshold, never
special-cased or dropped; a ``canonical_threshold`` field is recorded so the
choice can be inspected.  Thresholds are deduplicated and sorted ascending so
the report ordering is deterministic.  The CLI exits non-zero if any threshold
is ``FAIL`` (e.g. an overlap assertion), so a partial sweep is never mistaken
for a clean success.

Injectable runner
-----------------
``run_sensitivity`` and ``main`` accept a ``runner`` exposing two methods::

    runner.build_split(argv, spec) -> None
    runner.run_training(argv, spec) -> None

The default :class:`SensitivityRunner` executes the two existing CLIs via their
``main(argv)`` entry points.  Tests inject a fake runner that writes a tiny
synthetic manifest and synthetic npz archives; no real MMseqs binary and no real
training is ever invoked by the test suite.  The tool itself still reads the
manifest and npz files from disk and does all metric computation, so the fake
runner cannot fabricate reported numbers.

Honest scope
------------
**One seed x six epochs per threshold is a sensitivity check, not a definitive
performance estimate.**  A single short run is noisy; it cannot rank thresholds
with confidence and must not be quoted as a final score.  MMseqs clustering at
lower identity is a *heuristic* (set-cover, ``--cluster-mode 0``): "no homology
crosses splits" only means no cluster reported by these exact parameters
crosses, and distant relationships below the requested identity are not
detected.  Lower thresholds do not guarantee a cleaner split.  All metric
values are computed from real prediction archives; nothing is hard-coded or
cherry-picked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.metrics import compute_metrics  # noqa: E402
from src.real_split import (  # noqa: E402
    SPLITS as SPLIT_NAMES,
    OverlapError,
    default_mmseqs_path,
)

DEFAULT_DATA_DIR = "data/real"
DEFAULT_OUT_DIR = "results/real/sensitivity"
DEFAULT_WORK_ROOT = "data/real/sensitivity_work"
DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.2, 0.3, 0.4)
DEFAULT_SEEDS: Tuple[int, ...] = (666,)
DEFAULT_EPOCHS = 6
DEFAULT_BATCH_SIZE = 1024
DEFAULT_ABLATION = "both"
DEFAULT_THREADS = 8
CANONICAL_THRESHOLD = 0.3
ABLATION_CHOICES = ("seq_only", "seq_embedding", "both")
DUAL_ABLATIONS: Tuple[str, str] = ("seq_only", "seq_embedding")
REPORT_NAME = "sensitivity_report.json"
METRICS_SOURCE = "src.metrics.compute_metrics"
REPORTED_METRICS = ("roc_auc", "pr_auc", "f1", "mcc")


# ---------------------------------------------------------------------------
# Path / tag helpers
# ---------------------------------------------------------------------------


def _resolve_input(path: Path) -> Path:
    """Resolve a relative input from cwd first, then the repository root."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def _resolve_output(path: Path, default: str) -> Path:
    """Resolve an output path; the shipped default lives at the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if str(candidate).replace("\\", "/") == default:
        return REPO_ROOT / candidate
    return candidate


def threshold_tag(threshold: float) -> str:
    """Return the stable directory tag for a threshold (0.2 -> ``t020``)."""
    value = float(threshold)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"threshold must lie in (0, 1], got {threshold!r}")
    return "t{:03d}".format(int(round(value * 100.0)))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ablation_names(ablation: str) -> Tuple[str, ...]:
    """Mirror ``run_real_experiments``: ``both`` means the two single-feature runs."""
    if ablation == "both":
        return DUAL_ABLATIONS
    if ablation not in ABLATION_CHOICES:
        raise ValueError(f"unknown ablation {ablation!r}; expected {ABLATION_CHOICES}")
    return (ablation,)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Resolved, validated configuration shared by every threshold."""

    data_dir: Path
    mmseqs: Path
    work_root: Path
    out_dir: Path
    thresholds: Tuple[float, ...]
    seeds: Tuple[int, ...]
    epochs: int
    batch_size: int
    ablation: str
    threads: int

    @property
    def split_seed(self) -> int:
        """The first requested seed fixes the whole-cluster split assignment."""
        return self.seeds[0]


@dataclass(frozen=True)
class ThresholdSpec:
    """Per-threshold paths, derived deterministically from :class:`Settings`."""

    threshold: float
    settings: Settings

    @property
    def tag(self) -> str:
        return threshold_tag(self.threshold)

    @property
    def split_dir(self) -> Path:
        return self.settings.data_dir / f"split_{self.tag}"

    @property
    def work_dir(self) -> Path:
        return self.settings.work_root / f"work_{self.tag}"

    @property
    def train_dir(self) -> Path:
        return self.settings.out_dir / self.tag


def build_split_argv(spec: ThresholdSpec) -> List[str]:
    """Exact argv handed to ``tools/make_real_split.py`` for this threshold."""
    settings = spec.settings
    return [
        "--data-dir", str(settings.data_dir),
        "--mmseqs", str(settings.mmseqs),
        "--work-dir", str(spec.work_dir),
        "--out-dir", str(spec.split_dir),
        "--min-seq-id", str(spec.threshold),
        "--seed", str(settings.split_seed),
        "--threads", str(settings.threads),
    ]


def build_train_argv(spec: ThresholdSpec, seed: int) -> List[str]:
    """Exact argv handed to ``tools/run_real_experiments.py`` for one seed."""
    settings = spec.settings
    return [
        "--split-dir", str(spec.split_dir),
        "--out-dir", str(spec.train_dir),
        "--seeds", "1",
        "--seed-base", str(seed),
        "--folds", "0",
        "--epochs", str(settings.epochs),
        "--batch-size", str(settings.batch_size),
        "--ablation", str(settings.ablation),
    ]


# ---------------------------------------------------------------------------
# Injectable runner
# ---------------------------------------------------------------------------


class SensitivityRunner:
    """Default runner: delegate to the two existing, unmodified CLIs.

    Imports are deliberately lazy so importing this module (and running the
    offline tests) never loads torch or touches the MMseqs binary.
    """

    def build_split(self, argv: Sequence[str], spec: ThresholdSpec) -> None:
        from tools.make_real_split import main as make_split_main

        if make_split_main(list(argv)) != 0:
            raise RuntimeError(f"split CLI returned non-zero for {spec.threshold}")

    def run_training(self, argv: Sequence[str], spec: ThresholdSpec) -> None:
        from tools.run_real_experiments import main as run_experiments_main

        if run_experiments_main(list(argv)) != 0:
            raise RuntimeError(
                f"training CLI returned non-zero for {spec.threshold}: {list(argv)}"
            )


# ---------------------------------------------------------------------------
# Per-threshold orchestration
# ---------------------------------------------------------------------------


def _fail(entry: Dict[str, Any], message: str) -> Dict[str, Any]:
    entry["status"] = "FAIL"
    entry["failures"].append(message)
    return entry


def _read_split_manifest(spec: ThresholdSpec, entry: Dict[str, Any]) -> Mapping[str, Any]:
    """Read and hash the produced manifest, then copy its truthful counts."""
    manifest_path = spec.split_dir / "manifest.json"
    entry["provenance"]["split_manifest_path"] = str(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(manifest_path)
    entry["provenance"]["split_manifest_sha256"] = digest

    # cluster / protein / singleton counts are taken verbatim from the manifest.
    entry["cluster_counts"] = {
        "n_proteins": manifest.get("n_proteins"),
        "n_clusters": manifest.get("n_clusters"),
    }
    entry["singleton_counts"] = {
        "n_singletons": manifest.get("n_singletons"),
        "n_singleton_clusters": manifest.get("n_singleton_clusters"),
    }
    windows: Dict[str, Any] = {}
    proteins: Dict[str, Any] = {}
    clusters: Dict[str, Any] = {}
    positives: Dict[str, Any] = {}
    negatives: Dict[str, Any] = {}
    for split in SPLIT_NAMES:
        info = manifest.get("splits", {}).get(split, {})
        windows[split] = info.get("windows")
        proteins[split] = info.get("proteins")
        clusters[split] = info.get("clusters")
        positives[split] = info.get("positives")
        negatives[split] = info.get("negatives")
    entry["split_window_counts"] = windows
    entry["split_protein_counts"] = proteins
    entry["split_cluster_counts"] = clusters
    entry["split_positive_counts"] = positives
    entry["split_negative_counts"] = negatives

    # Propagate the no-overlap assertions verbatim; a false value or missing
    # block is a hard failure, never a silently included threshold.
    checks = manifest.get("no_overlap")
    entry["no_overlap"] = checks
    entry["overlap_ok"] = bool(
        isinstance(checks, dict) and checks and all(bool(value) for value in checks.values())
    )
    return manifest


def _read_npz_run(
    npz_path: Path, seed: int, argv: Sequence[str], spec: ThresholdSpec
) -> Dict[str, Any]:
    """Recompute test metrics from one saved archive; never invent a number."""
    with np.load(npz_path, allow_pickle=True) as data:
        metrics = compute_metrics(
            data["labels"], data["probabilities"], site_types=data["site_type"]
        )
        chosen_epoch = int(data["chosen_epoch"])
    return {
        "seed": int(seed),
        "epochs": int(spec.settings.epochs),
        "batch_size": int(spec.settings.batch_size),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "chosen_epoch": chosen_epoch,
        "test_metrics": metrics,
        "test": {name: metrics.get(name) for name in REPORTED_METRICS},
        "train_argv": list(argv),
    }


def _run_threshold(spec: ThresholdSpec, runner: Any) -> Dict[str, Any]:
    settings = spec.settings
    split_argv = build_split_argv(spec)
    entry: Dict[str, Any] = {
        "threshold": float(spec.threshold),
        "tag": spec.tag,
        "status": "OK",
        "failures": [],
        "split_dir": str(spec.split_dir),
        "work_dir": str(spec.work_dir),
        "train_dir": str(spec.train_dir),
        "cluster_counts": None,
        "singleton_counts": None,
        "split_window_counts": None,
        "split_protein_counts": None,
        "split_cluster_counts": None,
        "split_positive_counts": None,
        "split_negative_counts": None,
        "no_overlap": None,
        "overlap_ok": None,
        "ablations": {},
        "provenance": {
            "split_argv": split_argv,
            "train_argv": [],
            "split_manifest_path": None,
            "split_manifest_sha256": None,
            "npz_sha256": {},
            "seed": int(settings.split_seed),
            "seeds": [int(seed) for seed in settings.seeds],
            "epochs": int(settings.epochs),
            "batch_size": int(settings.batch_size),
            "ablation": str(settings.ablation),
            "threads": int(settings.threads),
            "folds": 0,
            "flags": {
                "data_dir": str(settings.data_dir),
                "mmseqs": str(settings.mmseqs),
                "thresholds": sorted({float(value) for value in settings.thresholds}),
                "seeds": [int(seed) for seed in settings.seeds],
                "epochs": int(settings.epochs),
                "batch_size": int(settings.batch_size),
                "ablation": str(settings.ablation),
                "work_root": str(settings.work_root),
                "out_dir": str(settings.out_dir),
            },
            "metrics_source": METRICS_SOURCE,
        },
    }

    # Stage 1: split.  An overlap assertion failure raises before any CSV or
    # manifest is written; record it as FAIL and do not train on this threshold.
    try:
        runner.build_split(split_argv, spec)
    except OverlapError as exc:
        return _fail(entry, f"split stage raised OverlapError: {exc}")
    except Exception as exc:  # any other split failure must be visible too
        return _fail(
            entry,
            f"split stage raised {type(exc).__name__}: {exc}",
        )

    try:
        _read_split_manifest(spec, entry)
    except Exception as exc:
        return _fail(entry, f"manifest stage raised {type(exc).__name__}: {exc}")

    if not entry["overlap_ok"]:
        return _fail(
            entry,
            f"no-overlap assertions did not all pass for threshold {spec.threshold}: "
            f"{json.dumps(entry['no_overlap'], sort_keys=True)}",
        )

    # Stage 2: training (one short run per requested seed), then honest metrics.
    for seed in settings.seeds:
        train_argv = build_train_argv(spec, int(seed))
        entry["provenance"]["train_argv"].append(train_argv)
        try:
            runner.run_training(train_argv, spec)
        except Exception as exc:
            _fail(
                entry,
                f"training stage raised {type(exc).__name__} for seed {seed}: {exc}",
            )
            continue

        for ablation in ablation_names(settings.ablation):
            npz_path = spec.train_dir / f"predictions_{ablation}_seed{int(seed)}.npz"
            if not npz_path.is_file():
                _fail(entry, f"missing predictions archive: {npz_path}")
                continue
            try:
                run = _read_npz_run(npz_path, int(seed), train_argv, spec)
            except Exception as exc:
                _fail(
                    entry,
                    f"metrics stage raised {type(exc).__name__} for {npz_path}: {exc}",
                )
                continue
            entry["ablations"].setdefault(ablation, {"runs": []})["runs"].append(run)
            entry["provenance"]["npz_sha256"][f"{ablation}_seed{int(seed)}"] = run["npz_sha256"]

    return entry


# ---------------------------------------------------------------------------
# Report + CLI
# ---------------------------------------------------------------------------


def print_table(report: Mapping[str, Any]) -> None:
    """Print a compact, faithful summary; FAIL rows keep their reason visible."""
    header = (
        f"{'threshold':>9}  {'status':<6}  {'clusters':>8}  {'singletons':>10}  "
        f"{'windows train/val/test':>26}  metrics (ROC-AUC / PR-AUC / F1 / MCC)"
    )
    print(header)
    print("-" * len(header))
    for entry in report["thresholds"]:
        counts = entry["cluster_counts"] or {}
        singles = entry["singleton_counts"] or {}
        windows = entry["split_window_counts"] or {}
        window_text = "/".join(str(windows.get(split)) for split in SPLIT_NAMES)
        bits: List[str] = []
        for ablation, block in sorted(entry["ablations"].items()):
            for run in block["runs"]:
                test = run["test"]
                fmt = lambda name: "n/a" if test.get(name) is None else f"{test[name]:.3f}"
                bits.append(
                    f"{ablation}@{run['seed']}: "
                    + "/".join(fmt(name) for name in REPORTED_METRICS)
                )
        detail = "; ".join(bits) if bits else (entry["failures"][0] if entry["failures"] else "")
        print(
            f"{entry['threshold']:>9.2f}  {entry['status']:<6}  "
            f"{str(counts.get('n_clusters')):>8}  {str(singles.get('n_singletons')):>10}  "
            f"{window_text:>26}  {detail}"
        )
    summary = report["summary"]
    print(
        f"\n{summary['n_thresholds']} threshold(s): "
        f"{summary['n_ok']} OK, {summary['n_failed']} FAIL. "
        f"Report: {report['report_path']}"
    )
    print("NOTE: 1 seed x 6 epochs is a sensitivity check, not a definitive estimate.")


def run_sensitivity(
    settings: Settings,
    *,
    runner: Optional[Any] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every threshold and write the JSON report; return it in memory."""
    runner = runner if runner is not None else SensitivityRunner()
    report_path = Path(report_path) if report_path is not None else settings.out_dir / REPORT_NAME

    thresholds = tuple(sorted({float(value) for value in settings.thresholds}))
    entries = [
        _run_threshold(ThresholdSpec(threshold, settings), runner)
        for threshold in thresholds
    ]
    n_ok = sum(1 for entry in entries if entry["status"] == "OK")
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "data_dir": str(settings.data_dir),
        "mmseqs": str(settings.mmseqs),
        "canonical_threshold": CANONICAL_THRESHOLD,
        "canonical_threshold_processed": CANONICAL_THRESHOLD in thresholds,
        "thresholds_requested": [float(value) for value in settings.thresholds],
        "thresholds_processed": list(thresholds),
        "seeds": [int(seed) for seed in settings.seeds],
        "epochs": int(settings.epochs),
        "batch_size": int(settings.batch_size),
        "ablation": str(settings.ablation),
        "metrics_source": METRICS_SOURCE,
        "reported_metrics": list(REPORTED_METRICS),
        "summary": {
            "n_thresholds": len(entries),
            "n_ok": n_ok,
            "n_failed": len(entries) - n_ok,
        },
        "caveats": [
            "One seed x six epochs per threshold is a sensitivity check, not a "
            "definitive performance estimate; a single short run is noisy.",
            "MMseqs clustering at lower identity is a heuristic (set-cover, "
            "--cluster-mode 0); only clusters reported by these exact parameters "
            "are guaranteed not to cross splits.",
            "The split is by cluster count, not window count; per-split window "
            "counts are read from the manifest and reported verbatim.",
            "All thresholds are reported, including the canonical 0.3; nothing "
            "is dropped or cherry-picked after seeing the metrics.",
        ],
        "thresholds": entries,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print_table(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep homology identity thresholds, retrain briefly at each one, "
            "and write a provenance-carrying sensitivity_report.json."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR),
        help=f"Directory holding data.csv and fasta_cache/ (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--mmseqs", type=Path, default=default_mmseqs_path(),
        help="Path to the MMseqs binary (default: tools/bin/mmseqs/bin/mmseqs.exe).",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        help=f"MMseqs --min-seq-id values to compare (default: {DEFAULT_THRESHOLDS}).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help=f"Actual seed values used for training and split assignment (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Epochs per training run (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--ablation", choices=list(ABLATION_CHOICES), default=DEFAULT_ABLATION,
        help=f"Architecture variant(s) to train (default: {DEFAULT_ABLATION}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR),
        help=f"Report and per-threshold prediction directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--work-root", type=Path, default=Path(DEFAULT_WORK_ROOT),
        help=f"Root for per-threshold MMseqs scratch dirs (default: {DEFAULT_WORK_ROOT}).",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"Fixed MMseqs thread count (default: {DEFAULT_THREADS}).",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    if not args.thresholds:
        raise ValueError("--thresholds needs at least one value")
    if any(not 0.0 < float(value) <= 1.0 for value in args.thresholds):
        raise ValueError("--thresholds must all lie in (0, 1]")
    if not args.seeds:
        raise ValueError("--seeds needs at least one value")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.threads < 1:
        raise ValueError(f"--threads must be >= 1, got {args.threads}")


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Any] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate(args)
    settings = Settings(
        data_dir=_resolve_input(Path(args.data_dir)),
        mmseqs=_resolve_input(Path(args.mmseqs)),
        work_root=_resolve_output(Path(args.work_root), DEFAULT_WORK_ROOT),
        out_dir=_resolve_output(Path(args.out_dir), DEFAULT_OUT_DIR),
        thresholds=tuple(float(value) for value in args.thresholds),
        seeds=tuple(int(value) for value in args.seeds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        ablation=str(args.ablation),
        threads=int(args.threads),
    )
    report = run_sensitivity(settings, runner=runner)
    return 0 if report["summary"]["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
