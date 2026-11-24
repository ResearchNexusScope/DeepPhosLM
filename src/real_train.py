"""Training utilities for the real homology-controlled phosphosite splits.

Scope
-----
This module trains the *existing* binary model from :mod:`src.model` on the
real train/val CSVs produced by :mod:`src.real_split` and evaluates with the
honest metrics in :mod:`src.metrics`.  No new architecture is introduced: the
one-hot encoder, the tokeniser, ``BinaryPhosphositeModel`` and ``ModelConfig``
are reused verbatim from :mod:`src.model`.

Selection rules
---------------
Each run trains with binary cross-entropy and keeps the model state from the
epoch with the best **validation** AUROC.  If a validation fold happens to hold
only one class, AUROC is undefined and validation BCE is used instead; the
chosen signal is recorded in the returned ``config``.  Test rows are predicted
only after the best state is restored and are **never** used for selection.

Cross-validation
----------------
:func:`cross_validate` performs k-fold CV on the **train split only**.  The
fold unit is ``cluster_id``: every cluster is assigned to exactly one fold, so
no homologous group is ever split between a fold's train and validation sets.

Cost and smoke checks
---------------------
On CPU the wall-clock cost grows roughly with ``seeds x ablations`` for the
held-out runs and with ``seeds x ablations x folds`` when CV is enabled.  The
mother session decides the actual run configuration (seed count, ablations,
epochs and whether CV is worth its cost); this module only implements the
mechanism.  The companion CLI ``tools/run_real_experiments.py`` exposes
``--limit-rows`` for fast smoke checks on truncated copies of the splits.
"""

from __future__ import annotations

import random
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import compute_metrics, summarize_folds  # noqa: E402
from src.model import (  # noqa: E402
    ABLATIONS,
    AMINO_ACID_ALPHABET,
    WINDOW_LENGTH,
    BinaryPhosphositeModel,
    ModelConfig,
    one_hot_encode,
    tokenize_sequences,
)
from src.real_split import sha256_file  # noqa: E402

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("sequence", "label", "site_type", "protein_acc", "cluster_id")
STRING_COLUMNS = ("sequence", "site_type", "protein_acc", "cluster_id")

# Exact keys written to each per-run predictions archive.
NPZ_KEYS = (
    "labels",
    "probabilities",
    "site_type",
    "protein_acc",
    "cluster_id",
    "seed",
    "ablation",
    "chosen_epoch",
    "config",
)

LEARNING_RATE = 1e-3
EPS = 1e-7
TASK = "binary phosphosite prediction (site vs non-site S/T/Y residue)"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducible CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_csv(path: Path) -> pd.DataFrame:
    """Load one real split CSV, validate its schema, and normalise its columns.

    String columns (notably ``cluster_id`` and ``protein_acc``) are kept as
    strings so identifiers such as ``A1X283`` or ``-`` are never coerced to
    numbers.  Labels must be binary.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required split CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    frame = frame.copy()
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy()
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError(f"{path} contains labels outside {{0, 1}}")
    frame["label"] = labels.astype(np.int64)
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype(str)
    return frame


def input_file_hashes(split_dir: Path) -> Dict[str, str]:
    """Return SHA-256 digests of the three input CSVs and the split manifest."""
    split_dir = Path(split_dir)
    hashes: Dict[str, str] = {}
    for split in SPLITS:
        csv_path = split_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Required split CSV not found: {csv_path}")
        hashes[f"{split}.csv"] = sha256_file(csv_path)
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    return {"csv": hashes, "manifest": sha256_file(manifest_path)}


def encode_rows(
    rows: pd.DataFrame, ablation: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Encode a frame into one-hot features and optional token ids."""
    if ablation not in ABLATIONS:
        raise ValueError(f"ablation must be one of {ABLATIONS}, got {ablation!r}")
    sequences = rows["sequence"].tolist()
    features = one_hot_encode(sequences)
    tokens = tokenize_sequences(sequences) if ablation == "seq_embedding" else None
    return features, tokens


def _make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    tokens: Optional[np.ndarray],
    batch_size: int,
    generator: torch.Generator,
) -> DataLoader:
    tensors = [torch.from_numpy(features)]
    if tokens is not None:
        tensors.append(torch.from_numpy(tokens))
    tensors.append(torch.from_numpy(labels))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def predict_probs(
    model: nn.Module,
    features: np.ndarray,
    tokens: Optional[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Return sigmoid probabilities for all rows, in input order."""
    model.eval()
    pieces: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            batch_tokens = None if tokens is None else torch.from_numpy(tokens[start:stop])
            probabilities = model(torch.from_numpy(features[start:stop]), batch_tokens)
            pieces.append(probabilities.numpy().astype(np.float64))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)


def _mean_bce(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )


def _selection_inputs(labels: np.ndarray, probabilities: np.ndarray):
    """Return ``(selection_metric, score)`` for one epoch.

    Which direction counts as an improvement is decided by the caller from
    whether the validation labels contain both classes; it is constant for
    the whole run, so it is not repeated per epoch here.
    """
    if len(np.unique(labels)) == 2:
        score = float(roc_auc_score(labels.astype(int), probabilities))
        return "validation_auroc", score
    return "validation_bce_fallback", _mean_bce(probabilities, labels)


def train_one_run(
    *,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    seed: int,
    ablation: str,
    epochs: int = 10,
    batch_size: int = 32,
    patience: int = 0,
    learning_rate: float = LEARNING_RATE,
    predict_rows: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Train one seeded run and return predictions from the best epoch.

    ``predict_rows`` defaults to ``val_rows``.  Passing the test frame here is
    safe: it is scored only after validation-based selection is finalised.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")
    for name, rows in (("train_rows", train_rows), ("val_rows", val_rows)):
        missing = [column for column in REQUIRED_COLUMNS if column not in rows.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
    target_rows = val_rows if predict_rows is None else predict_rows
    missing = [column for column in REQUIRED_COLUMNS if column not in target_rows.columns]
    if missing:
        raise ValueError(f"predict_rows is missing required columns: {missing}")

    model_config = ModelConfig(ablation=ablation)
    set_seed(int(seed))

    train_features, train_tokens = encode_rows(train_rows, ablation)
    val_features, val_tokens = encode_rows(val_rows, ablation)
    target_features, target_tokens = encode_rows(target_rows, ablation)

    y_train = train_rows["label"].to_numpy(dtype=np.float32)
    y_val = val_rows["label"].to_numpy(dtype=np.float32)

    model = BinaryPhosphositeModel(model_config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = _make_loader(
        train_features, y_train, train_tokens, batch_size, generator
    )

    val_is_binary = len(np.unique(y_val)) == 2
    selection_metric = "validation_auroc" if val_is_binary else "validation_bce_fallback"
    higher_is_better = val_is_binary
    best_score = -np.inf if higher_is_better else np.inf
    best_state = deepcopy(model.state_dict())
    best_epoch = 1
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            inputs = batch[0]
            batch_tokens = batch[1] if train_tokens is not None else None
            targets = batch[-1]
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(inputs, batch_tokens)
            loss = criterion(probabilities.clamp(EPS, 1.0 - EPS), targets)
            loss.backward()
            optimizer.step()

        val_probabilities = predict_probs(model, val_features, val_tokens, batch_size)
        _, score = _selection_inputs(y_val, val_probabilities)
        improved = score > best_score if higher_is_better else score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    probabilities = predict_probs(model, target_features, target_tokens, batch_size)

    config: Dict[str, Any] = {
        "task": TASK,
        "alphabet": AMINO_ACID_ALPHABET,
        "window_length": WINDOW_LENGTH,
        "ablation": model_config.ablation,
        "model": asdict(model_config),
        "epochs_requested": epochs,
        "epochs_run": best_epoch if patience else epochs,
        "batch_size": batch_size,
        "patience": patience,
        "learning_rate": learning_rate,
        "selection_metric": selection_metric,
        "best_selection_score": best_score,
        "chosen_epoch": best_epoch,
        "seed": int(seed),
        "n_train_rows": int(len(train_rows)),
        "n_val_rows": int(len(val_rows)),
        "n_predict_rows": int(len(target_rows)),
    }

    return {
        "probabilities": probabilities,
        "labels": target_rows["label"].to_numpy(dtype=np.int64),
        "site_type": target_rows["site_type"].astype(str).to_numpy(),
        "protein_acc": target_rows["protein_acc"].astype(str).to_numpy(),
        "cluster_id": target_rows["cluster_id"].astype(str).to_numpy(),
        "chosen_epoch": best_epoch,
        "config": config,
        "seed": int(seed),
        "ablation": model_config.ablation,
        "selection_metric": selection_metric,
        "best_score": best_score,
        "n_train_rows": int(len(train_rows)),
        "n_val_rows": int(len(val_rows)),
    }


def save_predictions(result: Mapping[str, Any], out_path: Path) -> Path:
    """Write the exact :data:`NPZ_KEYS` archive for one run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        labels=np.asarray(result["labels"], dtype=np.int64),
        probabilities=np.asarray(result["probabilities"], dtype=np.float64),
        site_type=np.asarray(result["site_type"], dtype=str),
        protein_acc=np.asarray(result["protein_acc"], dtype=str),
        cluster_id=np.asarray(result["cluster_id"], dtype=str),
        seed=np.int64(result["seed"]),
        ablation=np.asarray(result["ablation"], dtype=str),
        chosen_epoch=np.int64(result["chosen_epoch"]),
        config=np.asarray(dict(result["config"]), dtype=object),
    )
    return out_path


def cluster_cv_folds(
    train_rows: pd.DataFrame, n_folds: int, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Partition TRAIN rows into k folds whose unit is the whole cluster.

    Returns ``(folds, assignments)`` where each fold carries disjoint train /
    validation frames and ``assignments`` maps every cluster id to its fold.
    Because the partition is over ``cluster_id`` values, a cluster can never
    appear on both sides of one fold or in two different folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds!r}")
    clusters = sorted(set(train_rows["cluster_id"].astype(str)))
    if n_folds > len(clusters):
        raise ValueError(
            f"n_folds={n_folds} exceeds the {len(clusters)} distinct clusters in train"
        )

    rng = np.random.default_rng(int(seed))
    shuffled = [clusters[index] for index in rng.permutation(len(clusters))]
    base, remainder = divmod(len(shuffled), n_folds)
    sizes = [base + (1 if fold < remainder else 0) for fold in range(n_folds)]
    fold_clusters: List[List[str]] = []
    cursor = 0
    for size in sizes:
        fold_clusters.append(shuffled[cursor : cursor + size])
        cursor += size

    assignments: Dict[str, int] = {
        cluster: fold for fold, members in enumerate(fold_clusters) for cluster in members
    }
    cluster_column = train_rows["cluster_id"].astype(str)
    all_clusters = set(clusters)
    folds: List[Dict[str, Any]] = []
    for fold, members in enumerate(fold_clusters):
        val_ids = set(members)
        val_mask = cluster_column.isin(val_ids).to_numpy()
        folds.append(
            {
                "fold": fold,
                "val_clusters": sorted(val_ids),
                "train_clusters": sorted(all_clusters - val_ids),
                "n_val_clusters": len(val_ids),
                "n_train_clusters": len(all_clusters) - len(val_ids),
                "train_rows": train_rows.loc[~val_mask].reset_index(drop=True),
                "val_rows": train_rows.loc[val_mask].reset_index(drop=True),
            }
        )
    return folds, assignments


def cross_validate(
    *,
    train_rows: pd.DataFrame,
    n_folds: int,
    seed: int,
    ablation: str,
    epochs: int = 10,
    batch_size: int = 32,
    patience: int = 0,
    learning_rate: float = LEARNING_RATE,
) -> Dict[str, Any]:
    """Run whole-cluster k-fold CV on train rows only.

    Every fold's validation probabilities are scored with
    :func:`src.metrics.compute_metrics`; the full per-fold vectors and a
    mean/std summary (from ``summarize_folds``) are returned.  No test row is
    touched.
    """
    folds, assignments = cluster_cv_folds(train_rows, n_folds, seed)
    fold_records: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    for fold in folds:
        run = train_one_run(
            train_rows=fold["train_rows"],
            val_rows=fold["val_rows"],
            predict_rows=fold["val_rows"],
            seed=seed,
            ablation=ablation,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            learning_rate=learning_rate,
        )
        metrics = compute_metrics(
            run["labels"], run["probabilities"], site_types=fold["val_rows"]["site_type"].tolist()
        )
        fold_metrics.append(metrics)
        fold_records.append(
            {
                "fold": fold["fold"],
                "n_train_rows": int(len(fold["train_rows"])),
                "n_val_rows": int(len(fold["val_rows"])),
                "val_clusters": fold["val_clusters"],
                "chosen_epoch": run["chosen_epoch"],
                "selection_metric": run["selection_metric"],
                "config": run["config"],
                "metrics": metrics,
            }
        )
    return {
        "n_folds": int(n_folds),
        "seed": int(seed),
        "ablation": ablation,
        "fold_assignments": assignments,
        "folds": fold_records,
        "summary": summarize_folds(fold_metrics),
    }


__all__ = [
    "ABLATIONS",
    "EPS",
    "LEARNING_RATE",
    "NPZ_KEYS",
    "REQUIRED_COLUMNS",
    "SPLITS",
    "cluster_cv_folds",
    "cross_validate",
    "encode_rows",
    "input_file_hashes",
    "load_split_csv",
    "predict_probs",
    "save_predictions",
    "set_seed",
    "train_one_run",
]
