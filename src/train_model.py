"""Train the binary phosphosite classifier and save honest test predictions.

The task is binary phosphosite prediction: for each centred S/T/Y residue in a
15-residue window, decide site (label 1) versus non-site residue (label 0).
Features and the model live in :mod:`src.model`; this module only reads the
three generated CSV splits, trains with BCE loss, selects the epoch with the
best validation AUROC, and writes test predictions to an ``.npz`` plus a small
JSON config. AUROC from scikit-learn is used solely for model selection; the
saved file contains no hard-coded performance claims.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import (  # noqa: E402
    ABLATIONS,
    AMINO_ACID_ALPHABET,
    WINDOW_LENGTH,
    BinaryPhosphositeModel,
    ModelConfig,
    one_hot_encode,
    tokenize_sequences,
)

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("sequence", "label", "site_type", "kinase", "cluster_id")
DEFAULT_DATA_DIR = "data/honest"

# Exact set of keys written to the predictions archive.
NPZ_KEYS = (
    "labels",
    "probabilities",
    "site_type",
    "kinase",
    "cluster_id",
    "chosen_epoch",
    "config",
    "seed",
)

LEARNING_RATE = 1e-3
EPS = 1e-7


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch so repeated runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_dir(data_dir) -> Path:
    """Resolve a data directory relative to cwd, then to the repo root."""
    path = Path(data_dir)
    if not path.is_absolute() and not path.is_dir():
        candidate = REPO_ROOT / path
        if candidate.is_dir():
            return candidate
    return path


def read_split(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Read one split CSV and validate its schema and binary labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Required split CSV not found: {path}")
    frame = pd.read_csv(
        path,
        dtype={column: str for column in REQUIRED_COLUMNS if column != "label"},
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    labels = frame["label"].to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{path} contains labels outside {{0, 1}}")
    return frame, labels.astype(np.float32)


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
    pieces = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            batch_tokens = None
            if tokens is not None:
                batch_tokens = torch.from_numpy(tokens[start:stop])
            probabilities = model(torch.from_numpy(features[start:stop]), batch_tokens)
            pieces.append(probabilities.numpy().astype(np.float64))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)


def _mean_bce(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )


def train_and_predict(
    *,
    data_dir=DEFAULT_DATA_DIR,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 666,
    ablation: str = "seq_only",
    out=None,
    patience: int = 5,
) -> Dict[str, object]:
    """Train with a fixed seed and write test predictions to ``out``.

    The epoch with the highest validation AUROC is restored before predicting
    on the test split. If a validation split happens to contain only one class,
    AUROC is mathematically undefined; validation BCE is then used as the
    selection signal and the choice is recorded in the saved config.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience!r}")

    model_config = ModelConfig(ablation=ablation)
    set_seed(seed)

    base = resolve_data_dir(data_dir)
    frames, labels = {}, {}
    for split in SPLITS:
        frames[split], labels[split] = read_split(base / f"{split}.csv")

    features = {
        split: one_hot_encode(frames[split]["sequence"].tolist()) for split in SPLITS
    }
    tokens = None
    if model_config.ablation == "seq_embedding":
        tokens = {
            split: tokenize_sequences(frames[split]["sequence"].tolist())
            for split in SPLITS
        }

    def split_tokens(split: str) -> Optional[np.ndarray]:
        return None if tokens is None else tokens[split]

    model = BinaryPhosphositeModel(model_config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        features["train"],
        labels["train"],
        split_tokens("train"),
        batch_size,
        generator,
    )

    y_val = labels["val"]
    val_is_binary = len(np.unique(y_val)) == 2
    selection_metric = "validation_auroc" if val_is_binary else "validation_bce_fallback"
    best_score = -np.inf if val_is_binary else np.inf
    best_state = deepcopy(model.state_dict())
    best_epoch = 1
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            inputs = batch[0]
            batch_tokens = batch[1] if tokens is not None else None
            targets = batch[-1]
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(inputs, batch_tokens)
            loss = criterion(probabilities.clamp(EPS, 1.0 - EPS), targets)
            loss.backward()
            optimizer.step()

        val_probabilities = predict_probs(
            model, features["val"], split_tokens("val"), batch_size
        )
        if val_is_binary:
            score = float(roc_auc_score(y_val.astype(int), val_probabilities))
            improved = score > best_score
        else:
            score = _mean_bce(val_probabilities, y_val)
            improved = score < best_score

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
    test_probabilities = predict_probs(
        model, features["test"], split_tokens("test"), batch_size
    )

    config = {
        "task": "binary phosphosite prediction (site vs non-site S/T/Y residue)",
        "alphabet": AMINO_ACID_ALPHABET,
        "window_length": WINDOW_LENGTH,
        "ablation": model_config.ablation,
        "model": asdict(model_config),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "selection_metric": selection_metric,
        "chosen_epoch": best_epoch,
        "seed": seed,
        "data_dir": str(base),
    }

    npz_path = Path(out) if out is not None else (
        Path("results") / f"predictions_{model_config.ablation}.npz"
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        labels=labels["test"].astype(np.int64),
        probabilities=test_probabilities.astype(np.float64),
        site_type=frames["test"]["site_type"].to_numpy(dtype=str),
        kinase=frames["test"]["kinase"].to_numpy(dtype=str),
        cluster_id=frames["test"]["cluster_id"].to_numpy(dtype=str),
        chosen_epoch=np.int64(best_epoch),
        config=np.array(config, dtype=object),
        seed=np.int64(seed),
    )
    json_path = npz_path.with_suffix(".json")
    json_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "labels": labels["test"].astype(np.int64),
        "probabilities": test_probabilities,
        "site_type": frames["test"]["site_type"].to_numpy(dtype=str),
        "kinase": frames["test"]["kinase"].to_numpy(dtype=str),
        "cluster_id": frames["test"]["cluster_id"].to_numpy(dtype=str),
        "chosen_epoch": best_epoch,
        "config": config,
        "seed": seed,
        "npz_path": npz_path,
        "json_path": json_path,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary phosphosite classifier and save test predictions."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory with train/val/test CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="seq_only")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz path (default: results/predictions_<ablation>.npz)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after this many non-improving epochs; 0 disables early stopping",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    npz_path = (
        Path(args.out)
        if args.out is not None
        else Path("results") / f"predictions_{args.ablation}.npz"
    )
    result = train_and_predict(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        ablation=args.ablation,
        out=npz_path,
        patience=args.patience,
    )
    print(
        json.dumps(
            {
                "predictions": str(result["npz_path"]),
                "config": str(result["json_path"]),
                "chosen_epoch": result["chosen_epoch"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
