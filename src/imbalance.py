"""Honest class-imbalance handling for the real phosphosite binary task.

What this fix is -- and what it is not
--------------------------------------
The real homology-controlled split contains more labelled sites than labelled
non-site S/T/Y residues (train: 103,534 positives / 84,869 negatives; test:
11,482 / 9,867, in the pinned split this code reads at runtime).  The negative
pool is limited because most S/T/Y residues in these proteins are themselves
annotated as phosphorylation sites.

The fix implemented here is **statistical**, not synthetic:

* report the exact class counts and the prevalence on every split;
* report the prevalence-only baseline (a constant positive score) with its
  ROC-AUC / AUPRC interpretation, so any model number has an honest floor;
* train the *existing* binary model with ``BCEWithLogitsLoss`` and
  ``pos_weight = n_negative / n_positive`` computed from TRAIN labels only.

No negative is invented, relabelled, duplicated, sampled from another protein,
or dropped.  No positive is dropped.  A literal 1:1 ratio is impossible from
same-protein negatives alone: the number of unannotated S/T/Y residues inside
the same annotated proteins is a fixed property of the annotation.  Reaching
1:1 would require either fabricating negatives outside the supplied rows or
discarding real positives.  This module does neither, and says so explicitly
in :func:`imbalance_report`.

Reuse
-----
The model, encoders, seeding, CSV loading, data loader, prediction helper and
validation-selection helper come from :mod:`src.model` and :mod:`src.real_train`
without modification.  The only difference from ``src.real_train.train_one_run``
is the loss: unweighted ``BCELoss`` on sigmoid outputs becomes positive-weighted
``BCEWithLogitsLoss`` on raw logits (mathematically equivalent to reweighting
the positive term, numerically stable, sigmoid applied only for reporting).
Model selection still uses validation AUROC (validation BCE fallback when a
validation fold has one class), and test rows are scored only after the best
state is restored.  No metric value is hardcoded.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import DEFAULT_THRESHOLD, compute_metrics  # noqa: E402
from src.model import (  # noqa: E402
    AMINO_ACID_ALPHABET,
    WINDOW_LENGTH,
    BinaryPhosphositeModel,
    ModelConfig,
)
from src.real_train import (  # noqa: E402
    LEARNING_RATE,
    NPZ_KEYS,
    REQUIRED_COLUMNS,
    _make_loader,
    _selection_inputs,
    encode_rows,
    load_split_csv,
    predict_probs,
    set_seed,
)

# E-003 keys plus the single new scalar: the weighting used by this run.
BALANCED_NPZ_KEYS = NPZ_KEYS + ("pos_weight",)
LOSS_DESCRIPTION = "torch.nn.BCEWithLogitsLoss(pos_weight=n_train_negative/n_train_positive)"
FIX_DESCRIPTION = (
    "statistical fix only: positive-class weighting plus honest prevalence "
    "reporting; no synthetic negatives, no resampling, no dropped rows"
)


def _as_binary_labels(labels: Sequence[Any]) -> np.ndarray:
    """Validate a 1-D binary label vector and return it as int64."""
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("labels must not be empty")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError("labels must contain only binary values 0 and 1")
    return array.astype(np.int64)


def class_counts(labels: Sequence[Any]) -> Dict[str, Any]:
    """Return exact class counts and prevalence for one set of labels."""
    array = _as_binary_labels(labels)
    n_positive = int(np.count_nonzero(array == 1))
    n_negative = int(np.count_nonzero(array == 0))
    n = int(array.size)
    return {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": n_positive / n,
        "n_negative_per_positive": (n_negative / n_positive) if n_positive else None,
        "n_positive_per_negative": (n_positive / n_negative) if n_negative else None,
    }


def compute_pos_weight(labels: Sequence[Any]) -> float:
    """Return ``pos_weight = n_negative / n_positive`` for ``BCEWithLogitsLoss``.

    The weight is computed from the labels passed in, which the CLI always
    takes from ``train.csv`` only.  It is a plain ratio of observed counts, not
    a tuned hyperparameter and not a claim about the test split.
    """
    counts = class_counts(labels)
    if counts["n_positive"] == 0 or counts["n_negative"] == 0:
        raise ValueError(
            "pos_weight is undefined when train labels are single-class "
            f"(n_positive={counts['n_positive']}, n_negative={counts['n_negative']})"
        )
    return float(counts["n_negative"] / counts["n_positive"])


# Explicit alias for callers that prefer the torch-facing name.
positive_class_weight = compute_pos_weight


def prevalence_baseline(
    labels: Sequence[Any], *, threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """Score the no-skill baseline that predicts the constant train prevalence.

    The constant score has no ranking information: its ROC-AUC is 0.5 and its
    average precision (AUPRC) equals the positive prevalence whenever both
    classes are present.  Both expectations are derived from the observed
    counts; the reported metrics themselves are recomputed by
    :func:`src.metrics.compute_metrics`, never hardcoded.  With a single-class
    input, ``src.metrics`` reports undefined discrimination metrics as ``None``.
    """
    array = _as_binary_labels(labels)
    counts = class_counts(array)
    constant = np.full(array.size, counts["prevalence"], dtype=float)
    metrics = compute_metrics(array, constant, threshold=threshold)
    has_both_classes = counts["n_positive"] > 0 and counts["n_negative"] > 0
    interpretation: Dict[str, Any] = {
        "prediction_rule": "predict the observed positive prevalence for every row",
        "roc_auc": (
            "A constant score cannot order positives above negatives, so ROC-AUC "
            "is 0.5 by construction (undefined/None for a single-class input)."
        ),
        "pr_auc": (
            "With all scores tied, average precision equals the positive "
            "prevalence; this is the no-skill AUPRC floor a useful model must beat."
        ),
        "accuracy": (
            "A constant at the decision threshold predicts one class everywhere; "
            "accuracy then equals the majority-class share and is not evidence of "
            "discrimination."
        ),
        "expected_roc_auc": 0.5 if has_both_classes else None,
        "expected_pr_auc": counts["prevalence"] if has_both_classes else None,
        "expected_accuracy": (
            counts["prevalence"]
            if counts["prevalence"] >= threshold
            else 1.0 - counts["prevalence"]
        ),
    }
    return {
        "counts": counts,
        "constant_probability": counts["prevalence"],
        "threshold": float(threshold),
        "metrics": metrics,
        "interpretation": interpretation,
    }


def why_one_to_one_impossible(counts: Mapping[str, Any]) -> str:
    """Explain, with this split's exact counts, why 1:1 is not attainable."""
    n_positive = int(counts["n_positive"])
    n_negative = int(counts["n_negative"])
    if n_negative < n_positive:
        gap = n_positive - n_negative
        return (
            "A 1:1 ratio cannot be obtained from same-protein negatives: negatives "
            "are only S/T/Y residues of the annotated proteins that are not "
            f"labelled as sites. This split has {n_negative} real negatives and "
            f"{n_positive} real positives, a shortfall of {gap} negatives. Filling "
            f"that shortfall would require inventing {gap} negatives outside the "
            f"provided rows (e.g. from other proteins or from unannotated residues "
            f"assumed negative) or discarding {gap} real positives. Neither is "
            "done here: every supplied row is kept and the positive class is "
            "weighted instead."
        )
    if n_positive < n_negative:
        gap = n_negative - n_positive
        return (
            "A 1:1 ratio cannot be obtained from same-protein negatives: negatives "
            "are only S/T/Y residues of the annotated proteins that are not "
            f"labelled as sites. This split has {n_negative} real negatives and "
            f"{n_positive} real positives, an excess of {gap} negatives. Reaching "
            f"1:1 would require discarding {gap} real negatives or inventing "
            f"{gap} positives; neither is done here. Every supplied row is kept "
            "and the observed ratio is reported honestly."
        )
    return (
        f"This split is already 1:1 ({n_positive} positives, {n_negative} "
        "negatives); no rebalancing is needed and none is performed."
    )


def imbalance_report(
    train_labels: Sequence[Any], test_labels: Optional[Sequence[Any]] = None
) -> Dict[str, Any]:
    """Report exact counts, the weighting rule, and why no rows are invented."""
    train_counts = class_counts(train_labels)
    pos_weight = compute_pos_weight(train_labels)
    return {
        "train": train_counts,
        "test": class_counts(test_labels) if test_labels is not None else None,
        "pos_weight": pos_weight,
        "exact_rule": (
            "pos_weight = n_train_negative / n_train_positive "
            f"= {train_counts['n_negative']} / {train_counts['n_positive']} "
            f"= {pos_weight}"
        ),
        "one_to_one_possible_from_same_protein_negatives": (
            train_counts["n_negative"] == train_counts["n_positive"]
        ),
        "why_one_to_one_impossible": why_one_to_one_impossible(train_counts),
        "negative_synthesis": "none: no row is created, duplicated, relabelled or dropped",
        "fix": FIX_DESCRIPTION,
    }


class _LogitsBinaryModel(nn.Module):
    """Expose the unchanged binary model's raw logits for BCEWithLogitsLoss.

    The architecture is identical to :class:`src.model.BinaryPhosphositeModel`:
    its motif MLP, learned embedding path and classifier head are reused as-is.
    Only the final sigmoid is omitted (it is applied by ``_ProbabilityView``),
    which lets the trainer use the stable logits form of BCE with pos_weight.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.binary_model = BinaryPhosphositeModel(config)

    def forward(self, features, tokens=None):
        hidden = self.binary_model.motif_mlp(features)
        if self.binary_model.config.ablation == "seq_embedding":
            if tokens is None:
                raise ValueError("token_ids is required for the seq_embedding ablation")
            hidden = torch.cat([hidden, self.binary_model.sequence_embedding(tokens)], dim=1)
        return self.binary_model.classifier(hidden).squeeze(-1)


class _ProbabilityView(nn.Module):
    """Sigmoid wrapper so ``src.real_train.predict_probs`` can be reused unchanged."""

    def __init__(self, logits_model: _LogitsBinaryModel) -> None:
        super().__init__()
        self.logits_model = logits_model

    def forward(self, features, tokens=None):
        return torch.sigmoid(self.logits_model(features, tokens))


def train_weighted_run(
    *,
    train_rows,
    val_rows,
    seed: int,
    ablation: str,
    epochs: int = 10,
    batch_size: int = 32,
    patience: int = 0,
    learning_rate: float = LEARNING_RATE,
    predict_rows=None,
) -> Dict[str, Any]:
    """Train one seeded, positive-weighted run on real split frames.

    Reuses :func:`src.real_train.encode_rows`, :func:`src.real_train.set_seed`,
    ``src.real_train._make_loader``, :func:`src.real_train.predict_probs` and
    ``src.real_train._selection_inputs`` so preprocessing, shuffling, selection
    and scoring stay identical to E-003.  The only change is the loss: positive
    weighted BCEWithLogits instead of plain BCE on sigmoid outputs.

    Returns test probabilities for ``predict_rows`` (default ``val_rows``), the
    chosen epoch, the pos_weight actually used, and a provenance ``config``.
    Test labels are never used for epoch selection.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")

    target_rows = val_rows if predict_rows is None else predict_rows
    for name, rows in (
        ("train_rows", train_rows),
        ("val_rows", val_rows),
        ("predict_rows", target_rows),
    ):
        missing = [column for column in REQUIRED_COLUMNS if column not in rows.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    model_config = ModelConfig(ablation=ablation)
    set_seed(int(seed))

    train_features, train_tokens = encode_rows(train_rows, ablation)
    val_features, val_tokens = encode_rows(val_rows, ablation)
    target_features, target_tokens = encode_rows(target_rows, ablation)

    y_train = train_rows["label"].to_numpy(dtype=np.float32)
    y_val = val_rows["label"].to_numpy(dtype=np.float32)
    train_counts = class_counts(y_train)
    pos_weight = compute_pos_weight(y_train)

    logits_model = _LogitsBinaryModel(model_config)
    probability_view = _ProbabilityView(logits_model)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(logits_model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = _make_loader(
        train_features, y_train, train_tokens, batch_size, generator
    )

    val_is_binary = len(np.unique(y_val)) == 2
    selection_metric = "validation_auroc" if val_is_binary else "validation_bce_fallback"
    higher_is_better = val_is_binary
    best_score = -np.inf if higher_is_better else np.inf
    best_state = deepcopy(logits_model.state_dict())
    best_epoch = 1
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        logits_model.train()
        for batch in train_loader:
            inputs = batch[0]
            batch_tokens = batch[1] if train_tokens is not None else None
            targets = batch[-1]
            optimizer.zero_grad(set_to_none=True)
            logits = logits_model(inputs, batch_tokens)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        val_probabilities = predict_probs(
            probability_view, val_features, val_tokens, batch_size
        )
        _, score = _selection_inputs(y_val, val_probabilities)
        improved = score > best_score if higher_is_better else score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(logits_model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                break

    logits_model.load_state_dict(best_state)
    probabilities = predict_probs(
        probability_view, target_features, target_tokens, batch_size
    )

    config: Dict[str, Any] = {
        "task": "binary phosphosite prediction (site vs non-site S/T/Y residue)",
        "alphabet": AMINO_ACID_ALPHABET,
        "window_length": WINDOW_LENGTH,
        "ablation": model_config.ablation,
        "model": asdict(model_config),
        "loss": LOSS_DESCRIPTION,
        "pos_weight": float(pos_weight),
        "train_counts": train_counts,
        "train_prevalence": train_counts["prevalence"],
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
        "fix": FIX_DESCRIPTION,
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
        "pos_weight": float(pos_weight),
        "train_counts": train_counts,
        "n_train_rows": int(len(train_rows)),
        "n_val_rows": int(len(val_rows)),
    }


def train_weighted_from_csv(
    split_dir: Path,
    *,
    seed: int,
    ablation: str,
    epochs: int = 10,
    batch_size: int = 32,
    patience: int = 0,
    learning_rate: float = LEARNING_RATE,
) -> Dict[str, Any]:
    """Convenience wrapper: load train/val/test CSVs, then train and score test.

    This is the CSV-facing form of :func:`train_weighted_run` for callers that
    have a split directory rather than already-loaded frames.
    """
    split_dir = Path(split_dir)
    rows = {
        split: load_split_csv(split_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    return train_weighted_run(
        train_rows=rows["train"],
        val_rows=rows["val"],
        predict_rows=rows["test"],
        seed=seed,
        ablation=ablation,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
    )


def save_balanced_predictions(result: Mapping[str, Any], out_path: Path) -> Path:
    """Write the E-003 NPZ keys plus the scalar ``pos_weight`` for one run."""
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
        pos_weight=np.float64(result["pos_weight"]),
    )
    return out_path


__all__ = [
    "BALANCED_NPZ_KEYS",
    "FIX_DESCRIPTION",
    "LOSS_DESCRIPTION",
    "class_counts",
    "compute_pos_weight",
    "imbalance_report",
    "positive_class_weight",
    "prevalence_baseline",
    "save_balanced_predictions",
    "train_weighted_from_csv",
    "train_weighted_run",
    "why_one_to_one_impossible",
]
