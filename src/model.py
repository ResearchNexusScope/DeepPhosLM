"""Binary phosphosite prediction from a 15-residue sequence window.

Computational task
------------------
This module implements BINARY classification of a centred S/T/Y residue:
label 1 means the residue is a phosphorylation site, label 0 means it is a
non-site S/T/Y residue. The only input is the 15-residue amino-acid window
centred on the candidate residue. The trainer pairs this model with BCE loss.

Features
--------
* ``motif_onehot``: a one-hot encoding of all 15 window positions over the
  legacy-compatible 22-symbol alphabet ``XWGSAELQDMPFTRIHVNCY_K``. Symbols
  outside that alphabet are encoded as ``X`` (the ambiguity symbol).
* ``token_ids``: an optional integer encoding of the same 15 tokens. When the
  ``seq_embedding`` ablation is selected, these tokens feed a small learned
  embedding (``nn.Embedding`` + bidirectional LSTM) trained end-to-end from
  scratch. There are no external weights, downloads or network calls.

Ablations
---------
``seq_only``      -- motif one-hot path only (a small MLP).
``seq_embedding`` -- the motif MLP plus the learned token embedding path.

Both paths end in a single sigmoid probability per example. The architecture
is intentionally small so that training stays reproducible on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Union

import numpy as np
import torch

# Exact legacy alphabet, in this order. Do not reorder: indices are persisted
# in saved token arrays and must remain stable.
AMINO_ACID_ALPHABET = "XWGSAELQDMPFTRIHVNCY_K"
WINDOW_LENGTH = 15
ALPHABET_SIZE = len(AMINO_ACID_ALPHABET)  # 22
ONE_HOT_WIDTH = WINDOW_LENGTH * ALPHABET_SIZE  # 330

ABLATIONS = ("seq_only", "seq_embedding")
MAX_EMBEDDING_DIM = 100

_SYMBOL_TO_INDEX: Dict[str, int] = {
    symbol: index for index, symbol in enumerate(AMINO_ACID_ALPHABET)
}
_UNKNOWN_INDEX = _SYMBOL_TO_INDEX["X"]


def _validate_windows(sequences: Sequence[str]) -> list[str]:
    windows = [str(sequence).upper() for sequence in sequences]
    for window in windows:
        if len(window) != WINDOW_LENGTH:
            raise ValueError(
                f"Expected a {WINDOW_LENGTH}-residue window, got {len(window)}: {window!r}"
            )
    return windows


def one_hot_encode(sequences: Sequence[str]) -> np.ndarray:
    """Return a float32 array of shape ``[n, 15 * 22]``.

    Position ``p`` (0-based) occupies columns ``p * 22 ... p * 22 + 21``.
    Every window contributes exactly 15 active columns.
    """
    windows = _validate_windows(sequences)
    encoded = np.zeros((len(windows), ONE_HOT_WIDTH), dtype=np.float32)
    for row, window in enumerate(windows):
        for position, symbol in enumerate(window):
            index = _SYMBOL_TO_INDEX.get(symbol, _UNKNOWN_INDEX)
            encoded[row, position * ALPHABET_SIZE + index] = 1.0
    return encoded


def tokenize_sequences(sequences: Sequence[str]) -> np.ndarray:
    """Return int64 token ids of shape ``[n, 15]`` over ``AMINO_ACID_ALPHABET``.

    These are the same window characters used by :func:`one_hot_encode`, just
    in index form for the learned embedding path. Unknown symbols map to the
    ambiguity token ``X``.
    """
    windows = _validate_windows(sequences)
    tokens = np.full((len(windows), WINDOW_LENGTH), _UNKNOWN_INDEX, dtype=np.int64)
    for row, window in enumerate(windows):
        for position, symbol in enumerate(window):
            tokens[row, position] = _SYMBOL_TO_INDEX.get(symbol, _UNKNOWN_INDEX)
    return tokens


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for :class:`BinaryPhosphositeModel`."""

    ablation: str = "seq_only"
    embedding_dim: int = 32
    lstm_hidden_dim: int = 16
    mlp_hidden_dim: int = 32
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.ablation not in ABLATIONS:
            raise ValueError(
                f"ablation must be one of {ABLATIONS}, got {self.ablation!r}"
            )
        if not 1 <= self.embedding_dim <= MAX_EMBEDDING_DIM:
            raise ValueError(
                f"embedding_dim must be in [1, {MAX_EMBEDDING_DIM}], "
                f"got {self.embedding_dim!r}"
            )
        if self.lstm_hidden_dim < 1:
            raise ValueError(f"lstm_hidden_dim must be >= 1, got {self.lstm_hidden_dim!r}")
        if self.mlp_hidden_dim < 1:
            raise ValueError(f"mlp_hidden_dim must be >= 1, got {self.mlp_hidden_dim!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")

    @classmethod
    def from_mapping(cls, config: Union["ModelConfig", Mapping[str, object]]) -> "ModelConfig":
        """Build a config from a dataclass instance or a plain mapping."""
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            known = set(cls.__dataclass_fields__)
            unknown = set(config) - known
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            return cls(**config)
        raise TypeError(f"Expected ModelConfig or mapping, got {type(config).__name__}")


class SequenceEmbedding(torch.nn.Module):
    """Learned token embedding + BiLSTM over the 15 window tokens.

    Randomly initialised, trained end-to-end by the trainer. The mean over the
    time axis yields a fixed-size summary of the window; no external weights
    are involved.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, lstm_hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        outputs, _ = self.lstm(embedded)
        return outputs.mean(dim=1)


class BinaryPhosphositeModel(torch.nn.Module):
    """Small binary classifier over one-hot windows and optional token ids.

    ``forward(motif_onehot, token_ids=None)`` returns a tensor of shape
    ``[batch_size]`` with one sigmoid probability per example. ``token_ids``
    is required only for the ``seq_embedding`` ablation.
    """

    def __init__(self, config: Union[ModelConfig, Mapping[str, object]]) -> None:
        super().__init__()
        self.config = ModelConfig.from_mapping(config)

        self.motif_mlp = torch.nn.Sequential(
            torch.nn.Linear(ONE_HOT_WIDTH, self.config.mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.config.dropout),
        )
        if self.config.ablation == "seq_embedding":
            self.sequence_embedding = SequenceEmbedding(
                ALPHABET_SIZE,
                self.config.embedding_dim,
                self.config.lstm_hidden_dim,
            )
            combined_dim = self.config.mlp_hidden_dim + 2 * self.config.lstm_hidden_dim
        else:
            combined_dim = self.config.mlp_hidden_dim
        self.classifier = torch.nn.Linear(combined_dim, 1)

    def forward(self, motif_onehot, token_ids=None):
        if motif_onehot.ndim != 2 or motif_onehot.shape[1] != ONE_HOT_WIDTH:
            raise ValueError(
                f"motif_onehot must have shape [batch, {ONE_HOT_WIDTH}], "
                f"got {tuple(motif_onehot.shape)}"
            )
        hidden = self.motif_mlp(motif_onehot)
        if self.config.ablation == "seq_embedding":
            if token_ids is None:
                raise ValueError(
                    "token_ids is required for the seq_embedding ablation"
                )
            if token_ids.ndim != 2 or token_ids.shape[1] != WINDOW_LENGTH:
                raise ValueError(
                    f"token_ids must have shape [batch, {WINDOW_LENGTH}], "
                    f"got {tuple(token_ids.shape)}"
                )
            hidden = torch.cat([hidden, self.sequence_embedding(token_ids)], dim=1)
        return torch.sigmoid(self.classifier(hidden)).squeeze(-1)
