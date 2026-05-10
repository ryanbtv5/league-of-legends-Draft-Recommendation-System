"""
src/models/transformer.py
--------------------------
Transformer-based sequence model for draft recommendation.

Each draft is treated as an ordered sequence of champion tokens (picks and
bans interleaved in the true Bo5 draft order).  A causal Transformer encoder
reads the sequence and produces a distribution over the next pick.

Architecture:
  1. Champion token embedding (shared table)
  2. Learnable positional embedding (up to 20 positions — 10 bans + 10 picks)
  3. Transformer encoder (causal masking so position *t* can only attend to
     positions ≤ *t*)
  4. Final hidden state projected to champion logits

Usage:
    from src.models.transformer import DraftTransformer
    model = DraftTransformer()
    logits = model(tokens)         # (B, seq_len, num_champions + 1)
    next_pick_logits = logits[:, -1, :]  # last position for next-token prediction
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import concurrent.futures
import functools

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))

NUM_CHAMPIONS: int = get("data.num_champions", 165)
D_MODEL: int = get("model.transformer.d_model", 128)
NHEAD: int = get("model.transformer.nhead", 4)
NUM_LAYERS: int = get("model.transformer.num_layers", 3)
DIM_FF: int = get("model.transformer.dim_feedforward", 256)
DROPOUT: float = get("model.transformer.dropout", 0.1)
MAX_SEQ_LEN: int = 20  # 10 bans + 10 picks


def _infer_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int | float]:
    """Infer Transformer hyperparameters from a saved state dict.

    This keeps legacy checkpoints loadable even if config defaults change.
    """
    token_shape = state_dict.get("token_emb.weight")
    pos_shape = state_dict.get("pos_emb.weight")
    proj_shape = state_dict.get("proj.weight")
    linear1 = state_dict.get("transformer.layers.0.linear1.weight")
    layer_indices: set[int] = set()

    for key in state_dict:
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))

    num_layers = (max(layer_indices) + 1) if layer_indices else NUM_LAYERS
    ref_shape = token_shape if token_shape is not None else proj_shape
    out_shape = proj_shape if proj_shape is not None else token_shape
    d_model = int(ref_shape.shape[1]) if ref_shape is not None else D_MODEL
    num_champions = int(out_shape.shape[0] - 1) if out_shape is not None else NUM_CHAMPIONS
    dim_feedforward = int(linear1.shape[0]) if linear1 is not None else DIM_FF
    max_seq_len = int(pos_shape.shape[0]) if pos_shape is not None else MAX_SEQ_LEN

    return {
        "num_champions": num_champions,
        "d_model": d_model,
        "nhead": NHEAD,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": DROPOUT,
        "max_seq_len": max_seq_len,
    }


class DraftTransformer(nn.Module):
    """Causal Transformer for draft-sequence next-champion prediction.

    Args:
        num_champions:    Vocabulary size (champion pool).
        d_model:          Embedding / hidden dimension.
        nhead:            Number of attention heads.
        num_layers:       Number of Transformer encoder layers.
        dim_feedforward:  Feed-forward hidden size.
        dropout:          Dropout probability.
        max_seq_len:      Maximum sequence length (default 20).
    """

    def __init__(
        self,
        num_champions: int = NUM_CHAMPIONS,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FF,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.num_champions = num_champions
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Special tokens: 0 = padding, 1..num_champions = champion indices
        vocab_size = num_champions + 2
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # enables nested-tensor fast path on PyTorch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Output includes padding token at index 0.
        self.proj = nn.Linear(d_model, num_champions + 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = ignore) for causal attention."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute per-position logits over the champion pool.

        Args:
            tokens:               LongTensor ``(B, T)`` of token indices.
                                  0 = padding, 1..N = champion index + 1.
            src_key_padding_mask: BoolTensor ``(B, T)`` — ``True`` for padding
                                  positions (passed through to PyTorch).

        Returns:
            FloatTensor ``(B, T, num_champions + 1)`` — logits at each position.
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = x * math.sqrt(self.d_model)

        causal = self._causal_mask(T, tokens.device)
        x = self.transformer(x, mask=causal, src_key_padding_mask=src_key_padding_mask)
        return self.proj(x)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self,
        tokens: torch.Tensor,
        unavailable: set[int] | None = None,
        k: int = 5,
    ) -> list[int]:
        """Recommend the top-*k* champions for the *next* position in the draft.

        Args:
            tokens:      LongTensor ``(1, T)`` — the current draft sequence.
            unavailable: Set of champion indices that are already picked/banned.
            k:           Number of recommendations.

        Returns:
            List of *k* champion indices (dense, 0-based) sorted by probability.
        """
        self.eval()
        logits = self(tokens)[:, -1, :]  # (1, num_champions + 1)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if unavailable:
            for idx in unavailable:
                token = idx + 1
                if 0 <= token < len(probs):
                    probs[token] = 0.0
        ranked = np.argsort(probs[1:])[::-1][:k]
        return ranked.tolist()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model: DraftTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Each batch contains:
      - ``tokens``: LongTensor ``(B, T)`` — the input draft sequence.
      - ``targets``: LongTensor ``(B, T)`` — the target next token at each step.

    Returns:
        Mean cross-entropy loss.
    """
    model.train()
    total_loss = 0.0

    for tokens, targets in loader:
        tokens, targets = tokens.to(device), targets.to(device)
        padding_mask = (tokens == 0)
        logits = model(tokens, src_key_padding_mask=padding_mask)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        valid = targets_flat != 0
        if valid.any():
            loss = F.cross_entropy(logits_flat[valid], targets_flat[valid])
        else:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * B

    return total_loss / len(loader.dataset)


def evaluate_epoch(
    model: DraftTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss on a DataLoader without updating weights.

    Args:
        model:  The Transformer model (set to eval mode internally).
        loader: DataLoader yielding ``(tokens, targets)`` batches.
        device: Compute device.

    Returns:
        Mean cross-entropy loss over the dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tokens, targets in loader:
            tokens, targets = tokens.to(device), targets.to(device)
            padding_mask = (tokens == 0)
            logits = model(tokens,