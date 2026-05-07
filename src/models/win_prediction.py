"""
src/models/win_prediction.py
----------------------------
Transformer-based win-probability model that treats the draft as a sequence.

 Draft order (standard pick/ban phases):
  Ban phase 1:  B1, R1, B2, R2, B3, R3
  Pick phase 1: B1, R1, R2, B2, B3, R3
  Ban phase 2:  R4, B4, R5, B5
  Pick phase 2: R4, B4, B5, R5

Tokens are champion indices with 0 reserved for padding. Use the helper
``build_sequence`` to interleave team picks/bans into the ordered sequence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils.config import get

NUM_CHAMPIONS: int = get("data.num_champions", 165)
D_MODEL: int = get("model.transformer.d_model", 128)
NHEAD: int = get("model.transformer.nhead", 4)
NUM_LAYERS: int = get("model.transformer.num_layers", 3)
DIM_FF: int = get("model.transformer.dim_feedforward", 256)
DROPOUT: float = get("model.transformer.dropout", 0.1)
MAX_SEQ_LEN: int = 20

DRAFT_ORDER: list[tuple[str, str, int]] = [
    ("ban", "blue", 0),
    ("ban", "red", 0),
    ("ban", "blue", 1),
    ("ban", "red", 1),
    ("ban", "blue", 2),
    ("ban", "red", 2),
    ("pick", "blue", 0),
    ("pick", "red", 0),
    ("pick", "red", 1),
    ("pick", "blue", 1),
    ("pick", "blue", 2),
    ("pick", "red", 2),
    ("ban", "red", 3),
    ("ban", "blue", 3),
    ("ban", "red", 4),
    ("ban", "blue", 4),
    ("pick", "red", 3),
    ("pick", "blue", 3),
    ("pick", "blue", 4),
    ("pick", "red", 4),
]


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


class DraftWinPredictor(nn.Module):
    """Predict blue win probability from a draft sequence."""

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
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_champions + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
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
    def build_sequence(
        blue_picks: torch.Tensor,
        red_picks: torch.Tensor,
        blue_bans: torch.Tensor,
        red_bans: torch.Tensor,
        *,
        offset_tokens: bool = True,
    ) -> torch.Tensor:
        """Interleave pick/ban tensors into draft order.

        Inputs are expected as dense champion indices with 0 meaning "empty".
        When ``offset_tokens`` is ``True``, non-zero entries are shifted by +1 so
        that 0 can be reserved for padding.
        """
        blue_picks = _ensure_2d(blue_picks).long()
        red_picks = _ensure_2d(red_picks).long()
        blue_bans = _ensure_2d(blue_bans).long()
        red_bans = _ensure_2d(red_bans).long()

        sources = {
            ("pick", "blue"): blue_picks,
            ("pick", "red"): red_picks,
            ("ban", "blue"): blue_bans,
            ("ban", "red"): red_bans,
        }
        steps = [sources[(kind, team)][:, idx] for kind, team, idx in DRAFT_ORDER]
        sequence = torch.stack(steps, dim=1)
        if offset_tokens:
            sequence = sequence.clone()
            sequence[sequence != 0] += 1
        return sequence

    def forward(self, draft_sequence: torch.Tensor) -> torch.Tensor:
        """Return win logits for the blue team.

        Args:
            draft_sequence: LongTensor ``(B, T)`` with ordered draft tokens.
                            0 = padding, 1..N = champion index + 1.
        """
        draft_sequence = _ensure_2d(draft_sequence)
        if draft_sequence.size(1) > self.max_seq_len:
            raise ValueError(f"Sequence length {draft_sequence.size(1)} exceeds max_seq_len={self.max_seq_len}.")

        padding_mask = draft_sequence == 0
        B, T = draft_sequence.shape
        positions = torch.arange(T, device=draft_sequence.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(draft_sequence) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_emb(positions))
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = x.sum(dim=1) / lengths
        logits = self.head(pooled)
        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, draft_sequence: torch.Tensor) -> torch.Tensor:
        """Return blue win probabilities as a tensor."""
        self.eval()
        logits = self(draft_sequence)
        return torch.sigmoid(logits)
