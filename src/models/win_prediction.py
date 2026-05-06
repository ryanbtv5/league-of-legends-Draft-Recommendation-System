"""
src/models/win_prediction.py
----------------------------
PyTorch win-probability model using champion embeddings.

Architecture:
  1. Shared champion embedding table
  2. Self-attention per team to capture synergy within picks/bans
  3. Cross-attention between teams to capture counter interactions
  4. Attention pooling to build team representations
  5. Classifier head predicting blue win probability
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.utils.config import get

NUM_CHAMPIONS: int = get("data.num_champions", 165)
EMB_DIM: int = get("features.embedding_dim", 64)


class AttentionPool(nn.Module):
    """Learned attention pooling over a variable-length sequence."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return a pooled representation from (B, S, D) embeddings."""
        d_model = embs.size(-1)
        query = self.query.squeeze()
        scores = torch.matmul(embs, query) / (d_model**0.5)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        weights = torch.nan_to_num(weights)
        return torch.bmm(weights.unsqueeze(1), embs).squeeze(1)


class DraftWinPredictor(nn.Module):
    """Predict match win probability from a partial draft state.

    Args:
        num_champions: Champion vocabulary size.
        embedding_dim: Dimension of champion embeddings.
        team_hidden_dim: Hidden size for team representations.
        head_hidden_dims: Hidden sizes for the classifier head.
        dropout: Dropout probability for hidden layers.
    """

    def __init__(
        self,
        num_champions: int = NUM_CHAMPIONS,
        embedding_dim: int = EMB_DIM,
        team_hidden_dim: int | None = None,
        head_hidden_dims: Sequence[int] | None = None,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim
        team_hidden_dim = team_hidden_dim or embedding_dim
        head_hidden_dims = list(head_hidden_dims or [128, 64])

        self.champion_emb = nn.Embedding(
            num_embeddings=num_champions + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.self_attention = nn.MultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool = AttentionPool(embedding_dim, dropout=dropout)

        self.team_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, team_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        head_layers: list[nn.Module] = []
        prev = 2 * team_hidden_dim
        for hidden in head_hidden_dims:
            head_layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        head_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head_layers)

    def _encode_team(self, picks: torch.Tensor, bans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([picks, bans], dim=1)
        mask = tokens == 0
        return self.champion_emb(tokens), mask

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_mask,
            need_weights=False,
        )
        return attended

    def _team_representation(
        self,
        self_context: torch.Tensor,
        cross_context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self_context = self_context.masked_fill(mask.unsqueeze(-1), 0.0)
        cross_context = cross_context.masked_fill(mask.unsqueeze(-1), 0.0)
        synergy = self.pool(self_context, mask)
        counters = self.pool(cross_context, mask)
        return self.team_encoder(torch.cat([synergy, counters], dim=-1))

    def forward(
        self,
        blue_picks: torch.Tensor,
        red_picks: torch.Tensor,
        blue_bans: torch.Tensor,
        red_bans: torch.Tensor,
    ) -> torch.Tensor:
        """Return win logits for the blue team."""
        blue_embs, blue_mask = self._encode_team(blue_picks, blue_bans)
        red_embs, red_mask = self._encode_team(red_picks, red_bans)

        blue_self, _ = self.self_attention(
            query=blue_embs,
            key=blue_embs,
            value=blue_embs,
            key_padding_mask=blue_mask,
            need_weights=False,
        )
        red_self, _ = self.self_attention(
            query=red_embs,
            key=red_embs,
            value=red_embs,
            key_padding_mask=red_mask,
            need_weights=False,
        )

        blue_cross = self._attend(blue_embs, red_embs, red_embs, red_mask)
        red_cross = self._attend(red_embs, blue_embs, blue_embs, blue_mask)

        blue_team = self._team_representation(blue_self, blue_cross, blue_mask)
        red_team = self._team_representation(red_self, red_cross, red_mask)
        logits = self.head(torch.cat([blue_team, red_team], dim=-1))
        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        blue_picks: torch.Tensor,
        red_picks: torch.Tensor,
        blue_bans: torch.Tensor,
        red_bans: torch.Tensor,
    ) -> torch.Tensor:
        """Return blue win probabilities as a tensor."""
        self.eval()
        logits = self(blue_picks, red_picks, blue_bans, red_bans)
        return torch.sigmoid(logits)
