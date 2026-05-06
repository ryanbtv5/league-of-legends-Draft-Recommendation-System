"""
src/models/win_prediction.py
----------------------------
PyTorch win-probability model using champion embeddings.

Architecture:
  1. Shared champion embedding table
  2. Mean-pooled team embeddings for picks + bans
  3. Team-level projection for blue and red separately
  4. Classifier head predicting blue win probability
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.utils.config import get

NUM_CHAMPIONS: int = get("data.num_champions", 165)
EMB_DIM: int = get("features.embedding_dim", 64)


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

    def _aggregate(self, indices: torch.Tensor) -> torch.Tensor:
        """Mean-pool embeddings for champion index sequences."""
        embs = self.champion_emb(indices)
        mask = (indices != 0).unsqueeze(-1).float()
        summed = (embs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count

    def _team_representation(
        self,
        picks: torch.Tensor,
        bans: torch.Tensor,
    ) -> torch.Tensor:
        picks_emb = self._aggregate(picks)
        bans_emb = self._aggregate(bans)
        return self.team_encoder(torch.cat([picks_emb, bans_emb], dim=-1))

    def forward(
        self,
        blue_picks: torch.Tensor,
        red_picks: torch.Tensor,
        blue_bans: torch.Tensor,
        red_bans: torch.Tensor,
    ) -> torch.Tensor:
        """Return win logits for the blue team."""
        blue_team = self._team_representation(blue_picks, blue_bans)
        red_team = self._team_representation(red_picks, red_bans)
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
