"""
src/features/champion_encoder.py
---------------------------------
Champion and draft-state encoding utilities.

Provides:
  - ``ChampionEncoder``: maps champion IDs ↔ dense indices
  - ``DraftStateEncoder``: converts a draft state dict into a fixed-length
    feature vector suitable for tree-based or neural models
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence

import numpy as np

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

NUM_CHAMPIONS: int = get("data.num_champions", 165)
ROLES: list[str] = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
EXTERNAL_DIR = pathlib.Path(get("data.external_dir", "data/external"))


class ChampionEncoder:
    """Bidirectional mapping between champion IDs (ints from Riot API) and
    zero-based dense indices used as embedding row indices.

    Args:
        champion_ids: Ordered sequence of unique champion IDs.  If *None*,
                      a sequential mapping ``0..NUM_CHAMPIONS-1`` is used.
    """

    def __init__(self, champion_ids: Sequence[int] | None = None) -> None:
        if champion_ids is None:
            champion_ids = list(range(NUM_CHAMPIONS))
        self._id2idx: dict[int, int] = {cid: idx for idx, cid in enumerate(champion_ids)}
        self._idx2id: dict[int, int] = {idx: cid for cid, idx in self._id2idx.items()}
        self.num_champions = len(self._id2idx)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: pathlib.Path) -> None:
        """Serialise the encoder to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump({"id2idx": {str(k): v for k, v in self._id2idx.items()}}, fh)

    @classmethod
    def load(cls, path: pathlib.Path) -> "ChampionEncoder":
        """Load a previously saved encoder."""
        with path.open("r") as fh:
            data = json.load(fh)
        champion_ids = [int(k) for k in data["id2idx"]]
        return cls(champion_ids)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, champion_id: int) -> int:
        """Map a champion ID to its dense index.  Unknown IDs → 0."""
        return self._id2idx.get(champion_id, 0)

    def decode(self, idx: int) -> int:
        """Map a dense index back to the original champion ID."""
        return self._idx2id.get(idx, -1)

    def encode_many(self, champion_ids: Sequence[int]) -> list[int]:
        return [self.encode(cid) for cid in champion_ids]

    def multi_hot(self, champion_ids: Sequence[int]) -> np.ndarray:
        """Return a binary multi-hot vector of length ``num_champions``."""
        vec = np.zeros(self.num_champions, dtype=np.float32)
        for cid in champion_ids:
            idx = self.encode(cid)
            vec[idx] = 1.0
        return vec


class DraftStateEncoder:
    """Encode the current draft state into a fixed-length feature vector.

    The feature vector concatenates:
      1. Multi-hot vector of blue picks (``num_champions`` dims)
      2. Multi-hot vector of red picks  (``num_champions`` dims)
      3. Multi-hot vector of blue bans  (``num_champions`` dims)
      4. Multi-hot vector of red bans   (``num_champions`` dims)
      5. One-hot pick-order position    (5 dims)
      6. One-hot team indicator         (2 dims — blue / red)

    Total: ``4 * num_champions + 7`` dimensions.

    Args:
        champion_encoder: A fitted :class:`ChampionEncoder`.
    """

    def __init__(self, champion_encoder: ChampionEncoder) -> None:
        self.enc = champion_encoder
        self.n = champion_encoder.num_champions
        self.feature_dim = 4 * self.n + 7

    def encode(
        self,
        blue_picks: Sequence[int],
        red_picks: Sequence[int],
        blue_bans: Sequence[int],
        red_bans: Sequence[int],
        pick_order: int,
        team: str,
    ) -> np.ndarray:
        """Build the feature vector for a single pick event.

        Args:
            blue_picks:  Champion IDs already picked by the blue team.
            red_picks:   Champion IDs already picked by the red team.
            blue_bans:   Champion IDs banned by the blue team.
            red_bans:    Champion IDs banned by the red team.
            pick_order:  Zero-based index of the current pick (0–4).
            team:        ``"blue"`` or ``"red"``.

        Returns:
            1-D ``float32`` numpy array of length ``feature_dim``.
        """
        pick_order_onehot = np.zeros(5, dtype=np.float32)
        pick_order_onehot[min(pick_order, 4)] = 1.0

        team_onehot = np.array([1.0, 0.0] if team == "blue" else [0.0, 1.0], dtype=np.float32)

        return np.concatenate(
            [
                self.enc.multi_hot(blue_picks),
                self.enc.multi_hot(red_picks),
                self.enc.multi_hot(blue_bans),
                self.enc.multi_hot(red_bans),
                pick_order_onehot,
                team_onehot,
            ]
        )

    def encode_batch(self, rows: list[dict]) -> np.ndarray:
        """Encode a list of draft-event dicts (as produced by the preprocessor).

        Args:
            rows: List of dicts with keys matching the output of
                  :func:`src.data.preprocess._extract_draft`.

        Returns:
            2-D ``float32`` array of shape ``(len(rows), feature_dim)``.
        """
        return np.stack(
            [
                self.encode(
                    blue_picks=r.get("blue_picks_so_far", []),
                    red_picks=r.get("red_picks_so_far", []),
                    blue_bans=r.get("blue_bans", []),
                    red_bans=r.get("red_bans", []),
                    pick_order=r.get("pick_order", 0),
                    team=r.get("team", "blue"),
                )
                for r in rows
            ],
            axis=0,
        )


class DraftInteractionEncoder:
    """Encode draft state with team vectors and champion interaction features.

    The output concatenates:
      1. Multi-hot vector of blue picks (``num_champions`` dims)
      2. Multi-hot vector of red picks  (``num_champions`` dims)
      3. Multi-hot vector of blue bans  (``num_champions`` dims)
      4. Multi-hot vector of red bans   (``num_champions`` dims)
      5. Interaction features (12 dims):
         - blue synergy: sum/mean/max
         - red synergy:  sum/mean/max
         - blue vs red counters: sum/mean/max
         - red vs blue counters: sum/mean/max
    """

    def __init__(
        self,
        champion_encoder: ChampionEncoder,
        synergy_matrix: np.ndarray | None = None,
        counter_matrix: np.ndarray | None = None,
    ) -> None:
        self.enc = champion_encoder
        self.n = champion_encoder.num_champions
        self.synergy = self._init_matrix(synergy_matrix, "synergy")
        self.counter = self._init_matrix(counter_matrix, "counter")
        np.fill_diagonal(self.synergy, 0.0)
        self.feature_dim = 4 * self.n + 12

    def _init_matrix(self, matrix: np.ndarray | None, name: str) -> np.ndarray:
        if matrix is None:
            return np.zeros((0, 0), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} matrix must be square, got shape {matrix.shape}")
        if matrix.shape != (self.n, self.n):
            raise ValueError(f"{name} matrix shape must be ({self.n}, {self.n}), got {matrix.shape}")
        return matrix.astype(np.float32, copy=False)

    def _as_indices(self, champion_ids: Sequence[int]) -> np.ndarray:
        idxs = [self.enc.encode(cid) for cid in champion_ids if cid != 0]
        if not idxs:
            return np.array([], dtype=np.int64)
        return np.unique(np.array(idxs, dtype=np.int64))

    def encode_ids(self, champion_ids: Sequence[int], pad_to: int | None = None) -> np.ndarray:
        """Encode champion IDs into dense indices, optionally padding."""
        idxs = np.array(self.enc.encode_many(champion_ids), dtype=np.int64)
        if pad_to is None:
            return idxs
        if len(idxs) >= pad_to:
            return idxs[:pad_to]
        padded = np.zeros(pad_to, dtype=np.int64)
        padded[: len(idxs)] = idxs
        return padded

    def team_vector(self, champion_ids: Sequence[int]) -> np.ndarray:
        """Create a multi-hot team vector from champion IDs (0 = empty)."""
        vec = np.zeros(self.n, dtype=np.float32)
        for cid in champion_ids:
            if cid == 0:
                continue
            idx = self.enc.encode(cid)
            vec[idx] = 1.0
        return vec

    def _within_team_stats(self, idxs: np.ndarray) -> tuple[float, float, float]:
        if idxs.size < 2 or self.synergy.size == 0:
            return 0.0, 0.0, 0.0
        sub = self.synergy[np.ix_(idxs, idxs)]
        tri = sub[np.triu_indices(len(idxs), k=1)]
        if tri.size == 0:
            return 0.0, 0.0, 0.0
        total = float(tri.sum())
        return total, total / tri.size, float(tri.max())

    def _cross_team_stats(self, left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
        if left.size == 0 or right.size == 0 or self.counter.size == 0:
            return 0.0, 0.0, 0.0
        sub = self.counter[np.ix_(left, right)]
        total = float(sub.sum())
        return total, total / sub.size, float(sub.max())

    def interaction_features(self, blue_picks: Sequence[int], red_picks: Sequence[int]) -> np.ndarray:
        """Compute synergy/counter interaction features for current teams."""
        blue_idxs = self._as_indices(blue_picks)
        red_idxs = self._as_indices(red_picks)

        blue_sum, blue_mean, blue_max = self._within_team_stats(blue_idxs)
        red_sum, red_mean, red_max = self._within_team_stats(red_idxs)
        bvr_sum, bvr_mean, bvr_max = self._cross_team_stats(blue_idxs, red_idxs)
        rvb_sum, rvb_mean, rvb_max = self._cross_team_stats(red_idxs, blue_idxs)

        return np.array(
            [
                blue_sum,
                blue_mean,
                blue_max,
                red_sum,
                red_mean,
                red_max,
                bvr_sum,
                bvr_mean,
                bvr_max,
                rvb_sum,
                rvb_mean,
                rvb_max,
            ],
            dtype=np.float32,
        )

    def encode(
        self,
        blue_picks: Sequence[int],
        red_picks: Sequence[int],
        blue_bans: Sequence[int],
        red_bans: Sequence[int],
    ) -> np.ndarray:
        """Build the draft feature vector for a single state."""
        return np.concatenate(
            [
                self.team_vector(blue_picks),
                self.team_vector(red_picks),
                self.team_vector(blue_bans),
                self.team_vector(red_bans),
                self.interaction_features(blue_picks, red_picks),
            ]
        )

    def encode_batch(self, rows: list[dict]) -> np.ndarray:
        """Encode a list of draft-event dicts into a 2-D array."""
        features = np.zeros((len(rows), self.feature_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            features[i] = self.encode(
                blue_picks=row.get("blue_picks_so_far", []),
                red_picks=row.get("red_picks_so_far", []),
                blue_bans=row.get("blue_bans", []),
                red_bans=row.get("red_bans", []),
            )
        return features
