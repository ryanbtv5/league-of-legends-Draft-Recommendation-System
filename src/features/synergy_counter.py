"""
src/features/synergy_counter.py
--------------------------------
Build champion synergy and counter matrices from historical draft data.

A *synergy score* between champions A and B reflects how often they win
together on the same team.  A *counter score* of A against B captures
how often A's team wins when B is on the opposing team.

Both matrices are normalised to the range [0, 1].
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from src.features.champion_encoder import ChampionEncoder
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = pathlib.Path(get("data.processed_dir", "data/processed"))


def build_synergy_matrix(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
) -> np.ndarray:
    """Compute a (num_champions × num_champions) pairwise synergy matrix.

    ``synergy[i, j]`` = win-rate of teams containing both champion *i* and
    champion *j*.

    Args:
        df:      Processed draft DataFrame (one row per pick event).
        encoder: Fitted :class:`ChampionEncoder`.

    Returns:
        Symmetric ``float32`` matrix of shape ``(N, N)`` where ``N =
        encoder.num_champions``.
    """
    N = encoder.num_champions
    wins = np.zeros((N, N), dtype=np.float32)
    games = np.zeros((N, N), dtype=np.float32)

    for match_id, group in df.groupby("match_id"):
        blue_champs = group[group["team"] == "blue"]["champion_id"].tolist()
        red_champs = group[group["team"] == "red"]["champion_id"].tolist()
        blue_win = bool(group["blue_win"].iloc[0])

        for team, won in [(blue_champs, blue_win), (red_champs, not blue_win)]:
            idxs = [encoder.encode(c) for c in team]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    games[a, b] += 1
                    games[b, a] += 1
                    if won:
                        wins[a, b] += 1
                        wins[b, a] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(games > 0, wins / games, 0.0)
    return matrix.astype(np.float32)


def build_counter_matrix(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
) -> np.ndarray:
    """Compute a (num_champions × num_champions) counter matrix.

    ``counter[i, j]`` = win-rate of the team with champion *i* when the
    opposing team has champion *j*.  Values > 0.5 mean *i* counters *j*.

    Args:
        df:      Processed draft DataFrame.
        encoder: Fitted :class:`ChampionEncoder`.

    Returns:
        ``float32`` matrix of shape ``(N, N)``.
    """
    N = encoder.num_champions
    wins = np.zeros((N, N), dtype=np.float32)
    games = np.zeros((N, N), dtype=np.float32)

    for match_id, group in df.groupby("match_id"):
        blue_champs = group[group["team"] == "blue"]["champion_id"].tolist()
        red_champs = group[group["team"] == "red"]["champion_id"].tolist()
        blue_win = bool(group["blue_win"].iloc[0])

        blue_idxs = [encoder.encode(c) for c in blue_champs]
        red_idxs = [encoder.encode(c) for c in red_champs]

        for bi in blue_idxs:
            for ri in red_idxs:
                games[bi, ri] += 1
                games[ri, bi] += 1
                if blue_win:
                    wins[bi, ri] += 1
                else:
                    wins[ri, bi] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(games > 0, wins / games, 0.0)
    return matrix.astype(np.float32)


def save_matrices(
    synergy: np.ndarray,
    counter: np.ndarray,
    output_dir: pathlib.Path = PROCESSED_DIR,
) -> None:
    """Save synergy and counter matrices as ``.npy`` files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "synergy_matrix.npy", synergy)
    np.save(output_dir / "counter_matrix.npy", counter)
    logger.info("Saved synergy and counter matrices to %s", output_dir)


def load_matrices(
    input_dir: pathlib.Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load previously saved synergy and counter matrices."""
    synergy = np.load(input_dir / "synergy_matrix.npy")
    counter = np.load(input_dir / "counter_matrix.npy")
    return synergy, counter
