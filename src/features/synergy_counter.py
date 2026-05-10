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

    # Support two input formats:
    # 1) event-level rows containing 'team' and 'champion_id' columns
    # 2) processed final-row format with blue_pick_*/red_pick_* columns
    if "team" in df.columns and "champion_id" in df.columns:
        iter_groups = df.groupby("match_id")
        for match_id, group in iter_groups:
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
    else:
        # processed dataframe: one row per draft_step; take final row per match
        pick_cols_blue = [f"blue_pick_{i}" for i in range(1, get("data.picks_per_team", 5) + 1)]
        pick_cols_red = [f"red_pick_{i}" for i in range(1, get("data.picks_per_team", 5) + 1)]
        for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
            final_row = group.iloc[-1]
            blue_champs = [int(final_row.get(c, 0)) for c in pick_cols_blue if int(final_row.get(c, 0)) != 0]
            red_champs = [int(final_row.get(c, 0)) for c in pick_cols_red if int(final_row.get(c, 0)) != 0]
            blue_win = bool(final_row.get("blue_win", False))

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

    if "team" in df.columns and "champion_id" in df.columns:
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
    else:
        # processed final-row format
        pick_cols_blue = [f"blue_pick_{i}" for i in range(1, get("data.picks_per_team", 5) + 1)]
        pick_cols_red = [f"red_pick_{i}" for i in range(1, get("data.picks_per_team", 5) + 1)]
        for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
            final_row = group.iloc[-1]
            blue_champs = [int(final_row.get(c, 0)) for c in pick_cols_blue if int(final_row.get(c, 0)) != 0]
            red_champs = [int(final_row.get(c, 0)) for c in pick_cols_red if int(final_row.get(c, 0)) != 0]
            blue_win = bool(final_row.get("blue_win", False))

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


def build_role_winrate_matrix(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
) -> np.ndarray:
    """Compute per-champion win rates broken down by role.

    Returns an array of shape (N, R) where R = number of roles configured
    (TOP, JUNGLE, MID, ADC, SUPPORT). Entry [i, r] is the win-rate of games
    where champion i was played in role r.
    """
    ROLES = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
    R = len(ROLES)
    N = encoder.num_champions
    wins = np.zeros((N, R), dtype=np.float32)
    games = np.zeros((N, R), dtype=np.float32)

    pick_cols_blue = [f"blue_pick_{i}" for i in range(1, R + 1)]
    pick_cols_red = [f"red_pick_{i}" for i in range(1, R + 1)]

    for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
        final_row = group.iloc[-1]
        blue_win = bool(final_row["blue_win"])

        # Blue picks: pick position maps directly to role index (TOP..SUPPORT)
        for role_idx, col in enumerate(pick_cols_blue):
            champ_id = int(final_row.get(col, 0))
            if champ_id == 0:
                continue
            idx = encoder.encode(champ_id)
            games[idx, role_idx] += 1
            if blue_win:
                wins[idx, role_idx] += 1

        # Red picks: role index same ordering
        for role_idx, col in enumerate(pick_cols_red):
            champ_id = int(final_row.get(col, 0))
            if champ_id == 0:
                continue
            idx = encoder.encode(champ_id)
            games[idx, role_idx] += 1
            if not blue_win:
                wins[idx, role_idx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(games > 0, wins / games, 0.0)
    return matrix.astype(np.float32)


def build_role_counter_matrices(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
) -> np.ndarray:
    """Compute role-specific counter matrices.

    Returns an array of shape (R, N, N) where entry [r, i, j] is the win-rate
    of champion i (when played in role r) against opponent champion j.
    """
    ROLES = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
    R = len(ROLES)
    N = encoder.num_champions
    wins = np.zeros((R, N, N), dtype=np.float32)
    games = np.zeros((R, N, N), dtype=np.float32)

    pick_cols_blue = [f"blue_pick_{i}" for i in range(1, R + 1)]
    pick_cols_red = [f"red_pick_{i}" for i in range(1, R + 1)]

    for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
        final_row = group.iloc[-1]
        blue_win = bool(final_row["blue_win"])

        blue_idxs = [int(final_row.get(c, 0)) for c in pick_cols_blue]
        red_idxs = [int(final_row.get(c, 0)) for c in pick_cols_red]

        # Blue champions vs red champions
        for role_idx, champ_id in enumerate(blue_idxs):
            if champ_id == 0:
                continue
            i = encoder.encode(champ_id)
            for opp in red_idxs:
                if opp == 0:
                    continue
                j = encoder.encode(opp)
                games[role_idx, i, j] += 1
                if blue_win:
                    wins[role_idx, i, j] += 1

        # Red champions vs blue champions (role index is red's pick position)
        for role_idx, champ_id in enumerate(red_idxs):
            if champ_id == 0:
                continue
            i = encoder.encode(champ_id)
            for opp in blue_idxs:
                if opp == 0:
                    continue
                j = encoder.encode(opp)
                games[role_idx, i, j] += 1
                if not blue_win:
                    wins[role_idx, i, j] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        matrices = np.where(games > 0, wins / games, 0.0)
    return matrices.astype(np.float32)


def save_role_matrices(
    role_win: np.ndarray,
    role_counter: np.ndarray,
    output_dir: pathlib.Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "role_win_matrix.npy", role_win)
    np.save(output_dir / "role_counter_matrices.npy", role_counter)
    logger.info("Saved role win-rate and role-counter matrices to %s", output_dir)


def load_role_matrices(input_dir: pathlib.Path = PROCESSED_DIR) -> tuple[np.ndarray, np.ndarray]:
    role_win = np.load(input_dir / "role_win_matrix.npy")
    role_counter = np.load(input_dir / "role_counter_matrices.npy")
    return role_win, role_counter
