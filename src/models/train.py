"""
src/models/train.py
--------------------
End-to-end training script that covers all three model tiers:
  - Random Forest baseline
  - MLP with champion embeddings
  - Transformer sequence model

Usage:
    python -m src.models.train --model rf
    python -m src.models.train --model mlp --epochs 50
    python -m src.models.train --model transformer --epochs 100
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_processed
from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.models import neural as nm
from src.models import transformer as tm
from src.models.win_prediction import DraftWinPredictor
from src.utils.config import get, load_config
from src.utils.logger import get_logger

try:
    import mlflow
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    class _NullMlflowRun(contextlib.AbstractContextManager):
        def __exit__(self, exc_type, exc, tb):
            return False

    class _NullMlflow:
        def set_tracking_uri(self, *_args, **_kwargs):
            return None

        def set_experiment(self, *_args, **_kwargs):
            return None

        def start_run(self, *_args, **_kwargs):
            return _NullMlflowRun()

        def log_params(self, *_args, **_kwargs):
            return None

        def log_metric(self, *_args, **_kwargs):
            return None

        def log_metrics(self, *_args, **_kwargs):
            return None

        def log_artifact(self, *_args, **_kwargs):
            return None

    mlflow = _NullMlflow()

logger = get_logger(__name__)

SEED: int = get("project.random_seed", 42)
MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))

BLUE_PICK_COLS = [f"blue_pick_{i}" for i in range(1, 6)]
RED_PICK_COLS = [f"red_pick_{i}" for i in range(1, 6)]
BLUE_BAN_COLS = [f"blue_ban_{i}" for i in range(1, 6)]
RED_BAN_COLS = [f"red_ban_{i}" for i in range(1, 6)]
PICK_COLS = BLUE_PICK_COLS + RED_PICK_COLS
BAN_COLS = BLUE_BAN_COLS + RED_BAN_COLS
ROLES = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dataloader_kwargs(shuffle: bool = False) -> dict[str, object]:
    workers = int(get("training.num_workers", min(4, os.cpu_count() or 1)))
    workers = max(0, workers)
    pin_memory = bool(get("training.pin_memory", True))
    return {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }


def _nonzero_values(row: pd.Series, columns: list[str]) -> list[int]:
    return [int(row[col]) for col in columns if int(row[col]) != 0]


def _pick_order(row: pd.Series, team: str) -> int:
    columns = BLUE_PICK_COLS if team == "blue" else RED_PICK_COLS
    return sum(1 for col in columns if int(row[col]) != 0)


def _draft_state_from_row(row: pd.Series, team: str) -> dict[str, object]:
    return {
        "blue_picks_so_far": _nonzero_values(row, BLUE_PICK_COLS),
        "red_picks_so_far": _nonzero_values(row, RED_PICK_COLS),
        "blue_bans": _nonzero_values(row, BLUE_BAN_COLS),
        "red_bans": _nonzero_values(row, RED_BAN_COLS),
        "pick_order": _pick_order(row, team),
        "team": team,
    }


def _next_pick_target(current: pd.Series, nxt: pd.Series, team: str) -> int | None:
    columns = BLUE_PICK_COLS if team == "blue" else RED_PICK_COLS
    for col in columns:
        cur_val = int(current[col])
        next_val = int(nxt[col])
        if cur_val != next_val and next_val != 0:
            return next_val
    return None


def _build_recommendation_dataset(df: pd.DataFrame) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    targets: list[int] = []

    for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
        match_rows = group.reset_index(drop=True)
        for idx in range(len(match_rows) - 1):
            current = match_rows.iloc[idx]
            nxt = match_rows.iloc[idx + 1]
            next_team = "blue" if int(nxt["picking_team"]) == 0 else "red"
            target = _next_pick_target(current, nxt, next_team)
            if target is None:
                continue
            rows.append(_draft_state_from_row(current, next_team))
            targets.append(target)

    return rows, np.array(targets, dtype=np.int64)


def _unique_champion_ids(df: pd.DataFrame) -> list[int]:
    values = df[PICK_COLS + BAN_COLS].to_numpy(dtype=np.int64, copy=False)
    champion_ids = np.unique(values)
    return sorted(int(cid) for cid in champion_ids if cid != 0)


def _pad_encoded_ids(champion_ids: list[int], encoder: ChampionEncoder, pad_to: int = 5) -> np.ndarray:
    values = np.zeros(pad_to, dtype=np.int64)
    encoded = encoder.encode_many([int(cid) for cid in champion_ids if int(cid) != 0])
    if encoded:
        values[: min(len(encoded), pad_to)] = np.asarray(encoded[:pad_to], dtype=np.int64)
    return values


def _split_train_val(*arrays: np.ndarray, stratify: np.ndarray | None = None, test_size: float = 0.15):
    try:
        return train_test_split(*arrays, test_size=test_size, random_state=SEED, stratify=stratify)
    except ValueError:
        return train_test_split(*arrays, test_size=test_size, random_state=SEED)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _load_flat_data(df: pd.DataFrame, encoder: DraftStateEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Convert processed DataFrame to (X, y) arrays for tree/MLP models."""
    rows, targets = _build_recommendation_dataset(df)
    X = encoder.encode_batch(rows)
    y = np.array(encoder.enc.encode_many(targets.tolist()), dtype=np.int64)
    return X, y


def _load_mlp_data(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build role-aware champion-index batches for the embedding MLP."""
    rows, targets = _build_recommendation_dataset(df)
    n = len(rows)

    blue_picks = np.zeros((n, 5), dtype=np.int64)
    red_picks = np.zeros((n, 5), dtype=np.int64)
    blue_bans = np.zeros((n, 5), dtype=np.int64)
    red_bans = np.zeros((n, 5), dtype=np.int64)
    roles = np.zeros((n, len(ROLES)), dtype=np.float32)
    teams = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    for i, (row, target) in enumerate(zip(rows, targets)):
        blue_picks[i] = _pad_encoded_ids(row["blue_picks_so_far"], encoder)
        red_picks[i] = _pad_encoded_ids(row["red_picks_so_far"], encoder)
        blue_bans[i] = _pad_encoded_ids(row["blue_bans"], encoder)
        red_bans[i] = _pad_encoded_ids(row["red_bans"], encoder)

        role_idx = min(int(row.get("pick_order", 0)), len(ROLES) - 1)
        roles[i, role_idx] = 1.0
        teams[i, 0 if row.get("team", "blue") == "blue" else 1] = 1.0
        y[i] = encoder.encode(int(target))

    return blue_picks, red_picks, blue_bans, red_bans, roles, teams, y


def _load_sequence_data(df: pd.DataFrame, encoder: ChampionEncoder, max_len: int = 20) -> np.ndarray:
    """Build padded token sequences (one per match) for the Transformer."""
    sequences: list[list[int]] = []
    draft_order = [
        "blue_pick_1",
        "red_pick_1",
        "red_pick_2",
        "blue_pick_2",
        "blue_pick_3",
        "red_pick_3",
        "red_pick_4",
        "blue_pick_4",
        "blue_pick_5",
        "red_pick_5",
    ]
    for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
        final_row = group.iloc[-1]
        seq = [encoder.encode(int(final_row[col])) + 1 for col in draft_order if int(final_row[col]) != 0]

        # Pad / truncate to max_len + 1 (input + target)
        seq = seq[: max_len + 1]
        seq += [0] * (max_len + 1 - len(seq))
        sequences.append(seq)

    return np.array(sequences, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_rf(df: pd.DataFrame, encoder: DraftStateEncoder) -> None:
    cfg = load_config()["model"]["baseline"]
    rf_params = {k: v for k, v in cfg.items() if k != "type"}
    X, y = _load_flat_data(df, encoder)
    X_tr, X_val, y_tr, y_val = _split_train_val(X, y, stratify=y)

    mlflow.set_tracking_uri(get("mlflow.tracking_uri", "mlruns"))
    mlflow.set_experiment(get("mlflow.experiment_name", "draft-recommendation"))

    with mlflow.start_run(run_name="rf_baseline"):
        mlflow.log_params(rf_params)
        model = bm.RandomForestRecommender(num_champions=encoder.enc.num_champions, params=rf_params)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)
        top1 = (probs.argmax(axis=1) == y_val).mean()
        mlflow.log_metric("val_top1_acc", top1)
        logger.info("Random Forest validation top-1 accuracy: %.4f", top1)

        path = model.save(MODEL_DIR / "rf_recommender.pkl")
        mlflow.log_artifact(str(path))


def train_mlp(
    df: pd.DataFrame,
    encoder: DraftStateEncoder,
    epochs: int = 50,
    resume_from: pathlib.Path | None = None,
) -> None:
    cfg = load_config()["model"]["neural"]
    (
        blue_picks,
        red_picks,
        blue_bans,
        red_bans,
        roles,
        teams,
        y,
    ) = _load_mlp_data(df, encoder.enc)
    (
        bp_tr,
        bp_val,
        rp_tr,
        rp_val,
        bb_tr,
        bb_val,
        rb_tr,
        rb_val,
        role_tr,
        role_val,
        team_tr,
        team_val,
        y_tr,
        y_val,
    ) = _split_train_val(blue_picks, red_picks, blue_bans, red_bans, roles, teams, y, stratify=y)

    device = _get_device()
    train_loader_kwargs = _dataloader_kwargs(shuffle=True)
    val_loader_kwargs = _dataloader_kwargs(shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(bp_tr, dtype=torch.long),
            torch.tensor(rp_tr, dtype=torch.long),
            torch.tensor(bb_tr, dtype=torch.long),
            torch.tensor(rb_tr, dtype=torch.long),
            torch.tensor(role_tr, dtype=torch.float32),
            torch.tensor(team_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        ),
        batch_size=cfg["batch_size"],
        **train_loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(bp_val, dtype=torch.long),
            torch.tensor(rp_val, dtype=torch.long),
            torch.tensor(bb_val, dtype=torch.long),
            torch.tensor(rb_val, dtype=torch.long),
            torch.tensor(role_val, dtype=torch.float32),
            torch.tensor(team_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=cfg["batch_size"],
        **val_loader_kwargs,
    )

    start_epoch = 1
    best_acc = 0.0
    opt_state = None
    sched_state = None
    last_epoch = 0
    min_delta = get("training.early_stopping_min_delta", 0.001)
    expected_input_dim = 4 * nm.EMB_DIM + len(RO