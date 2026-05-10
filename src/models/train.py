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
    expected_input_dim = 4 * nm.EMB_DIM + len(ROLES) + 2

    if resume_from is not None and resume_from.exists():
        logger.info("Resuming MLP from checkpoint: %s", resume_from)
        try:
            model, opt_state, sched_state, ckpt_epoch, best_acc = nm.load_checkpoint(resume_from, device)
            if model.net[0].in_features != expected_input_dim:
                logger.warning(
                    "Checkpoint input dim %d does not match embedding trainer input dim %d; starting fresh.",
                    model.net[0].in_features,
                    expected_input_dim,
                )
                raise ValueError("incompatible checkpoint shape")
            start_epoch = ckpt_epoch + 1
            last_epoch = ckpt_epoch
        except Exception:
            model = nm.DraftMLP(num_champions=encoder.enc.num_champions)
            opt_state = None
            sched_state = None
            best_acc = 0.0
    else:
        model = nm.DraftMLP(num_champions=encoder.enc.num_champions)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)

    mlflow.set_tracking_uri(get("mlflow.tracking_uri", "mlruns"))
    mlflow.set_experiment(get("mlflow.experiment_name", "draft-recommendation"))

    patience = get("training.early_stopping_patience", 10)
    no_improve = 0

    with mlflow.start_run(run_name="mlp_neural"):
        mlflow.log_params({**cfg, "epochs": epochs, "resumed_from": str(resume_from) if resume_from else "", "start_epoch": start_epoch})
        if start_epoch > epochs:
            logger.info("Resume checkpoint is already beyond target epoch (%d > %d); nothing to train.", start_epoch, epochs)
        for epoch in range(start_epoch, epochs + 1):
            last_epoch = epoch
            train_loss = nm.train_epoch(model, train_loader, optimizer, device)
            val_metrics = nm.evaluate(model, val_loader, device)
            scheduler.step()

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_metrics["loss"], "val_top1_acc": val_metrics["top1_acc"]},
                step=epoch,
            )
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_top1=%.4f",
                epoch, epochs, train_loss, val_metrics["loss"], val_metrics["top1_acc"],
            )

            if val_metrics["top1_acc"] > best_acc + min_delta:
                best_acc = val_metrics["top1_acc"]
                nm.save_checkpoint(
                    model,
                    MODEL_DIR / "mlp_recommender_best.pt",
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    epoch=epoch,
                    best_acc=best_acc,
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        mlflow.log_metric("best_val_top1_acc", best_acc)
        nm.save_checkpoint(
            model,
            MODEL_DIR / "mlp_recommender.pt",
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=last_epoch,
            best_acc=best_acc,
        )


def train_transformer(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
    epochs: int = 100,
    resume_from: pathlib.Path | None = None,
) -> None:
    cfg = load_config()["model"]["transformer"]
    sequences = _load_sequence_data(df, encoder)
    split_ratio = get("training.val_split", 0.15)
    train_seq, val_seq = train_test_split(sequences, test_size=split_ratio, random_state=SEED)

    device = _get_device()
    workers = int(get("training.num_workers", min(4, os.cpu_count() or 1)))
    pin_memory = bool(get("training.pin_memory", True))
    train_loader = tm.build_sequence_dataloader(
        train_seq,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=max(0, workers),
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    val_loader = tm.build_sequence_dataloader(
        val_seq,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=max(0, workers),
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    start_epoch = 1
    best_loss = float("inf")
    no_improve = 0
    min_delta = get("training.early_stopping_min_delta", 0.001)

    if resume_from is not None and resume_from.exists():
        logger.info("Resuming from checkpoint: %s", resume_from)
        model, opt_state, sched_state, ckpt_epoch = tm.load_checkpoint(resume_from, device)
        start_epoch = ckpt_epoch + 1
        no_improve = 0  # reset patience counter on resume
        # Note: best_loss not recovered; training will continue fresh from this point
    else:
        model = tm.DraftTransformer(num_champions=encoder.num_champions).to(device)
        opt_state = None
        sched_state = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)

    mlflow.set_tracking_uri(get("mlflow.tracking_uri", "mlruns"))
    mlflow.set_experiment(get("mlflow.experiment_name", "draft-recommendation"))

    patience = get("training.early_stopping_patience", 10)

    with mlflow.start_run(run_name="transformer"):
        mlflow.log_params({**cfg, "epochs": epochs, "resumed_from_epoch": start_epoch - 1})
        for epoch in range(start_epoch, epochs + 1):
            train_loss = tm.train_epoch(model, train_loader, optimizer, device)
            val_loss = tm.evaluate_epoch(model, val_loader, device)
            scheduler.step()

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f", epoch, epochs, train_loss, val_loss)

            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                tm.save_model(
                    model,
                    MODEL_DIR / "transformer_recommender_best.pt",
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    epoch=epoch,
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        tm.save_model(
            model,
            MODEL_DIR / "transformer_recommender.pt",
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=epoch,
        )


# ---------------------------------------------------------------------------
# Win prediction training
# ---------------------------------------------------------------------------

def _load_win_prediction_data(
    df: pd.DataFrame,
    encoder: ChampionEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Build draft sequences and win labels for the DraftWinPredictor.
    
    Args:
        df: DataFrame with draft and outcome columns
        encoder: ChampionEncoder to map champion IDs to indices
    
    Returns:
        (sequences, labels) where sequences is (N, 20) token array and labels is (N,) binary array
    """
    sequences: list[list[int]] = []
    labels: list[int] = []
    
    draft_order_cols = [
        "blue_ban_1", "red_ban_1", "blue_ban_2", "red_ban_2", "blue_ban_3", "red_ban_3",
        "blue_pick_1", "red_pick_1", "red_pick_2", "blue_pick_2", "blue_pick_3", "red_pick_3",
        "red_ban_4", "blue_ban_4", "red_ban_5", "blue_ban_5",
        "red_pick_4", "blue_pick_4", "blue_pick_5", "red_pick_5",
    ]
    
    for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
        final_row = group.iloc[-1]
        blue_win = int(final_row["blue_win"])
        
        # Build token sequence from full draft (encoded champion indices + 1 for padding offset)
        seq = []
        for col in draft_order_cols:
            champ_id = int(final_row[col])
            if champ_id != 0:
                # Encode the champion ID and add 1 for padding offset (0 is reserved for padding)
                encoded_idx = encoder.encode(champ_id) + 1
                seq.append(encoded_idx)
        
        # Truncate to 20 tokens max
        seq = seq[:20]
        # Pad with 0s
        seq += [0] * (20 - len(seq))
        
        sequences.append(seq)
        labels.append(blue_win)
    
    return np.array(sequences, dtype=np.int64), np.array(labels, dtype=np.float32)


def train_win_predictor(df: pd.DataFrame, encoder: ChampionEncoder, epochs: int = 50) -> None:
    """Train the DraftWinPredictor model on match outcomes."""
    cfg = load_config()["model"].get("win_predictor", {})
    sequences, labels = _load_win_prediction_data(df, encoder)
    # Build role-aware matrices and counters to produce aggregated extra features
    from src.features.synergy_counter import (
        build_role_winrate_matrix,
        build_counter_matrix,
    )
    role_win = build_role_winrate_matrix(df, encoder)
    counter_mat = build_counter_matrix(df, encoder)

    # Reconstruct extra features per final match row (blue_role_mean, red_role_mean, cross_counter_mean)
    pick_cols_blue = [f"blue_pick_{i}" for i in range(1, len(ROLES) + 1)]
    pick_cols_red = [f"red_pick_{i}" for i in range(1, len(ROLES) + 1)]
    extra_feats = []
    for _, group in df.sort_values(["match_id", "draft_step"]).groupby("match_id", sort=False):
        final_row = group.iloc[-1]
        blue_rates = []
        red_rates = []
        for role_idx, col in enumerate(pick_cols_blue):
            cid = int(final_row.get(col, 0))
            if cid == 0:
                continue
            idx = encoder.encode(cid)
            blue_rates.append(float(role_win[idx, role_idx]))
        for role_idx, col in enumerate(pick_cols_red):
            cid = int(final_row.get(col, 0))
            if cid == 0:
                continue
            idx = encoder.encode(cid)
            red_rates.append(float(role_win[idx, role_idx]))

        blue_mean = float(sum(blue_rates) / len(blue_rates)) if blue_rates else 0.5
        red_mean = float(sum(red_rates) / len(red_rates)) if red_rates else 0.5

        # cross-team counter mean (average over all blue vs red pairs)
        bidxs = [encoder.encode(int(final_row.get(c, 0))) for c in pick_cols_blue if int(final_row.get(c, 0)) != 0]
        ridxs = [encoder.encode(int(final_row.get(c, 0))) for c in pick_cols_red if int(final_row.get(c, 0)) != 0]
        pair_vals = []
        for bi in bidxs:
            for rj in ridxs:
                pair_vals.append(float(counter_mat[bi, rj]))
        cross_mean = float(sum(pair_vals) / len(pair_vals)) if pair_vals else 0.5
        extra_feats.append([blue_mean, red_mean, cross_mean])
    extra_feats = np.array(extra_feats, dtype=np.float32)
    
    split_ratio = get("training.val_split", 0.15)
    train_seq, val_seq, train_labels, val_labels, train_extra, val_extra = train_test_split(
        sequences,
        labels,
        extra_feats,
        test_size=split_ratio,
        random_state=SEED,
        stratify=labels,
    )

    device = _get_device()
    train_loader_kwargs = _dataloader_kwargs(shuffle=True)
    val_loader_kwargs = _dataloader_kwargs(shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_seq).long(),
            torch.from_numpy(train_labels).float(),
            torch.from_numpy(train_extra).float(),
        ),
        batch_size=cfg.get("batch_size", 32),
        **train_loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(val_seq).long(),
            torch.from_numpy(val_labels).float(),
            torch.from_numpy(val_extra).float(),
        ),
        batch_size=cfg.get("batch_size", 32),
        **val_loader_kwargs,
    )

    model = DraftWinPredictor(
        num_champions=encoder.num_champions,
        d_model=cfg.get("d_model", get("model.win_predictor.d_model", 128)),
        nhead=cfg.get("nhead", get("model.win_predictor.nhead", 4)),
        num_layers=cfg.get("num_layers", get("model.win_predictor.num_layers", 3)),
        dim_feedforward=cfg.get("dim_feedforward", get("model.win_predictor.dim_feedforward", 256)),
        dropout=cfg.get("dropout", get("model.win_predictor.dropout", 0.1)),
        extra_feat_dim=extra_feats.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-3), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    mlflow.set_tracking_uri(get("mlflow.tracking_uri", "mlruns"))
    mlflow.set_experiment(get("mlflow.experiment_name", "draft-recommendation"))
    
    best_loss = float("inf")
    patience = get("training.early_stopping_patience", 10)
    min_delta = get("training.early_stopping_min_delta", 0.001)
    no_improve = 0
    
    with mlflow.start_run(run_name="win_predictor"):
        mlflow.log_params({**cfg, "epochs": epochs})
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_seq, batch_labels, batch_extra in train_loader:
                batch_seq = batch_seq.to(device)
                batch_labels = batch_labels.to(device)
                batch_extra = batch_extra.to(device)

                optimizer.zero_grad()
                logits = model(batch_seq, batch_extra)
                loss = criterion(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch_seq, batch_labels, batch_extra in val_loader:
                    batch_seq = batch_seq.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_extra = batch_extra.to(device)

                    logits = model(batch_seq, batch_extra)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()

                    preds = (logits.squeeze(-1) > 0).float()
                    val_acc += (preds == batch_labels).float().mean().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            scheduler.step()
            
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch, epochs, train_loss, val_loss, val_acc,
            )
            
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                torch.save(model.state_dict(), MODEL_DIR / "win_predictor_best.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        
        # Save final model
        torch.save(model.state_dict(), MODEL_DIR / "win_predictor.pt")
        mlflow.log_metric("best_val_loss", best_loss)
        logger.info("Win predictor training complete. Best val_loss: %.4f", best_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a draft recommendation model")
    parser.add_argument("--model", choices=["rf", "mlp", "transformer", "win"], default="rf")
    parser.add_argument("--data", default="data/processed/drafts.parquet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume-from", type=pathlib.Path, default=None, help="Resume training from a checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = load_processed(pathlib.Path(args.data))
    all_champion_ids = _unique_champion_ids(df)
    champ_enc = ChampionEncoder(all_champion_ids)
    
    if args.model == "win":
        # Win predictor training
        epochs = args.epochs or get("model.win_predictor.epochs", 50)
        train_win_predictor(df, champ_enc, epochs=epochs)
    else:
        state_enc = DraftStateEncoder(champ_enc)

        if args.model == "rf":
            train_rf(df, state_enc)
        elif args.model == "mlp":
            epochs = args.epochs or get("model.neural.epochs", 50)
            train_mlp(df, state_enc, epochs=epochs, resume_from=args.resume_from)
        elif args.model == "transformer":
            epochs = args.epochs or get("model.transformer.epochs", 100)
            train_transformer(df, champ_enc, epochs=epochs, resume_from=args.resume_from)
            # Win predictor training disabled: draft state alone cannot reliably predict match outcomes (~50% baseline)
            # To enable: logger.info("Training win predictor..."); train_win_predictor(...)
