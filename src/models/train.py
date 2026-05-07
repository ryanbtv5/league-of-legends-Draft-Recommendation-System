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
import pathlib

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_processed
from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.models import neural as nm
from src.models import transformer as tm
from src.utils.config import get, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED: int = get("project.random_seed", 42)
MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _load_flat_data(df: pd.DataFrame, encoder: DraftStateEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Convert processed DataFrame to (X, y) arrays for tree/MLP models."""
    rows = df.to_dict("records")
    X = encoder.encode_batch(rows)
    y = np.array(encoder.enc.encode_many(df["champion_id"].tolist()), dtype=np.int64)
    return X, y


def _load_sequence_data(df: pd.DataFrame, encoder: ChampionEncoder, max_len: int = 20) -> np.ndarray:
    """Build padded token sequences (one per match) for the Transformer."""
    sequences: list[list[int]] = []
    for _, group in df.groupby("match_id"):
        # Interleave blue bans, red bans, blue picks, red picks in a simple order
        blue_bans = group[group["team"] == "blue"]["champion_id"].tolist()[:5]
        red_bans = group[group["team"] == "red"]["champion_id"].tolist()[:5]
        blue_picks = group[group["team"] == "blue"]["champion_id"].tolist()[:5]
        red_picks = group[group["team"] == "red"]["champion_id"].tolist()[:5]

        seq: list[int] = []
        for b, r in zip(blue_bans, red_bans):
            seq.extend([encoder.encode(b) + 1, encoder.encode(r) + 1])
        for b, r in zip(blue_picks, red_picks):
            seq.extend([encoder.encode(b) + 1, encoder.encode(r) + 1])

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
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED)

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


def train_mlp(df: pd.DataFrame, encoder: DraftStateEncoder, epochs: int = 50) -> None:
    cfg = load_config()["model"]["neural"]
    X, y = _load_flat_data(df, encoder)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = nm.build_dataloader(X_tr, y_tr, batch_size=cfg["batch_size"])
    val_loader = nm.build_dataloader(X_val, y_val, batch_size=cfg["batch_size"], shuffle=False)

    # Adapt input dim to match DraftStateEncoder output
    feature_dim = X_tr.shape[1]
    model = nm.DraftMLP(num_champions=encoder.enc.num_champions)
    # Override first layer to accept flat feature vector
    model.net[0] = torch.nn.Linear(feature_dim, cfg["hidden_dims"][0])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri(get("mlflow.tracking_uri", "mlruns"))
    mlflow.set_experiment(get("mlflow.experiment_name", "draft-recommendation"))

    best_acc = 0.0
    patience = get("training.early_stopping_patience", 10)
    no_improve = 0

    with mlflow.start_run(run_name="mlp_neural"):
        mlflow.log_params({**cfg, "epochs": epochs})
        for epoch in range(1, epochs + 1):
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

            if val_metrics["top1_acc"] > best_acc:
                best_acc = val_metrics["top1_acc"]
                nm.save_model(model, MODEL_DIR / "mlp_recommender_best.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        mlflow.log_metric("best_val_top1_acc", best_acc)
        nm.save_model(model, MODEL_DIR / "mlp_recommender.pt")


def train_transformer(df: pd.DataFrame, encoder: ChampionEncoder, epochs: int = 100) -> None:
    cfg = load_config()["model"]["transformer"]
    sequences = _load_sequence_data(df, encoder)
    n = len(sequences)
    split = int(n * 0.85)
    train_seq, val_seq = sequences[:split], sequences[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = tm.build_sequence_dataloader(train_seq, batch_size=cfg["batch_size"])
    val_loader = tm.build_sequence_dataloader(val_seq, batch_size=cfg["batch_size"], shuffle=False)

    model = tm.DraftTransformer(num_champions=encoder.num_champions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri(get("mlflow.tracking_uri", "mlruns"))
    mlflow.set_experiment(get("mlflow.experiment_name", "draft-recommendation"))

    best_loss = float("inf")
    patience = get("training.early_stopping_patience", 10)
    no_improve = 0

    with mlflow.start_run(run_name="transformer"):
        mlflow.log_params({**cfg, "epochs": epochs})
        for epoch in range(1, epochs + 1):
            train_loss = tm.train_epoch(model, train_loader, optimizer, device)
            val_loss = tm.evaluate_epoch(model, val_loader, device)
            scheduler.step()

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f", epoch, epochs, train_loss, val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                tm.save_model(model, MODEL_DIR / "transformer_recommender_best.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        tm.save_model(model, MODEL_DIR / "transformer_recommender.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a draft recommendation model")
    parser.add_argument("--model", choices=["rf", "mlp", "transformer"], default="rf")
    parser.add_argument("--data", default="data/processed/drafts.parquet")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = load_processed(pathlib.Path(args.data))
    all_champion_ids = sorted(df["champion_id"].unique().tolist())
    champ_enc = ChampionEncoder(all_champion_ids)
    state_enc = DraftStateEncoder(champ_enc)

    if args.model == "rf":
        train_rf(df, state_enc)
    elif args.model == "mlp":
        epochs = args.epochs or get("model.neural.epochs", 50)
        train_mlp(df, state_enc, epochs=epochs)
    elif args.model == "transformer":
        epochs = args.epochs or get("model.transformer.epochs", 100)
        train_transformer(df, champ_enc, epochs=epochs)
