"""
evaluation/evaluate.py
-----------------------
Run a full evaluation sweep across all saved models and print a comparison table.

Usage:
    python -m evaluation.evaluate --data data/processed/drafts.parquet
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.metrics import compute_all
from src.data.preprocess import load_processed
from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.models import neural as nm
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED: int = get("project.random_seed", 42)
MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))
K_VALUES: list[int] = get("evaluation.top_k", [1, 3, 5])


def _flat_data(df: pd.DataFrame, state_enc: DraftStateEncoder) -> tuple[np.ndarray, np.ndarray]:
    rows = df.to_dict("records")
    X = state_enc.encode_batch(rows)
    y = np.array(state_enc.enc.encode_many(df["champion_id"].tolist()), dtype=np.int64)
    return X, y


def evaluate_all(data_path: pathlib.Path = pathlib.Path("data/processed/drafts.parquet")) -> pd.DataFrame:
    """Evaluate all available models and return a comparison DataFrame.

    Args:
        data_path: Path to the processed Parquet file.

    Returns:
        :class:`pandas.DataFrame` with one row per model and metric columns.
    """
    df = load_processed(data_path)
    all_ids = sorted(df["champion_id"].unique().tolist())
    champ_enc = ChampionEncoder(all_ids)
    state_enc = DraftStateEncoder(champ_enc)

    X, y = _flat_data(df, state_enc)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)

    results: list[dict] = []

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_path = MODEL_DIR / "xgb_recommender.pkl"
    if xgb_path.exists():
        logger.info("Evaluating XGBoost …")
        model = bm.XGBoostRecommender.load(xgb_path)
        scores = model.predict_proba(X_test)
        metrics = compute_all(y_test, scores, k_values=K_VALUES)
        results.append({"model": "XGBoost", **metrics})
    else:
        logger.warning("XGBoost model not found at %s", xgb_path)

    # ── MLP ──────────────────────────────────────────────────────────────────
    mlp_path = MODEL_DIR / "mlp_recommender_best.pt"
    if mlp_path.exists():
        import torch
        logger.info("Evaluating MLP …")
        device = torch.device("cpu")
        mlp = nm.load_model(mlp_path, device)
        loader = nm.build_dataloader(X_test, y_test, batch_size=512, shuffle=False)
        all_probs: list[np.ndarray] = []
        mlp.eval()
        with torch.no_grad():
            for X_b, _ in loader:
                logits = mlp.net(X_b.to(device))
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        scores = np.concatenate(all_probs, axis=0)
        metrics = compute_all(y_test, scores, k_values=K_VALUES)
        results.append({"model": "MLP", **metrics})
    else:
        logger.warning("MLP model not found at %s", mlp_path)

    if not results:
        logger.error("No models found. Train at least one model first.")
        return pd.DataFrame()

    report = pd.DataFrame(results).set_index("model")
    print("\n=== Draft Recommendation Evaluation ===")
    print(report.to_string(float_format="{:.4f}".format))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate draft recommendation models")
    parser.add_argument("--data", default="data/processed/drafts.parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_all(pathlib.Path(args.data))
