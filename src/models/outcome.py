"""
src/models/outcome.py
---------------------
Train a Random Forest to predict match outcome from engineered draft features.

Usage:
    python -m src.models.outcome --data data/processed/drafts.parquet
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data.preprocess import load_processed
from src.features import ChampionEncoder, DraftInteractionEncoder
from src.utils.config import get
from src.utils.logger import get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger(__name__)

SEED: int = get("project.random_seed", 42)
PICKS_PER_TEAM: int = get("data.picks_per_team", 5)
BANS_PER_TEAM: int = get("data.bans_per_team", 5)
MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))


def _team_columns(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, count + 1)]


def _draft_columns() -> dict[str, list[str]]:
    return {
        "blue_picks": _team_columns("blue_pick", PICKS_PER_TEAM),
        "red_picks": _team_columns("red_pick", PICKS_PER_TEAM),
        "blue_bans": _team_columns("blue_ban", BANS_PER_TEAM),
        "red_bans": _team_columns("red_ban", BANS_PER_TEAM),
    }


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _load_interaction_matrices() -> tuple[np.ndarray | None, np.ndarray | None]:
    processed_dir = pathlib.Path(get("data.processed_dir", "data/processed"))
    synergy_path = processed_dir / "synergy_matrix.npy"
    counter_path = processed_dir / "counter_matrix.npy"
    if synergy_path.exists() and counter_path.exists():
        return np.load(synergy_path), np.load(counter_path)
    logger.warning("Synergy/counter matrices not found in %s; using zeros.", processed_dir)
    return None, None


def _unique_champion_ids(df: pd.DataFrame, columns: Iterable[str]) -> list[int]:
    values = df[list(columns)].to_numpy(dtype=np.int64, copy=False)
    champion_ids = np.unique(values)
    return sorted(int(cid) for cid in champion_ids if cid != 0)


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: DraftInteractionEncoder,
    columns: dict[str, list[str]],
) -> np.ndarray:
    blue_picks = df[columns["blue_picks"]].to_numpy(dtype=np.int64, copy=False)
    red_picks = df[columns["red_picks"]].to_numpy(dtype=np.int64, copy=False)
    blue_bans = df[columns["blue_bans"]].to_numpy(dtype=np.int64, copy=False)
    red_bans = df[columns["red_bans"]].to_numpy(dtype=np.int64, copy=False)

    features = np.zeros((len(df), encoder.feature_dim), dtype=np.float32)
    for i in range(len(df)):
        features[i] = encoder.encode(
            blue_picks=blue_picks[i],
            red_picks=red_picks[i],
            blue_bans=blue_bans[i],
            red_bans=red_bans[i],
        )
    return features


def build_feature_names(champ_enc: ChampionEncoder) -> list[str]:
    champ_ids = [champ_enc.decode(idx) for idx in range(champ_enc.num_champions)]
    names = [f"blue_pick_{cid}" for cid in champ_ids]
    names += [f"red_pick_{cid}" for cid in champ_ids]
    names += [f"blue_ban_{cid}" for cid in champ_ids]
    names += [f"red_ban_{cid}" for cid in champ_ids]
    names += [
        "blue_synergy_sum",
        "blue_synergy_mean",
        "blue_synergy_max",
        "red_synergy_sum",
        "red_synergy_mean",
        "red_synergy_max",
        "blue_vs_red_counter_sum",
        "blue_vs_red_counter_mean",
        "blue_vs_red_counter_max",
        "red_vs_blue_counter_sum",
        "red_vs_blue_counter_mean",
        "red_vs_blue_counter_max",
    ]
    return names


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    output_path: pathlib.Path,
    *,
    top_k: int = 20,
) -> None:
    top_k = min(top_k, len(importances))
    order = np.argsort(importances)[::-1][:top_k]
    labels = [feature_names[i] for i in order][::-1]
    values = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * top_k)))
    ax.barh(range(top_k), values, color="#4C72B0")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest Feature Importance (Top-K)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", output_path)


def prepare_dataset(
    data_path: pathlib.Path,
    *,
    final_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], DraftInteractionEncoder]:
    df = load_processed(data_path)
    if df.empty:
        raise ValueError(f"No data found at {data_path}")
    if final_only:
        max_step = int(df["draft_step"].max())
        df = df[df["draft_step"] == max_step].copy()

    columns = _draft_columns()
    all_cols = sum(columns.values(), [])
    _validate_columns(df, all_cols)

    champion_ids = _unique_champion_ids(df, all_cols)
    champ_enc = ChampionEncoder(champion_ids)
    synergy, counter = _load_interaction_matrices()
    feature_enc = DraftInteractionEncoder(champ_enc, synergy, counter)

    X = build_feature_matrix(df, feature_enc, columns)
    y = df["blue_win"].astype(int).to_numpy()
    feature_names = build_feature_names(champ_enc)
    return X, y, feature_names, feature_enc


def cross_validate_model(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: int = 5,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"auc": "roc_auc", "accuracy": "accuracy"},
        n_jobs=-1,
    )
    return {
        "auc_mean": float(scores["test_auc"].mean()),
        "auc_std": float(scores["test_auc"].std()),
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 400,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, model.predict(X_test))
    return float(auc), float(acc)


def run_training(
    *,
    data_path: pathlib.Path,
    test_split: float,
    cv_splits: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    final_only: bool,
    plot_path: pathlib.Path,
    top_features: int,
) -> None:
    X, y, feature_names, _ = prepare_dataset(data_path, final_only=final_only)
    if len(np.unique(y)) < 2:
        raise ValueError("Outcome labels contain a single class; cannot compute AUC.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_split,
        stratify=y,
        random_state=SEED,
    )

    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",
    )
    cv_results = cross_validate_model(base_model, X_train, y_train, splits=cv_splits)
    logger.info(
        "CV AUC %.4f ± %.4f | CV Acc %.4f ± %.4f",
        cv_results["auc_mean"],
        cv_results["auc_std"],
        cv_results["accuracy_mean"],
        cv_results["accuracy_std"],
    )

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    auc, acc = evaluate_model(model, X_test, y_test)
    logger.info("Test AUC: %.4f | Test Accuracy: %.4f", auc, acc)

    plot_feature_importance(model.feature_importances_, feature_names, plot_path, top_k=top_features)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest outcome model")
    parser.add_argument("--data", default="data/processed/drafts.parquet")
    parser.add_argument("--test-split", type=float, default=get("training.test_split", 0.15))
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--all-steps", action="store_true", help="Use all draft steps instead of final draft only.")
    parser.add_argument(
        "--plot",
        default=str(MODEL_DIR / "outcome_feature_importance.png"),
        help="Path to save the feature-importance plot.",
    )
    parser.add_argument("--top-features", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        data_path=pathlib.Path(args.data),
        test_split=args.test_split,
        cv_splits=args.cv_splits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        final_only=not args.all_steps,
        plot_path=pathlib.Path(args.plot),
        top_features=args.top_features,
    )
