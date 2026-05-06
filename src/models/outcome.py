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
from typing import Iterable, Sequence

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


def _as_index_array(encoder: ChampionEncoder, champion_ids: Sequence[int]) -> np.ndarray:
    filtered = [cid for cid in champion_ids if cid != 0]
    if not filtered:
        return np.array([], dtype=np.int64)
    idxs = np.array([encoder.encode(cid) for cid in filtered], dtype=np.int64)
    if idxs.size == 0:
        return np.array([], dtype=np.int64)
    return np.unique(idxs)


def _within_team_sum_max(idxs: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    if idxs.size < 2:
        return 0.0, 0.0
    sub = matrix[np.ix_(idxs, idxs)]
    tri = sub[np.triu_indices(len(idxs), k=1)]
    if tri.size == 0:
        return 0.0, 0.0
    return float(tri.sum()), float(tri.max())


def _cross_team_sum_max(left: np.ndarray, right: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    if left.size == 0 or right.size == 0:
        return 0.0, 0.0
    sub = matrix[np.ix_(left, right)]
    return float(sub.sum()), float(sub.max())


def _pair_additions_left(
    matrix: np.ndarray,
    candidate_idxs: np.ndarray,
    fixed_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_idxs.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if fixed_right.size == 0:
        zeros = np.zeros(candidate_idxs.size, dtype=np.float32)
        return zeros, zeros
    sub = matrix[np.ix_(candidate_idxs, fixed_right)]
    return sub.sum(axis=1).astype(np.float32), sub.max(axis=1).astype(np.float32)


def _pair_additions_right(
    matrix: np.ndarray,
    fixed_left: np.ndarray,
    candidate_idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_idxs.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if fixed_left.size == 0:
        zeros = np.zeros(candidate_idxs.size, dtype=np.float32)
        return zeros, zeros
    sub = matrix[np.ix_(fixed_left, candidate_idxs)]
    return sub.sum(axis=0).astype(np.float32), sub.max(axis=0).astype(np.float32)


def _safe_divide(values: np.ndarray, denom: int) -> np.ndarray:
    if denom <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / float(denom)).astype(np.float32)


def _blue_win_probabilities(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(features)
    if probs.ndim == 1:
        return probs.astype(np.float32)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            return probs[:, classes.index(1)].astype(np.float32)
    if probs.shape[1] > 1:
        return probs[:, 1].astype(np.float32)
    return probs[:, 0].astype(np.float32)


def recommend_champions(
    model: RandomForestClassifier,
    encoder: DraftInteractionEncoder,
    *,
    blue_picks: Sequence[int],
    red_picks: Sequence[int],
    blue_bans: Sequence[int],
    red_bans: Sequence[int],
    team: str,
    top_k: int = 5,
    candidate_ids: Sequence[int] | None = None,
) -> list[tuple[int, float]]:
    """Recommend champions that maximize win probability for the given team."""
    if team not in {"blue", "red"}:
        raise ValueError("team must be 'blue' or 'red'")

    excluded_ids = set(blue_picks) | set(red_picks) | set(blue_bans) | set(red_bans) | {0}
    if candidate_ids is None:
        champion_ids = [encoder.enc.decode(i) for i in range(encoder.n)]
        candidate_ids = [cid for cid in champion_ids if cid not in excluded_ids and cid != -1]
    else:
        candidate_ids = [cid for cid in candidate_ids if cid not in excluded_ids]
    candidate_ids = list(dict.fromkeys(candidate_ids))

    if not candidate_ids:
        return []

    candidate_idxs = np.array([encoder.enc.encode(cid) for cid in candidate_ids], dtype=np.int64)

    blue_vec = encoder.team_vector(blue_picks)
    red_vec = encoder.team_vector(red_picks)
    blue_ban_vec = encoder.team_vector(blue_bans)
    red_ban_vec = encoder.team_vector(red_bans)

    base_vec = np.concatenate([blue_vec, red_vec, blue_ban_vec, red_ban_vec]).astype(np.float32)
    features = np.repeat(base_vec[None, :], candidate_idxs.size, axis=0)
    offset = 0 if team == "blue" else encoder.n
    features[np.arange(candidate_idxs.size), offset + candidate_idxs] = 1.0

    blue_idxs = _as_index_array(encoder.enc, blue_picks)
    red_idxs = _as_index_array(encoder.enc, red_picks)

    base_blue_sum, base_blue_max = _within_team_sum_max(blue_idxs, encoder.synergy)
    base_red_sum, base_red_max = _within_team_sum_max(red_idxs, encoder.synergy)
    base_bvr_sum, base_bvr_max = _cross_team_sum_max(blue_idxs, red_idxs, encoder.counter)
    base_rvb_sum, base_rvb_max = _cross_team_sum_max(red_idxs, blue_idxs, encoder.counter)

    blue_pair_count = int(len(blue_idxs) * (len(blue_idxs) - 1) / 2)
    red_pair_count = int(len(red_idxs) * (len(red_idxs) - 1) / 2)
    cross_pair_count = int(len(blue_idxs) * len(red_idxs))

    if team == "blue":
        blue_add_sum, blue_add_max = _pair_additions_left(encoder.synergy, candidate_idxs, blue_idxs)
        blue_sum = base_blue_sum + blue_add_sum
        blue_mean = _safe_divide(blue_sum, blue_pair_count + len(blue_idxs))
        blue_max = np.maximum(base_blue_max, blue_add_max)

        red_sum = np.full(candidate_idxs.size, base_red_sum, dtype=np.float32)
        red_mean = _safe_divide(red_sum, red_pair_count)
        red_max = np.full(candidate_idxs.size, base_red_max, dtype=np.float32)

        bvr_add_sum, bvr_add_max = _pair_additions_left(encoder.counter, candidate_idxs, red_idxs)
        bvr_sum = base_bvr_sum + bvr_add_sum
        bvr_mean = _safe_divide(bvr_sum, cross_pair_count + len(red_idxs))
        bvr_max = np.maximum(base_bvr_max, bvr_add_max)

        rvb_add_sum, rvb_add_max = _pair_additions_right(encoder.counter, red_idxs, candidate_idxs)
        rvb_sum = base_rvb_sum + rvb_add_sum
        rvb_mean = _safe_divide(rvb_sum, cross_pair_count + len(red_idxs))
        rvb_max = np.maximum(base_rvb_max, rvb_add_max)
    else:
        blue_sum = np.full(candidate_idxs.size, base_blue_sum, dtype=np.float32)
        blue_mean = _safe_divide(blue_sum, blue_pair_count)
        blue_max = np.full(candidate_idxs.size, base_blue_max, dtype=np.float32)

        red_add_sum, red_add_max = _pair_additions_left(encoder.synergy, candidate_idxs, red_idxs)
        red_sum = base_red_sum + red_add_sum
        red_mean = _safe_divide(red_sum, red_pair_count + len(red_idxs))
        red_max = np.maximum(base_red_max, red_add_max)

        bvr_add_sum, bvr_add_max = _pair_additions_right(encoder.counter, blue_idxs, candidate_idxs)
        bvr_sum = base_bvr_sum + bvr_add_sum
        bvr_mean = _safe_divide(bvr_sum, cross_pair_count + len(blue_idxs))
        bvr_max = np.maximum(base_bvr_max, bvr_add_max)

        rvb_add_sum, rvb_add_max = _pair_additions_left(encoder.counter, candidate_idxs, blue_idxs)
        rvb_sum = base_rvb_sum + rvb_add_sum
        rvb_mean = _safe_divide(rvb_sum, cross_pair_count + len(blue_idxs))
        rvb_max = np.maximum(base_rvb_max, rvb_add_max)

    interaction = np.column_stack(
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
        ]
    ).astype(np.float32)

    features = np.concatenate([features, interaction], axis=1)
    blue_probs = _blue_win_probabilities(model, features)
    win_probs = blue_probs if team == "blue" else 1.0 - blue_probs

    top_k = min(top_k, len(candidate_ids))
    order = np.argsort(win_probs)[::-1][:top_k]
    return [(int(candidate_ids[i]), float(win_probs[i])) for i in order]


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
