"""
evaluation/evaluate.py
-----------------------
Run a full evaluation sweep across saved recommendation models and print a comparison table.

Usage:
    python -m evaluation.evaluate --data data/processed/drafts.parquet
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
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


def _align_scores(scores: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    if scores.shape[1] == num_classes and np.array_equal(classes, np.arange(num_classes)):
        return scores
    aligned = np.zeros((scores.shape[0], num_classes), dtype=np.float32)
    aligned[:, classes] = scores
    return aligned


def _multiclass_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _draft_summary(row: pd.Series) -> dict[str, object]:
    summary: dict[str, object] = {}

    def _values(prefix: str) -> list[int] | None:
        cols = [f"{prefix}_{i}" for i in range(1, 6)]
        if all(col in row.index for col in cols):
            return [int(row[col]) for col in cols]
        return None

    if "blue_picks_so_far" in row.index:
        summary["blue_picks"] = _to_serializable(row["blue_picks_so_far"])
    if "red_picks_so_far" in row.index:
        summary["red_picks"] = _to_serializable(row["red_picks_so_far"])
    if "blue_bans" in row.index:
        summary["blue_bans"] = _to_serializable(row["blue_bans"])
    if "red_bans" in row.index:
        summary["red_bans"] = _to_serializable(row["red_bans"])

    for key in ("blue_pick", "red_pick", "blue_ban", "red_ban"):
        values = _values(key)
        if values is not None:
            summary.setdefault(f"{key}s", values)

    for key in ("team", "role", "pick_order", "draft_step"):
        if key in row.index:
            summary[key] = _to_serializable(row[key])

    return summary


def _to_serializable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _error_examples(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    champ_enc: ChampionEncoder,
    *,
    k: int = 5,
    max_examples: int = 5,
) -> list[dict[str, object]]:
    y_pred = scores.argmax(axis=1)
    wrong = np.where(y_pred != y_true)[0][:max_examples]
    examples: list[dict[str, object]] = []
    for idx in wrong:
        row = df.iloc[idx]
        top_k = np.argsort(scores[idx])[::-1][:k]
        examples.append(
            {
                "draft": _draft_summary(row),
                "true_champion": champ_enc.decode(int(y_true[idx])),
                "predicted_champion": champ_enc.decode(int(y_pred[idx])),
                "top_k": [champ_enc.decode(int(i)) for i in top_k],
            }
        )
    return examples


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    champ_enc: ChampionEncoder,
    output_path: pathlib.Path,
) -> None:
    labels = list(range(champ_enc.num_champions))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    champ_ids = [champ_enc.decode(idx) for idx in labels]
    df = pd.DataFrame(matrix, index=champ_ids, columns=champ_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def _flat_data(df: pd.DataFrame, state_enc: DraftStateEncoder) -> tuple[np.ndarray, np.ndarray]:
    rows = df.to_dict("records")
    X = state_enc.encode_batch(rows)
    y = np.array(state_enc.enc.encode_many(df["champion_id"].tolist()), dtype=np.int64)
    return X, y


def evaluate_all(
    data_path: pathlib.Path = pathlib.Path("data/processed/drafts.parquet"),
    *,
    output_dir: pathlib.Path = pathlib.Path("evaluation/reports"),
    error_examples: int = 5,
) -> pd.DataFrame:
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
    indices = np.arange(len(df))
    _, test_idx, _, y_test = train_test_split(indices, y, test_size=0.15, random_state=SEED)
    X_test = X[test_idx]
    df_test = df.iloc[test_idx].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    def _evaluate_model(name: str, scores: np.ndarray) -> None:
        y_pred = scores.argmax(axis=1)
        metrics = compute_all(y_test, scores, k_values=K_VALUES)
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["auc"] = _multiclass_auc(y_test, scores)
        results.append({"model": name, **metrics})

        confusion_path = output_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv"
        _save_confusion_matrix(y_test, y_pred, champ_enc, confusion_path)
        logger.info("Saved %s confusion matrix to %s", name, confusion_path)

        examples = _error_examples(df_test, y_test, scores, champ_enc, k=max(K_VALUES), max_examples=error_examples)
        examples_path = output_dir / f"{name.lower().replace(' ', '_')}_errors.json"
        examples_path.write_text(json.dumps(examples, indent=2))
        logger.info("Saved %s error examples to %s", name, examples_path)
        if examples:
            logger.info("Sample %s errors: %s", name, examples[:3])

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_path = MODEL_DIR / "xgb_recommender.pkl"
    if xgb_path.exists():
        logger.info("Evaluating XGBoost …")
        model = bm.XGBoostRecommender.load(xgb_path)
        scores = model.predict_proba(X_test)
        _evaluate_model("XGBoost", scores)
    else:
        logger.warning("XGBoost model not found at %s", xgb_path)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = MODEL_DIR / "rf_recommender.pkl"
    if rf_path.exists():
        logger.info("Evaluating Random Forest …")
        model = bm.RandomForestRecommender.load(rf_path)
        scores = model.predict_proba(X_test)
        scores = _align_scores(scores, model.model.classes_, champ_enc.num_champions)
        _evaluate_model("Random Forest", scores)
    else:
        logger.warning("Random Forest model not found at %s", rf_path)

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
        _evaluate_model("MLP", scores)
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
    parser.add_argument("--output-dir", default="evaluation/reports")
    parser.add_argument("--error-examples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_all(
        pathlib.Path(args.data),
        output_dir=pathlib.Path(args.output_dir),
        error_examples=args.error_examples,
    )
