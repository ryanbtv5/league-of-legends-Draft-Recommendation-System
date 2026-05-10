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
import time
import os
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from evaluation.metrics import compute_all
from src.data.preprocess import load_processed
from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.models import neural as nm
from src.models.transformer import MAX_SEQ_LEN, build_sequence_dataloader, load_model as load_transformer_model
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED: int = get("project.random_seed", 42)
MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))
K_VALUES: list[int] = get("evaluation.top_k", [1, 3, 5])

BLUE_PICK_COLS = [f"blue_pick_{i}" for i in range(1, 6)]
RED_PICK_COLS = [f"red_pick_{i}" for i in range(1, 6)]
BLUE_BAN_COLS = [f"blue_ban_{i}" for i in range(1, 6)]
RED_BAN_COLS = [f"red_ban_{i}" for i in range(1, 6)]
PICK_COLS = BLUE_PICK_COLS + RED_PICK_COLS
BAN_COLS = BLUE_BAN_COLS + RED_BAN_COLS


def _unique_champion_ids(df: pd.DataFrame) -> list[int]:
    values = df[PICK_COLS + BAN_COLS].to_numpy(dtype=np.int64, copy=False)
    champion_ids = np.unique(values)
    return sorted(int(cid) for cid in champion_ids if cid != 0)


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


def _transformer_sequence_from_state(row: pd.Series, champ_enc: ChampionEncoder) -> np.ndarray:
    """Build a Transformer input sequence from a partial draft state.

    The Transformer was trained on ordered champion pick tokens only, so we
    reconstruct the observed prefix in true pick order and then pad to the
    model's expected max length.
    """
    blue_picks = list(row.get("blue_picks_so_far", []))
    red_picks = list(row.get("red_picks_so_far", []))

    draft_order = [
        ("blue", 0),
        ("red", 0),
        ("red", 1),
        ("blue", 1),
        ("blue", 2),
        ("red", 2),
        ("red", 3),
        ("blue", 3),
        ("blue", 4),
        ("red", 4),
    ]

    sequence: list[int] = []
    blue_idx = 0
    red_idx = 0
    for team, _ in draft_order:
        if team == "blue":
            if blue_idx < len(blue_picks):
                sequence.append(champ_enc.encode(int(blue_picks[blue_idx])) + 1)
                blue_idx += 1
        else:
            if red_idx < len(red_picks):
                sequence.append(champ_enc.encode(int(red_picks[red_idx])) + 1)
                red_idx += 1

    sequence = sequence[: MAX_SEQ_LEN + 1]
    sequence += [0] * (MAX_SEQ_LEN + 1 - len(sequence))
    return np.array(sequence, dtype=np.int64)


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


def _align_scores(scores: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    if scores.shape[1] == num_classes and np.array_equal(classes, np.arange(num_classes)):
        return scores
    aligned = np.zeros((scores.shape[0], num_classes), dtype=np.float32)
    aligned[:, classes] = scores
    return aligned


def _multiclass_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute macro OVR AUC for multiclass predictions, returning NaN on failure."""
    try:
        return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _draft_summary(row: pd.Series) -> dict[str, object]:
    """Summarize available draft fields from a row for error analysis output."""
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
    """Convert numpy/pandas values into JSON-serializable Python types."""
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


def _flat_data(df: pd.DataFrame, state_enc: DraftStateEncoder) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    rows, targets = _build_recommendation_dataset(df)
    X = state_enc.encode_batch(rows)
    y = np.array(state_enc.enc.encode_many(targets.tolist()), dtype=np.int64)
    return rows, X, y


def _mlp_data(rows: list[dict[str, object]], champ_enc: ChampionEncoder) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    blue_picks = np.zeros((n, 5), dtype=np.int64)
    red_picks = np.zeros((n, 5), dtype=np.int64)
    blue_bans = np.zeros((n, 5), dtype=np.int64)
    red_bans = np.zeros((n, 5), dtype=np.int64)
    roles = np.zeros((n, 5), dtype=np.float32)
    teams = np.zeros((n, 2), dtype=np.float32)

    def _pad(ids: list[int], pad_to: int = 5) -> np.ndarray:
        enc = champ_enc.encode_many([int(cid) for cid in ids if int(cid) != 0])[:pad_to]
        out = np.zeros(pad_to, dtype=np.int64)
        if enc:
            out[: len(enc)] = np.asarray(enc, dtype=np.int64)
        return out

    for i, row in enumerate(rows):
        blue_picks[i] = _pad(row.get("blue_picks_so_far", []))
        red_picks[i] = _pad(row.get("red_picks_so_far", []))
        blue_bans[i] = _pad(row.get("blue_bans", []))
        red_bans[i] = _pad(row.get("red_bans", []))
        roles[i, min(int(row.get("pick_order", 0)), 4)] = 1.0
        teams[i, 0 if row.get("team", "blue") == "blue" else 1] = 1.0

    return blue_picks, red_picks, blue_bans, red_bans, roles, teams


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
    all_ids = _unique_champion_ids(df)
    champ_enc = ChampionEncoder(all_ids)
    state_enc = DraftStateEncoder(champ_enc)

    rows, X, y = _flat_data(df, state_enc)
    indices = np.arange(len(y))
    _, test_idx, _, y_test = train_test_split(indices, y, test_size=0.15, random_state=SEED)
    X_test = X[test_idx]
    df_test = pd.DataFrame(rows).iloc[test_idx].reset_index(drop=True)
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

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_path = MODEL_DIR / "rf_recommender.pkl"
    if rf_path.exists():
        logger.info("Evaluating Random Forest …")
        model = bm.RandomForestRecommender.load(rf_path)
        # ensure RF uses all CPUs where possible
        try:
            model.model.n_jobs = os.cpu_count() or -1
        except Exception:
            pass
        t0 = time.time()
        scores = model.predict_proba(X_test)
        logger.info("Random Forest predict_proba time: %.2fs", time.time() - t0)
        scores = _align_scores(scores, model.model.classes_, champ_enc.num_champions)
        _evaluate_model("Random Forest", scores)
    else:
        logger.warning("Random Forest model not found at %s", rf_path)

    # ── MLP ──────────────────────────────────────────────────────────────────
    mlp_path = MODEL_DIR / "mlp_recommender_best.pt"
    if mlp_path.exists():
        logger.info("Evaluating MLP …")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp = nm.load_model(mlp_path, device)
        batch_size = max(512, get("model.neural.batch_size", 512))
        num_workers = 0
        pin_memory = False
        if torch.cuda.is_available():
            num_workers = min(4, (os.cpu_count() or 1) - 1)
            pin_memory = True

        test_rows = df_test.to_dict(orient="records")
        bp, rp, bb, rb, role, team = _mlp_data(test_rows, champ_enc=champ_enc)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(bp, dtype=torch.long),
            torch.tensor(rp, dtype=torch.long),
            torch.tensor(bb, dtype=torch.long),
            torch.tensor(rb, dtype=torch.long),
            torch.tensor(role, dtype=torch.float32),
            torch.tensor(team, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        all_probs: list[np.ndarray] = []
        mlp.eval()
        t0 = time.time()
        with torch.no_grad():
            for bp_b, rp_b, bb_b, rb_b, role_b, team_b in loader:
                bp_b = bp_b.to(device)
                rp_b = rp_b.to(device)
                bb_b = bb_b.to(device)
                rb_b = rb_b.to(device)
                role_b = role_b.to(device)
                team_b = team_b.to(device)
                logits = mlp(bp_b, rp_b, bb_b, rb_b, role_b, team_b)
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        logger.info("MLP inference time: %.2fs", time.time() - t0)
        scores = np.concatenate(all_probs, axis=0)
        _evaluate_model("MLP", scores)
    else:
        logger.warning("MLP model not found at %s", mlp_path)

    # ── Transformer (if present) ────────────────────────────────────────────
    tr_path = MODEL_DIR / "transformer_recommender_best.pt"
    if tr_path.exists():
        logger.info("Evaluating Transformer …")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tm_device = device
        transformer = load_transformer_model(tr_path, tm_device)
        sequences = np.array([
            _transformer_sequence_from_state(row, champ_enc)
            for _, row in df_test.iterrows()
        ], dtype=np.int64)

        if len(sequences) > 0:
            batch_size = get("model.transformer.batch_size", 256)
            num_workers = 0
            pin_memory = False
            if torch.cuda.is_available():
                num_workers = min(4, (os.cpu_count() or 1) - 1)
                pin_memory = True

            tr_loader = build_sequence_dataloader(
                sequences,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            all_probs: list[np.ndarray] = []
            transformer.eval()
            t0 = time.time()
            with torch.no_grad():
                for tokens, _ in tr_loader:
                    tokens = tokens.to(tm_device)
                    logits = transformer(tokens)
                    probs = torch.softmax(logits[:, -1, 1:], dim=-1).cpu().numpy()
                    all_probs.append(probs)
            logger.info("Transformer inference time: %.2fs", time.time() - t0)
            scores = np.concatenate(all_probs, axis=0)
            _evaluate_model("Transformer", scores)
        else:
            logger.warning("Could not build sequences for Transformer evaluation; skipping.")

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
