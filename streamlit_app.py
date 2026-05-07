"""
streamlit_app.py
----------------
Interactive Streamlit UI for draft recommendations and win-probability signals.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.models import neural as nm
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))
DEFAULT_DATA_PATH = pathlib.Path(get("data.processed_dir", "data/processed")) / "drafts.parquet"
NUM_CHAMPIONS: int = get("data.num_champions", 165)
ROLES: list[str] = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])


@st.cache_data(show_spinner=False)
def load_champion_ids(data_path: pathlib.Path) -> list[int]:
    if data_path.exists():
        try:
            df = pd.read_parquet(data_path, columns=["champion_id"])
            champion_ids = sorted(
                {
                    int(cid)
                    for cid in df["champion_id"].dropna().unique().tolist()
                    if int(cid) >= 0
                }
            )
            if champion_ids:
                return champion_ids
        except Exception as exc:
            logger.warning("Could not read champion IDs from %s: %s", data_path, exc)
    return list(range(NUM_CHAMPIONS))


@st.cache_resource(show_spinner=False)
def load_models(champion_ids: tuple[int, ...]) -> tuple[ChampionEncoder, DraftStateEncoder, dict[str, object]]:
    champ_enc = ChampionEncoder(champion_ids)
    state_enc = DraftStateEncoder(champ_enc)
    models: dict[str, object] = {}

    xgb_path = MODEL_DIR / "xgb_recommender.pkl"
    if xgb_path.exists():
        models["XGBoost"] = bm.XGBoostRecommender.load(xgb_path)

    mlp_path = MODEL_DIR / "mlp_recommender_best.pt"
    if mlp_path.exists():
        models["MLP"] = nm.load_model(mlp_path, torch.device("cpu"))

    return champ_enc, state_enc, models


def _predict_probabilities(model_name: str, model: object, features: np.ndarray) -> np.ndarray:
    if model_name == "MLP":
        with torch.no_grad():
            logits = model.net(torch.tensor(features, dtype=torch.float32))
            return torch.softmax(logits, dim=-1).cpu().numpy()
    return model.predict_proba(features)


def _find_duplicates(groups: Iterable[Iterable[int]]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for group in groups:
        for champ in group:
            if champ in seen:
                duplicates.add(champ)
            else:
                seen.add(champ)
    return sorted(duplicates)


def _mask_scores(scores: np.ndarray, unavailable: list[int]) -> np.ndarray:
    masked = scores.copy()
    if unavailable:
        masked[unavailable] = 0.0
    return masked


def main() -> None:
    st.set_page_config(page_title="LoL Draft Assistant", layout="wide")
    st.title("LoL Draft Recommendation Assistant")
    st.caption("Select the current draft state to get top-5 picks and a win-probability signal.")

    st.sidebar.header("Draft Settings")
    data_path = pathlib.Path(
        st.sidebar.text_input("Champion source (drafts.parquet)", str(DEFAULT_DATA_PATH))
    )
    champion_ids = load_champion_ids(data_path)
    champ_enc, state_enc, models = load_models(tuple(champion_ids))

    if not models:
        st.error(f"No trained models found in {MODEL_DIR}. Train a model before running the app.")
        st.stop()

    model_order = list(models.keys())
    if "MLP" in model_order:
        model_order.insert(0, model_order.pop(model_order.index("MLP")))
    model_name = st.sidebar.selectbox("Recommendation model", model_order)
    team = st.sidebar.radio("Picking team", ["blue", "red"], horizontal=True)
    role = st.sidebar.selectbox("Role slot", ROLES, index=2 if "MID" in ROLES else 0)
    pick_order = st.sidebar.slider("Pick order (0-4)", min_value=0, max_value=4, value=0, step=1)
    top_k = st.sidebar.slider("Top-K recommendations", min_value=1, max_value=10, value=5, step=1)

    col_blue, col_red = st.columns(2)
    with col_blue:
        st.subheader("Blue team")
        blue_picks = st.multiselect(
            "Blue picks",
            options=champion_ids,
            default=[],
            max_selections=5,
        )
        blue_bans = st.multiselect(
            "Blue bans",
            options=champion_ids,
            default=[],
            max_selections=5,
        )
    with col_red:
        st.subheader("Red team")
        red_picks = st.multiselect(
            "Red picks",
            options=champion_ids,
            default=[],
            max_selections=5,
        )
        red_bans = st.multiselect(
            "Red bans",
            options=champion_ids,
            default=[],
            max_selections=5,
        )

    duplicates = _find_duplicates([blue_picks, red_picks, blue_bans, red_bans])
    if duplicates:
        st.warning(f"Duplicate champion IDs detected across picks/bans: {duplicates}")

    if st.button("Recommend"):
        row = {
            "blue_picks_so_far": blue_picks,
            "red_picks_so_far": red_picks,
            "blue_bans": blue_bans,
            "red_bans": red_bans,
            "pick_order": pick_order,
            "team": team,
        }
        features = state_enc.encode_batch([row])
        scores = _predict_probabilities(model_name, models[model_name], features)[0]

        unavailable_ids = set(blue_picks + red_picks + blue_bans + red_bans)
        unavailable_idx = [champ_enc.encode(cid) for cid in unavailable_ids]
        masked_scores = _mask_scores(scores, unavailable_idx)

        top_idx = np.argsort(masked_scores)[::-1][:top_k]
        recommendations = [
            {
                "Champion ID": champ_enc.decode(int(idx)),
                "Win Probability (signal)": float(masked_scores[idx]),
                "Rank": rank + 1,
            }
            for rank, idx in enumerate(top_idx)
        ]

        if recommendations:
            st.metric(
                "Win probability signal (top pick)",
                f"{recommendations[0]['Win Probability (signal)']:.1%}",
            )
            st.subheader("Top recommendations")
            st.dataframe(pd.DataFrame(recommendations), hide_index=True, use_container_width=True)
        else:
            st.info("No recommendations available. Check your draft inputs.")


if __name__ == "__main__":
    main()
