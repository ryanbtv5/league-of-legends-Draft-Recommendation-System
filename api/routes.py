"""
api/routes.py
--------------
FastAPI route handlers for the draft recommendation service.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    ChampionRecommendation,
    DraftState,
    HealthResponse,
    RecommendationResponse,
)
from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))
NUM_CHAMPIONS: int = get("data.num_champions", 165)


# ---------------------------------------------------------------------------
# Model registry (lazy-loaded singletons)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Holds references to loaded models and encoders."""

    def __init__(self) -> None:
        self.champ_enc: ChampionEncoder = ChampionEncoder()
        self.state_enc: DraftStateEncoder = DraftStateEncoder(self.champ_enc)
        self._rf: bm.RandomForestRecommender | None = None
        self._mlp: Any | None = None
        self._transformer: Any | None = None

    def load_all(self) -> None:
        """Attempt to load all available model checkpoints."""
        rf_path = MODEL_DIR / "rf_recommender.pkl"
        if rf_path.exists():
            self._rf = bm.RandomForestRecommender.load(rf_path)
            logger.info("Loaded Random Forest model")

        mlp_path = MODEL_DIR / "mlp_recommender_best.pt"
        if mlp_path.exists():
            try:
                import torch
                from src.models.neural import load_model as load_mlp
                self._mlp = load_mlp(mlp_path, torch.device("cpu"))
                logger.info("Loaded MLP model")
            except Exception as exc:
                logger.warning("Could not load MLP: %s", exc)

        tx_path = MODEL_DIR / "transformer_recommender_best.pt"
        if tx_path.exists():
            try:
                import torch
                from src.models.transformer import load_model as load_tx
                self._transformer = load_tx(tx_path, torch.device("cpu"))
                logger.info("Loaded Transformer model")
            except Exception as exc:
                logger.warning("Could not load Transformer: %s", exc)

    @property
    def loaded_models(self) -> list[str]:
        names: list[str] = []
        if self._rf is not None:
            names.append("random_forest")
        if self._mlp is not None:
            names.append("mlp")
        if self._transformer is not None:
            names.append("transformer")
        return names

    def recommend(self, state: DraftState, model_name: str = "auto") -> tuple[np.ndarray, str]:
        """Run inference and return (scores_array, model_name_used)."""
        # Build flat feature vector
        row = {
            "blue_picks_so_far": state.blue_picks,
            "red_picks_so_far": state.red_picks,
            "blue_bans": state.blue_bans,
            "red_bans": state.red_bans,
            "pick_order": state.pick_order,
            "team": state.team,
        }
        X = self.state_enc.encode_batch([row])

        if model_name == "auto":
            model_name = self.loaded_models[-1] if self.loaded_models else "none"

        if model_name == "random_forest" and self._rf is not None:
            scores = self._rf.predict_proba(X)[0]
            return scores, "random_forest"
        if model_name == "mlp" and self._mlp is not None:
            import torch
            with torch.no_grad():
                logits = self._mlp.net(torch.tensor(X, dtype=torch.float32))
                scores = torch.softmax(logits, dim=-1).numpy()[0]
            return scores, "mlp"

        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not available. Train a model first.")


@lru_cache(maxsize=1)
def _registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.load_all()
    return reg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Check service health and which models are loaded."""
    reg = _registry()
    return HealthResponse(
        status="ok",
        models_loaded=reg.loaded_models,
        version=get("project.version", "0.1.0"),
    )


@router.post("/recommend", response_model=RecommendationResponse, tags=["draft"])
def recommend(
    state: DraftState,
    model: str = Query(default="auto", description="Model to use: random_forest | mlp | transformer | auto"),
) -> RecommendationResponse:
    """Return top-k champion recommendations for the current draft state.

    Pass the current picks/bans for both teams, which pick order slot is being
    filled, and which team is picking.  The API returns an ordered list of
    champion recommendations with estimated win probabilities.
    """
    reg = _registry()
    scores, model_used = reg.recommend(state, model_name=model)

    # Mask already picked/banned champions
    unavailable = set(
        reg.champ_enc.encode_many(
            state.blue_picks + state.red_picks + state.blue_bans + state.red_bans
        )
    )
    masked_scores = scores.copy()
    for idx in unavailable:
        if 0 <= idx < len(masked_scores):
            masked_scores[idx] = 0.0

    top_indices = np.argsort(masked_scores)[::-1][: state.top_k]

    recommendations = [
        ChampionRecommendation(
            champion_id=reg.champ_enc.decode(int(idx)),
            champion_idx=int(idx),
            win_probability=float(masked_scores[idx]),
            rank=rank + 1,
        )
        for rank, idx in enumerate(top_indices)
    ]

    return RecommendationResponse(
        recommendations=recommendations,
        model_used=model_used,
        draft_state_summary={
            "blue_picks": state.blue_picks,
            "red_picks": state.red_picks,
            "blue_bans": state.blue_bans,
            "red_bans": state.red_bans,
            "team": state.team,
            "role": state.role,
            "pick_order": state.pick_order,
        },
    )
