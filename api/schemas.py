"""
api/schemas.py
--------------
Pydantic request / response models for the draft recommendation API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DraftState(BaseModel):
    """Current state of the draft passed to the recommendation endpoint.

    All champion values are **Riot API champion IDs** (integers).

    Example::

        {
            "blue_picks": [157, 64],
            "red_picks": [238],
            "blue_bans": [11, 99],
            "red_bans": [517, 235],
            "pick_order": 2,
            "team": "blue",
            "role": "MID",
            "top_k": 5
        }
    """

    blue_picks: list[int] = Field(default_factory=list, description="Champion IDs picked by blue team so far")
    red_picks: list[int] = Field(default_factory=list, description="Champion IDs picked by red team so far")
    blue_bans: list[int] = Field(default_factory=list, description="Champion IDs banned by blue team")
    red_bans: list[int] = Field(default_factory=list, description="Champion IDs banned by red team")
    pick_order: int = Field(default=0, ge=0, le=4, description="Zero-based index of the current pick (0–4)")
    team: str = Field(default="blue", description="Which team is picking: 'blue' or 'red'")
    role: str = Field(default="MID", description="Role slot being filled")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")

    @field_validator("team")
    @classmethod
    def _validate_team(cls, v: str) -> str:
        if v not in ("blue", "red"):
            raise ValueError("team must be 'blue' or 'red'")
        return v


class ChampionRecommendation(BaseModel):
    """A single champion recommendation with its estimated win probability."""

    champion_id: int
    champion_idx: int
    win_probability: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class RecommendationResponse(BaseModel):
    """Response returned by the /recommend endpoint."""

    recommendations: list[ChampionRecommendation]
    model_used: str
    draft_state_summary: dict


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""

    status: str
    models_loaded: list[str]
    version: str
