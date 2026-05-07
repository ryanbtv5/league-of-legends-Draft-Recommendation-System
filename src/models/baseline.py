"""
src/models/baseline.py
-----------------------
Random Forest baseline for champion pick recommendation.

Framing: multi-class classification over the champion pool.
  Input:  draft-state feature vector (from DraftStateEncoder)
  Output: probability distribution over all champions

Usage:
    from src.models.baseline import RandomForestRecommender
    model = RandomForestRecommender()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)   # shape (n_samples, num_champions)
    top5  = model.top_k(X_test, k=5)     # shape (n_samples, 5)
"""

from __future__ import annotations

import pathlib
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))

_DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": get("model.baseline.n_estimators", 300),
    "max_depth": get("model.baseline.max_depth", 10),
    "random_state": get("project.random_seed", 42),
    "n_jobs": -1,
}


class RandomForestRecommender:
    """Scikit-learn Random Forest baseline.

    Args:
        num_champions: Size of the output class space.
        n_estimators:  Number of trees.
    """

    def __init__(self, num_champions: int = get("data.num_champions", 165), params: dict | None = None) -> None:
        self.num_champions = num_champions
        merged = {**_DEFAULT_RF_PARAMS}
        if params:
            merged.update(params)
        self.model = RandomForestClassifier(**merged)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRecommender":
        self.model.fit(X, y)
        logger.info("RandomForestRecommender trained on %d samples", len(X))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def top_k(self, X: np.ndarray, k: int = 5) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argsort(probs, axis=1)[:, ::-1][:, :k]

    def save(self, path: pathlib.Path | None = None) -> pathlib.Path:
        path = path or MODEL_DIR / "rf_recommender.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved RandomForestRecommender to %s", path)
        return path

    @classmethod
    def load(cls, path: pathlib.Path) -> "RandomForestRecommender":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.num_champions = len(instance.model.classes_)
        return instance
