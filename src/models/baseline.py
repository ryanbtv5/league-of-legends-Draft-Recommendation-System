"""
src/models/baseline.py
-----------------------
XGBoost / Random-Forest baseline for champion pick recommendation.

Framing: multi-class classification over the champion pool.
  Input:  draft-state feature vector (from DraftStateEncoder)
  Output: probability distribution over all champions

Usage:
    from src.models.baseline import XGBoostRecommender
    model = XGBoostRecommender()
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
from xgboost import XGBClassifier

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))

_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": get("model.baseline.n_estimators", 300),
    "max_depth": get("model.baseline.max_depth", 6),
    "learning_rate": get("model.baseline.learning_rate", 0.05),
    "subsample": get("model.baseline.subsample", 0.8),
    "colsample_bytree": get("model.baseline.colsample_bytree", 0.8),
    "objective": "multi:softprob",
    "tree_method": "hist",
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": get("project.random_seed", 42),
}


class XGBoostRecommender:
    """XGBoost-based champion recommendation model.

    Args:
        num_champions: Size of the output class space.
        params:        XGBoost hyperparameters (merged with defaults).
    """

    def __init__(self, num_champions: int = get("data.num_champions", 165), params: dict | None = None) -> None:
        self.num_champions = num_champions
        merged = {**_DEFAULT_XGB_PARAMS, "num_class": num_champions}
        if params:
            merged.update(params)
        self.model = XGBClassifier(**merged)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list | None = None) -> "XGBoostRecommender":
        """Train the model.

        Args:
            X:        Feature matrix, shape ``(n_samples, feature_dim)``.
            y:        Champion index labels, shape ``(n_samples,)``.
            eval_set: Optional list of ``(X_val, y_val)`` tuples for early stopping.
        """
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = 50
        self.model.fit(X, y, **fit_kwargs)
        logger.info("XGBoostRecommender trained on %d samples", len(X))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution over champions.

        Returns:
            Array of shape ``(n_samples, num_champions)``.
        """
        return self.model.predict_proba(X)

    def top_k(self, X: np.ndarray, k: int = 5, mask: np.ndarray | None = None) -> np.ndarray:
        """Return the top-*k* champion indices for each sample.

        Args:
            X:    Feature matrix.
            k:    Number of recommendations.
            mask: Boolean array ``(n_samples, num_champions)`` where ``True``
                  means the champion is *unavailable* (already picked/banned).
                  Unavailable champions receive probability 0 before ranking.

        Returns:
            Integer array of shape ``(n_samples, k)``.
        """
        probs = self.predict_proba(X).copy()
        if mask is not None:
            probs[mask] = 0.0
        return np.argsort(probs, axis=1)[:, ::-1][:, :k]

    def save(self, path: pathlib.Path | None = None) -> pathlib.Path:
        """Persist model weights with joblib."""
        path = path or MODEL_DIR / "xgb_recommender.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved XGBoostRecommender to %s", path)
        return path

    @classmethod
    def load(cls, path: pathlib.Path) -> "XGBoostRecommender":
        """Load a previously saved model."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.num_champions = instance.model.n_classes_
        logger.info("Loaded XGBoostRecommender from %s", path)
        return instance


class RandomForestRecommender:
    """Scikit-learn Random Forest baseline (lighter alternative to XGBoost).

    Args:
        num_champions: Size of the output class space.
        n_estimators:  Number of trees.
    """

    def __init__(self, num_champions: int = get("data.num_champions", 165), n_estimators: int = 200) -> None:
        self.num_champions = num_champions
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            n_jobs=-1,
            random_state=get("project.random_seed", 42),
        )

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
