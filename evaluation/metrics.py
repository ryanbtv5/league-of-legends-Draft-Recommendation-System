"""
evaluation/metrics.py
----------------------
Ranking-aware evaluation metrics for draft recommendation.

Implements:
  - Top-k accuracy (hit rate)
  - MRR  (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)
"""

from __future__ import annotations

import numpy as np


def top_k_accuracy(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
    """Fraction of samples where the correct champion appears in the top-*k* predictions.

    Args:
        y_true:   Integer array of shape ``(n_samples,)`` — ground-truth champion indices.
        y_scores: Float array of shape ``(n_samples, num_champions)`` — predicted scores.
        k:        Cutoff.

    Returns:
        Hit rate in ``[0, 1]``.
    """
    top_k_idx = np.argsort(y_scores, axis=1)[:, ::-1][:, :k]
    hits = np.any(top_k_idx == y_true[:, np.newaxis], axis=1)
    return float(hits.mean())


def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Mean Reciprocal Rank (MRR) over all samples.

    MRR = mean(1 / rank_of_correct_champion).

    Args:
        y_true:   Integer array of shape ``(n_samples,)``.
        y_scores: Float array of shape ``(n_samples, num_champions)``.

    Returns:
        MRR in ``[0, 1]``.
    """
    sorted_idx = np.argsort(y_scores, axis=1)[:, ::-1]
    rr_list: list[float] = []
    for i, true_idx in enumerate(y_true):
        ranks = np.where(sorted_idx[i] == true_idx)[0]
        if len(ranks) > 0:
            rr_list.append(1.0 / (ranks[0] + 1))
        else:
            rr_list.append(0.0)
    return float(np.mean(rr_list))


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
    """NDCG@k treating the correct champion as a single relevant item (relevance = 1).

    Args:
        y_true:   Integer array of shape ``(n_samples,)``.
        y_scores: Float array of shape ``(n_samples, num_champions)``.
        k:        Cutoff.

    Returns:
        Mean NDCG@k in ``[0, 1]``.
    """
    sorted_idx = np.argsort(y_scores, axis=1)[:, ::-1][:, :k]
    ndcg_list: list[float] = []
    ideal_dcg = 1.0  # Only one relevant item → ideal rank = 1

    for i, true_idx in enumerate(y_true):
        ranks = np.where(sorted_idx[i] == true_idx)[0]
        if len(ranks) > 0:
            rank = ranks[0] + 1  # 1-based
            dcg = 1.0 / np.log2(rank + 1)
            ndcg_list.append(dcg / ideal_dcg)
        else:
            ndcg_list.append(0.0)

    return float(np.mean(ndcg_list))


def compute_all(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all ranking metrics in one call.

    Args:
        y_true:   Ground-truth champion indices, shape ``(n_samples,)``.
        y_scores: Predicted scores, shape ``(n_samples, num_champions)``.
        k_values: List of cutoffs for top-k accuracy and NDCG (default ``[1, 3, 5]``).

    Returns:
        Dict mapping metric names to float values.
    """
    k_values = k_values or [1, 3, 5]
    results: dict[str, float] = {"mrr": mean_reciprocal_rank(y_true, y_scores)}
    for k in k_values:
        results[f"top{k}_acc"] = top_k_accuracy(y_true, y_scores, k=k)
        results[f"ndcg@{k}"] = ndcg_at_k(y_true, y_scores, k=k)
    return results
