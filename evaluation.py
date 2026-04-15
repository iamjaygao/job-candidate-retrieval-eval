"""
evaluation.py — Retrieval evaluation metrics for job-candidate matching.

Implements standard IR metrics with graded relevance support (scores 0, 1, 2):
  - Precision@K     — fraction of top-K results that are relevant (score > 0)
  - Recall@K        — fraction of all relevant items captured in top-K
  - MRR             — mean reciprocal rank of the first relevant result
  - NDCG@K          — normalized discounted cumulative gain (graded relevance)

Each metric is computed per candidate and then macro-averaged across the evaluation set.
Using graded relevance (rather than binary) rewards strategies that surface
the strongest matches at the top, not just any relevant match.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Per-query metric helpers
# ---------------------------------------------------------------------------

def _precision_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """
    Fraction of top-K retrieved items with relevance > 0.

    Binary relevance: an item is either relevant (score ≥ 1) or not (score = 0).
    """
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for jid in top_k if relevant.get(jid, 0) > 0)
    return hits / k


def _recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """
    Fraction of all relevant items recovered in top-K.

    Total relevant = items with any positive relevance score (≥ 1).
    Returns 0.0 when no relevant items exist to avoid division by zero.
    """
    total_relevant = sum(1 for v in relevant.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for jid in top_k if relevant.get(jid, 0) > 0)
    return hits / total_relevant


def _mrr(ranked_ids: list[str], relevant: dict[str, int]) -> float:
    """
    Reciprocal rank of the *first* relevant result (score ≥ 1).

    MRR = 1/rank_first_relevant, or 0 if no relevant item appears in the list.
    """
    for rank, jid in enumerate(ranked_ids, start=1):
        if relevant.get(jid, 0) > 0:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """
    Discounted Cumulative Gain using graded relevance.

    DCG@K = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)

    The log2 denominator discounts gains at lower ranks; the 2^rel numerator
    amplifies the difference between strong (2) and borderline (1) matches.
    """
    dcg = 0.0
    for rank, jid in enumerate(ranked_ids[:k], start=1):
        rel = relevant.get(jid, 0)
        dcg += (2 ** rel - 1) / math.log2(rank + 1)
    return dcg


def _idcg_at_k(relevant: dict[str, int], k: int) -> float:
    """
    Ideal DCG: maximum achievable DCG if results were perfectly ranked.

    Computed by sorting all known relevance scores in descending order.
    """
    sorted_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(sorted_rels, start=1):
        idcg += (2 ** rel - 1) / math.log2(rank + 1)
    return idcg


def _ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """
    Normalized DCG@K in [0, 1].

    Returns 0 if the ideal DCG is zero (i.e., all relevant scores are 0).
    """
    idcg = _idcg_at_k(relevant, k)
    if idcg == 0.0:
        return 0.0
    return _dcg_at_k(ranked_ids, relevant, k) / idcg


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    ranked_results: dict[str, list[str]],
    relevance: dict[tuple[str, str], int],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate a retrieval strategy across all candidates.

    Args:
        ranked_results: Dict mapping candidate_id → ordered list of job_ids.
        relevance:      Dict mapping (candidate_id, job_id) → relevance score.
        k_values:       List of K cutoffs to evaluate. Default [5, 10].

    Returns:
        Dict of metric names to macro-averaged scores, e.g.:
          {"P@5": 0.42, "P@10": 0.38, "R@5": 0.21, "R@10": 0.36,
           "MRR": 0.55, "NDCG@5": 0.48, "NDCG@10": 0.44}
    """
    if k_values is None:
        k_values = [5, 10]

    # Accumulate per-candidate scores
    per_candidate: dict[str, list[float]] = {
        f"P@{k}": [] for k in k_values
    }
    per_candidate.update({f"R@{k}": [] for k in k_values})
    per_candidate.update({f"NDCG@{k}": [] for k in k_values})
    per_candidate["MRR"] = []

    for cand_id, ranked_jobs in ranked_results.items():
        # Build candidate-specific relevance dict: job_id → score
        cand_rel = {
            job_id: score
            for (cid, job_id), score in relevance.items()
            if cid == cand_id
        }

        for k in k_values:
            per_candidate[f"P@{k}"].append(_precision_at_k(ranked_jobs, cand_rel, k))
            per_candidate[f"R@{k}"].append(_recall_at_k(ranked_jobs, cand_rel, k))
            per_candidate[f"NDCG@{k}"].append(_ndcg_at_k(ranked_jobs, cand_rel, k))

        per_candidate["MRR"].append(_mrr(ranked_jobs, cand_rel))

    # Macro-average across all candidates
    return {metric: float(np.mean(scores)) for metric, scores in per_candidate.items()}


def run_all_strategies(
    strategies: dict[str, Any],
    candidates: list[dict[str, Any]],
    relevance: dict[tuple[str, str], int],
    top_k: int = 20,
    k_values: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Run retrieval and evaluation for multiple strategies in one pass.

    Args:
        strategies:  Dict mapping strategy_name → retriever object (must have .retrieve()).
        candidates:  List of candidate dicts to use as queries.
        relevance:   Ground-truth relevance labels.
        top_k:       Number of results to retrieve per candidate.
        k_values:    Metric cutoffs. Default [5, 10].

    Returns:
        Dict mapping strategy_name → metric_dict.
    """
    if k_values is None:
        k_values = [5, 10]

    results: dict[str, dict[str, float]] = {}
    for name, retriever in strategies.items():
        ranked: dict[str, list[str]] = {
            c["id"]: retriever.retrieve(c, top_k=top_k) for c in candidates
        }
        results[name] = evaluate(ranked, relevance, k_values=k_values)
        print(f"[eval] {name}: NDCG@10={results[name].get('NDCG@10', 0):.4f}")
    return results
