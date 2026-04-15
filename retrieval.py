"""
retrieval.py — Retrieval strategies for job-candidate matching.

Implements three strategies with a shared interface:
  1. BM25        — lexical matching via term frequency statistics
  2. Dense       — semantic matching via sentence embeddings + cosine similarity
  3. Hybrid      — linear interpolation of BM25 and dense scores

All retrievers accept a candidate query and return a ranked list of job IDs.
Jobs are indexed at construction time; retrieval is query-time only.

Design note: We retrieve top-k jobs for each candidate (not candidates per job)
because the evaluation task is "given this candidate, which jobs are most relevant?"
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, punctuation-strip tokenizer for BM25."""
    import re
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def _job_to_text(job: dict[str, Any]) -> str:
    """Convert a job dict to a single searchable string."""
    skills_str = ", ".join(job.get("skills", []))
    return f"{job['title']} {skills_str} {job.get('description', '')}"


def _candidate_to_text(candidate: dict[str, Any]) -> str:
    """Convert a candidate dict to a single query string."""
    skills_str = ", ".join(candidate.get("skills", []))
    return f"{skills_str} {candidate.get('text', '')}"


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Lexical retriever using BM25Okapi term-frequency scoring.

    Indexes each job's full text (title + skills + description).
    Query is the candidate's skill list + experience summary.
    """

    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.job_ids: list[str] = [j["id"] for j in jobs]
        corpus = [_tokenize(_job_to_text(j)) for j in jobs]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, candidate: dict[str, Any], top_k: int = 20) -> list[str]:
        """
        Return top_k job IDs ranked by BM25 score.

        Args:
            candidate: Candidate dict with 'skills' and 'text'.
            top_k:     Number of results to return.

        Returns:
            Ordered list of job IDs (most relevant first).
        """
        query_tokens = _tokenize(_candidate_to_text(candidate))
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [self.job_ids[i] for i in ranked_indices]

    def get_scores(self, candidate: dict[str, Any]) -> np.ndarray:
        """Return raw BM25 scores for all jobs (aligned with self.job_ids)."""
        query_tokens = _tokenize(_candidate_to_text(candidate))
        return self.bm25.get_scores(query_tokens)


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Semantic retriever using sentence-transformer embeddings.

    Uses all-MiniLM-L6-v2 (22M params, CPU-friendly) to encode both
    job descriptions and candidate profiles into a shared embedding space,
    then ranks by cosine similarity.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        print(f"[dense] Loading model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.job_ids: list[str] = [j["id"] for j in jobs]

        # Pre-encode all job texts at index time to avoid repeated computation
        job_texts = [_job_to_text(j) for j in jobs]
        self.job_embeddings: np.ndarray = self.model.encode(
            job_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        print(f"[dense] Indexed {len(jobs)} jobs "
              f"(embedding dim={self.job_embeddings.shape[1]})")

    def retrieve(self, candidate: dict[str, Any], top_k: int = 20) -> list[str]:
        """
        Return top_k job IDs ranked by cosine similarity.

        Args:
            candidate: Candidate dict with 'skills' and 'text'.
            top_k:     Number of results to return.

        Returns:
            Ordered list of job IDs (most relevant first).
        """
        scores = self.get_scores(candidate)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [self.job_ids[i] for i in ranked_indices]

    def get_scores(self, candidate: dict[str, Any]) -> np.ndarray:
        """Return cosine similarity scores for all jobs."""
        query_emb = self.model.encode(
            [_candidate_to_text(candidate)],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # cosine_similarity returns shape (1, n_jobs); flatten to 1-D
        return cosine_similarity(query_emb, self.job_embeddings)[0]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retriever combining BM25 (lexical) and dense (semantic) scores.

    Scores are min-max normalized independently before interpolation so that
    the alpha parameter has an intuitive, scale-invariant effect.

    hybrid_score = alpha * dense_norm + (1 - alpha) * bm25_norm
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.5,
    ) -> None:
        """
        Args:
            bm25:  Pre-built BM25Retriever.
            dense: Pre-built DenseRetriever.
            alpha: Weight for the dense component [0, 1]. Default 0.5.
        """
        assert bm25.job_ids == dense.job_ids, "Retrievers must index the same jobs."
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.job_ids = bm25.job_ids

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]; returns zeros if all values are equal."""
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def retrieve(self, candidate: dict[str, Any], top_k: int = 20) -> list[str]:
        """
        Return top_k job IDs ranked by hybrid score.

        Args:
            candidate: Candidate dict with 'skills' and 'text'.
            top_k:     Number of results to return.

        Returns:
            Ordered list of job IDs (most relevant first).
        """
        bm25_scores = self._minmax(self.bm25.get_scores(candidate))
        dense_scores = self._minmax(self.dense.get_scores(candidate))
        hybrid_scores = self.alpha * dense_scores + (1 - self.alpha) * bm25_scores
        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self.job_ids[i] for i in ranked_indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retrievers(
    jobs: list[dict[str, Any]],
    alpha: float = 0.5,
) -> dict[str, BM25Retriever | DenseRetriever | HybridRetriever]:
    """
    Build and return all three retrievers sharing a single dense index.

    Args:
        jobs:  List of job dicts.
        alpha: Hybrid interpolation weight for dense scores.

    Returns:
        Dict with keys 'bm25', 'dense', 'hybrid'.
    """
    bm25 = BM25Retriever(jobs)
    dense = DenseRetriever(jobs)
    hybrid = HybridRetriever(bm25, dense, alpha=alpha)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}
