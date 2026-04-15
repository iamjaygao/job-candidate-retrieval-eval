"""
pipeline.py — End-to-end retrieval evaluation pipeline.

Runs six retrieval strategies and prints a comparison report:

  Baselines (original job index):
    1. BM25    — lexical baseline
    2. Dense   — semantic embedding baseline
    3. Hybrid  — BM25 + dense interpolation

  Enhanced (expanded job index, candidates unchanged):
    4. BM25-Enhanced   — BM25 over expanded job descriptions
    5. Dense-Enhanced  — dense over expanded job descriptions
    6. Hybrid-Enhanced — hybrid over expanded job descriptions

Also runs a hybrid alpha grid search over [0.3, 0.5, 0.7, 0.9].

Usage:
  python pipeline.py
  OPENAI_API_KEY=sk-... python pipeline.py   # enables GPT-4o-mini enhancement
"""

import os
import time
from typing import Any

from data_loader import load_dataset
from retrieval import build_retrievers, BM25Retriever, DenseRetriever, HybridRetriever
from llm_enhancer import enhance_jobs
from evaluation import run_all_strategies, evaluate
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_STRATEGY_ORDER = [
    "BM25", "Dense", "Hybrid",
    "BM25-Enhanced", "Dense-Enhanced", "Hybrid-Enhanced",
]

_HEADER = """
========================================================
  JOB-CANDIDATE RETRIEVAL — EVALUATION REPORT
========================================================
  {:<22}  {:>5}  {:>5}  {:>5}  {:>6}  {:>5}  {:>7}
--------------------------------------------------------""".format(
    "Strategy", "P@5", "P@10", "R@5", "R@10", "MRR", "NDCG@10"
)

_SEPARATOR = "  " + "-" * 54
_ROW_FMT = "  {:<22}  {:>5.3f}  {:>5.3f}  {:>5.3f}  {:>6.3f}  {:>5.3f}  {:>7.4f}"
_FOOTER_FMT = """--------------------------------------------------------
  Best strategy: {name} (+{pct:.1f}% NDCG@10 vs BM25)
========================================================"""


def _format_report(metrics: dict[str, dict[str, float]]) -> str:
    lines = [_HEADER]
    for i, strategy in enumerate(_STRATEGY_ORDER):
        if i == 3:
            lines.append(_SEPARATOR)
        m = metrics[strategy]
        lines.append(_ROW_FMT.format(
            strategy,
            m["P@5"], m["P@10"],
            m["R@5"], m["R@10"],
            m["MRR"], m["NDCG@10"],
        ))

    best_name = max(metrics, key=lambda s: metrics[s]["NDCG@10"])
    bm25_ndcg = metrics["BM25"]["NDCG@10"]
    best_ndcg = metrics[best_name]["NDCG@10"]
    pct_gain = ((best_ndcg - bm25_ndcg) / bm25_ndcg * 100) if bm25_ndcg > 0 else 0.0

    lines.append(_FOOTER_FMT.format(name=best_name, pct=pct_gain))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alpha grid search
# ---------------------------------------------------------------------------

def _tune_alpha(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidates: list[dict[str, Any]],
    relevance: dict[tuple[str, str], int],
    alphas: list[float] | None = None,
) -> float:
    """
    Grid-search hybrid alpha on the given candidates and return the best value.

    Args:
        bm25:       Pre-built BM25Retriever (shared, not rebuilt).
        dense:      Pre-built DenseRetriever (shared, not rebuilt).
        candidates: Candidate query dicts.
        relevance:  Ground-truth relevance labels.
        alphas:     Alpha values to try. Default [0.3, 0.5, 0.7, 0.9].

    Returns:
        Alpha value with highest NDCG@10.
    """
    if alphas is None:
        alphas = [0.3, 0.5, 0.7, 0.9]

    print("\n[alpha_search] Hybrid alpha grid search:")
    best_alpha, best_ndcg = alphas[0], -1.0

    for alpha in alphas:
        hybrid = HybridRetriever(bm25, dense, alpha=alpha)
        ranked = {c["id"]: hybrid.retrieve(c, top_k=20) for c in candidates}
        m = evaluate(ranked, relevance)
        ndcg = m["NDCG@10"]
        print(f"  [alpha={alpha}] Hybrid NDCG@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg, best_alpha = ndcg, alpha

    print(f"[alpha_search] Best alpha={best_alpha} (NDCG@10={best_ndcg:.4f})")
    return best_alpha


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(n_jobs: int = 100, seed: int = 42) -> dict[str, dict[str, float]]:
    """
    Execute the full retrieval evaluation pipeline.

    Steps:
      1. Generate synthetic dataset
      2. Build baseline retrievers (original job index)
      3. Tune hybrid alpha via grid search
      4. Evaluate BM25, Dense, Hybrid baselines (best alpha)
      5. Expand job descriptions via LLM/rule-based enhancement
      6. Build enhanced retrievers (expanded job index, best alpha)
      7. Evaluate BM25-Enhanced, Dense-Enhanced, Hybrid-Enhanced
      8. Print comparative report

    Candidates are used as queries throughout and are never modified.

    Args:
        n_jobs: Number of job descriptions (default 100).
        seed:   Random seed for data generation (default 42).

    Returns:
        Dict mapping strategy name → evaluation metrics.
    """
    load_dotenv()
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 1/7: Loading dataset...")
    jobs, candidates, relevance = load_dataset(n_jobs=n_jobs, seed=seed)

    # ------------------------------------------------------------------
    # 2. Build baseline retrievers (original job index)
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 2/7: Building baseline retrievers...")
    retrievers = build_retrievers(jobs, alpha=0.5)  # alpha overridden after search

    # ------------------------------------------------------------------
    # 3. Alpha grid search (reuses shared BM25 + Dense, no reloading)
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 3/7: Tuning hybrid alpha...")
    best_alpha = _tune_alpha(
        retrievers["bm25"], retrievers["dense"], candidates, relevance
    )

    # Rebuild hybrid with the best alpha (BM25 + Dense are reused)
    from retrieval import HybridRetriever
    best_hybrid = HybridRetriever(retrievers["bm25"], retrievers["dense"], alpha=best_alpha)

    # ------------------------------------------------------------------
    # 4. Baseline evaluation
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 4/7: Running baseline strategies...")
    baseline_strategies: dict[str, Any] = {
        "BM25":   retrievers["bm25"],
        "Dense":  retrievers["dense"],
        "Hybrid": best_hybrid,
    }
    metrics = run_all_strategies(
        baseline_strategies, candidates, relevance, top_k=20
    )

    # ------------------------------------------------------------------
    # 5. Expand job descriptions (candidates not touched)
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 5/7: Expanding job descriptions...")
    # enhance_jobs forces rule-based expansion internally (api_key=None);
    # api_key is read here only to preserve the LLM path for future use.
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
    print(f"[llm_enhancer] Mode: rule-based | {len(jobs)} jobs")

    enhanced_jobs = enhance_jobs(jobs, api_key=api_key)

    # ------------------------------------------------------------------
    # 6. Build enhanced retrievers (expanded job index, best alpha)
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 6/7: Building enhanced retrievers...")
    enhanced_retrievers = build_retrievers(enhanced_jobs, alpha=best_alpha)

    # ------------------------------------------------------------------
    # 7. Evaluate all enhanced strategies
    # ------------------------------------------------------------------
    print("\n[pipeline] Step 7/7: Evaluating enhanced strategies...")
    enhanced_strategies: dict[str, Any] = {
        "BM25-Enhanced":   enhanced_retrievers["bm25"],
        "Dense-Enhanced":  enhanced_retrievers["dense"],
        "Hybrid-Enhanced": enhanced_retrievers["hybrid"],
    }
    enhanced_metrics = run_all_strategies(
        enhanced_strategies, candidates, relevance, top_k=20
    )
    metrics.update(enhanced_metrics)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(_format_report(metrics))
    print(f"\n[pipeline] Total runtime: {time.time() - t_start:.1f}s")

    return metrics


if __name__ == "__main__":
    run_pipeline()
