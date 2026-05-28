# Job-Candidate Retrieval & Ranking Evaluation

A retrieval and ranking evaluation system for job-candidate matching, benchmarking BM25, dense retrieval, hybrid retrieval, and index-side expansion under controlled vocabulary mismatch and graded multi-relevance evaluation.

---

## Background

This project is part of a three-part search ranking portfolio:

| Project | Focus |
|---------|-------|
| [1 — ESCI Search Ranking](https://github.com/iamjaygao/esci-search-ranking-system) | Multi-stage ranking (BM25 + neural reranking on Amazon ESCI) |
| [2 — A/B Test Simulation](https://github.com/iamjaygao/search-ranking-ab-testing )| A/B testing simulation for ranking validation |
| **3 — This project** | **Retrieval evaluation and representation learning** |

The goal is to isolate how **retrieval strategy and input representation** affect ranking quality, particularly under **controlled vocabulary mismatch**.

---

## Key Results

- Built a controlled retrieval evaluation framework with graded multi-relevance labels across 3,411 job-candidate relevance pairs.
- Benchmarked BM25, dense retrieval, hybrid retrieval, and index-side expansion under vocabulary mismatch conditions.
- Achieved best performance with Hybrid Retrieval (α=0.3), reaching NDCG@10 = 0.8637 (+2.0% over BM25 baseline).
- Demonstrated that query expansion can reduce ranking discrimination in precision-sensitive multi-relevance settings.

---

## Dataset Design

### Controlled Multi-Relevance Evaluation Dataset

| Property | Value |
|----------|-------|
| Job descriptions | 100 |
| Candidate profiles | 300 |
| Total relevance pairs | 3,411 |
| Strong matches (rel=2) | 216 |
| Borderline matches (rel=1) | 3,195 |
| Candidates with ≥2 relevant jobs | 279 / 300 |
| Random seed | 42 |

Unlike typical setups where each query has a single correct answer, this dataset uses **multi-relevance labeling**: a candidate can be relevant to multiple jobs with different strengths.

### Relevance Definition

Relevance is derived from **normalized skill overlap** (not heuristic annotation):

```
overlap_ratio = |candidate_skills ∩ job_skills| / |job_skills|
```

| Label | Threshold | Meaning |
|-------|-----------|---------|
| Strong (2) | ≥ 0.60 | High skill alignment |
| Borderline (1) | 0.25 – 0.59 | Partial alignment |
| Weak (0) | < 0.25 | Minimal alignment |

### Vocabulary Mismatch (Key Design)

To simulate real-world retrieval conditions, candidate profiles use **surface-level skill variants** while job descriptions use **canonical terms**:

| Job description | Candidate profile |
|-----------------|-------------------|
| "deep learning" | "DL", "neural net training" |
| "BERT" | "language models", "LLM fine-tuning" |
| "transformers" | "HuggingFace", "attention models" |

This controlled mismatch:
- Reduces BM25 effectiveness (fewer exact matches)
- Creates opportunity for semantic retrieval to show value
- Simulates real-world vocabulary gap between job postings and resumes

Critically, **relevance labels are computed on canonical skills** — vocabulary substitution only affects the retrieval text, not the ground truth.

---

## Methodology

### Pipeline (7 Steps)

1. Generate synthetic dataset (jobs + candidates + relevance labels)
2. Build baseline retrievers (original job index)
3. Tune hybrid alpha via grid search over {0.3, 0.5, 0.7, 0.9}
4. Evaluate BM25, Dense, Hybrid baselines (best alpha)
5. Expand job descriptions (rule-based or GPT-4o-mini)
6. Build enhanced retrievers (expanded job index, same best alpha)
7. Evaluate BM25-Enhanced, Dense-Enhanced, Hybrid-Enhanced

Candidates are used as queries throughout. **Only the job index is modified** in the enhanced setting — candidates are never touched.

### Retrieval Strategies

#### Baselines

**BM25** — Term frequency scoring via `rank_bm25`. Strong precision when vocabulary aligns; brittle under synonym mismatch.

**Dense** — `all-MiniLM-L6-v2` sentence-transformer embeddings (22M params, CPU-friendly). Captures semantic similarity across vocabulary variants.

**Hybrid** — Linear interpolation of min-max normalized scores:

```
hybrid_score = α × dense_norm + (1 − α) × bm25_norm
```

Alpha tuned via grid search; best value selected by NDCG@10 on full candidate set.

#### Enhanced (Index-Side Expansion)

Job descriptions are expanded **before indexing** — candidates are never modified. This isolates the effect of index-side representation on retrieval metrics.

Expansion adds at most **1 high-confidence related skill** per job:

```
"data pipelines"     → + "Airflow"
"deep learning"      → + "PyTorch"
"distributed systems"→ + "Kafka"
```

Two modes:
- **Rule-based** (default): deterministic, no API required
- **LLM-based**: GPT-4o-mini adds 1 specific implied skill when `OPENAI_API_KEY` is set

### What This Is (and Isn't)

| This IS | This is NOT |
|---------|-------------|
| Index-side representation improvement | Reranking |
| Additive skill augmentation | Query rewriting |
| Interpretable and controlled | Black-box rescoring |
| Applied once before indexing | Applied at scoring time |

---

## Evaluation Metrics

All metrics computed per candidate and macro-averaged. Graded relevance (0/1/2) enables true ranking evaluation across multiple relevant items — not binary hit/miss.

| Metric | Definition |
|--------|-----------|
| Precision@K | Fraction of top-K with relevance ≥ 1 |
| Recall@K | Fraction of all relevant items in top-K |
| MRR | Mean Reciprocal Rank of first relevant result |
| **NDCG@K** | **Primary metric** — rewards strong matches ranked higher |

K values: 5, 10.

---

## Results

```
========================================================
  JOB-CANDIDATE RETRIEVAL — EVALUATION REPORT
========================================================
  Strategy               P@5   P@10   R@5   R@10   MRR   NDCG@10
--------------------------------------------------------
  BM25                  0.755  0.607  0.499  0.687  0.950   0.8464
  Dense                 0.587  0.476  0.352  0.517  0.848   0.6628
  Hybrid (α=0.3)        0.771  0.618  0.512  0.703  0.958   0.8637
  --------------------------------------------------------
  BM25-Enhanced         0.734  0.588  0.482  0.668  0.942   0.8239
  Dense-Enhanced        0.578  0.461  0.346  0.493  0.842   0.6436
  Hybrid-Enhanced       0.749  0.595  0.494  0.679  0.949   0.8379
--------------------------------------------------------
  Best strategy: Hybrid (α=0.3), NDCG@10 = 0.8637
  Hybrid vs BM25: +2.0% NDCG@10
========================================================
```

**Alpha grid search results:**

| Alpha | Hybrid NDCG@10 |
|-------|---------------|
| 0.3 | **0.8637** ← selected |
| 0.5 | 0.8506 |
| 0.7 | 0.8028 |
| 0.9 | 0.7185 |

---

## Key Findings

- **Hybrid retrieval consistently outperforms** both BM25 and dense alone — tuned α=0.3 confirms this dataset benefits from lexical-dominant blending
- **Dense retrieval underperforms BM25** on skill-matching tasks where canonical terms appear in both indexes — semantic retrieval adds value mainly under vocabulary mismatch
- **Query expansion did not improve over tuned hybrid** — rule-based expansion introduced vocabulary overlap across jobs, reducing ranking discrimination
- **Multi-relevance evaluation reveals differentiation invisible in binary settings** — graded NDCG separates strategies that appear equivalent under hit/miss metrics

---

## Key Insight

> Query expansion is beneficial when recall is the bottleneck, but can degrade performance when ranking precision dominates.

In multi-relevance settings with strong lexical signals, adding common skill terms increases cross-job overlap and hurts discriminative ranking. This highlights a core trade-off:

| Setting | Expansion effect |
|---------|-----------------|
| Single-label, low-recall task | Helps |
| Multi-relevance, precision-critical task | Can hurt |

**The takeaway: better evaluation design reveals failure modes that binary metrics hide.**

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (CPU only, no API key required)
python pipeline.py

# Run with LLM-based job expansion (GPT-4o-mini)
OPENAI_API_KEY=sk-... python pipeline.py
```

**Requirements**: Python 3.9+, CPU only. Runs on Google Colab free tier. Expected runtime: ~25 seconds (rule-based) or ~75 seconds (LLM mode).

---

## Project Structure

| File | Description |
|------|-------------|
| `data_loader.py` | Synthetic dataset generation + graded multi-relevance labels |
| `retrieval.py` | BM25, Dense, and Hybrid retrievers with shared interface |
| `llm_enhancer.py` | Rule-based and GPT-4o-mini job description expansion |
| `evaluation.py` | Precision@K, Recall@K, MRR, NDCG@K with graded relevance |
| `pipeline.py` | End-to-end runner + alpha tuning + formatted report |
| `requirements.txt` | CPU-only dependencies |
| `README.md` | This file |