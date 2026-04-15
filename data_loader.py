"""
data_loader.py — Controlled synthetic dataset for job-candidate retrieval evaluation.

Generates job descriptions and candidate profiles with graded relevance labels
derived from normalized skill overlap:

    overlap_ratio = |candidate_canonical_skills ∩ job_skills| / |job_skills|

Relevance scale (multi-job):
    2  strong     overlap_ratio >= 0.6
    1  borderline 0.25 <= overlap_ratio < 0.6
    (not stored)  overlap_ratio < 0.25

Relevance is computed for ALL (candidate, job) pairs — not just the generating
job. Candidates may therefore be relevant to multiple jobs, which makes NDCG,
Recall@K, and MRR semantically meaningful.

Candidate distribution:
    ~100 strong / ~100 borderline / ~100 weak  (1 per tier per job, by generation)

Vocabulary mismatch (for strong and borderline candidates):
    Some canonical skills are replaced in the displayed profile text and
    skills list with synonym variants from SYNONYM_MAP. Overlap/label
    computation always uses the original canonical skills, so labels are
    unaffected. This mismatch forces semantic retrieval and query expansion
    to work harder than pure lexical matching.
"""

from __future__ import annotations

import math
import random
from typing import Any

SEED = 42

# ---------------------------------------------------------------------------
# Skill pool — 63 realistic technical skills across four domains
# ---------------------------------------------------------------------------

SKILL_POOL: list[str] = [
    # ML / AI (20)
    "Python", "scikit-learn", "PyTorch", "TensorFlow", "deep learning",
    "neural networks", "machine learning", "NLP", "computer vision",
    "reinforcement learning", "transformers", "BERT", "MLflow",
    "feature engineering", "model evaluation", "hyperparameter tuning",
    "XGBoost", "LightGBM", "Keras", "model deployment",
    # Data (17)
    "SQL", "pandas", "NumPy", "data analysis", "data visualization",
    "Tableau", "Power BI", "matplotlib", "seaborn", "statistics",
    "A/B testing", "data cleaning", "Spark", "data modeling", "Jupyter",
    "R", "SAS",
    # Engineering (15)
    "Docker", "Kubernetes", "REST APIs", "microservices", "Java",
    "Go", "system design", "distributed systems", "Redis", "PostgreSQL",
    "CI/CD", "Terraform", "AWS", "GCP", "Linux",
    # Analytics / Data Platform (11)
    "dbt", "Airflow", "BigQuery", "Snowflake", "ETL",
    "data pipelines", "Kafka", "data warehousing", "Looker",
    "business intelligence", "Azure",
]

SKILL_POOL_SET: frozenset[str] = frozenset(SKILL_POOL)

# ---------------------------------------------------------------------------
# Synonym map — vocabulary mismatch for realistic retrieval challenge
# ---------------------------------------------------------------------------
#
# Keys:   canonical skills (must be in SKILL_POOL)
# Values: one or more surface variants used in candidate profiles.
#         Variants do NOT need to be in SKILL_POOL — they represent the kind
#         of vocabulary gap that semantic search and query expansion should bridge.
#
# Overlap / relevance labels are always computed on canonical skills.

SYNONYM_MAP: dict[str, list[str]] = {
    "deep learning":          ["DL", "neural net training"],
    "transformers":           ["HuggingFace", "attention models"],
    "BERT":                   ["language models", "LLM fine-tuning"],
    "data pipelines":         ["data workflows", "pipeline orchestration"],
    "data warehousing":       ["cloud data warehouse", "analytical storage"],
    "distributed systems":    ["fault-tolerant systems", "scalable infrastructure"],
    "business intelligence":  ["BI reporting", "executive dashboards"],
    "model deployment":       ["ML serving", "model productionization"],
    "feature engineering":    ["feature extraction", "data featurization"],
    "reinforcement learning": ["RL", "reward-based learning"],
    "computer vision":        ["image recognition", "visual AI"],
    "hyperparameter tuning":  ["model tuning", "AutoML"],
    "data modeling":          ["schema design", "dimensional modeling"],
    "microservices":          ["service-oriented architecture", "SOA"],
    "system design":          ["architecture design", "large-scale systems"],
}

# Probability that a given expandable skill is replaced in strong/borderline profiles
_SYNONYM_REPLACE_PROB = 0.4

# ---------------------------------------------------------------------------
# Job roles — 10 roles with preferred skill sets
# ---------------------------------------------------------------------------

JOB_ROLES: list[dict[str, Any]] = [
    {
        "title": "Data Scientist",
        "preferred": [
            "Python", "scikit-learn", "machine learning", "pandas",
            "statistics", "data visualization", "SQL", "NumPy",
            "feature engineering", "model evaluation",
        ],
    },
    {
        "title": "ML Engineer",
        "preferred": [
            "PyTorch", "TensorFlow", "deep learning", "model deployment",
            "MLflow", "feature engineering", "Docker", "Python",
            "Kubernetes", "CI/CD",
        ],
    },
    {
        "title": "Data Analyst",
        "preferred": [
            "SQL", "data analysis", "Tableau", "Power BI",
            "A/B testing", "data visualization", "statistics",
            "data cleaning", "Python", "R",
        ],
    },
    {
        "title": "NLP Engineer",
        "preferred": [
            "NLP", "transformers", "BERT", "Python", "deep learning",
            "Keras", "PyTorch", "machine learning", "TensorFlow", "model evaluation",
        ],
    },
    {
        "title": "AI Researcher",
        "preferred": [
            "deep learning", "reinforcement learning", "neural networks",
            "PyTorch", "machine learning", "Python", "statistics",
            "computer vision", "TensorFlow", "model evaluation",
        ],
    },
    {
        "title": "Backend Engineer",
        "preferred": [
            "Java", "Go", "REST APIs", "microservices", "Docker",
            "PostgreSQL", "Redis", "system design", "distributed systems",
            "Kubernetes",
        ],
    },
    {
        "title": "Data Engineer",
        "preferred": [
            "Spark", "Airflow", "ETL", "data pipelines", "SQL",
            "Kafka", "Python", "data warehousing", "BigQuery", "Snowflake",
        ],
    },
    {
        "title": "Analytics Engineer",
        "preferred": [
            "dbt", "BigQuery", "Snowflake", "SQL", "data modeling",
            "Airflow", "data pipelines", "Python", "Looker", "business intelligence",
        ],
    },
    {
        "title": "Applied Scientist",
        "preferred": [
            "machine learning", "Python", "scikit-learn", "XGBoost",
            "feature engineering", "A/B testing", "statistics",
            "model evaluation", "LightGBM", "data analysis",
        ],
    },
    {
        "title": "Research Scientist",
        "preferred": [
            "deep learning", "PyTorch", "neural networks", "computer vision",
            "NLP", "Python", "reinforcement learning", "Keras",
            "TensorFlow", "model evaluation",
        ],
    },
]

_PROFILE_TEMPLATES: list[str] = [
    "{years} years of experience as a {domain} professional. Proficient in {skills}.",
    "Experienced {domain} specialist with {years} years in the field. Skills include {skills}.",
    "{domain} practitioner with {years} years of hands-on experience. Expertise: {skills}.",
    "Seasoned {domain} professional ({years} years). Core competencies: {skills}.",
]

_DOMAIN_LABELS: list[str] = [
    "data science", "machine learning", "engineering", "analytics",
    "data engineering", "software", "AI", "backend", "data analysis",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlap_ratio(candidate_skills: list[str], job_skills: list[str]) -> float:
    """Normalized intersection: |C ∩ J| / |J|."""
    return len(set(candidate_skills) & set(job_skills)) / len(job_skills)


def _borderline_k_range(n: int) -> tuple[int, int]:
    """
    Compute the valid range of overlapping skill counts for a borderline candidate.

    We need: 0.25 <= k/n < 0.6
      min_k = ceil(0.25 * n)     — smallest k with ratio >= 0.25
      max_k = ceil(0.6 * n) - 1  — largest k with ratio strictly < 0.6
    """
    min_k = math.ceil(0.25 * n)
    max_k = math.ceil(0.6 * n) - 1
    return min_k, max_k


def _apply_synonym_mismatch(skills: list[str], rng: random.Random) -> list[str]:
    """
    Replace a subset of canonical skills with surface variants from SYNONYM_MAP.

    The returned list is used only in the candidate's displayed profile and
    skills field — never for overlap/label computation. This simulates the
    vocabulary gap between how candidates describe themselves and how jobs
    are posted.
    """
    result: list[str] = []
    for skill in skills:
        if skill in SYNONYM_MAP and rng.random() < _SYNONYM_REPLACE_PROB:
            result.append(rng.choice(SYNONYM_MAP[skill]))
        else:
            result.append(skill)
    return result


def _make_candidate(
    cid: str,
    canonical_skills: list[str],
    domain_label: str,
    years: int,
    tier: str,
    rng: random.Random,
    apply_mismatch: bool = False,
) -> dict[str, Any]:
    """
    Build a candidate dict.

    Args:
        canonical_skills: Skills used for overlap/label computation (stored as
                          'canonical_skills' in the returned dict).
        apply_mismatch:   If True, replace some skills in the displayed profile
                          with synonyms to create vocabulary mismatch.
    """
    display_skills = (
        _apply_synonym_mismatch(canonical_skills, rng) if apply_mismatch
        else list(canonical_skills)
    )
    template = rng.choice(_PROFILE_TEMPLATES)
    skill_sample = ", ".join(display_skills[:4])
    text = template.format(years=years, domain=domain_label, skills=skill_sample)
    return {
        "id": cid,
        "skills": display_skills,              # used for retrieval queries
        "canonical_skills": canonical_skills,  # used for label computation only
        "text": text,
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# Job generation
# ---------------------------------------------------------------------------

def _generate_jobs(rng: random.Random) -> list[dict[str, Any]]:
    """
    Generate 100 unique job descriptions — 10 per role.

    Skill set uniqueness is enforced: any duplicate draw is discarded and
    re-sampled. Each job takes 3–5 skills from the role's preferred list and
    fills the remainder from the broader pool, giving natural cross-domain overlap.
    """
    jobs: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()

    for role in JOB_ROLES:
        preferred = [s for s in role["preferred"] if s in SKILL_POOL_SET]
        remaining_pool = [s for s in SKILL_POOL if s not in preferred]
        count = 0
        attempts = 0

        while count < 10:
            attempts += 1
            if attempts > 1_000:
                raise RuntimeError(
                    f"Could not generate 10 unique jobs for {role['title']} — "
                    "consider expanding the skill pool."
                )

            n_skills = rng.randint(5, 8)
            n_preferred = min(rng.randint(3, 5), len(preferred), n_skills)
            core = rng.sample(preferred, k=n_preferred)
            n_extra = n_skills - len(core)
            extra = rng.sample(remaining_pool, k=min(n_extra, len(remaining_pool)))

            seen_in_job: set[str] = set()
            skills: list[str] = []
            for s in core + extra:
                if s not in seen_in_job:
                    seen_in_job.add(s)
                    skills.append(s)

            skill_set = frozenset(skills)
            if skill_set in seen:
                continue
            seen.add(skill_set)

            job_id = f"job_{len(jobs):03d}"
            description = (
                f"We are looking for a {role['title']} with experience in "
                f"{', '.join(skills[:3])}. "
                f"The role involves {', '.join(skills[3:])} to build and "
                f"maintain scalable data and ML systems."
            )
            jobs.append({
                "id": job_id,
                "title": role["title"],
                "skills": skills,
                "description": description,
            })
            count += 1

    return jobs


# ---------------------------------------------------------------------------
# Candidate generation — rejection sampling per tier
# ---------------------------------------------------------------------------

def _gen_strong(
    job: dict[str, Any], cid: str, rng: random.Random, max_tries: int = 200
) -> dict[str, Any]:
    """
    Generate a strong candidate (overlap_ratio >= 0.6) for the given job.

    Overlap is computed on canonical skills. Display skills may include synonyms.
    """
    job_skills = job["skills"]
    n = len(job_skills)
    non_job = [s for s in SKILL_POOL if s not in job_skills]
    min_k = math.ceil(0.6 * n)

    for _ in range(max_tries):
        k = rng.randint(min_k, n)
        overlap = rng.sample(job_skills, k=k)
        n_extra = rng.randint(1, min(2, len(non_job)))
        extra = rng.sample(non_job, k=n_extra)

        canonical = list(dict.fromkeys(overlap + extra))
        if _overlap_ratio(canonical, job_skills) >= 0.6:
            years = rng.randint(4, 8)
            return _make_candidate(
                cid, canonical, job["title"], years, "strong", rng,
                apply_mismatch=True,
            )

    canonical = list(job_skills) + ([rng.choice(non_job)] if non_job else [])
    return _make_candidate(
        cid, canonical, job["title"], rng.randint(4, 8), "strong", rng,
        apply_mismatch=True,
    )


def _gen_borderline(
    job: dict[str, Any], cid: str, rng: random.Random, max_tries: int = 200
) -> dict[str, Any]:
    """
    Generate a borderline candidate (0.25 <= overlap_ratio < 0.6).

    Overlap is computed on canonical skills. Display skills may include synonyms.
    """
    job_skills = job["skills"]
    n = len(job_skills)
    non_job = [s for s in SKILL_POOL if s not in job_skills]
    min_k, max_k = _borderline_k_range(n)
    min_k = max(1, min(min_k, n))
    max_k = max(min_k, min(max_k, n))

    for _ in range(max_tries):
        k = rng.randint(min_k, max_k)
        overlap = rng.sample(job_skills, k=k)

        n_total = rng.randint(max(3, len(overlap)), min(6, len(overlap) + len(non_job)))
        n_extra = n_total - len(overlap)
        extra = rng.sample(non_job, k=min(n_extra, len(non_job)))

        canonical = list(dict.fromkeys(overlap + extra))
        ratio = _overlap_ratio(canonical, job_skills)
        if 0.25 <= ratio < 0.6:
            years = rng.randint(1, 5)
            domain = rng.choice(_DOMAIN_LABELS)
            return _make_candidate(
                cid, canonical, domain, years, "borderline", rng,
                apply_mismatch=True,
            )

    overlap = rng.sample(job_skills, k=min(min_k, len(job_skills)))
    extra = rng.sample(non_job, k=min(3, len(non_job)))
    canonical = list(dict.fromkeys(overlap + extra))
    return _make_candidate(
        cid, canonical, rng.choice(_DOMAIN_LABELS), rng.randint(1, 5), "borderline", rng,
        apply_mismatch=True,
    )


def _gen_weak(
    job: dict[str, Any], cid: str, rng: random.Random, max_tries: int = 200
) -> dict[str, Any]:
    """
    Generate a weak candidate (overlap_ratio < 0.25).

    No synonym substitution — weak candidates stay lexically distinct from
    relevant jobs, so no extra mismatch is needed.
    """
    job_skills = job["skills"]
    non_job = [s for s in SKILL_POOL if s not in job_skills]

    for _ in range(max_tries):
        k_overlap = rng.randint(0, 1)
        overlap = rng.sample(job_skills, k=k_overlap) if k_overlap else []

        n_total = rng.randint(3, 5)
        n_extra = max(n_total - len(overlap), 3)
        extra = rng.sample(non_job, k=min(n_extra, len(non_job)))

        canonical = list(dict.fromkeys(overlap + extra))
        if _overlap_ratio(canonical, job_skills) < 0.25:
            years = rng.randint(1, 7)
            domain = rng.choice(_DOMAIN_LABELS)
            return _make_candidate(cid, canonical, domain, years, "weak", rng)

    canonical = rng.sample(non_job, k=min(4, len(non_job)))
    return _make_candidate(
        cid, canonical, rng.choice(_DOMAIN_LABELS), rng.randint(1, 7), "weak", rng
    )


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _generate_candidates(
    jobs: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """
    Generate exactly 3 candidates per job (1 per tier) using rejection sampling.

    Returns 300 candidate dicts. Relevance labels are computed separately
    via _compute_multi_relevance to cover all (candidate, job) pairs.

    Returns:
        candidates: 300 dicts with id, skills (display), canonical_skills, text, tier.
    """
    candidates: list[dict[str, Any]] = []

    generators = [
        ("strong",     _gen_strong),
        ("borderline", _gen_borderline),
        ("weak",       _gen_weak),
    ]

    for job in jobs:
        for _tier, gen_fn in generators:
            cid = f"cand_{len(candidates):04d}"
            candidate = gen_fn(job, cid, rng)
            candidates.append(candidate)

    return candidates


def _compute_multi_relevance(
    candidates: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[tuple[str, str], int]:
    """
    Compute relevance labels for ALL (candidate, job) pairs using canonical skills.

    Only pairs with overlap_ratio >= 0.25 are stored.
    Pairs with overlap_ratio < 0.25 are omitted (treated as irrelevant).

    Returns:
        Dict mapping (candidate_id, job_id) → score (1 or 2).
    """
    relevance: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        canonical = candidate["canonical_skills"]
        for job in jobs:
            ratio = _overlap_ratio(canonical, job["skills"])
            if ratio >= 0.6:
                relevance[(candidate["id"], job["id"])] = 2
            elif ratio >= 0.25:
                relevance[(candidate["id"], job["id"])] = 1
            # ratio < 0.25: omitted — not relevant
    return relevance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    n_jobs: int = 100,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], int]]:
    """
    Build the synthetic evaluation dataset.

    Relevance is computed for ALL (candidate, job) pairs so that each
    candidate may have multiple relevant jobs. Only pairs with
    overlap_ratio >= 0.25 appear in the relevance dict.

    Args:
        n_jobs: Number of job descriptions to generate. Must be a multiple of
                10 (one per role). Default 100.
        seed:   Random seed for full reproducibility. Default 42.

    Returns:
        jobs:       List of job dicts — id, title, skills, description.
        candidates: List of candidate dicts — id, skills (display),
                    canonical_skills, text, tier.
        relevance:  Dict mapping (candidate_id, job_id) → score (1 or 2).
                    Pairs with overlap < 0.25 are omitted.

    Raises:
        ValueError: If n_jobs is not a multiple of 10.
    """
    if n_jobs % 10 != 0:
        raise ValueError(f"n_jobs must be a multiple of 10 (one per role), got {n_jobs}.")

    rng = random.Random(seed)
    jobs = _generate_jobs(rng)
    jobs = jobs[:n_jobs]
    candidates = _generate_candidates(jobs, rng)
    relevance = _compute_multi_relevance(candidates, jobs)

    # Stats
    n_strong = sum(1 for v in relevance.values() if v == 2)
    n_borderline = sum(1 for v in relevance.values() if v == 1)
    n_gen_strong = sum(1 for c in candidates if c["tier"] == "strong")
    n_gen_borderline = sum(1 for c in candidates if c["tier"] == "borderline")
    n_gen_weak = sum(1 for c in candidates if c["tier"] == "weak")

    # Candidates with more than one relevant job (score > 0)
    from collections import Counter
    cand_rel_counts = Counter(cid for (cid, _) in relevance)
    n_multi_relevant = sum(1 for cnt in cand_rel_counts.values() if cnt > 1)

    n_mismatched = sum(
        1 for c in candidates if c["skills"] != c["canonical_skills"]
    )

    print(f"[data_loader] {len(jobs)} jobs | {len(candidates)} candidates "
          f"(generated: strong={n_gen_strong}, borderline={n_gen_borderline}, "
          f"weak={n_gen_weak})")
    print(f"[data_loader] Multi-relevance pairs: "
          f"{n_strong} strong + {n_borderline} borderline = {len(relevance)} total")
    print(f"[data_loader] Candidates with ≥2 relevant jobs: {n_multi_relevant}")
    print(f"[data_loader] Vocabulary mismatch applied to {n_mismatched} candidate profiles")

    return jobs, candidates, relevance
