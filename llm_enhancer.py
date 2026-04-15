"""
llm_enhancer.py — Query-side (job) expansion for improved retrieval.

Strategy: expand job descriptions before indexing so that candidate skills
align more precisely with job vocabulary. Candidate profiles are never touched.

Design principles:
    1. Preserve originals — never remove or replace existing job skills/text
    2. Controlled scope   — add at most 1–2 terms per job
    3. Discriminative only — expand specific tools (PyTorch, BERT, Kafka),
       never generic terms (Python, SQL, machine learning)
    4. Vocabulary-aligned — every added term exists in data_loader.SKILL_POOL

Two modes:
    Rule-based (default) — deterministic, no API required
    GPT-4o-mini          — additive-only LLM expansion when OPENAI_API_KEY is set
"""

from __future__ import annotations

import copy
import os
from typing import Any

# ---------------------------------------------------------------------------
# Skill expansion map (job-side only)
# ---------------------------------------------------------------------------
#
# Keys:   specific/discriminative skills that appear in job descriptions
# Values: 1–2 tightly related terms a job requiring the key also implies
# All values must exist in data_loader.SKILL_POOL

QUERY_EXPANSIONS: dict[str, list[str]] = {
    # ML / AI — framework implications
    "deep learning":          ["PyTorch"],
    "neural networks":        ["PyTorch"],
    "NLP":                    ["transformers", "BERT"],
    "transformers":           ["BERT"],
    "BERT":                   ["transformers"],
    "computer vision":        ["PyTorch"],
    "reinforcement learning": ["PyTorch"],
    "model deployment":       ["MLflow"],
    "hyperparameter tuning":  ["XGBoost"],
    # Data Platform — tool implications
    "data pipelines":         ["Airflow"],
    "ETL":                    ["Airflow"],
    "data warehousing":       ["Snowflake"],
    "distributed systems":    ["Kafka"],
    "business intelligence":  ["Looker"],
    # Engineering — tool implications
    "Kubernetes":             ["CI/CD"],
    "microservices":          ["Docker"],
}

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a technical job description enricher. "
    "Given a job's required skills, output 1–2 additional specific technical skills "
    "that this role almost certainly also requires. "
    "Rules: (1) return ONLY a comma-separated list of skill names, no explanation; "
    "(2) do NOT repeat skills already listed; "
    "(3) use specific technical terms — never generic ones like 'Python', 'SQL', "
    "'machine learning', 'software', or 'technology'; "
    "(4) only add skills with very high confidence given the listed skills."
)

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def enhance_query(job: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    """
    Return an enhanced copy of a job dict with expansion terms added to skills.

    Expansion terms are injected into the skills field so they receive full
    BM25 weight. The original job dict is never mutated.

    Args:
        job:     Job dict with 'skills', 'title', and 'description'.
        api_key: OpenAI API key. If provided, uses GPT-4o-mini; otherwise
                 falls back to the rule-based mapping.

    Returns:
        Deep copy of job with expanded skills list. Unchanged if no terms added.
    """
    original_skills: list[str] = job.get("skills", [])
    original_set: set[str] = set(original_skills)

    if api_key:
        added = _llm_expand(original_skills, original_set, api_key)
    else:
        added = _rule_expand(original_skills, original_set)

    if not added:
        return copy.deepcopy(job)

    job_copy = copy.deepcopy(job)
    job_copy["skills"] = original_skills + added  # preserve order; no dedup needed (added filtered against original_set)
    return job_copy


def _job_to_text(job: dict[str, Any]) -> str:
    skills_str = ", ".join(job.get("skills", []))
    parts = [job.get("title", ""), skills_str, job.get("description", "")]
    return " ".join(p for p in parts if p).strip()


def _rule_expand(original_skills: list[str], original_set: set[str]) -> list[str]:
    """Rule-based expansion: add at most 1 term total across all matched skills."""
    added: list[str] = []
    added_set: set[str] = set()

    for skill in original_skills:
        if len(added) >= 1:
            break
        for term in QUERY_EXPANSIONS.get(skill, []):
            if term not in original_set and term not in added_set and len(added) < 1:
                added.append(term)
                added_set.add(term)

    return added


def _llm_expand(
    original_skills: list[str],
    original_set: set[str],
    api_key: str,
) -> list[str]:
    """GPT-4o-mini expansion; falls back to rule-based on any error."""
    try:
        from openai import OpenAI  # type: ignore[import]
        client = OpenAI(api_key=api_key)
        user_msg = f"Job required skills: {', '.join(original_skills)}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=40,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        new_terms = [t.strip() for t in raw.split(",") if t.strip()]
        return [t for t in new_terms if t not in original_set][:2]
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_enhancer] API error ({exc}); falling back to rule-based.")
        return _rule_expand(original_skills, original_set)


# ---------------------------------------------------------------------------
# Batch helper: expand all jobs and return enhanced job dicts
# ---------------------------------------------------------------------------

def enhance_jobs(
    jobs: list[dict[str, Any]],
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return enhanced copies of all job dicts with expansion terms in the skills field.
    Original job dicts are not mutated.

    Args:
        jobs:    List of job dicts.
        api_key: OpenAI API key (None → rule-based).

    Returns:
        List of enhanced job dicts aligned with `jobs`.
    """
    # api_key is intentionally ignored here: force rule-based expansion so that
    # LLM-generated terms (which can be noisy and environment-dependent) never
    # contaminate the indexed job skills. LLM path is preserved for future use.
    enhanced = [enhance_query(job, api_key=None) for job in jobs]
    _log_sample(jobs, enhanced)
    return enhanced


def _log_sample(
    original: list[dict[str, Any]],
    enhanced: list[dict[str, Any]],
) -> None:
    """Log expansion stats and one before/after example."""
    n_expanded = sum(
        1 for o, e in zip(original, enhanced)
        if set(e.get("skills", [])) != set(o.get("skills", []))
    )
    print(f"[llm_enhancer] Expanded {n_expanded}/{len(original)} job descriptions")

    for orig, enh in zip(original, enhanced):
        orig_set = set(orig.get("skills", []))
        added_skills = [s for s in enh.get("skills", []) if s not in orig_set]
        if added_skills:
            print(f"[llm_enhancer] Example expansion:")
            print(f"  Job skills : {orig.get('skills', [])}")
            print(f"  Added      : {added_skills}")
            break
