"""
agents.py — the LangGraph agents.

FOUR AGENTS, deliberately split by COST:

  classifier   cheap model · runs on EVERY job      (high volume, clear rules)
  fit scorer   cheap model · runs on every job      (high volume)
  salary       strong model + web search · GATED    (expensive, low volume)
  resume       strong model · only on APPROVAL      (expensive, ~5/week)

That ordering is the whole cost strategy: cheap filters gate expensive
work. Running salary research on a job scoring 20/100 is pure waste.
"""

import json
import os
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

FAST = os.getenv("FAST_MODEL", "gpt-4o-mini")
STRONG = os.getenv("STRONG_MODEL", "gpt-4o")
FIT_GATE = 60          # below this, skip salary research entirely


def fast(temp: float = 0):
    return ChatOpenAI(model=FAST, temperature=temp)


def strong(temp: float = 0):
    return ChatOpenAI(model=STRONG, temperature=temp)


# ============================================================
# AGENT 1 — CLASSIFIER
# ============================================================

class Classification(TypedDict):
    employment_type: Literal["full_time", "contract", "freelance", "internship", "unknown"]
    relocation_tier: Literal["VISA_SPONSORSHIP", "GLOBAL_REMOTE", "REGION_REMOTE",
                             "INDIA_REMOTE", "ONSITE_ONLY", "BLOCKED"]
    visa_evidence: str
    work_region: str
    seniority: Literal["junior", "mid", "senior", "staff", "principal", "unknown"]
    archetype: Literal["backend", "ai_agentic", "platform"]


CLASSIFY_SYS = """Classify this job for a Gurgaon-based senior engineer (6 yrs)
who works remotely and actively wants to relocate to Europe, the US or Singapore.

RELOCATION TIER — read the MEANING, these are easy to get backwards:
- VISA_SPONSORSHIP: sponsors visas, offers relocation, or "we can sponsor for
  the right candidate"
- GLOBAL_REMOTE: hires from anywhere / worldwide / no location restriction
- REGION_REMOTE: remote but region-locked ("Remote - Europe", "US timezones")
- INDIA_REMOTE: remote within India, or an Indian office
- ONSITE_ONLY: must be physically present, no sponsorship mentioned
- BLOCKED: requires existing work authorization / citizenship the candidate
  lacks ("must have US work authorization", "no sponsorship available")

⚠️ "we sponsor visas" and "must already have work authorization" contain the
SAME WORDS and mean OPPOSITE things. Read the sentence, not the keywords.

visa_evidence: quote the EXACT sentence you based relocation_tier on. Empty
string if nothing explicit was stated. This lets a human verify your reasoning.

ARCHETYPE — which resume variant fits:
- backend: distributed systems, APIs, databases, scale
- ai_agentic: LLMs, agents, RAG, ML engineering
- platform: infra, devops, developer tooling, SRE"""


async def classify(title: str, company: str, location: str, description: str) -> dict:
    """One CHEAP call per job. High volume, clear rules — no need for a big model."""
    try:
        llm = fast().with_structured_output(Classification)
        return await llm.ainvoke([HumanMessage(content=(
            f"{CLASSIFY_SYS}\n\nTitle: {title}\nCompany: {company}\n"
            f"Location: {location}\n\nDescription:\n{description[:4000]}"
        ))])
    except Exception as e:
        print(f"[classify] {company}/{title}: {e}")
        # "unknown" is honest — it surfaces as unclassified rather than
        # silently mislabeling a job as something it isn't.
        return {"employment_type": "unknown", "relocation_tier": "unknown",
                "visa_evidence": "", "work_region": "", "seniority": "unknown",
                "archetype": "backend"}


# ============================================================
# AGENT 2 — FIT SCORER
# ============================================================

class Fit(TypedDict):
    fit_score: int
    reason: str
    matched_skills: list[str]
    missing_skills: list[str]


async def score_fit(title: str, description: str, profile: dict) -> dict:
    """
    Semantic scoring, not keyword counting.

    "built production AI agents with LangGraph" should score highly
    against "experience with agentic frameworks" despite zero literal
    overlap — and a JD that repeats a buzzword ten times shouldn't score
    higher for it.

    missing_skills matters as much as matched: an 80% fit missing one
    learnable thing is often a BETTER target than a 95% fit, and it tells
    the resume agent what to emphasize.
    """
    sys = f"""Score this job against the candidate.

CANDIDATE: {profile.get('seniority', 'senior')} engineer, {profile.get('years', 6)} years.
Target roles: {profile.get('roles')}
Core skills: {profile.get('skills')}

85-100 strong match on role, seniority and most core skills
70-84  good match, minor gaps
50-69  partial — related role or notable skill gaps
<50    wrong role or wrong seniority

Judge by MEANING. Equivalent experience counts even when worded differently."""
    try:
        llm = fast().with_structured_output(Fit)
        return await llm.ainvoke([HumanMessage(content=(
            f"{sys}\n\nJob: {title}\n\n{description[:4000]}"
        ))])
    except Exception as e:
        print(f"[fit] {title}: {e}")
        return {"fit_score": 0, "reason": "scoring failed",
                "matched_skills": [], "missing_skills": []}


# ============================================================
# AGENT 3 — SALARY (the careful one)
# ============================================================

class Salary(TypedDict):
    base_lpa: int
    total_lpa: int
    base_confidence: Literal["high", "medium", "low", "unknown"]
    currency_note: str
    sources: list[str]
    note: str


SALARY_SYS = """Research compensation for this role and report BASE and TOTAL
SEPARATELY, in INR lakhs per annum (LPA).

⚠️ THIS DISTINCTION IS THE WHOLE POINT. Most public sources report TOTAL comp.
A "₹58L" figure is often ₹34L base + ₹18L ESOPs + ₹6L bonus. The candidate's
bar is on BASE, so conflating them gives a dangerously wrong answer.

base_confidence:
  high    — found base explicitly stated for this company+role
  medium  — base inferred from a reliable split for comparable roles
  low     — only total comp found; base is a guess
  unknown — no usable data

If you can only find total comp, set base_lpa to your best estimate, set
base_confidence to "low", and SAY SO in `note`. Do not present a guess as fact.

For non-INR salaries, convert to LPA at roughly USD 1 = INR 84, EUR 1 = INR 91,
SGD 1 = INR 62, GBP 1 = INR 106, and record the original in currency_note."""


async def research_salary(company: str, title: str, location: str,
                          salary_raw: str = "") -> dict:
    """
    STRONG model + web search — the expensive step, gated behind fit score.

    If the posting already states comp, short-circuit: parse it instead of
    searching. Free, and high confidence.
    """
    try:
        if salary_raw and any(ch.isdigit() for ch in salary_raw):
            llm = strong().with_structured_output(Salary)
            return await llm.ainvoke([HumanMessage(content=(
                f"{SALARY_SYS}\n\nThe posting states: '{salary_raw}'\n"
                f"Role: {title} at {company}, {location}\n"
                "Parse it. If it states base explicitly, confidence is 'high'. "
                "If it's a total/range without a base split, confidence is 'low'. "
                "sources = ['posted in JD']."
            ))])

        # Web-search grounded. NOT scraping Glassdoor/levels.fyi — the model
        # searches, reads public results, and synthesizes with a confidence
        # level. Different activity, and the only one that actually works.
        llm = strong().bind_tools([{"type": "web_search_preview"}]) \
            if os.getenv("ENABLE_WEB_SEARCH") else strong()
        structured = strong().with_structured_output(Salary)
        return await structured.ainvoke([HumanMessage(content=(
            f"{SALARY_SYS}\n\nCompany: {company}\nRole: {title}\n"
            f"Location: {location}\n\nUse what you know about levels.fyi, "
            f"Glassdoor, AmbitionBox and Blind data for this company and level. "
            f"If you don't have company-specific data, use comparable companies "
            f"of similar size and market, and lower the confidence accordingly."
        ))])
    except Exception as e:
        print(f"[salary] {company}/{title}: {e}")
        return {"base_lpa": 0, "total_lpa": 0, "base_confidence": "unknown",
                "currency_note": "", "sources": [], "note": "research failed"}


# ============================================================
# AGENT 4 — RESUME TAILOR (runs only on approval)
# ============================================================

class Tailored(TypedDict):
    tailored_resume: str
    cover_letter: str
    changes_summary: str


TAILOR_SYS = """Tailor a resume to a specific job description.

STRICT RULES — rule 1 is non-negotiable:
1. NEVER invent experience, skills, employers, dates or metrics. Every line
   must trace back to the master resume.
2. You MAY reorder bullets so the most JD-relevant experience appears first.
3. You MAY re-word real experience using the JD's vocabulary — if the JD says
   "agentic frameworks" and the candidate wrote "LangGraph", surface both.
4. You MAY de-emphasize less relevant items.
5. If the candidate genuinely LACKS a required skill, do NOT fabricate it.
   Surface the closest real adjacent experience instead.
6. Everything must be defensible in an interview.

changes_summary: a bullet list of exactly what you changed and why. This is
how a human audits your edits in ten seconds instead of re-reading the whole
document. Be specific."""


async def tailor_resume(master: str, title: str, company: str,
                        description: str, missing_skills: list[str]) -> dict:
    """
    LAYER 3 — per-JD, and ONLY on approval.

    Not per discovered job: 200 rewrites is real money for applications
    you'll never send, and 200 slightly different versions of your career
    is a consistency risk if two reach the same company.
    """
    try:
        llm = strong(0.2).with_structured_output(Tailored)
        gaps = (f"\n\nThe fit scorer flagged these as missing: {missing_skills}. "
                f"Do NOT claim them. Where there's genuinely adjacent real "
                f"experience, surface that instead.") if missing_skills else ""
        return await llm.ainvoke([HumanMessage(content=(
            f"{TAILOR_SYS}\n\nMASTER RESUME:\n{master}\n\n---\n"
            f"TARGET: {title} at {company}\n\n{description[:5000]}{gaps}"
        ))])
    except Exception as e:
        print(f"[tailor] {company}/{title}: {e}")
        return {"tailored_resume": "", "cover_letter": "",
                "changes_summary": f"tailoring failed: {str(e)[:120]}"}