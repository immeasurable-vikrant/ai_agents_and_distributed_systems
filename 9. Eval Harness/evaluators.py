"""
evaluators.py — the four kinds of evaluator, from cheapest to most useful.

THE QUESTION THIS PROJECT ANSWERS: "how do you know your agent works?"

Not "does it run" — you have that. Does it give GOOD answers? Did your
prompt change make things better or quietly worse? Without evals you're
shipping on vibes, and vibes don't catch a regression that only affects
12% of inputs.

FOUR EVALUATOR TYPES:

  1. EXACT / HEURISTIC   free, instant, brittle
     String match after normalization. Only works when there IS one
     right answer.

  2. LLM-AS-JUDGE        costs a call, handles nuance
     A model scores the output against a rubric. Handles paraphrase,
     tone, completeness — the things string matching can't.

  3. TRAJECTORY          free, checks the PROCESS not the answer
     Did the agent call the right tools, in a sane order? An agent can
     get the right answer through an absurd path — that's a latent bug.

  4. REGRESSION          the one that actually saves you
     Compare two runs. Did this prompt change break anything that used
     to work?

Types 1 and 3 are FREE and DETERMINISTIC. Reach for them first. LLM
judges are powerful and also non-deterministic and billed per call —
which is why type 4 exists, since a judge's own noise can look like a
regression.
"""

import os
import re
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")


# ============================================================
# 1. HEURISTIC EVALUATORS — free, deterministic
# ============================================================

def normalize(text: str) -> str:
    """
    Normalize before comparing. Without this, "Paris." != "paris" and
    your eval fails on a correct answer — which is worse than useless,
    because you'll start distrusting the eval instead of the agent.
    """
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)       # punctuation → space
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def exact_match(output: str, expected: str) -> dict:
    ok = normalize(output) == normalize(expected)
    return {"key": "exact_match", "score": 1.0 if ok else 0.0,
            "comment": "identical after normalization" if ok else "differs"}


def contains_answer(output: str, expected: str) -> dict:
    """
    Substring check. The right tool when the answer is a FACT embedded in
    prose: "The capital is Paris, a city of 2 million" contains "Paris".

    Its weakness is real: "The capital is NOT Paris" also passes. Use it
    for facts, never for judgments.
    """
    ok = normalize(expected) in normalize(output)
    return {"key": "contains", "score": 1.0 if ok else 0.0,
            "comment": f"'{expected}' {'found' if ok else 'missing'}"}


def no_refusal(output: str, expected: str = "") -> dict:
    """
    Catches the agent bailing out. A refusal is often technically safe
    and completely unhelpful — and it won't show up in an accuracy metric
    if you only measure the cases it DID answer.
    """
    markers = ["i don't have", "i cannot", "i can't help",
               "not able to", "no information", "unable to"]
    n = normalize(output)
    refused = any(m in n for m in markers)
    return {"key": "no_refusal", "score": 0.0 if refused else 1.0,
            "comment": "refused" if refused else "answered"}


def length_sane(output: str, expected: str = "", max_words: int = 200) -> dict:
    """Rambling is a real failure mode, and it's free to detect."""
    n = len(output.split())
    ok = 1 <= n <= max_words
    return {"key": "length_sane", "score": 1.0 if ok else 0.0,
            "comment": f"{n} words"}


# ============================================================
# 2. LLM-AS-JUDGE — costs a call, handles nuance
# ============================================================

class Judgement(TypedDict):
    score: Literal[1, 2, 3, 4, 5]
    reasoning: str


RUBRIC = """Score the ANSWER against the REFERENCE on a 1-5 scale:

5 — fully correct and complete; nothing important missing
4 — correct, but missing a minor detail
3 — partially correct, or correct with a notable omission
2 — mostly wrong, though it touches the topic
1 — wrong, irrelevant, or a refusal when an answer was available

Judge FACTUAL AGREEMENT with the reference, not writing style. Different
wording that means the same thing scores 5."""


async def llm_judge(output: str, expected: str, question: str = "") -> dict:
    """
    ⚠️ THE JUDGE IS ITSELF A MODEL, with all that implies:
      • NON-DETERMINISTIC — the same input can score 4 then 5. temperature=0
        reduces this but does not eliminate it.
      • BIASED — judges tend to reward longer, more confident answers.
      • COSTS MONEY — one call per test case, every run.

    So: use heuristics wherever they suffice, and treat a 1-point judge
    delta as noise rather than a regression. `aggregate()` below reports
    the spread for exactly this reason.
    """
    judge = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Judgement)
    r = await judge.ainvoke([HumanMessage(content=(
        f"{RUBRIC}\n\nQUESTION: {question}\n\nREFERENCE: {expected}\n\n"
        f"ANSWER: {output}"
    ))])
    return {"key": "llm_judge", "score": r["score"] / 5.0,
            "comment": f"{r['score']}/5 — {r['reasoning'][:90]}"}


# ============================================================
# 3. TRAJECTORY EVALUATOR — checks the PROCESS
# ============================================================

def trajectory_match(actual_tools: list[str], expected_tools: list[str],
                     strict_order: bool = False) -> dict:
    """
    Did the agent take a sensible PATH, not just land on a good answer?

    WHY THIS MATTERS: an agent can produce the right answer through a
    terrible route — calling a tool 5 times, or calling an expensive one
    it didn't need. The output eval says "pass". Your bill says otherwise,
    and the day the data shifts, the bad path starts giving bad answers.

    strict_order=False (default) compares tool SETS, because many valid
    plans differ only in ordering. Turn it on when sequence genuinely
    matters (authorize-then-capture, not capture-then-authorize).
    """
    if strict_order:
        ok = actual_tools == expected_tools
        detail = f"{actual_tools} vs {expected_tools}"
    else:
        a, e = set(actual_tools), set(expected_tools)
        ok = a == e
        missing, extra = e - a, a - e
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)}")
        if extra:
            detail.append(f"unexpected {sorted(extra)}")
        detail = "; ".join(detail) or "exact match"

    return {"key": "trajectory", "score": 1.0 if ok else 0.0, "comment": detail}


def no_tool_loops(actual_tools: list[str], max_repeats: int = 2) -> dict:
    """
    Catches the agent calling the same tool over and over — usually a
    sign it isn't understanding the result it's getting back.
    """
    counts = {t: actual_tools.count(t) for t in set(actual_tools)}
    worst = max(counts.values()) if counts else 0
    ok = worst <= max_repeats
    return {"key": "no_loops", "score": 1.0 if ok else 0.0,
            "comment": f"max repeat = {worst}"}


# ============================================================
# 4. AGGREGATION + REGRESSION
# ============================================================

def aggregate(results: list[dict]) -> dict:
    """
    Per-evaluator averages across the whole dataset.

    WHY report the spread, not just the mean: a mean of 0.8 could be
    "everything scored 0.8" or "80% perfect, 20% catastrophic". Those
    demand completely different responses, and the mean alone hides which
    one you have.
    """
    by_key: dict[str, list[float]] = {}
    for r in results:
        for e in r["evals"]:
            by_key.setdefault(e["key"], []).append(e["score"])

    summary = {}
    for key, scores in by_key.items():
        summary[key] = {
            "mean": round(sum(scores) / len(scores), 3),
            "passed": sum(1 for s in scores if s >= 0.999),
            "total": len(scores),
            "min": round(min(scores), 3),
        }
    return summary


def compare_runs(baseline: list[dict], candidate: list[dict],
                 threshold: float = 0.15) -> dict:
    """
    ⭐ THE EVALUATOR THAT ACTUALLY SAVES YOU.

    You changed a prompt. Overall score went 0.82 → 0.84. Ship it?

    Maybe not. That average can hide 3 cases that went from perfect to
    broken, offset by 5 that improved slightly. Aggregate metrics are
    exactly where regressions go to hide.

    So this compares PER TEST CASE and names the ones that got worse.
    `threshold` exists because LLM judges are noisy — a 0.1 wobble is
    noise, a 0.4 drop is a regression.
    """
    base = {r["id"]: r for r in baseline}
    regressions, improvements = [], []

    for cand in candidate:
        b = base.get(cand["id"])
        if not b:
            continue
        bs = sum(e["score"] for e in b["evals"]) / max(len(b["evals"]), 1)
        cs = sum(e["score"] for e in cand["evals"]) / max(len(cand["evals"]), 1)
        delta = round(cs - bs, 3)

        if delta <= -threshold:
            regressions.append({"id": cand["id"], "question": cand["question"],
                                "before": round(bs, 3), "after": round(cs, 3),
                                "delta": delta})
        elif delta >= threshold:
            improvements.append({"id": cand["id"], "before": round(bs, 3),
                                 "after": round(cs, 3), "delta": delta})

    b_mean = sum(sum(e["score"] for e in r["evals"]) / max(len(r["evals"]), 1)
                 for r in baseline) / max(len(baseline), 1)
    c_mean = sum(sum(e["score"] for e in r["evals"]) / max(len(r["evals"]), 1)
                 for r in candidate) / max(len(candidate), 1)

    return {
        "baseline_mean": round(b_mean, 3),
        "candidate_mean": round(c_mean, 3),
        "overall_delta": round(c_mean - b_mean, 3),
        "regressions": regressions,
        "improvements": improvements,
        # The headline: a rising average does NOT mean nothing broke.
        "verdict": ("❌ BLOCKED — regressions found" if regressions
                    else "✅ safe to ship"),
    }