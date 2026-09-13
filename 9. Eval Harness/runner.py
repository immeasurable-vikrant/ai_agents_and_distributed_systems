"""
runner.py — run the dataset through the agent and score every case.

RUN OFFLINE (no API key, no cost):
    python runner.py

Offline mode uses a deterministic FAKE agent, so you can develop and
debug your eval harness without paying for a single LLM call. That's not
a demo shortcut — it's how you should build evals in real life. You want
to know your evaluator is correct BEFORE you trust it to judge a model.
"""

import asyncio
import json
import os
import time

import evaluators as ev
from dataset import DATASET, run_agent

USE_REAL_AGENT = bool(os.getenv("OPENAI_API_KEY"))


# ============================================================
# THE FAKE AGENT — for offline development
# ============================================================

def fake_agent(question: str, version: str) -> dict:
    """
    Deterministic stand-in. v2 is DELIBERATELY made worse on one case so
    the regression detector has something real to find.
    """
    q = question.lower()

    if "hello" in q or "who are you" in q:
        if version == "v2":
            # ⚠️ THE PLANTED REGRESSION: "ALWAYS use a tool" pushed the
            # agent into calling a tool on a greeting. The ANSWER still
            # looks fine — only the TRAJECTORY eval catches this.
            return {"answer": "I am an assistant.", "tools_used": ["get_weather"]}
        return {"answer": "I am an assistant that can check weather and do math.",
                "tools_used": []}

    if "delhi" in q and ("times" in q or "12" in q):
        return {"answer": "Delhi is 34C and 12 times 12 is 144.",
                "tools_used": ["get_weather", "calculate"]}
    if "atlantis" in q:
        return {"answer": "Atlantis is 20C.", "tools_used": ["get_weather"]}
    if "doubled" in q:
        return {"answer": "That is 68.", "tools_used": ["calculate"]}
    if "weather" in q or "delhi" in q:
        return {"answer": "It is 34C in Delhi.", "tools_used": ["get_weather"]}
    if "17" in q and "23" in q:
        return {"answer": "17 times 23 is 391.", "tools_used": ["calculate"]}

    return {"answer": "I don't have that information.", "tools_used": []}


# ============================================================
# SCORING ONE CASE
# ============================================================

async def evaluate_case(case: dict, result: dict, use_judge: bool) -> dict:
    evals = [
        ev.contains_answer(result["answer"], case["expected"]),
        ev.no_refusal(result["answer"]),
        ev.length_sane(result["answer"]),
        ev.trajectory_match(result["tools_used"], case["expected_tools"]),
        ev.no_tool_loops(result["tools_used"]),
    ]
    # The LLM judge is opt-in because it's the only evaluator that costs
    # money and the only one that isn't deterministic.
    if use_judge:
        evals.append(await ev.llm_judge(
            result["answer"], case["expected"], case["question"]))

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": result["answer"],
        "tools_used": result["tools_used"],
        "expected_tools": case["expected_tools"],
        "note": case["note"],
        "evals": evals,
    }


# ============================================================
# RUNNING THE WHOLE DATASET
# ============================================================

async def run_suite(version: str = "v1", use_judge: bool = False,
                    real: bool | None = None) -> dict:
    real = USE_REAL_AGENT if real is None else real
    t0 = time.perf_counter()

    async def one(case):
        result = (await run_agent(case["question"], version) if real
                  else fake_agent(case["question"], version))
        return await evaluate_case(case, result, use_judge)

    # Cases are independent → run them concurrently.
    # A 200-case suite run serially is a suite nobody runs.
    results = await asyncio.gather(*[one(c) for c in DATASET])

    return {
        "version": version,
        "mode": "real" if real else "offline (fake agent)",
        "judge": use_judge,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": ev.aggregate(results),
        "results": results,
    }


async def run_comparison(use_judge: bool = False, real: bool | None = None) -> dict:
    """Run BOTH prompt versions and diff them per test case."""
    v1 = await run_suite("v1", use_judge, real)
    v2 = await run_suite("v2", use_judge, real)
    return {
        "v1": v1,
        "v2": v2,
        "comparison": ev.compare_runs(v1["results"], v2["results"]),
    }


# ============================================================
# CLI
# ============================================================

def _print(report: dict):
    print(f"\n{'='*66}")
    print(f"  {report['version']}  ·  {report['mode']}  ·  {report['elapsed_s']}s")
    print("=" * 66)
    for key, s in report["summary"].items():
        bar = "█" * int(s["mean"] * 20)
        print(f"  {key:<14} {s['mean']:.2f} {bar:<20} {s['passed']}/{s['total']}")

    failed = [r for r in report["results"]
              if any(e["score"] < 0.999 for e in r["evals"])]
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed:
            bad = [e for e in r["evals"] if e["score"] < 0.999]
            print(f"   ✗ {r['id']}")
            for e in bad:
                print(f"       {e['key']}: {e['comment']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        out = asyncio.run(run_comparison())
        _print(out["v1"])
        _print(out["v2"])
        c = out["comparison"]
        print(f"\n{'='*66}")
        print("  REGRESSION CHECK")
        print("=" * 66)
        print(f"  v1 mean: {c['baseline_mean']}   v2 mean: {c['candidate_mean']}"
              f"   delta: {c['overall_delta']:+}")
        if c["regressions"]:
            print(f"\n  ⚠️  {len(c['regressions'])} REGRESSION(S):")
            for r in c["regressions"]:
                print(f"     {r['id']}: {r['before']} → {r['after']} ({r['delta']:+})")
                print(f"        {r['question']}")
        for i in c["improvements"]:
            print(f"  ✓ improved: {i['id']} {i['before']} → {i['after']}")
        print(f"\n  {c['verdict']}\n")
    else:
        _print(asyncio.run(run_suite(sys.argv[1] if len(sys.argv) > 1 else "v1")))