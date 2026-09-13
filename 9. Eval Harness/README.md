# Project 9 — Eval Harness

> "How do you know your agent works?" — the question you can't answer by
> looking at it. This project builds the four kinds of evaluator and then
> catches a regression that the overall average completely hides.

---

## Files (4)

```
evaluators.py   the 4 evaluator types + regression comparison
dataset.py      test cases + the agent under test (two prompt versions)
runner.py       runs suites, prints results  ← works OFFLINE
main.py         FastAPI
index.html      eval dashboard
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn langgraph langchain-openai

python runner.py           # v1, offline, no API key needed
python runner.py compare   # ⭐ the regression demo

uvicorn main:app --reload  # the dashboard
```

**Offline by default.** With no `OPENAI_API_KEY`, the suite runs against
a deterministic fake agent. That's not a demo shortcut — it's how you
should build evals for real. You want to know your *evaluator* is correct
before you trust it to judge a *model*.

---

## The headline result

```
$ python runner.py compare

  v1  ·  offline
  contains       1.00 ████████████████████ 6/6
  trajectory     1.00 ████████████████████ 6/6

  v2  ·  offline
  contains       1.00 ████████████████████ 6/6
  trajectory     0.83 ████████████████     5/6
   ✗ no-tool-needed
       trajectory: unexpected ['get_weather']

  REGRESSION CHECK
  v1 mean: 1.0   v2 mean: 0.967   delta: -0.033

  ⚠️  1 REGRESSION(S):
     no-tool-needed: 1.0 → 0.8 (-0.2)
        Hello, who are you?

  ❌ BLOCKED — regressions found
```

**Read that carefully.** The overall delta is `-0.033` — the kind of
number you'd shrug off as noise and ship. But per case, one test dropped
`-0.2`.

**What broke:** v2's prompt says *"ALWAYS use a tool"*. So it calls
`get_weather` on *"Hello, who are you?"*

**What makes it sneaky:** the **answer is still correct**. `contains`,
`no_refusal`, `length_sane` all pass. Only the **trajectory** eval —
which checks the *process*, not the output — catches it.

Two lessons in one output:
1. **Aggregate metrics hide regressions.** Always compare per case.
2. **Output-only evals miss process bugs.** A right answer via a wrong
   path is a latent bug and a real bill.

---

## The four evaluator types

| Type | Cost | Deterministic | Catches |
|---|---|---|---|
| **Heuristic** (exact, contains, refusal, length) | free | ✅ | factual misses, refusals, rambling |
| **LLM-as-judge** | 1 call/case | ❌ | nuance, paraphrase, completeness |
| **Trajectory** | free | ✅ | wrong tools, wrong order, loops |
| **Regression** | free | ✅ | *"did my change break something?"* |

**Reach for the free deterministic ones first.** The LLM judge is powerful
and also non-deterministic and billed per call — which is *why* the
regression comparator has a `threshold`: a 0.1 judge wobble is noise, a
0.4 drop is real.

---

## A brittleness bug I hit and kept as a lesson

The first dataset had:

```python
{"question": "Weather in Delhi, and 12 times 12?", "expected": "34 and 144"}
```

It **failed on a correct answer.** The agent said *"Delhi is 34C and 12
times 12 is 144"* — which contains `34` and contains `144`, but not the
literal string `"34 and 144"`. `contains_answer` is a substring check.

Fixed by expecting `"144"`, with the reasoning written into the dataset:
**heuristic evaluators need expectations shaped to fit them.** For
genuinely multi-part answers, use several single-fact checks or an LLM
judge.

And note what actually guards that case properly: the **trajectory**
eval, which verifies both tools were called.

> A brittle eval that fails correct output is worse than no eval — you
> stop trusting the harness instead of the agent.

---

## Why `aggregate()` reports the spread

A mean of `0.8` could be "everything scored 0.8" or "80% perfect, 20%
catastrophic". Those demand completely different responses, and the mean
alone hides which one you have. So the summary carries `mean`, `passed`,
`total`, and `min`.

---

## Wiring this into CI

`/api/compare` is the endpoint you'd call from a pipeline:

```
fail the build when comparison.regressions is non-empty
— regardless of whether the overall mean went up
```

That one rule is the difference between "we have evals" and "our evals
actually stop bad changes."

---

## Try these

**1. Run the comparison.** `python runner.py compare`. Sit with the fact
that `-0.033` overall was hiding a `-0.2` case failure.

**2. Delete the trajectory eval.** Comment out `ev.trajectory_match(...)`
in `runner.py`'s `evaluate_case`. Run compare again — **the regression
vanishes**. Every remaining eval looks at the output, and the output was
fine. That's how process bugs ship.

**3. Raise the threshold.** Set `threshold=0.3` in `compare_runs`. The
`-0.2` regression is no longer reported. Threshold choice *is* a policy
decision — too low and judge noise blocks every build, too high and real
regressions slip through.

**4. Make the dataset bad.** Delete `no-tool-needed` and `two-tools` —
the cases that test *failure* modes. Everything passes forever. **A
dataset of only happy paths can tell you when you broke something, never
that you fixed something.**

**5. Turn on the LLM judge** (needs a key). Run the same suite twice with
`use_judge=True` and compare scores. They may differ on identical input —
that's the non-determinism the threshold exists to absorb.

**6. Break `contains_answer` deliberately.** Change the `math-simple`
expected value to `"391."` (with a period). It still passes — because
`normalize()` strips punctuation. Now remove the punctuation stripping
from `normalize()` and watch a correct answer fail.

**7. Add a real test case.** Think of a question this agent would get
wrong, add it to `DATASET` with the answer you *want*, and watch it fail.
That failing case is now a specification. **Production failures you've
fixed are the highest-value source of test cases there is.**

---

## Check yourself

1. Why compare per test case instead of using the overall mean?
2. What does a trajectory eval catch that an output eval can't?
3. Why is the LLM judge opt-in rather than always on?
4. Why does `compare_runs` take a `threshold`?
5. What's wrong with a dataset where everything passes on day one?
6. Why build the harness against a fake agent first?

<details><summary>Answers</summary>

1. Averages hide regressions. A few cases going from perfect to broken
   can be offset by several improving slightly — demonstrated above with
   `-0.033` overall concealing `-0.2` on one case.
2. Process bugs: wrong tools, unnecessary calls, loops, wrong ordering.
   The answer can be perfectly correct via an expensive or fragile path.
3. It costs a call per case and is non-deterministic. Heuristic and
   trajectory evals are free and repeatable — use them wherever they work.
4. Because judge scores wobble. Without a threshold, normal noise would
   block every build; with too high a threshold, real regressions pass.
5. It can only ever detect breakage, never improvement. You need cases
   that currently fail — especially real production failures.
6. So you can verify the *evaluator* is correct before trusting it to
   judge a model. A buggy eval is worse than none — it destroys trust in
   the harness.
</details>

---

**Next:** Project 10 — Planner Agent. Decomposing a goal into steps,
executing them, and **replanning** when a step fails.