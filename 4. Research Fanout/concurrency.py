"""
concurrency.py — the backend half of Project 4.

Four runnable experiments. The POINT is the numbers they print, not the
code. Run them, read the timings, then go look at agent.py and notice the
graph is doing the same thing at a higher level.

    python concurrency.py
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


# ============================================================
# The two kinds of work — everything follows from this distinction
# ============================================================

def cpu_work(n: int = 4_000_000) -> int:
    """
    CPU-BOUND. Pure Python, no I/O, holds the GIL the whole time.
    Module-level (not a lambda) because ProcessPoolExecutor must PICKLE
    the function to send it to another process — lambdas can't be pickled.
    """
    return sum(i * i for i in range(n))


async def io_work(seconds: float = 0.3) -> str:
    """I/O-BOUND. Waiting, not working. The CPU is idle."""
    await asyncio.sleep(seconds)
    return "done"


def blocking_io(seconds: float = 0.3) -> str:
    """A blocking library (requests, psycopg2, open()). Releases the GIL."""
    time.sleep(seconds)
    return "done"


# ============================================================
# EXPERIMENT 1 — `await` in a loop is SEQUENTIAL
# ============================================================
# The most common async performance bug there is. The code LOOKS async
# and runs one-at-a-time, because each `await` completes before the next
# iteration begins.

async def exp1_sequential_vs_gather(n: int = 8):
    t0 = time.perf_counter()
    for _ in range(n):
        await io_work()                       # ❌ one at a time
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*[io_work() for _ in range(n)])   # ✅ all at once
    par = time.perf_counter() - t0

    return {"n": n, "sequential_s": round(seq, 2), "gather_s": round(par, 2),
            "speedup": f"{seq/par:.1f}x"}


# ============================================================
# EXPERIMENT 2 — the GIL: threads vs processes on CPU work
# ============================================================
# The single most misunderstood thing about Python concurrency.
#
#   THREADS on CPU work  → no speedup. The GIL lets only ONE thread
#                          execute Python bytecode at a time. They take
#                          turns; total work is unchanged.
#   PROCESSES on CPU work→ real speedup. Each process has its OWN GIL
#                          and runs on a different core.
#
# ⚠️ You need MULTIPLE CORES to see this. On a 1-core machine (some
# containers, small VMs) processes won't be faster either — there's
# only one core to share. Check os.cpu_count() first.

def exp2_gil(n_tasks: int = 4):
    t0 = time.perf_counter()
    for _ in range(n_tasks):
        cpu_work()
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(cpu_work, [4_000_000] * n_tasks))
    threads = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_tasks) as pool:
        list(pool.map(cpu_work, [4_000_000] * n_tasks))
    procs = time.perf_counter() - t0

    import os
    return {
        "cpu_count": os.cpu_count(),
        "sequential_s": round(seq, 2),
        "threads_s": round(threads, 2),
        "processes_s": round(procs, 2),
        "verdict": ("threads ≈ sequential (GIL); processes faster"
                    if os.cpu_count() > 1 else
                    "1 CPU available — processes can't help here either"),
    }


# ============================================================
# EXPERIMENT 3 — threads DO help I/O
# ============================================================
# The nuance people miss: the GIL is RELEASED during I/O. So threads are
# genuinely useful for blocking I/O — just not for CPU work.
#
# Then why prefer async over threads for I/O? MEMORY.
#   10 threads    = fine
#   10,000 threads = ~80 GB of stack
#   10,000 coroutines = ~50 MB
# For a web server holding many concurrent connections, that's decisive.

def exp3_threads_on_io(n: int = 8):
    t0 = time.perf_counter()
    for _ in range(n):
        blocking_io()
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(blocking_io, [0.3] * n))
    threaded = time.perf_counter() - t0

    return {"n": n, "sequential_s": round(seq, 2), "threaded_s": round(threaded, 2),
            "note": "GIL is released during I/O, so threads genuinely overlap here"}


# ============================================================
# EXPERIMENT 4 — bounded concurrency (semaphore)
# ============================================================
# Unbounded gather() on 10,000 URLs will exhaust file descriptors, get
# you rate-limited, or take down the upstream. A semaphore is a bouncer
# with N wristbands: everyone gets in eventually, the venue never gets
# crushed. This is BACKPRESSURE.

async def exp4_bounded(n: int = 20, limit: int = 5):
    t0 = time.perf_counter()
    await asyncio.gather(*[io_work() for _ in range(n)])
    unbounded = time.perf_counter() - t0

    sem = asyncio.Semaphore(limit)

    async def guarded():
        async with sem:
            return await io_work()

    t0 = time.perf_counter()
    await asyncio.gather(*[guarded() for _ in range(n)])
    bounded = time.perf_counter() - t0

    return {"n": n, "limit": limit,
            "unbounded_s": round(unbounded, 2),
            "bounded_s": round(bounded, 2),
            "note": f"bounded ≈ ceil({n}/{limit}) batches — slower, but safe"}


# ============================================================
# EXPERIMENT 5 — blocking the event loop (the #1 FastAPI incident)
# ============================================================
# `time.sleep` inside `async def` never yields. For its whole duration
# the worker serves NOBODY — not just this request, every concurrent one.
# In real code this arrives disguised as requests.get() or a sync DB driver.

async def exp5_blocking_loop(n: int = 5):
    async def bad():
        time.sleep(0.3)            # ❌ blocks the loop
    async def good():
        await asyncio.sleep(0.3)   # ✅ yields

    t0 = time.perf_counter()
    await asyncio.gather(*[bad() for _ in range(n)])
    blocked = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*[good() for _ in range(n)])
    ok = time.perf_counter() - t0

    return {"n": n, "blocking_s": round(blocked, 2), "nonblocking_s": round(ok, 2),
            "note": "identical 'work', but blocking serializes everything"}


async def run_all() -> dict:
    return {
        "exp1_sequential_vs_gather": await exp1_sequential_vs_gather(),
        "exp2_gil_threads_vs_processes": exp2_gil(),
        "exp3_threads_on_io": exp3_threads_on_io(),
        "exp4_bounded_concurrency": await exp4_bounded(),
        "exp5_blocking_the_loop": await exp5_blocking_loop(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(asyncio.run(run_all()), indent=2))