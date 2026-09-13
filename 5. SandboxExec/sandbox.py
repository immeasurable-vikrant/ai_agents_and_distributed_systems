"""
sandbox.py — running UNTRUSTED code safely.

THE PROBLEM: an LLM writes Python. You want to run it. That code is
untrusted input — exactly like a form field, just executable.

WHY NOT exec() or eval():
  • runs with YOUR process's permissions — same files, same network,
    same env vars (which hold your API keys and DB password)
  • an infinite loop hangs YOUR event loop, freezing every other request
  • `import os; os.system("rm -rf ~")` would genuinely run
  That is the textbook definition of a remote code execution hole.

TWO BACKENDS, auto-detected:

  DOCKER      Strong isolation: separate kernel namespaces, NO network,
              capped memory/CPU/pids, read-only filesystem, auto-removed.
              Use this in anything real.

  SUBPROCESS  Weaker. A separate PROCESS with a scrubbed environment and
              OS resource limits. It stops file writes and fork bombs,
              caps CPU, and protects the parent. But it SHARES your
              kernel, filesystem view, and network — and it CANNOT cap
              memory reliably (see the RLIMIT_AS note below). It is a
              FALLBACK for machines without Docker, not a boundary to
              trust hostile code with.

Being honest about that gap matters more than the code.
"""

import asyncio
import os
import shutil
import sys
import tempfile

TIMEOUT_SECONDS = 5
MEMORY_MB = 256          # docker backend only — see note in the preamble
MAX_OUTPUT_CHARS = 4000

DOCKER_AVAILABLE = shutil.which("docker") is not None
FORCE_SUBPROCESS = os.getenv("SANDBOX_BACKEND", "").lower() == "subprocess"


def backend_name() -> str:
    return "docker" if (DOCKER_AVAILABLE and not FORCE_SUBPROCESS) else "subprocess"


# ============================================================
# BACKEND 1 — DOCKER (the real one)
# ============================================================

async def _run_docker(code: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    cmd = [
        "docker", "run",
        "--rm",
        #   Delete the container the instant it exits — success, crash or
        #   kill. No leftovers, no state carried between runs.

        "--network", "none",
        #   ⭐ THE MOST IMPORTANT FLAG. Not "firewalled" — the container
        #   gets NO network interface at all. The code CANNOT exfiltrate
        #   data or fetch a payload, because there is no network stack
        #   for it to try.

        "--memory", f"{MEMORY_MB}m",
        #   Runaway allocation gets OOM-killed by the KERNEL, cleanly.
        #   Note this is enforced from OUTSIDE the process — which is why
        #   it works where the subprocess backend's RLIMIT_AS does not.

        "--cpus", "0.5",
        #   An infinite loop burns half a core and CANNOT starve the host.

        "--pids-limit", "50",        # fork bombs
        "--read-only",                # immutable filesystem
        "--tmpfs", "/tmp:size=16m",   # small bounded scratch space
        "--user", "nobody",           # unprivileged in-container
        "--security-opt", "no-new-privileges",

        "-v", f"{path}:/code.py:ro",  # :ro — can't rewrite its own script
        "python:3.12-slim", "python", "/code.py",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # ⚠️ wait_for timing out only stops US waiting — the container
            # keeps running. Kill it explicitly or an infinite loop burns
            # CPU forever, invisibly.
            proc.kill()
            await proc.wait()
            return _result(False, "", f"Timed out after {TIMEOUT_SECONDS}s", None)

        return _result(proc.returncode == 0,
                       out.decode(errors="replace"),
                       err.decode(errors="replace"),
                       proc.returncode)
    finally:
        os.unlink(path)


# ============================================================
# BACKEND 2 — SUBPROCESS + RLIMITS (the fallback)
# ============================================================
#
# ⚠️ TWO REAL WEAKNESSES, both found by actually breaking this while
#    building it. Documented because the failures are the lesson.
#
# 1. CPU STARVATION. A child running `while True: pass` on a 1-core
#    machine starved the PARENT so badly it could not run its own
#    timeout to kill the child. The whole environment locked up.
#    Docker's `--cpus 0.5` prevents this structurally. Here we can only
#    ask nicely — hence os.nice(19).
#
# 2. MEMORY CANNOT BE CAPPED SAFELY. The obvious move is
#    RLIMIT_AS = 128MB. Don't. Python 3.12 maps far more VIRTUAL address
#    space than that just to start, so the limit kills the interpreter
#    before your code ever runs — `print(1+1)` dies. RLIMIT_AS limits
#    address space, not resident memory, and modern runtimes reserve a
#    lot of the former. Real memory limiting needs cgroups, i.e. Docker.
#
# "Structurally prevented" vs "asked nicely" is exactly why Docker is
# the real backend and this is a fallback.

_LIMITS_PREAMBLE = f"""
import resource, os

# Lowest scheduling priority, so a CPU-spinning child can never starve
# the parent out of running its own timeout.
try: os.nice(19)
except Exception: pass

# RLIMIT_CPU — max CPU seconds. The kernel SIGKILLs on overrun, which
# catches busy-loops that a wall-clock timeout might race.
resource.setrlimit(resource.RLIMIT_CPU, ({TIMEOUT_SECONDS}, {TIMEOUT_SECONDS}))

# RLIMIT_FSIZE 0 — cannot WRITE CONTENT to any file.
# ⚠️ NUANCE found by testing: the file can still be CREATED (it just
# stays 0 bytes). And because Python buffers writes, a small write can
# LOOK like it succeeded — the failure only surfaces at flush/close.
# So: this blocks data exfiltration to disk, but it is not "no
# filesystem access". Docker's --read-only is the stronger guarantee.
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

# RLIMIT_NPROC — no fork bombs.
try: resource.setrlimit(resource.RLIMIT_NPROC, (24, 24))
except Exception: pass

# NOTE: deliberately NO RLIMIT_AS — see weakness #2 above.
"""


async def _run_subprocess(code: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_LIMITS_PREAMBLE + "\n" + code)
        path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", path,
            #   -I = isolated mode: ignores PYTHON* env vars and the user
            #   site-packages directory. Small hardening step.
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir(),
            env={"PATH": "/usr/bin:/bin"},
            #   ⭐ DELIBERATELY EMPTY ENVIRONMENT. Your real env holds
            #   OPENAI_API_KEY, DATABASE_URL and friends. Inheriting it
            #   would hand every secret you own to the untrusted code.
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), TIMEOUT_SECONDS + 2)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _result(False, "", f"Timed out after {TIMEOUT_SECONDS}s", None)

        return _result(proc.returncode == 0,
                       out.decode(errors="replace"),
                       err.decode(errors="replace"),
                       proc.returncode)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _result(ok: bool, out: str, err: str, code: int | None) -> dict:
    # Truncate. A `while True: print("x")` must not fill your database or
    # logs just because it eventually got killed.
    return {
        "success": ok,
        "stdout": out[:MAX_OUTPUT_CHARS],
        "stderr": err[:MAX_OUTPUT_CHARS],
        "exit_code": code,
        "backend": backend_name(),
    }


async def run_code(code: str) -> dict:
    """Execute untrusted code. Never raises — failures come back as data."""
    if DOCKER_AVAILABLE and not FORCE_SUBPROCESS:
        return await _run_docker(code)
    return await _run_subprocess(code)