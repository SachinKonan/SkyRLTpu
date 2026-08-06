"""Parent-side grader: fork a plain subprocess per candidate (the
SandboxRewardEvaluator.execute_code idiom — no containers), with:

  * wall-clock timeout, TERM -> KILL on the whole process group;
  * RLIMIT_AS on the child (parallel Mosaic compiles of pathological
    unrolls can OOM a host — timeouts alone don't catch it);
  * the hidden seed delivered over STDIN only (never argv/env/config);
  * hash->reward cache so repeat kernels get instant, byte-identical
    rewards (de-noises advantage estimation).

Compile + trace + run all happen in the child; a candidate can never touch
the judge's timer, references, or cache in this (parent) process.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ARENA_ROOT = Path(__file__).resolve().parents[1]  # tpu/pallas_arena
IMPORT_ROOT = ARENA_ROOT.parent  # tpu/ (pallas_arena.*)
REPO_ROOT = IMPORT_ROOT.parent
CHILD_RUNNER = Path(__file__).with_name("child_runner.py")

DEFAULT_RLIMIT_GB = 16.0
DEFAULT_TIMEOUT_S = 240.0


def normalize_code(code: str) -> str:
    lines = [ln.rstrip() for ln in code.strip().splitlines()]
    return "\n".join(lines) + "\n"


def cache_key(problem_name: str, problem_version: str, mode: str, code: str, smoke: bool) -> str:
    h = hashlib.sha256()
    h.update(f"{problem_name}:{problem_version}:{mode}:{int(smoke)}:".encode())
    h.update(normalize_code(code).encode())
    return h.hexdigest()


def _kill_group(proc: subprocess.Popen, hard: bool) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL if hard else signal.SIGTERM)
    except Exception:
        pass


def grade(
    problem_name: str,
    code: str,
    *,
    mode: str = "full",
    smoke: bool = False,
    cases: list[str] | None = None,
    seed: int | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    rlimit_gb: float = DEFAULT_RLIMIT_GB,
    enforce_pallas: bool | None = None,
    noise_floor: float | None = None,
    timing_pairs: int = 20,
    timing_warmup: int = 3,
    determinism_runs: int = 5,
    correctness_seeds: int = 3,
    fail_reward: float = 0.0,
    cache=None,
    child_env: dict[str, str] | None = None,
    workdir: str | None = None,
    problem_version: str | None = None,
) -> dict:
    """Grade one candidate. Returns the child's result dict, augmented with
    parent-side bookkeeping (wall time, cache hit, timeout/oom classification).
    """
    if problem_version is None:
        # resolve lazily without importing jax in the server process
        from pallas_arena.judge.problems import get_problem

        problem_version = get_problem(problem_name).version

    key = cache_key(problem_name, problem_version, mode, code, smoke)
    if cache is not None and mode in ("full", "gates"):
        hit = cache.get(key)
        if hit is not None:
            hit = dict(hit)
            hit["cache_hit"] = True
            return hit

    if seed is None:
        seed = secrets.randbits(31)  # hidden; never logged to the client

    own_workdir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="arena-grade-")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    result_path = wd / "result.json"
    config = {
        "problem": problem_name,
        "mode": mode,
        "code": code,
        "smoke": smoke,
        "cases": cases,
        "result_path": str(result_path),
        "pythonpath": [str(IMPORT_ROOT), str(REPO_ROOT)],
        "enforce_pallas": enforce_pallas,
        "noise_floor": noise_floor,
        "timing_pairs": timing_pairs,
        "timing_warmup": timing_warmup,
        "determinism_runs": determinism_runs,
        "correctness_seeds": correctness_seeds,
        "fail_reward": fail_reward,
        # NB: no seed in this file — stdin only.
    }
    cfg_path = wd / "config.json"
    cfg_path.write_text(json.dumps(config))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(IMPORT_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    # TPU-host split: the judge parent runs jax-CPU (JAX_PLATFORMS=cpu) so
    # the exclusive TPU chip stays free for the grading child. Setting
    # ARENA_CHILD_JAX_PLATFORMS=tpu on the judge redirects children to the
    # chip; unset means children inherit the parent's platform (CPU battery).
    child_platforms = env.pop("ARENA_CHILD_JAX_PLATFORMS", None)
    if child_platforms is not None:
        if child_platforms:
            env["JAX_PLATFORMS"] = child_platforms
        else:
            env.pop("JAX_PLATFORMS", None)
    for var, val in (child_env or {}).items():
        env[var] = val
    env.pop("ARENA_HIDDEN_SEED", None)  # belt & suspenders: never via env

    rlimit_bytes = int(rlimit_gb * 1024**3)

    def _preexec():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (rlimit_bytes, rlimit_bytes))

    t_start = time.perf_counter()
    out_f = open(wd / "child.stdout", "wb")
    err_f = open(wd / "child.stderr", "wb")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(CHILD_RUNNER), str(cfg_path)],
            stdin=subprocess.PIPE,
            stdout=out_f,
            stderr=err_f,
            env=env,
            preexec_fn=_preexec,
        )
    finally:
        out_f.close()
        err_f.close()
    timed_out = False
    try:
        proc.stdin.write(json.dumps({"seed": seed}).encode() + b"\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc, hard=False)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _kill_group(proc, hard=True)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
    except BrokenPipeError:
        proc.wait(timeout=timeout_s)
    wall_s = time.perf_counter() - t_start

    result: dict
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except Exception as e:
            result = {"ok": False, "error": f"unreadable result: {e}"}
    else:
        stderr_tail = ""
        try:
            stderr_tail = (wd / "child.stderr").read_text()[-2000:]
        except Exception:
            pass
        if timed_out:
            result = {
                "ok": True,
                "gate": "timeout",
                "passed": False,
                "violations": [f"timed out after {timeout_s}s"],
                "reward": fail_reward,
            }
        elif proc.returncode and -proc.returncode == signal.SIGKILL:
            result = {
                "ok": True,
                "gate": "rlimit",
                "passed": False,
                "violations": [f"child killed (rc={proc.returncode}; " f"likely RLIMIT_AS {rlimit_gb}G exceeded)"],
                "reward": fail_reward,
                "stderr_tail": stderr_tail,
            }
        else:
            result = {
                "ok": False,
                "gate": "harness",
                "passed": False,
                "violations": [f"child died rc={proc.returncode} with " f"no result"],
                "reward": fail_reward,
                "stderr_tail": stderr_tail,
            }
    if timed_out:
        result.setdefault("gate", "timeout")
        result["timed_out"] = True
        result["passed"] = False
        result.setdefault("reward", fail_reward)
    result["wall_s"] = wall_s
    result["returncode"] = proc.returncode
    result["cache_hit"] = False

    if cache is not None and mode in ("full", "gates") and result.get("ok"):
        try:
            store = {k: v for k, v in result.items() if k not in ("wall_s", "returncode", "cache_hit")}
            cache.put(key, store)
        except Exception:
            pass

    if own_workdir:
        shutil.rmtree(wd, ignore_errors=True)
    return result


def measure_noise_floor(
    problem_name: str,
    *,
    smoke: bool = False,
    timing_pairs: int = 20,
    timeout_s: float = 600.0,
    child_env: dict[str, str] | None = None,
) -> dict:
    """Boot-time per-chip noise floor: ref-vs-ref through the identical
    interleaved protocol (DESIGN.md test layer 3)."""
    return grade(
        problem_name,
        code="",
        mode="noise_floor",
        smoke=smoke,
        timing_pairs=timing_pairs,
        timeout_s=timeout_s,
        cache=None,
        child_env=child_env,
    )
