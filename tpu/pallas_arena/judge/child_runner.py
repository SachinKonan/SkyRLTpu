"""Child-side grading harness: runs ONE candidate in a forked subprocess.

Invoked by grader.py as `python child_runner.py <config.json>`; the hidden
seed arrives on STDIN (single JSON line) and stdin is closed before any
candidate code runs — the seed is never in the config file, argv, or the
environment, so a seed-reading candidate finds nothing by construction.

Anti-tamper order of operations (DESIGN.md adversarial addendum #2/#4):
  1. read seed, close stdin;
  2. import jax + the problem module; bind DIRECT references to the timer,
     block_until_ready, RNG helpers, reference/baseline callables;
  3. precompute all correctness/adversarial fixtures and WARM the jitted
     baseline per timing shape (compiled executables cannot be monkeypatched
     afterwards);
  4. install poison stubs over every banned module prefix;
  5. only then exec the candidate.
Timing regenerates inputs on-device per timed iteration and verifies
correctness ON OUTPUTS PRODUCED BY TIMED INVOCATIONS. The parent enforces
the wall-clock timeout and RLIMIT_AS; process isolation guarantees nothing
the candidate does in here can touch the judge's own state.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import types


def _write_result(path: str, result: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, default=str)
    os.replace(tmp, path)


def main() -> int:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    result: dict = {"ok": False, "mode": cfg["mode"], "problem": cfg["problem"],
                    "phase": "boot"}
    try:
        # ---- 1. hidden seed over stdin, then close it for good
        seed_line = sys.stdin.readline()
        try:
            os.close(0)
        except OSError:
            pass
        seed = int(json.loads(seed_line)["seed"]) if seed_line.strip() else 0

        for p in cfg.get("pythonpath", []):
            if p not in sys.path:
                sys.path.insert(0, p)

        rc = _run(cfg, seed, result)
        _write_result(cfg["result_path"], result)
        return rc
    except Exception:
        result["error"] = traceback.format_exc(limit=20)
        try:
            _write_result(cfg["result_path"], result)
        except Exception:
            pass
        return 1


def _run(cfg: dict, seed: int, result: dict) -> int:
    # ---- 2. bind everything the harness needs BEFORE candidate code exists
    import jax
    import jax.numpy as jnp  # noqa: F401
    import numpy as np

    from pallas_arena.judge import timing as timing_mod
    from pallas_arena.judge.gates import (
        ArenaBannedImport,
        ast_gate,
        install_poison_stubs,
    )
    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.base import (
        BaselineUnavailable,
        check_tolerance,
        error_stats,
        tolerance_from_reference,
    )

    import zlib

    perf = time.perf_counter            # held ref: candidate patches to
    block = jax.block_until_ready       # time.perf_counter can't reach us
    fold_in = jax.random.fold_in
    prng_key = jax.random.PRNGKey

    def case_salt(name: str) -> int:
        # stable across processes (hash() is PYTHONHASHSEED-salted)
        return zlib.crc32(name.encode()) & 0x7FFF

    problem = get_problem(cfg["problem"])
    mode = cfg["mode"]                  # pregate | gates | full | noise_floor
    code = cfg.get("code", "")
    result["problem_version"] = problem.version
    result["backend"] = jax.default_backend()
    device = jax.local_devices()[0]
    result["device_kind"] = getattr(device, "device_kind", "?")
    chip = ("v6e" if "v6e" in result["device_kind"].lower() else
            "v5p" if "v5p" in result["device_kind"].lower() else
            result["backend"])
    smoke = bool(cfg.get("smoke", False))

    def _mem_stats():
        try:
            s = device.memory_stats()
            return int(s.get("peak_bytes_in_use", 0))
        except Exception:
            return None

    if cfg.get("cases"):
        cases = [problem.case_by_name(n) for n in cfg["cases"]]
    else:
        cases = problem.scored_cases(smoke) + problem.holdout_cases(smoke)
    scored_cases = [c for c in cases if not c.holdout]
    holdout_cases = [c for c in cases if c.holdout]

    key0 = prng_key(seed & 0x7FFFFFFF)
    k_corr, k_adv, k_time, k_floor = jax.random.split(key0, 4)

    n_pairs = int(cfg.get("timing_pairs", 20))
    n_warmup = int(cfg.get("timing_warmup", 3))
    n_det = int(cfg.get("determinism_runs", 5))
    n_seeds = int(cfg.get("correctness_seeds", 3))
    enforce_pallas = cfg.get("enforce_pallas")
    if enforce_pallas is None:
        enforce_pallas = problem.require_pallas

    # ---------------- baseline availability + warm jit per timing case
    result["phase"] = "baseline"
    baseline_fn = None
    baseline_reason = "ok"
    warmed: dict[str, tuple] = {}
    if mode in ("full", "noise_floor"):
        try:
            baseline_fn = jax.jit(problem.baseline)
            for case in cases:
                w_inputs = problem.make_inputs(fold_in(k_time, 999), case)
                block(w_inputs)
                block(baseline_fn(*w_inputs))   # compile + warm NOW
                warmed[case.name] = ()
        except BaselineUnavailable as e:
            baseline_fn, baseline_reason = None, str(e)
        except Exception as e:
            baseline_fn = None
            baseline_reason = f"baseline error: {type(e).__name__}: {e}"
    result["baseline_available"] = baseline_fn is not None
    result["baseline_reason"] = baseline_reason

    # ---------------- noise-floor-only mode: ref vs ref, no candidate at all
    if mode == "noise_floor":
        if baseline_fn is None:
            result["error"] = f"baseline unavailable: {baseline_reason}"
            return 1
        floors, med_ratios = {}, {}
        for case in scored_cases or cases:
            pairs = []
            for i in range(n_warmup + n_pairs):
                inputs = problem.make_inputs(fold_in(k_floor, i), case)
                block(inputs)
                t0 = perf(); a = baseline_fn(*inputs); block(a); t1 = perf()
                b = baseline_fn(*inputs); block(b); t2 = perf()
                if i >= n_warmup:
                    pairs.append((t1 - t0, t2 - t1))
            floors[case.name] = timing_mod.noise_floor_from_ref_pairs(pairs)
            med_ratios[case.name] = timing_mod.interleaved_score(pairs)
        result.update(ok=True, phase="done", noise_floors=floors,
                      ref_vs_ref_scores=med_ratios,
                      noise_floor=max(floors.values()),
                      peak_hbm_bytes=_mem_stats())
        return 0

    # ---------------- AST gate
    result["phase"] = "ast_gate"
    violations = ast_gate(
        code,
        banned_import_prefixes=problem.all_banned_prefixes,
        banned_call_names=problem.banned_call_names,
        require_pallas=enforce_pallas,
    )
    if violations:
        result.update(ok=True, gate="ast", passed=False,
                      violations=violations, reward=cfg.get("fail_reward", 0.0))
        return 0

    # ---------------- correctness / adversarial fixtures BEFORE exec
    result["phase"] = "fixtures"
    corr_fixtures = []  # (label, inputs, ref32, tol)
    for case in scored_cases:
        for g in range(n_seeds):
            key = fold_in(fold_in(k_corr, g), case_salt(case.name))
            inputs = problem.make_inputs(key, case)
            block(inputs)
            ref32 = problem.reference(*inputs)
            tol = tolerance_from_reference(ref32, problem.reference_bf16(*inputs))
            corr_fixtures.append((f"{case.name}#seed{g}", inputs, ref32, tol))
    adv_fixtures = []
    for i, adv in enumerate(problem.adversarial_cases()):
        inputs = adv.make_inputs(fold_in(k_adv, i))
        block(inputs)
        ref32 = problem.reference(*inputs)
        if adv.expect is not None:
            adv.expect(ref32, inputs)  # reference sanity (also pytest-covered)
        tol = tolerance_from_reference(ref32, problem.reference_bf16(*inputs))
        adv_fixtures.append((f"adv:{adv.name}", inputs, ref32, tol))

    grad_fixture = None
    if problem.has_bwd and corr_fixtures:
        _, g_inputs, _, _ = corr_fixtures[0]
        ref_g = problem.grad_outputs(
            lambda *i: problem.reference(*i), *g_inputs)
        cal_g = problem.grad_outputs(
            lambda *i: problem.reference_bf16(*i), *g_inputs)
        grad_fixture = (g_inputs, ref_g, tolerance_from_reference(ref_g, cal_g))

    # ---------------- 4. poison the reference modules; 5. exec candidate
    result["phase"] = "exec"
    result["poisoned_modules"] = len(
        install_poison_stubs(problem.all_banned_prefixes))

    cand_mod = types.ModuleType("candidate")
    cand_mod.__dict__["__name__"] = "candidate"
    try:
        exec(compile(code, "<candidate>", "exec"), cand_mod.__dict__)
    except ArenaBannedImport as e:
        result.update(ok=True, gate="poison_stub", passed=False,
                      violations=[str(e)], reward=cfg.get("fail_reward", 0.0))
        return 0
    except Exception:
        result.update(ok=True, gate="exec", passed=False,
                      violations=[traceback.format_exc(limit=6)],
                      reward=cfg.get("fail_reward", 0.0))
        return 0
    fn = getattr(cand_mod, problem.kernel_entrypoint, None)
    if fn is None:
        result.update(ok=True, gate="exec", passed=False,
                      violations=[f"no `{problem.kernel_entrypoint}` function "
                                  f"defined"],
                      reward=cfg.get("fail_reward", 0.0))
        return 0

    def _call(inputs):
        return fn(*inputs)

    def _fail(gate: str, why: str) -> int:
        result.update(ok=True, gate=gate, passed=False, violations=[why],
                      reward=cfg.get("fail_reward", 0.0))
        return 0

    # ---------------- pregate mode: trace/lower only, zero chip time
    if mode == "pregate":
        try:
            case = scored_cases[0]
            abstract = problem.abstract_inputs(case)
            lowered = jax.jit(fn).lower(*abstract)
            if jax.default_backend() == "tpu":
                t0 = perf()
                lowered.compile()
                result["compile_s"] = perf() - t0
            result.update(ok=True, gate="pregate", passed=True)
            return 0
        except ArenaBannedImport as e:
            return _fail("poison_stub", str(e))
        except Exception as e:
            return _fail("pregate", f"{type(e).__name__}: {e}")

    # ---------------- correctness gates on fresh hidden seeds
    result["phase"] = "correctness"
    t_first = perf()
    for label, inputs, ref32, tol in corr_fixtures + adv_fixtures:
        try:
            out = _call(inputs)
            block(out)
        except ArenaBannedImport as e:
            return _fail("poison_stub", str(e))
        except Exception as e:
            return _fail("correctness",
                         f"{label}: runtime error {type(e).__name__}: {e}")
        stats = error_stats(out, ref32)
        okay, why = check_tolerance(stats, tol)
        if not okay:
            return _fail("correctness", f"{label}: {why}")
    result["first_call_plus_correctness_s"] = perf() - t_first

    # ---------------- determinism: N bitwise-identical runs on same inputs
    result["phase"] = "determinism"
    _, det_inputs, _, _ = corr_fixtures[0]
    outs = []
    for _ in range(n_det):
        o = _call(det_inputs)
        block(o)
        leaves = o if isinstance(o, (tuple, list)) else (o,)
        outs.append(b"".join(np.asarray(x).tobytes() for x in leaves))
    if len(set(outs)) != 1:
        return _fail("determinism",
                     f"outputs not bitwise identical across {n_det} runs")

    # ---------------- gradient contract
    if grad_fixture is not None:
        result["phase"] = "gradient"
        g_inputs, ref_g, g_tol = grad_fixture
        try:
            cand_g = problem.grad_outputs(fn, *g_inputs)
            block(cand_g)
        except Exception as e:
            return _fail("gradient",
                         f"grad failed: {type(e).__name__}: {e}")
        stats = error_stats(cand_g, ref_g)
        okay, why = check_tolerance(stats, g_tol)
        if not okay:
            return _fail("gradient", why)

    if mode == "gates":
        result.update(ok=True, gate="all", passed=True,
                      peak_hbm_bytes=_mem_stats())
        return 0

    # ---------------- timing: interleaved R,C, fresh inputs per iteration,
    # ---------------- correctness verified on timed outputs
    result["phase"] = "timing"
    if baseline_fn is None:
        result.update(ok=True, gate="all", passed=True, score=None,
                      reward=None,
                      error=f"baseline unavailable: {baseline_reason}",
                      peak_hbm_bytes=_mem_stats())
        return 0

    noise_floor = cfg.get("noise_floor")
    if noise_floor is None:
        # quick self-measure on the first scored case (judge normally passes
        # the boot-time per-chip floor instead)
        pairs = []
        for i in range(n_warmup + n_pairs):
            inputs = problem.make_inputs(fold_in(k_floor, i), scored_cases[0])
            block(inputs)
            t0 = perf(); a = baseline_fn(*inputs); block(a); t1 = perf()
            b = baseline_fn(*inputs); block(b); t2 = perf()
            if i >= n_warmup:
                pairs.append((t1 - t0, t2 - t1))
        noise_floor = timing_mod.noise_floor_from_ref_pairs(pairs)
        result["noise_floor_source"] = "self-measured"
    else:
        result["noise_floor_source"] = "judge-provided"

    case_timings = []
    sol_fracs = {}
    for case in scored_cases + holdout_cases:
        pairs = []
        check_iters = sorted({0, n_pairs // 2, n_pairs - 1})
        checks = {}  # iter -> (inputs, timed_output)
        try:
            for i in range(n_warmup + n_pairs):
                inputs = problem.make_inputs(
                    fold_in(fold_in(k_time, i), case_salt(case.name)), case)
                block(inputs)
                t0 = perf()
                r_out = baseline_fn(*inputs)
                block(r_out)
                t1 = perf()
                c_out = _call(inputs)
                block(c_out)
                t2 = perf()
                if i >= n_warmup:
                    it = i - n_warmup
                    pairs.append((t1 - t0, t2 - t1))
                    if it in check_iters:
                        checks[it] = (inputs, c_out)
        except Exception as e:
            return _fail("timing",
                         f"{case.name}: runtime error during timing: "
                         f"{type(e).__name__}: {e}")
        # correctness on outputs produced by TIMED invocations (kills
        # fast-garbage/slow-correct split-personality kernels + memoizers)
        for it, (inputs, c_out) in checks.items():
            ref32 = problem.reference(*inputs)
            tol = tolerance_from_reference(
                ref32, problem.reference_bf16(*inputs))
            stats = error_stats(c_out, ref32)
            okay, why = check_tolerance(stats, tol)
            if not okay:
                return _fail("timed_output_correctness",
                             f"{case.name} timed iter {it}: {why}")
        ct = timing_mod.CaseTiming(case=case.name, pairs=pairs,
                                   holdout=case.holdout)
        case_timings.append(ct)
        bm = problem.bytes_moved(case)
        if problem.memory_bound and bm:
            frac = timing_mod.speed_of_light_fraction(
                bm, ct.cand_median_s, chip)
            if frac is not None:
                sol_fracs[case.name] = frac

    reward_frame = timing_mod.final_reward(case_timings, noise_floor)
    result.update(
        ok=True, gate="all", passed=True,
        **reward_frame,
        speed_of_light_fracs=sol_fracs,
        latencies={t.case: {"ref_median_s": t.ref_median_s,
                            "cand_median_s": t.cand_median_s}
                   for t in case_timings},
        peak_hbm_bytes=_mem_stats(),
        phase="done",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
