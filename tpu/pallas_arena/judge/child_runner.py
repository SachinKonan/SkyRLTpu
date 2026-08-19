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

import functools
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


def _tpu_export_device_shim(platforms) -> str | None:
    """Make `jax._src.pallas.mosaic.tpu_info` answer for a TPU while we export
    FOR tpu from a CPU-only child. Returns the pretended device kind, or None.

    MEASURED CONTAMINATION (job 3687041). 17 of megablox_gmm's 96 ladder
    candidates died at gate `aot_export` with

        ValueError: Unsupported TPU device kind: cpu

    which is not a statement about the kernel at all. Every one of them wrote a
    RANK-1 `BlockSpec` for the per-row expert-id array --
    `pl.BlockSpec((bm,), lambda i, j: (i,))` -- and Pallas's rank-1 block-shape
    check (jax/_src/pallas/mosaic/lowering.py) asks the *current device* for its
    sublane and lane counts:

        sublane_count = tpu_info.get_tpu_info().num_sublanes
        lane_count    = tpu_info.get_tpu_info().num_lanes

    `get_tpu_info()` resolves `get_device_kind()`, which under
    `JAX_PLATFORMS=cpu` is the string `"cpu"`, and raises before the check can
    run. GMM is the only task whose natural formulation has a rank-1 operand,
    which is why only GMM hit it. On a real v6e those blocks (bm a power of two
    >= 128) satisfy the check, so those candidates would have exported: the
    error was the export ENVIRONMENT, not the model.

    The shim pins the numbers the lowering asks for to the judge's own chip.
    It is applied ONLY when the child is exporting for `tpu` while its own
    backend is not tpu -- on a TPU host nothing is patched and the real device
    answers. `get_tpu_info` is `jax_util.cache`d, so this must run before the
    first lowering; it does, being immediately before the export loop.
    """
    import jax

    if "tpu" not in tuple(platforms) or jax.default_backend() == "tpu":
        return None
    kind = os.environ.get("ARENA_EXPORT_DEVICE_KIND", "TPU v6e")
    cores = int(os.environ.get("ARENA_EXPORT_DEVICE_CORES", "1"))
    try:
        from jax._src.pallas.mosaic import tpu_info as _tpu_info

        if _tpu_info.chip_version_from_device_kind(kind) is None:
            return None
        _tpu_info.get_device_kind = lambda: kind
        _tpu_info.get_num_device_cores = lambda: cores
    except Exception:
        return None
    return kind


def main() -> int:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)

    result: dict = {"ok": False, "mode": cfg["mode"], "problem": cfg["problem"], "phase": "boot"}
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
    import zlib

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
        check_grad_tolerance,
        check_tolerance,
        error_stats,
        grad_leaf_tolerances,
    )

    perf = time.perf_counter  # held ref: candidate patches to
    block = jax.block_until_ready  # time.perf_counter can't reach us
    fold_in = jax.random.fold_in
    prng_key = jax.random.PRNGKey

    def case_salt(name: str) -> int:
        # stable across processes (hash() is PYTHONHASHSEED-salted)
        return zlib.crc32(name.encode()) & 0x7FFF

    problem = get_problem(cfg["problem"])
    mode = cfg["mode"]  # pregate | gates | full | noise_floor
    code = cfg.get("code", "")
    result["problem_version"] = problem.version
    result["backend"] = jax.default_backend()
    device = jax.local_devices()[0]
    result["device_kind"] = getattr(device, "device_kind", "?")
    chip = (
        "v6e"
        if "v6e" in result["device_kind"].lower()
        else "v5p" if "v5p" in result["device_kind"].lower() else result["backend"]
    )
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
                block(baseline_fn(*w_inputs))  # compile + warm NOW
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
                pair, _, _ = timing_mod.counterbalanced_pair(
                    i, lambda: baseline_fn(*inputs), lambda: baseline_fn(*inputs), perf, block
                )
                if i >= n_warmup:
                    pairs.append(pair)
            floors[case.name] = timing_mod.noise_floor_from_ref_pairs(pairs)
            med_ratios[case.name] = timing_mod.interleaved_score(pairs)
        result.update(
            ok=True,
            phase="done",
            noise_floors=floors,
            ref_vs_ref_scores=med_ratios,
            noise_floor=max(floors.values()),
            peak_hbm_bytes=_mem_stats(),
        )
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
        result.update(ok=True, gate="ast", passed=False, violations=violations, reward=cfg.get("fail_reward", 0.0))
        return 0

    # ---------------- correctness / adversarial fixtures BEFORE exec
    # (skipped for trace/export-only modes: they never touch concrete data)
    result["phase"] = "fixtures"
    corr_fixtures = []  # (label, inputs, ref32, tol)
    fixture_cases = [] if mode in ("pregate", "aot_export") else scored_cases
    for case in fixture_cases:
        # Per-case view: reference, baseline, honest variants and candidates
        # all carry this case's static features, or it IS `problem` when the
        # case has none. Never mix the two -- see Problem.for_case.
        pc = problem.for_case(case)
        for g in range(n_seeds):
            key = fold_in(fold_in(k_corr, g), case_salt(case.name))
            inputs = problem.make_inputs(key, case)
            block(inputs)
            ref32 = pc.reference(*inputs)
            tol = pc.calibrated_tolerance(inputs, ref32)
            corr_fixtures.append((f"{case.name}#seed{g}", inputs, ref32, tol, case.feature_kwargs))
    adv_fixtures = []
    adv_list = [] if mode in ("pregate", "aot_export") else problem.adversarial_cases()
    for i, adv in enumerate(adv_list):
        inputs = adv.make_inputs(fold_in(k_adv, i))
        block(inputs)
        ref32 = problem.reference(*inputs)
        if adv.expect is not None:
            adv.expect(ref32, inputs)  # reference sanity (also pytest-covered)
        tol = problem.calibrated_tolerance(inputs, ref32)
        # Adversarial vectors are built on one plain shape case and carry no
        # features, so they use the unbound problem.
        adv_fixtures.append((f"adv:{adv.name}", inputs, ref32, tol, {}))

    grad_fixture = None
    if problem.has_bwd and corr_fixtures:
        _, g_inputs, _, _, g_feats = corr_fixtures[0]
        pg = problem.for_case(fixture_cases[0]) if fixture_cases else problem
        ref_g = pg.grad_outputs(lambda *i: pg.reference(*i), *g_inputs)
        cal_grads = [pg.grad_outputs(lambda *i: pg.reference_bf16(*i), *g_inputs)]
        # the honest variants' AUTODIFF backwards belong in the band (see
        # grad_leaf_tolerances) -- a variant that cannot express this shape
        # or cannot be differentiated simply does not constrain it
        for _variant in pg.grad_calibration_variants():
            try:
                cal_grads.append(pg.grad_outputs(_variant, *g_inputs))
            except Exception:  # noqa: BLE001
                continue
        grad_fixture = (g_inputs, ref_g, grad_leaf_tolerances(ref_g, *cal_grads), g_feats)

    # ---------------- 4. poison the reference modules; 5. exec candidate
    result["phase"] = "exec"
    result["poisoned_modules"] = len(install_poison_stubs(problem.all_banned_prefixes))

    cand_mod = types.ModuleType("candidate")
    cand_mod.__dict__["__name__"] = "candidate"
    try:
        exec(compile(code, "<candidate>", "exec"), cand_mod.__dict__)
    except ArenaBannedImport as e:
        result.update(
            ok=True, gate="poison_stub", passed=False, violations=[str(e)], reward=cfg.get("fail_reward", 0.0)
        )
        return 0
    except Exception:
        result.update(
            ok=True,
            gate="exec",
            passed=False,
            violations=[traceback.format_exc(limit=6)],
            reward=cfg.get("fail_reward", 0.0),
        )
        return 0
    fn = getattr(cand_mod, problem.kernel_entrypoint, None)
    if fn is None:
        result.update(
            ok=True,
            gate="exec",
            passed=False,
            violations=[f"no `{problem.kernel_entrypoint}` function " f"defined"],
            reward=cfg.get("fail_reward", 0.0),
        )
        return 0

    def _call(inputs, feats=None):
        # Static features are bound per fixture, matching what the judge
        # exported for that case.
        return fn(*inputs, **(feats or {}))

    def _fail(gate: str, why: str) -> int:
        result.update(ok=True, gate=gate, passed=False, violations=[why], reward=cfg.get("fail_reward", 0.0))
        return 0

    # ---------------- aot_export mode (persistent-worker path, phase 3):
    # serialize the candidate per input signature; NO concrete data, NO
    # device — the worker loads and times the artifact on the chip. All
    # candidate python (and any Mosaic-compile bomb) stays in THIS
    # throwaway child under the timeout + RLIMIT_AS.
    if mode == "aot_export":
        from jax import export as jax_export

        artifact_dir = cfg["artifact_dir"]
        os.makedirs(artifact_dir, exist_ok=True)
        platforms = tuple(cfg.get("platforms") or (jax.default_backend(),))
        # see _tpu_export_device_shim: without this, a rank-1 BlockSpec exported
        # for tpu from a cpu child raises `Unsupported TPU device kind: cpu`,
        # which reads as a model failure and is not one (17 GMM candidates).
        shimmed = _tpu_export_device_shim(platforms)
        if shimmed:
            result["export_device_kind"] = shimmed
        artifacts = {}
        t0 = perf()
        try:
            for sig in cfg["signatures"]:
                args = tuple(jax.ShapeDtypeStruct(tuple(a["shape"]), a["dtype"]) for a in sig["args"])
                # Static features (sliding window, soft-cap) are bound BEFORE
                # export so the candidate can specialize on them -- a window it
                # cannot see at compile time is a window it cannot skip blocks
                # with, which is the whole point of the feature.
                feats = sig.get("features") or {}
                bound = functools.partial(fn, **feats) if feats else fn
                if sig.get("kind") == "grad":
                    target = jax.jit(lambda *i: problem.grad_outputs(bound, *i))
                else:
                    target = jax.jit(bound)
                if sig.get("optional"):
                    # Non-differentiable candidate: record WHY and move on. The
                    # forward reward still stands; the backward component is
                    # forfeited. Recording the reason keeps "would this have
                    # passed a hard gate?" answerable after the fact.
                    try:
                        exp = jax_export.export(target, platforms=platforms)(*args)
                    except ArenaBannedImport:
                        raise
                    except Exception as e:
                        result[f"{sig['name']}_export_error"] = f"{type(e).__name__}: {e}"[:400]
                        continue
                else:
                    exp = jax_export.export(target, platforms=platforms)(*args)
                path = os.path.join(artifact_dir, sig["name"] + ".jaxexport")
                with open(path, "wb") as fh:
                    fh.write(bytes(exp.serialize()))
                artifacts[sig["name"]] = path
        except ArenaBannedImport as e:
            return _fail("poison_stub", str(e))
        except Exception as e:
            return _fail("aot_export", f"{type(e).__name__}: {e}")
        result.update(ok=True, gate="aot_export", passed=True, artifacts=artifacts, export_s=perf() - t0)
        return 0

    # ---------------- pregate mode: trace/lower only, zero chip time.
    # The parent bounds this child with the SAME compile budget the judge
    # applies to its own compile-warm, so a compile bomb dies here (killable
    # subprocess, no chip) instead of stalling a judge lane. The gradient
    # functional is lowered too: that is where phase 4's 569.5 s compile
    # lived, and a fwd-only probe would wave it straight through.
    if mode == "pregate":
        try:
            case = scored_cases[0]
            abstract = problem.abstract_inputs(case)
            t0 = perf()
            lowered = jax.jit(fn).lower(*abstract)
            result["lower_fwd_s"] = perf() - t0
            if problem.has_bwd:
                t0 = perf()
                try:
                    jax.jit(lambda *i: problem.grad_outputs(fn, *i)).lower(*abstract)
                except Exception as e:
                    # The pre-gate must PREDICT the judge. If the judge only
                    # scores the backward, a non-differentiable candidate is
                    # not rejected there, so rejecting it here would make the
                    # pre-gate stricter than the contract it screens for.
                    if bool(cfg.get("bwd_gates", getattr(problem, "bwd_gates", True))):
                        raise
                    result["lower_grad_error"] = f"{type(e).__name__}: {e}"[:400]
                result["lower_grad_s"] = perf() - t0
            # Backend compile too, when the client's backend can do it. On a
            # v5p training host that is a real Mosaic/XLA compile, so an
            # XLA-optimization bomb dies here at zero JUDGE chip cost; on a
            # CPU-only client it is the XLA:CPU compile, which shares the
            # platform-independent passes where these bombs actually live.
            # A candidate whose lowering the local backend cannot handle at
            # all (a real pallas_call on CPU) is NOT a failure of the
            # candidate — record it and let the judge decide.
            t0 = perf()
            try:
                lowered.compile()
                result["compile_s"] = perf() - t0
            except NotImplementedError as e:
                result["compile_skipped"] = f"{type(e).__name__}: {e}"[:200]
            except Exception as e:
                if jax.default_backend() == "tpu":
                    raise
                result["compile_skipped"] = f"{type(e).__name__}: {e}"[:200]
            result.update(ok=True, gate="pregate", passed=True)
            return 0
        except ArenaBannedImport as e:
            return _fail("poison_stub", str(e))
        except Exception as e:
            return _fail("pregate", f"{type(e).__name__}: {e}")

    # ---------------- correctness gates on fresh hidden seeds
    result["phase"] = "correctness"
    t_first = perf()
    for label, inputs, ref32, tol, feats in corr_fixtures + adv_fixtures:
        try:
            out = _call(inputs, feats)
            block(out)
        except ArenaBannedImport as e:
            return _fail("poison_stub", str(e))
        except Exception as e:
            return _fail("correctness", f"{label}: runtime error {type(e).__name__}: {e}")
        stats = error_stats(out, ref32)
        okay, why = check_tolerance(stats, tol)
        if not okay:
            return _fail("correctness", f"{label}: {why}")
    result["first_call_plus_correctness_s"] = perf() - t_first

    # ---------------- determinism: N bitwise-identical runs on same inputs
    result["phase"] = "determinism"
    _, det_inputs, _, _, det_feats = corr_fixtures[0]
    outs = []
    for _ in range(n_det):
        o = _call(det_inputs, det_feats)
        block(o)
        leaves = o if isinstance(o, (tuple, list)) else (o,)
        outs.append(b"".join(np.asarray(x).tobytes() for x in leaves))
    if len(set(outs)) != 1:
        return _fail("determinism", f"outputs not bitwise identical across {n_det} runs")

    # ---------------- gradient contract
    if grad_fixture is not None:
        result["phase"] = "gradient"
        # From the CONFIG, not from our own import: the parent decides the
        # contract and built the signatures around it (see grader.grade).
        gates = bool(cfg.get("bwd_gates", getattr(problem, "bwd_gates", True)))
        g_inputs, ref_g, g_tol, g_feats = grad_fixture
        try:
            import functools as _ft
            cand_g = problem.grad_outputs(
                _ft.partial(fn, **g_feats) if g_feats else fn, *g_inputs
            )
            block(cand_g)
        except Exception as e:
            why = f"grad failed: {type(e).__name__}: {e}"
            if gates:
                return _fail("gradient", why)
            result["grad_ok"], result["grad_error"] = False, why[:400]
        else:
            okay, why = check_grad_tolerance(cand_g, ref_g, g_tol)
            if not okay and gates:
                return _fail("gradient", why)
            result["grad_ok"] = bool(okay)
            if not okay:
                # Differentiable but WRONG is a distinct outcome from
                # non-differentiable, and worth separating in the histogram.
                result["grad_error"] = why[:400]

    if mode == "gates":
        result.update(ok=True, gate="all", passed=True, peak_hbm_bytes=_mem_stats())
        return 0

    # ---------------- timing: interleaved R,C, fresh inputs per iteration,
    # ---------------- correctness verified on timed outputs
    result["phase"] = "timing"
    if baseline_fn is None:
        result.update(
            ok=True,
            gate="all",
            passed=True,
            score=None,
            reward=None,
            error=f"baseline unavailable: {baseline_reason}",
            peak_hbm_bytes=_mem_stats(),
        )
        return 0

    noise_floor = cfg.get("noise_floor")
    if noise_floor is None:
        # quick self-measure on the first scored case (judge normally passes
        # the boot-time per-chip floor instead)
        pairs = []
        for i in range(n_warmup + n_pairs):
            inputs = problem.make_inputs(fold_in(k_floor, i), scored_cases[0])
            block(inputs)
            pair, _, _ = timing_mod.counterbalanced_pair(
                i, lambda: baseline_fn(*inputs), lambda: baseline_fn(*inputs), perf, block
            )
            if i >= n_warmup:
                pairs.append(pair)
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
                inputs = problem.make_inputs(fold_in(fold_in(k_time, i), case_salt(case.name)), case)
                block(inputs)
                pair, _r_out, c_out = timing_mod.counterbalanced_pair(
                    i, lambda: baseline_fn(*inputs), lambda: _call(inputs), perf, block
                )
                if i >= n_warmup:
                    it = i - n_warmup
                    pairs.append(pair)
                    if it in check_iters:
                        checks[it] = (inputs, c_out)
        except Exception as e:
            return _fail("timing", f"{case.name}: runtime error during timing: " f"{type(e).__name__}: {e}")
        # correctness on outputs produced by TIMED invocations (kills
        # fast-garbage/slow-correct split-personality kernels + memoizers)
        for it, (inputs, c_out) in checks.items():
            ref32 = problem.reference(*inputs)
            tol = problem.calibrated_tolerance(inputs, ref32)
            stats = error_stats(c_out, ref32)
            okay, why = check_tolerance(stats, tol)
            if not okay:
                return _fail("timed_output_correctness", f"{case.name} timed iter {it}: {why}")
        ct = timing_mod.CaseTiming(
            case=case.name, pairs=pairs, holdout=case.holdout,
            blind=getattr(case, "blind", False),
        )
        case_timings.append(ct)
        bm = problem.bytes_moved(case)
        if problem.memory_bound and bm:
            frac = timing_mod.speed_of_light_fraction(bm, ct.cand_median_s, chip)
            if frac is not None:
                sol_fracs[case.name] = frac

    reward_frame = timing_mod.final_reward(case_timings, noise_floor)
    result.update(
        ok=True,
        gate="all",
        passed=True,
        **reward_frame,
        speed_of_light_fracs=sol_fracs,
        latencies={t.case: {"ref_median_s": t.ref_median_s, "cand_median_s": t.cand_median_s} for t in case_timings},
        peak_hbm_bytes=_mem_stats(),
        phase="done",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
