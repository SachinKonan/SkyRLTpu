"""Persistent judge worker (phase-3 redesign of the grading hot path).

The phase-2 shakedown measured 112 s mean per candidate: a fresh fork paid
jax+libtpu init, reference compile, and full-shape fixtures every time.
This worker splits the work:

  * BOOT (once): jax/TPU init, reference + baseline compiled and warmed per
    shape case, per-case noise floor measured with the counterbalanced
    protocol. The worker owns the chip for its whole life.
  * PER CANDIDATE: the candidate's python runs ONLY in a throwaway sandbox
    child (grader.grade mode="aot_export": AST gate, poison stubs, timeout,
    RLIMIT_AS — no device), which serializes the kernel per input signature
    via jax.export. The worker deserializes the artifacts and does gates +
    counterbalanced timing on-chip. Target ~4 s warm chip time.

Anti-cheat is STRICTLY stronger than the fork-per-candidate path: no
candidate python ever executes in the worker process — only compiled
tensor programs — so the judge's timer, references, hidden seeds and cache
are unreachable by construction; the hidden fixture seed never exists in
the child at all (the child only ever sees abstract shapes).

Pull-queue mode: the worker polls GET /work on the client-side queue,
heartbeats while grading, POSTs the result, repeats. Stateless: kill it
any time; the lease expires and the item regrades elsewhere.

Run:
  JAX_PLATFORMS=cpu ARENA_CHILD_JAX_PLATFORMS=cpu \
      python -m pallas_arena.judge.worker --problem rmsnorm --smoke \
      --queue http://127.0.0.1:8770 --sim-mode real     # CPU sim
  (on a TPU judge host: no JAX_PLATFORMS override — worker owns the chip —
   and ARENA_CHILD_JAX_PLATFORMS=cpu so export children stay off it.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from pallas_arena.judge import grader
from pallas_arena.judge import observation
from pallas_arena.judge import timing as timing_mod

# Compile-warm deadline (phase-5 prerequisite). Phase 4 measured the honest
# distribution of `candidate_compile_s` on a v6e-1: median 2.98 s, typical
# band 2.2-3.7 s, worst HONEST candidate 46.2 s (`nondeterministic`) and
# 45.7 s in attempt 1. One adversarial candidate spent 569.5 s inside a
# single XLA compile while using 3.1 s of chip time.
#
# 90 s = ~2x the worst honest compile ever measured (46.2 s) and ~30x the
# median. Rationale for each side of the bound:
#   * high enough that no honest kernel is at risk: 2x margin over the worst
#     observation absorbs chip-to-chip and boot-to-boot compiler variance,
#     and a candidate needing >90 s of compile has already lost on score;
#   * low enough to be worth having: it caps a poison candidate's lane cost
#     at ~6x the median item wall (5.01 s) instead of 114x, so a fully
#     adversarial corpus costs a 5-judge fleet at most 7.5 judge-minutes of
#     compile instead of an unbounded stall;
#   * strictly under the 240 s lease timeout, so the rejection is POSTed
#     before the queue would requeue the item — the poison pill is never
#     handed to a second judge.
DEFAULT_COMPILE_BUDGET_S = 90.0


def build_signatures(problem, scored_cases, holdout_cases, adv_cases):
    """Deduped input signatures the sandbox child must export: shape cases +
    adversarial shapes + the gradient functional. Shapes are public.

    Module-level so a CLIENT-side AOT pre-gate exports the EXACT same
    signature set the judge will demand. If the two ever diverge the pre-gate
    stops being a pre-gate: it would pass candidates the judge rejects (or
    worse, reject ones it would have accepted) and the gate histogram would
    be measuring two different contracts.
    """
    import jax

    sigs, seen = [], {}

    def sig_of(abstract, kind, label, optional=False, features=None):
        args = [{"shape": list(a.shape), "dtype": str(a.dtype)} for a in abstract]
        # Features are part of the signature IDENTITY, not just metadata: they
        # are bound statically, so a windowed case and a plain case that happen
        # to share shapes are DIFFERENT programs. Deduping on shapes alone
        # would hand both cases one artifact and silently grade the windowed
        # case against the plain kernel.
        feats = dict(features or {})
        key = (kind, json.dumps(args), json.dumps(feats, sort_keys=True))
        if key in seen:
            return seen[key]
        name = f"{kind}{len(sigs):02d}"
        entry = {"name": name, "kind": kind, "args": args, "label": label}
        if feats:
            entry["features"] = feats
        if optional:
            # An OPTIONAL signature that fails to export costs the candidate
            # that component of the reward; it does not fail the export gate.
            entry["optional"] = True
        sigs.append(entry)
        seen[key] = name
        return name

    case_sig = {}
    for case in list(scored_cases) + list(holdout_cases):
        # A TP case exports at PER-SHARD shapes: under shard_map the kernel is
        # handed one device's slice, so an artifact exported at the full shape
        # could never be called. The width is DECLARED on the case, so the CPU
        # pre-gate (1 device) and the judge (8) derive identical signatures.
        w = problem.tp_declared_width(case)
        abstract = problem.abstract_inputs_tp(case, w) if w else problem.abstract_inputs(case)
        case_sig[case.name] = sig_of(abstract, "fwd", case.name, features=case.feature_kwargs)
    adv_sig = {}
    for adv in adv_cases:
        abstract = jax.eval_shape(adv.make_inputs, jax.ShapeDtypeStruct((2,), "uint32"))
        adv_sig[adv.name] = sig_of(abstract, "fwd", f"adv:{adv.name}")
    grad_sig = None
    if problem.has_bwd and scored_cases:
        grad_sig = sig_of(
            problem.abstract_inputs(scored_cases[0]),
            "grad",
            "grad",
            optional=not problem.bwd_gates,
        )
    return sigs, case_sig, adv_sig, grad_sig


class Fixture(NamedTuple):
    """One correctness fixture, by NAME rather than by position.

    Positional tuples cost three separate breakages in one change when this
    grew a `case` field: the compile-warm loop, the correctness loop and the
    determinism unpack each failed with `too many values to unpack`, one
    battery run apart. Attribute access makes adding a field a non-event.
    """

    label: str
    sig_name: str
    inputs: tuple
    case: object  # ShapeCase, or None for adversarial vectors (no features)
    # The callable correctness/determinism/warm must invoke: fns[sig] for a
    # plain case, the shard_map-WRAPPED artifact for a TP case. A TP artifact
    # is exported at PER-SHARD shapes (build_signatures), so calling it with
    # full-shape fixture inputs is a shape error -- every TP correctness
    # check would have crashed the first time a TP case was selected on a
    # multi-chip judge. Resolved once, at fixture-build time.
    call: object


class CompileBudgetExceeded(Exception):
    """Candidate blew the compile-warm deadline. Terminal: never requeued."""


class PersistentWorker:
    def __init__(
        self,
        problem_name: str,
        *,
        smoke: bool = False,
        cases: list[str] | None = None,
        use_adversarial: bool = True,
        adversarial_case: str | None = None,
        timing_pairs: int = 20,
        timing_warmup: int = 3,
        determinism_runs: int = 5,
        correctness_seeds: int = 2,
        export_timeout_s: float = 240.0,
        export_rlimit_gb: float | None = None,
        grade_budget_s: float | None = 900.0,
        compile_budget_s: float | None = DEFAULT_COMPILE_BUDGET_S,
        cache=None,
        worker_id: str = "worker-0",
    ):
        import jax

        from pallas_arena.judge.problems import get_problem

        self.jax = jax
        self.problem = get_problem(problem_name)
        self.problem_name = problem_name
        self.smoke = smoke
        # The adversarial library is built on ONE shape case (`tiny` by
        # default). Its shapes are exported alongside the scored ones, so a
        # judge running a non-default case set would silently require every
        # candidate to trace at shapes no prompt ever declared. Both knobs
        # exist to keep the graded contract equal to the published one.
        self.use_adversarial = use_adversarial
        if adversarial_case:
            self.problem.adversarial_case_name = adversarial_case
        self.timing_pairs = timing_pairs
        self.timing_warmup = timing_warmup
        self.determinism_runs = determinism_runs
        self.correctness_seeds = correctness_seeds
        self.export_timeout_s = export_timeout_s
        self.export_rlimit_gb = export_rlimit_gb
        self.grade_budget_s = grade_budget_s
        self.compile_budget_s = compile_budget_s
        self.cache = cache
        self.worker_id = worker_id
        # Set by poll_loop before each grade: posts a TERMINAL result for the
        # item currently leased and then hard-exits. Only the compile-warm
        # watchdog uses it (see _compile_warm).
        self.terminal_poster = None
        self.compile_timeouts = 0

        self.perf = time.perf_counter
        self.block = jax.block_until_ready
        self.device = jax.local_devices()[0]
        self.platform = jax.default_backend()
        # A v5p-8 (4 chips) or v6e-8 host hands ONE process every local chip.
        # Pin uncommitted computations to chip 0 so a multi-chip host grades
        # exactly like the single-chip judge the timing protocol was
        # calibrated on, and so peak_hbm_bytes below is the whole story
        # rather than one chip's share of a silently-sharded computation.
        try:
            jax.config.update("jax_default_device", self.device)
        except Exception:
            pass

        if cases:
            sel = [self.problem.case_by_name(n) for n in cases]
        else:
            sel = self.problem.scored_cases(smoke) + self.problem.holdout_cases(smoke)
        self.scored_cases = [c for c in sel if not c.holdout]
        self.holdout_cases = [c for c in sel if c.holdout]
        self.boot_report: dict = {}
        self._baseline_fn = None
        self.noise_floors: dict[str, float] = {}
        self.noise_floor: float | None = None

    # ------------------------------------------------------------------ boot
    def boot(self) -> dict:
        jax = self.jax
        t0 = self.perf()
        try:
            self._baseline_fn = jax.jit(self.problem.baseline)
            for case in self.scored_cases + self.holdout_cases:
                w = self.problem.make_inputs(jax.random.PRNGKey(0), case)
                self.block(w)
                self.block(self._baseline_fn(*w))
        except Exception as e:
            self.boot_report = {"ok": False, "error": f"baseline: {type(e).__name__}: {e}"}
            return self.boot_report

        # GENERAL mode: pick the denominator per shape by MEASUREMENT. Every
        # named candidate is timed here, at boot, off the candidate clock, and
        # the fastest wins that case. This is the structural fix for the defect
        # that invalidated every historical score: splash ran ~10x slow and
        # megablox ~38x slow on library-default tuning, so "beat the baseline"
        # meant "beat a crippled kernel". Which one won is recorded per case,
        # so a score always says what it beat.
        # Can this judge device-time (tokamax's TPU default) or must it fall
        # back to the Python stopwatch? Probed once, here, so a whole run's
        # numbers are labelled with how they were measured.
        self.device_timing = timing_mod.device_timing_available()
        print(f"[boot] device timing (xprof DIU): "
              f"{'AVAILABLE' if self.device_timing else 'unavailable -> wallclock'}", flush=True)

        self._general_baselines = {}
        if getattr(self.problem, "general_mode", False):
            t_bl = self.perf()
            for case in self.scored_cases + self.holdout_cases:
                # Candidates are re-fetched PER CASE: with static features the
                # set is case-dependent (each carries that case's window /
                # soft-cap), so hoisting it out of the loop would elect a
                # baseline computing a different function than the case asks
                # for. Featureless cases return the same objects as before.
                cands = self.problem.for_case(case).baseline_candidates()
                w = self.problem.make_inputs(jax.random.PRNGKey(1), case)
                self.block(w)
                timings = {}
                for nm, fn in cands.items():
                    try:
                        f = jax.jit(fn)
                        for _ in range(2):
                            self.block(f(*w))
                        ts = []
                        for _ in range(5):
                            t = self.perf()
                            self.block(f(*w))
                            ts.append(self.perf() - t)
                        ts.sort()
                        timings[nm] = ts[len(ts) // 2]
                    except Exception as e:  # noqa: BLE001 -- an unavailable candidate is not a boot failure
                        timings[nm] = None
                        print(f"[boot] baseline candidate {nm!r} unavailable at {case.name}: "
                              f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                live = {k: v for k, v in timings.items() if v}
                if not live:
                    self.boot_report = {"ok": False, "error": f"no usable baseline at {case.name}"}
                    return self.boot_report
                best = min(live, key=live.get)
                self._general_baselines[case.name] = {
                    "impl": best, "fn": jax.jit(cands[best]), "raw": cands[best],
                    "median_s": live[best], "all_s": live,
                }
                print(f"[boot] {case.name}: baseline={best} "
                      f"({', '.join(f'{k}={v * 1e3:.2f}ms' for k, v in live.items())})", flush=True)
                # Free this case's inputs before the next one allocates: with a
                # 6-shape sweep these accumulate across boot and the judge
                # segfaults (rc=139, job 3699504). The elected jitted fn is
                # kept; only the fixtures are dropped.
                del w
                import gc

                gc.collect()
            # Every LOSING candidate left a compiled executable in jax's global
            # cache -- 2 candidates x N shapes of dead programs held on a 32 GB
            # chip. Boot segfaults right after this loop with a 6-shape sweep
            # (rc=139, jobs 3699910/3699504), so drop them before calibration
            # warm allocates its fixtures. The winners recompile on first use,
            # which is off the candidate clock.
            self.jax.clear_caches()
            import gc as _gc

            _gc.collect()
            self.general_baseline_s = self.perf() - t_bl

        # Warm the CALIBRATION path too, not just the baseline. Every
        # correctness check runs the fp32 reference, the three honest
        # variants and the fused error-stats kernel — one XLA compile per
        # (shape, dtype) apiece. Off the clock here, they would otherwise
        # all land inside candidate #1's warm_chip_s.
        t_cal = self.perf()
        cal_warm_err = None
        try:
            from pallas_arena.judge.problems.base import error_stats

            k_warm = jax.random.PRNGKey(1)
            probes = [
                (c, lambda c=c, i=i: self.problem.make_inputs(jax.random.fold_in(k_warm, i), c))
                for i, c in enumerate(self.scored_cases + self.holdout_cases)
            ]
            probes += [
                (a, lambda a=a, i=i: a.make_inputs(jax.random.fold_in(k_warm, 1000 + i)))
                for i, a in enumerate(self.adversarial_cases())
            ]
            for _what, make in probes:
                inputs = make()
                self.block(inputs)
                # Warm the SAME callables the graded path will use: a featured
                # case compiles a different reference and different variants,
                # and warming the unbound ones would leave that compile inside
                # candidate #1's warm_chip_s -- the exact cost this block exists
                # to hoist out.
                pw = self.problem.for_case(_what) if hasattr(_what, "feature_kwargs") else self.problem
                ref32 = pw.reference(*inputs)
                self.block(ref32)
                pw.calibrated_tolerance(inputs, ref32)
                error_stats(ref32, ref32)
                del inputs, ref32  # keep boot peak HBM to one fixture
        except Exception as e:  # never fatal: a warm-up is an optimization
            cal_warm_err = f"{type(e).__name__}: {e}"
        cal_warm_s = self.perf() - t_cal

        floors, ref_scores = {}, {}
        k_floor = jax.random.PRNGKey(secrets.randbits(31))
        for case in self.scored_cases:
            pairs = []
            for i in range(self.timing_warmup + self.timing_pairs):
                inputs = self.problem.make_inputs(jax.random.fold_in(k_floor, i), case)
                self.block(inputs)
                pair, _, _ = timing_mod.counterbalanced_pair(
                    i, lambda: self._baseline_fn(*inputs), lambda: self._baseline_fn(*inputs), self.perf, self.block
                )
                if i >= self.timing_warmup:
                    pairs.append(pair)
            floors[case.name] = timing_mod.noise_floor_from_ref_pairs(pairs)
            ref_scores[case.name] = timing_mod.interleaved_score(pairs)
        self.noise_floors = floors
        self.noise_floor = max(floors.values()) if floors else None
        self.boot_report = {
            "ok": True,
            "platform": self.platform,
            "device_kind": getattr(self.device, "device_kind", "?"),
            "noise_floors": floors,
            "ref_vs_ref_scores": ref_scores,
            "noise_floor": self.noise_floor,
            "calibration_warm_s": cal_warm_s,
            "device_timing": self.device_timing,
            "calibration_warm_error": cal_warm_err,
            "boot_s": self.perf() - t0,
            "baseline_impl": getattr(self.problem, "baseline_impl", None),
            "cases": [c.name for c in self.scored_cases],
            "holdout_cases": [c.name for c in self.holdout_cases],
            "adversarial": self.use_adversarial,
        }
        return self.boot_report

    def adversarial_cases(self):
        """The adversarial vector library, or nothing when it is disabled."""
        return self.problem.adversarial_cases() if self.use_adversarial else []

    # ------------------------------------------------------------ signatures
    def _signatures(self):
        return build_signatures(self.problem, self.scored_cases, self.holdout_cases, self.adversarial_cases())

    # ----------------------------------------------------------------- grade
    def grade_code(self, code: str, *, enforce_pallas: bool | None = None, tag=None) -> dict:
        jax = self.jax
        problem = self.problem
        result: dict = {
            "ok": True,
            "problem": self.problem_name,
            "problem_version": problem.version,
            "worker_id": self.worker_id,
            "backend": self.platform,
            "device_kind": getattr(self.device, "device_kind", "?"),
            "tag": tag,
            # boot invariants travel with every result so a queue client can
            # assert ref-vs-ref/floor without host access (phase-4 driver)
            "worker_boot": {
                k: self.boot_report.get(k) for k in ("noise_floor", "noise_floors", "ref_vs_ref_scores", "boot_s")
            },
        }

        if self.cache is not None:
            key = grader.cache_key(
                self.problem_name, problem.version, "worker_full", code, self.smoke,
                contract=self._contract_tag(),
            )
            hit = self.cache.get(key)
            if hit is not None:
                hit = dict(hit)
                hit["cache_hit"] = True
                return hit

        def fail(gate, why, terminal=False, judge_fault=False):
            result.update(passed=False, gate=gate, violations=[why], reward=0.0)
            observation.attach_observation(result)
            if terminal:
                # A terminal verdict is a property of the CANDIDATE, not of
                # the judge that happened to draw it, so the queue must not
                # hand it to anyone else and the cache must serve it to every
                # future judge for free.
                result["terminal"] = True
            if judge_fault:
                # The exact opposite: a property of the JUDGE, not the
                # candidate. Phase-5 lesson -- one candidate halted the TPU
                # core at Mosaic compile time and the worker survived it, then
                # failed the next 35 items in fixture setup at ~1.1s each while
                # still reporting them as ordinary verdicts. Two things went
                # wrong and both are fixed here:
                #   * the zero was CACHED against each candidate's hash, so a
                #     dead chip permanently condemned 35 innocent candidates
                #     for every future judge -> never store a judge fault;
                #   * it was returned as a verdict, so the run looked healthy
                #     -> the poll loop now requeues the item and exits, and the
                #     supervisor brings the judge back on a fresh chip.
                result["judge_fault"] = True
                return result
            self._store(code, result)
            return result

        # Wall budget for the whole grade. Phase 4 measured why this is not
        # optional: one candidate (a 24-deep unjitted waste chain) stayed on
        # the chip for 19+ minutes and never returned, so its lease expired
        # and the item requeued — onto the next judge, which would wedge the
        # same way. Unbounded grading turns a single pathological kernel into
        # a fleet-wide poison pill. Checked between phases only: XLA work
        # already dispatched still has to finish, so the bound is "one op
        # late", not a hard kill. A candidate this slow has already lost on
        # score, so failing it here costs nothing legitimate.
        t_budget = self.perf()

        def over_budget() -> bool:
            return self.grade_budget_s is not None and (self.perf() - t_budget) > self.grade_budget_s

        def budget_fail(phase):
            return fail("budget", f"grade exceeded {self.grade_budget_s:.0f}s wall budget during {phase}")

        # ---- 1. sandbox child: AST + stubs + exec + jax.export (no device)
        sigs, case_sig, adv_sig, grad_sig = self._signatures()
        wd = tempfile.mkdtemp(prefix="arena-worker-")
        t_export = self.perf()
        child = grader.grade(
            self.problem_name,
            code,
            mode="aot_export",
            smoke=self.smoke,
            enforce_pallas=enforce_pallas,
            timeout_s=self.export_timeout_s,
            rlimit_gb=self.export_rlimit_gb,
            cache=None,
            workdir=wd,
            export_signatures=sigs,
            export_platforms=[self.platform],
            child_env={"JAX_PLATFORMS": "cpu"},  # the child NEVER gets the chip
        )
        result["export_s"] = self.perf() - t_export
        if not child.get("passed"):
            result.update(
                passed=False,
                gate=child.get("gate", "aot_export"),
                violations=child.get("violations", [child.get("error", "export failed")]),
                reward=0.0,
            )
            observation.attach_observation(result)
            self._store(code, result)
            return result

        # ---- 2. load artifacts (compile happens here, off the timed path)
        from jax import export as jax_export

        t_load = self.perf()
        fns: dict[str, object] = {}
        try:
            for name, path in child["artifacts"].items():
                exported = jax_export.deserialize(bytearray(Path(path).read_bytes()))
                fns[name] = jax.jit(exported.call)
        except Exception as e:
            return fail("artifact_load", f"{type(e).__name__}: {e}")
        result["load_s"] = self.perf() - t_load

        perf, block = self.perf, self.block
        fold_in = jax.random.fold_in
        seed_key = jax.random.PRNGKey(secrets.randbits(31))  # never leaves us
        k_corr, k_adv, k_time = jax.random.split(seed_key, 3)

        # ---- 2b. build fixtures + compile-warm EVERY artifact off the timed
        # path (the compile-vs-chip split the throughput table assumes:
        # warm_chip_s below is pure steady-state chip time)
        try:
            fixtures = []
            skipped_tp_fixture = {}
            n_dev = len(jax.local_devices())
            for case in self.scored_cases:
                tp_w = problem.tp_declared_width(case)
                call = fns[case_sig[case.name]]
                if tp_w:
                    if n_dev < tp_w:
                        # The artifact is shard-shaped; without the mesh it
                        # cannot be invoked AT ALL. Skip loudly rather than
                        # crash into it or silently grade at another width.
                        skipped_tp_fixture[case.name] = f"needs {tp_w} devices, judge has {n_dev}"
                        continue
                    tp_in, tp_out = problem.tp_specs(case)
                    tp_mesh = timing_mod.make_mesh(tp_w)
                    call = jax.jit(timing_mod.shard_mapped(call, tp_mesh, tp_in, tp_out))
                for g in range(self.correctness_seeds):
                    inputs = problem.make_inputs(fold_in(fold_in(k_corr, g), hash_stable(case.name)), case)
                    block(inputs)
                    fixtures.append(Fixture(f"{case.name}#seed{g}", case_sig[case.name], inputs, case, call))
            if skipped_tp_fixture:
                result["skipped_tp_correctness"] = skipped_tp_fixture
            if not fixtures and self.scored_cases:
                return fail(
                    "fixtures",
                    f"every scored case is a TP case this judge cannot run: {skipped_tp_fixture}",
                    judge_fault=True,
                )
            for i, adv in enumerate(self.adversarial_cases()):
                inputs = adv.make_inputs(fold_in(k_adv, i))
                block(inputs)
                # adversarial vectors carry no features -> case is None
                fixtures.append(Fixture(f"adv:{adv.name}", adv_sig[adv.name], inputs, None, fns[adv_sig[adv.name]]))
        except Exception as e:
            # Fixtures are built from the PROBLEM's own make_inputs and a
            # block() -- no candidate code is consulted -- so a failure here
            # can only mean this judge's chip is gone.
            return fail("fixtures", f"{type(e).__name__}: {e}", judge_fault=True)

        # Compile-warm units: one per distinct exported signature, plus the
        # gradient functional. Built as thunks so the deadline can be checked
        # BETWEEN them (see _compile_warm).
        warm_units, warmed = [], set()
        for f in fixtures:
            if f.sig_name not in warmed:
                # warm the callable the fixture will actually run -- for a TP
                # case that is the shard_map wrapper, whose compile is the one
                # that must land off the timed path
                warm_units.append((f.sig_name, lambda c=f.call, i=f.inputs: block(c(*i))))
                warmed.add(f.sig_name)
        try:
            for case in self.holdout_cases:
                sig_name = case_sig[case.name]
                if sig_name not in warmed:
                    # Same shard-shape rule as fixtures: a TP holdout's artifact
                    # can only be invoked through shard_map, and a small judge
                    # skips it (the timing loop will record skipped_tp).
                    tp_w = problem.tp_declared_width(case)
                    w_call = fns[sig_name]
                    if tp_w:
                        if len(jax.local_devices()) < tp_w:
                            continue
                        h_in, h_out = problem.tp_specs(case)
                        w_call = jax.jit(timing_mod.shard_mapped(w_call, timing_mod.make_mesh(tp_w), h_in, h_out))
                    w_in = problem.make_inputs(fold_in(k_time, 10_000), case)
                    block(w_in)
                    warm_units.append((sig_name, lambda c=w_call, i=w_in: block(c(*i))))
                    warmed.add(sig_name)
        except Exception as e:
            return fail("fixtures", f"{type(e).__name__}: {e}", judge_fault=True)
        if grad_sig is not None and fixtures:
            warm_units.append(("grad", lambda: block(fns[grad_sig](*fixtures[0].inputs))))

        t_warmc = self.perf()
        try:
            self._compile_warm(warm_units, code, tag)
        except CompileBudgetExceeded as e:
            result["candidate_compile_s"] = self.perf() - t_warmc
            result["compile_budget_s"] = self.compile_budget_s
            return fail("compile_budget", str(e), terminal=True)
        except Exception as e:
            result["candidate_compile_s"] = self.perf() - t_warmc
            return fail("candidate_compile", f"{type(e).__name__}: {e}")
        result["candidate_compile_s"] = self.perf() - t_warmc
        if over_budget():
            return budget_fail("candidate_compile")

        t_chip = self.perf()
        try:
            # ---- 3. correctness on fresh hidden seeds + adversarial vectors
            from pallas_arena.judge.problems.base import check_tolerance, error_stats

            for label, sig_name, inputs, case, call in ((f.label, f.sig_name, f.inputs, f.case, f.call) for f in fixtures):
                if over_budget():
                    return budget_fail(f"correctness ({label})")
                pc = problem.for_case(case)
                ref32 = pc.reference(*inputs)
                tol = pc.calibrated_tolerance(inputs, ref32)
                try:
                    out = call(*inputs)
                    block(out)
                except Exception as e:
                    return fail("correctness", f"{label}: runtime error {type(e).__name__}: {e}")
                okay, why = check_tolerance(error_stats(out, ref32), tol)
                if not okay:
                    return fail("correctness", f"{label}: {why}")

            # ---- 4. determinism: N bitwise-identical runs
            import numpy as np

            det_call, det_inputs = fixtures[0].call, fixtures[0].inputs
            outs = []
            for _ in range(self.determinism_runs):
                o = det_call(*det_inputs)
                block(o)
                leaves = o if isinstance(o, (tuple, list)) else (o,)
                outs.append(b"".join(np.asarray(x).tobytes() for x in leaves))
            if len(set(outs)) != 1:
                return fail("determinism", f"outputs not bitwise identical across {self.determinism_runs} runs")

            # ---- 5. gradient contract via the exported grad artifact
            if over_budget():
                return budget_fail("gradient")
            # A non-differentiable candidate exports no grad artifact when the
            # backward is scored rather than gated. That is a recorded outcome
            # (grad_reward stays absent), not a judge fault.
            if grad_sig is not None and grad_sig not in fns:
                result["grad_ok"] = False
                result["grad_error"] = (
                    child.get(f"{grad_sig}_export_error") or "backward not exported"
                )
                grad_sig = None
            if grad_sig is not None:
                from pallas_arena.judge.problems.base import check_grad_tolerance as check_grad_tol
                from pallas_arena.judge.problems.base import grad_leaf_tolerances as tolerance_from_reference_leaves

                g_case = self.scored_cases[0]
                pg = problem.for_case(g_case)
                g_inputs = problem.make_inputs(fold_in(k_corr, 999), g_case)
                block(g_inputs)
                ref_g = pg.grad_outputs(lambda *i: pg.reference(*i), *g_inputs)
                cal_g = pg.grad_outputs(lambda *i: pg.reference_bf16(*i), *g_inputs)
                g_tol = tolerance_from_reference_leaves(ref_g, cal_g)
                gates = problem.bwd_gates
                try:
                    cand_g = fns[grad_sig](*g_inputs)
                    block(cand_g)
                except Exception as e:
                    why = f"grad failed: {type(e).__name__}: {e}"
                    if gates:
                        return fail("gradient", why)
                    result["grad_ok"], result["grad_error"] = False, why[:400]
                    grad_sig = None
                if grad_sig is not None:
                    okay, why = check_grad_tol(cand_g, ref_g, g_tol)
                    if not okay and gates:
                        return fail("gradient", why)
                    result["grad_ok"] = bool(okay)
                    if not okay:
                        result["grad_error"] = why[:400]
                        # Wrong gradients must not be TIMED into a reward --
                        # a fast wrong backward is worth nothing.
                        grad_sig = None

                # ---- 5b. TIME the backward as its own benchmark.
                # tokamax treats forward and forward+vjp as two separate
                # benchmarks with separately tuned configs
                # (dot_product_attention vs dot_product_attention_vjp are
                # distinct autotuning cache entries), and for training the
                # backward is often the more expensive half. Until now our
                # backward was only a pass/fail gate and the reward timed the
                # forward alone -- so a kernel with a correct but catastrophic
                # backward scored exactly like one with a fast backward.
                # Reported separately, never folded into the forward reward:
                # they are different numbers about different things.
                #
                # Only timed when the backward exists AND is correct: a missing
                # or wrong gradient has already cleared grad_sig, and timing a
                # wrong backward would pay reward for the wrong thing.
                if grad_sig is not None:
                    try:
                        _gb = self._general_baselines.get(self.scored_cases[0].name, {})
                        base_grad_raw = _gb.get("raw") or problem.baseline
                        # WHICH backward the grad_reward was measured against.
                        # The forward already records `baseline_impl` for exactly
                        # this reason -- a ratio against a fallback must never be
                        # read as a ratio against the production kernel. The
                        # backward had no such record, so a grad_reward could not
                        # be interpreted at all: differentiating recurrentgemma's
                        # Pallas scan and differentiating lax.associative_scan
                        # are different bars, and only one of them is the claim
                        # anyone cares about.
                        result["grad_baseline_impl"] = _gb.get("impl") or getattr(
                            problem, "baseline_impl", "?"
                        )
                        gb_fn = self.jax.jit(lambda *i: problem.grad_outputs(base_grad_raw, *i))
                        gc_fn = self.jax.jit(lambda *i: fns[grad_sig](*i))
                        gpairs = []
                        for i in range(self.timing_warmup + self.timing_pairs):
                            gi = problem.make_inputs(fold_in(fold_in(k_time, 7000 + i), 31), self.scored_cases[0])
                            block(gi)
                            gp, _, _ = timing_mod.counterbalanced_pair(
                                i, lambda: gb_fn(*gi), lambda: gc_fn(*gi), perf, block
                            )
                            if i >= self.timing_warmup:
                                gpairs.append(gp)
                        if gpairs:
                            gt = timing_mod.CaseTiming(case=f"grad:{self.scored_cases[0].name}", pairs=gpairs)
                            result["grad_score"] = gt.score
                            result["grad_reward"] = timing_mod.gate_reward(gt.score, self.noise_floor or 0.0)
                            result["grad_latencies"] = {
                                "ref_median_s": gt.ref_median_s, "cand_median_s": gt.cand_median_s
                            }
                    except Exception as e:  # noqa: BLE001 -- a backward TIMING failure must not fail a correct kernel
                        result["grad_timing_error"] = f"{type(e).__name__}: {str(e)[:160]}"

            # ---- 6. counterbalanced interleaved timing, fresh inputs per
            # ---- iteration, correctness verified on TIMED outputs
            case_timings, sol_fracs, baseline_impls, timer_used = [], {}, {}, {}
            skipped_tp, tp_widths = {}, {}
            for case in self.scored_cases + self.holdout_cases:
                if over_budget():
                    return budget_fail(f"timing ({case.name})")
                pairs = []
                check_iters = sorted({0, self.timing_pairs // 2, self.timing_pairs - 1})
                checks = {}
                cand_fn = fns[case_sig[case.name]]
                # GENERAL mode grades against the fastest honest implementation
                # AT THIS SHAPE, chosen by measurement at boot; otherwise the
                # single named production kernel.
                gb = self._general_baselines.get(case.name)
                base_fn = gb["fn"] if gb else self._baseline_fn
                if gb:
                    baseline_impls[case.name] = gb["impl"]

                # TENSOR PARALLEL. The candidate was EXPORTED at per-shard
                # shapes (build_signatures), so it can only be called through
                # shard_map -- executing it on full-shape inputs would be a
                # shape mismatch, and grading a TP case unsharded would report
                # a number that looks comparable and is not. A judge with fewer
                # devices than the declared width must skip and say so.
                tp_w = problem.tp_declared_width(case)
                tp_mesh = None
                if tp_w:
                    n_dev = len(self.jax.local_devices())
                    if n_dev < tp_w:
                        skipped_tp[case.name] = f"needs {tp_w} devices, judge has {n_dev}"
                        continue
                    specs = problem.tp_specs(case)
                    if specs is None:
                        skipped_tp[case.name] = "task declares no tp_specs"
                        continue
                    tp_in, tp_out = specs
                    tp_mesh = timing_mod.make_mesh(tp_w)
                    raw_base = gb["raw"] if gb and gb.get("raw") else problem.for_case(case).baseline
                    cand_fn = self.jax.jit(timing_mod.shard_mapped(cand_fn, tp_mesh, tp_in, tp_out))
                    base_fn = self.jax.jit(timing_mod.shard_mapped(raw_base, tp_mesh, tp_in, tp_out))
                    tp_widths[case.name] = tp_w

                    # SHARDED == UNSHARDED, checked before anything is timed
                    # (tokamax's api_sharding invariant; theirs is atol=0.0).
                    # Our TP specs are collective-free, so the sharded baseline
                    # must reproduce the unsharded result to calibrated
                    # tolerance -- if it does not, the specs compute a
                    # DIFFERENT function and any timing ratio would be a
                    # comparison between two different problems. Skip + record,
                    # never grade through a broken spec.
                    chk_in = problem.make_inputs(fold_in(k_time, 90_000 + hash_stable(case.name) % 1000), case)
                    block(chk_in)
                    un_out = raw_base(*chk_in)
                    block(un_out)
                    sh_out = base_fn(*timing_mod.shard_inputs(chk_in, tp_mesh, tp_in))
                    block(sh_out)
                    from pallas_arena.judge.problems.base import check_tolerance as _ct
                    from pallas_arena.judge.problems.base import error_stats as _es
                    _pc = problem.for_case(case)
                    _stats = _es(sh_out, un_out)
                    _tol = _pc.calibrated_tolerance(chk_in, _pc.reference(*chk_in))
                    result.setdefault("tp_agreement", {})[case.name] = {
                        "max": _stats.get("max"), "q99": _stats.get("q99"),
                        "bitwise": bool(_stats.get("max") == 0.0),
                    }
                    _ok, _why = _ct(_stats, _tol)
                    if not _ok:
                        skipped_tp[case.name] = f"sharded != unsharded baseline: {_why}"
                        del chk_in, un_out, sh_out
                        continue
                    del chk_in, un_out, sh_out
                for i in range(self.timing_warmup + self.timing_pairs):
                    inputs = problem.make_inputs(fold_in(fold_in(k_time, i), hash_stable(case.name)), case)
                    block(inputs)
                    if tp_mesh is not None:
                        inputs = timing_mod.shard_inputs(inputs, tp_mesh, problem.tp_specs(case)[0])
                    pair, _r, c_out = timing_mod.counterbalanced_pair(
                        i, lambda: base_fn(*inputs), lambda: cand_fn(*inputs), perf, block
                    )
                    # Prefer DEVICE time (tokamax's TPU default) when this judge
                    # can profile: the wallclock pair above already produced the
                    # outputs the correctness checks need, and the device pair
                    # replaces only the two latencies -- so a ratio is not
                    # squashed toward 1.0 by dispatch overhead common to both.
                    if self.device_timing:
                        dp = timing_mod.counterbalanced_pair_device(
                            i, base_fn, cand_fn, inputs, timing_mod.device_timer
                        )
                        if dp is not None:
                            pair = dp
                            timer_used["device"] = timer_used.get("device", 0) + 1
                        else:
                            timer_used["wallclock"] = timer_used.get("wallclock", 0) + 1
                    else:
                        timer_used["wallclock"] = timer_used.get("wallclock", 0) + 1
                    if i >= self.timing_warmup:
                        it = i - self.timing_warmup
                        pairs.append(pair)
                        if it in check_iters:
                            checks[it] = (inputs, c_out)
                for it, (inputs, c_out) in checks.items():
                    pcase = problem.for_case(case)
                    ref32 = pcase.reference(*inputs)
                    tol = pcase.calibrated_tolerance(inputs, ref32)
                    okay, why = check_tolerance(error_stats(c_out, ref32), tol)
                    if not okay:
                        return fail("timed_output_correctness", f"{case.name} timed iter {it}: {why}")
                ct = timing_mod.CaseTiming(
                    case=case.name, pairs=pairs, holdout=case.holdout,
                    blind=getattr(case, "blind", False),
                )
                case_timings.append(ct)
                bm = problem.bytes_moved(case)
                chip = (
                    "v6e"
                    if "v6e" in result["device_kind"].lower()
                    else "v5p" if "v5p" in result["device_kind"].lower() else self.platform
                )
                if problem.memory_bound and bm:
                    frac = timing_mod.speed_of_light_fraction(bm, ct.cand_median_s, chip)
                    if frac is not None:
                        sol_fracs[case.name] = frac
        except Exception as e:
            return fail("worker", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}")

        result["warm_chip_s"] = self.perf() - t_chip
        reward_frame = timing_mod.final_reward(
            case_timings, self.noise_floor or 0.0, general=getattr(problem, "general_mode", False)
        )
        try:
            mem = self.device.memory_stats()
            result["peak_hbm_bytes"] = int(mem.get("peak_bytes_in_use", 0))
        except Exception:
            result["peak_hbm_bytes"] = None
        result.update(
            passed=True,
            gate="all",
            **reward_frame,
            speed_of_light_fracs=sol_fracs,
            baseline_impl_per_case=baseline_impls,
            timer=timer_used,
            tp_widths=tp_widths,
            skipped_tp=skipped_tp,
            latencies={
                t.case: {"ref_median_s": t.ref_median_s, "cand_median_s": t.cand_median_s} for t in case_timings
            },
            export_child_s=child.get("export_s"),
            cache_hit=False,
        )
        observation.attach_observation(result)
        self._store(code, result)
        return result

    # --------------------------------------------------- compile-warm budget
    def _compile_warm(self, units, code: str, tag) -> None:
        """Run the compile-warm units under their OWN deadline.

        The per-grade wall budget cannot bound this: it is checked between
        phases, and a single `jax.jit(exported.call)(...)` first call is one
        un-cancellable C++ XLA compile. Phase 4 measured one candidate at
        569.5 s inside exactly that call while spending 3.1 s on the chip,
        with the 900 s wall budget looking on. Under lease-requeue that is a
        fleet-wide poison pill: the judge stalls, the lease expires, the item
        migrates to the next judge and wedges it too.

        Two layers, in order of preference:

          1. **Between units** (clean, no process loss): each exported
             signature and the gradient functional is its own compile, and
             the deadline is checked before and after each. Costs "one
             compile unit late", exactly the idiom the wall budget uses.
          2. **Watchdog thread** (backstop for an overrun INSIDE one unit,
             which is the shape of the 569.5 s case): XLA releases the GIL
             while it compiles, so this thread still runs. There is no way to
             cancel the compile, so the only bound left is process death —
             but the watchdog POSTS THE TERMINAL VERDICT FIRST, so the queue
             marks the item done (never requeued, and cached for every future
             judge) and only then hard-exits. The supervisor restarts the
             judge in ~1 min. Bounded cost on ONE judge, once, versus an
             unbounded stall on ALL of them.
        """
        budget = self.compile_budget_s
        if not budget:
            for _label, run in units:
                run()
            return

        t0 = self.perf()
        deadline = t0 + budget
        stop = threading.Event()
        state = {"unit": "?"}

        def watchdog():
            while not stop.wait(0.05):
                if self.perf() >= deadline:
                    self._on_compile_timeout(code, tag, state["unit"], self.perf() - t0)
                    return

        wd = threading.Thread(target=watchdog, daemon=True, name="compile-budget-watchdog")
        wd.start()
        try:
            for label, run in units:
                state["unit"] = label
                self._check_compile_deadline(t0, deadline, label)
                run()
                self._check_compile_deadline(t0, deadline, label)
        finally:
            stop.set()

    def _check_compile_deadline(self, t0, deadline, label) -> None:
        if self.perf() >= deadline:
            raise CompileBudgetExceeded(
                f"candidate compile exceeded the {self.compile_budget_s:.0f}s compile budget "
                f"({self.perf() - t0:.1f}s at unit {label!r})"
            )

    def _on_compile_timeout(self, code: str, tag, unit, elapsed: float) -> None:
        self.compile_timeouts += 1
        why = (
            f"candidate compile exceeded the {self.compile_budget_s:.0f}s compile budget "
            f"({elapsed:.1f}s inside a single un-cancellable XLA compile at unit {unit!r}); "
            f"judge restarting"
        )
        result = {
            "ok": True,
            "problem": self.problem_name,
            "problem_version": self.problem.version,
            "worker_id": self.worker_id,
            "backend": self.platform,
            "tag": tag,
            "passed": False,
            "gate": "compile_budget",
            "terminal": True,
            "violations": [why],
            "reward": 0.0,
            "compile_budget_s": self.compile_budget_s,
            "candidate_compile_s": elapsed,
            "judge_restarted": True,
        }
        observation.attach_observation(result)
        self._store(code, result)
        print(f"[worker {self.worker_id}] COMPILE BUDGET: {why}", flush=True)
        poster = self.terminal_poster
        if poster is None:
            # In-process use (tests, or a judge with no queue): nothing owns
            # the process lifecycle here, so let the between-unit check
            # deliver the same verdict when the compile finally returns.
            return
        # The poster owns BOTH recording the verdict and ending the process —
        # process death is a lifecycle decision, and keeping it out of the
        # grading path is what makes this mechanism testable at all.
        try:
            poster(result)
        except Exception:
            pass

    def _contract_tag(self) -> str:
        """Identity of the GRADING CONTRACT, not of the code.

        Cached verdicts are only interchangeable when the contract that
        produced them matches. The backward is the part that moves: under
        ``bwd_gates`` a non-differentiable kernel is zeroed, and without it the
        same kernel keeps its forward reward. Sharing one cache across both
        would replay the wrong verdict and make a contract change look like a
        no-op.
        """
        return f"bwdgate={int(self.problem.bwd_gates)}" if self.problem.has_bwd else ""

    def _store(self, code: str, result: dict) -> None:
        if self.cache is None:
            return
        try:
            key = grader.cache_key(
                self.problem_name, self.problem.version, "worker_full", code, self.smoke,
                contract=self._contract_tag(),
            )
            self.cache.put(key, {k: v for k, v in result.items() if k != "cache_hit"})
        except Exception:
            pass


def hash_stable(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:4], 16)


# ------------------------------------------------------------- mock grading
def mock_grade(payload: dict, grade_s: float, worker_id: str) -> dict:
    """Deterministic pseudo-grade for queue/chaos simulation: same code ->
    same reward, wall time controlled by --mock-grade-s.

    When the payload carries a phase-5 corpus tag (`<variant>#s0007`) the
    mock also reproduces that variant's EXPECTED verdict, gate and rough
    cost, so the fleet driver's full acceptance logic — verdicts, cross-judge
    agreement, regrade spread, throughput accounting — is exercised on CPU
    before a single chip is provisioned. Scores carry a small deterministic
    per-worker offset so cross-judge agreement is a real check, not a
    tautology."""
    tag = payload.get("tag") or ""
    base = tag.split("#", 1)[0]
    code = payload.get("code", "")
    h = hashlib.sha256(code.encode()).hexdigest()

    profile = MOCK_VARIANT_PROFILES.get(base)
    if profile is None:
        time.sleep(grade_s)
        passed = "MOCK_FAIL" not in code
        return {
            "ok": True,
            "passed": passed,
            "gate": "all" if passed else "correctness",
            "reward": (int(h[:6], 16) % 1000) / 1000.0 if passed else 0.0,
            "score": (int(h[:6], 16) % 1000) / 1000.0 if passed else None,
            "mock": True,
            "worker_id": worker_id,
        }

    passed, gate, base_score, cost = profile
    time.sleep(min(grade_s * cost, 30.0))
    res = {
        "ok": True,
        "passed": passed,
        "gate": gate,
        "mock": True,
        "worker_id": worker_id,
        "backend": "tpu",
        "device_kind": "mock v6e",
        "tag": tag,
        "export_s": 0.8 * cost,
        "load_s": 0.004,
        "candidate_compile_s": 3.0 * cost,
        "warm_chip_s": 1.9 * cost if passed else None,
        "peak_hbm_bytes": 18 * 2**30 if passed else None,
        "worker_boot": {"ref_vs_ref_scores": {"mock": 1.0 + (int(h[6:8], 16) - 128) / 128 * 0.004}},
    }
    if passed:
        # +-0.6% spread within a judge, +-0.4% offset between judges: inside
        # the +-3% acceptance band, wide enough that a broken comparison shows.
        jitter = (int(h[:6], 16) % 1201 - 600) / 1e5
        wjit = (int(hashlib.sha256(worker_id.encode()).hexdigest()[:4], 16) % 801 - 400) / 1e5
        res["score"] = base_score * (1.0 + jitter + wjit)
        res["reward"] = 1.0 if 0.99 < res["score"] < 1.01 else res["score"]
    else:
        res["reward"] = 0.0
        res["score"] = None
        if gate == "compile_budget":
            res["terminal"] = True
            res["compile_budget_s"] = DEFAULT_COMPILE_BUDGET_S
    return res


# variant -> (passed, gate, base_score, relative cost). Costs are the phase-4
# measured item_wall ratios (AST reject 0.74 s .. full grade 5.9 s .. the
# compile bomb pinned at its budget).
MOCK_VARIANT_PROFILES: dict[str, tuple] = {
    "honest-xla": (True, "all", 1.002, 1.0),
    "honest-xla-b": (True, "all", 1.001, 1.0),
    "honest-xla-f32out": (True, "all", 0.783, 1.0),
    "unjitted-robust": (True, "all", 0.828, 1.0),
    "pallas-br16": (True, "all", 0.474, 1.0),
    "pallas-br256": (True, "all", 0.826, 1.0),
    "pallas-br512": (True, "all", 0.826, 1.0),
    "split-personality": (True, "all", 0.778, 1.0),
    "nondeterministic": (True, "all", 0.568, 10.0),
    "timer-tamperer": (True, "all", 0.062, 3.0),
    "wrong-eps": (False, "correctness", 0.0, 0.7),
    "cached-output": (False, "aot_export", 0.0, 0.2),
    "constant-output": (False, "correctness", 0.0, 0.7),
    "aliased-reference": (False, "ast", 0.0, 0.05),
    "obfuscated-import": (False, "poison_stub", 0.0, 0.15),
    "seed-reader": (False, "correctness", 0.0, 0.7),
    "memoizer": (False, "aot_export", 0.0, 0.2),
    "wrong-grad": (False, "gradient", 0.0, 0.8),
    "no-kernel": (False, "exec", 0.0, 0.05),
    "compile-bomb": (False, "compile_budget", 0.0, 4.0),
}


# ------------------------------------------------------------ queue polling
def _http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status == 204:
            return None
        return json.loads(resp.read())


def poll_loop(
    queue_url: str,
    grade_fn,
    *,
    worker_id: str,
    poll_s: float = 1.0,
    max_items: int | None = None,
    heartbeat_frac: float = 3.0,
    attach_terminal_poster=None,
) -> int:
    """GET /work -> grade (heartbeating) -> POST /result, forever (or until
    max_items). Every network error is survivable: sleep and re-poll.

    `attach_terminal_poster` is handed a one-shot poster for the item
    currently leased; the compile-budget watchdog uses it to record a
    terminal verdict from a thread before the judge hard-exits."""
    done = 0
    base = queue_url.rstrip("/")
    while max_items is None or done < max_items:
        try:
            item = _http_json(f"{base}/work?worker_id={worker_id}")
        except Exception:
            time.sleep(poll_s)
            continue
        if item is None:
            time.sleep(poll_s)
            continue

        lease_id = item["lease_id"]
        t_item = time.time()
        tag = (item.get("payload") or {}).get("tag")
        print(f"[worker {worker_id}] lease {item['work_id']} tag={tag} attempt={item.get('attempt')}", flush=True)
        stop = threading.Event()
        interval = max(item.get("lease_timeout_s", 60.0) / heartbeat_frac, 0.5)

        def beat():
            while not stop.wait(interval):
                try:
                    _http_json(f"{base}/heartbeat", {"lease_id": lease_id})
                except Exception:
                    pass  # queue down or lease expired: result post decides

        hb = threading.Thread(target=beat, daemon=True)
        hb.start()
        if attach_terminal_poster is not None:

            def poster(res, _lease=lease_id, _wid=item["work_id"], _t0=t_item):
                """Record a TERMINAL verdict for the item in flight and end
                this judge. Only the compile-budget watchdog calls it, from
                a thread, while the main thread is stuck inside one
                un-cancellable XLA compile — so the result MUST be posted
                before the process dies, otherwise the lease expires and the
                poison candidate migrates to the next judge, which is the
                exact failure this whole mechanism exists to prevent."""
                import os

                res["item_wall_s"] = time.time() - _t0
                try:
                    _http_json(f"{base}/result", {"lease_id": _lease, "work_id": _wid, "result": res, "terminal": True})
                    print(f"[worker {worker_id}] terminal verdict posted for {_wid}; exiting", flush=True)
                except Exception as e:
                    print(f"[worker {worker_id}] terminal post FAILED ({e!r}); exiting anyway", flush=True)
                os._exit(17)

            attach_terminal_poster(poster)
        try:
            result = grade_fn(item["payload"])
        except Exception as e:
            result = {"ok": False, "passed": False, "gate": "worker", "reward": 0.0, "violations": [repr(e)]}
        finally:
            stop.set()
            hb.join(timeout=2.0)
        # end-to-end lease wall: the number the fleet throughput table is
        # actually built from (export child + artifact compile + chip time),
        # as opposed to warm_chip_s which is the steady-state chip slice only.
        if isinstance(result, dict):
            result["item_wall_s"] = time.time() - t_item

        # ---- judge fault: this chip is gone, and NOTHING about that is the
        # candidate's fault. Phase 5 had a candidate halt the TPU core at
        # Mosaic compile time; the worker stayed alive and reported the next 35
        # items as ordinary zero-reward verdicts in ~1.1s each. Never post the
        # verdict (it would be a lie, and the queue would mark the item done);
        # instead let the lease lapse so the item requeues onto a live judge,
        # and exit so the supervisor rebuilds this one on a fresh chip.
        if isinstance(result, dict) and result.get("judge_fault"):
            stop.set()
            print(
                f"[worker {worker_id}] JUDGE FAULT on {item['work_id']} "
                f"(gate={result.get('gate')}): {'; '.join(result.get('violations') or [])[:200]}\n"
                f"[worker {worker_id}] chip presumed dead; NOT posting a verdict. "
                f"Lease {lease_id} will lapse and requeue. Exiting for a clean restart.",
                flush=True,
            )
            os._exit(18)
        print(
            f"[worker {worker_id}] done {item['work_id']} tag={tag} wall={time.time() - t_item:.1f}s "
            f"passed={result.get('passed')} gate={result.get('gate')} "
            f"export={result.get('export_s')} load={result.get('load_s')} "
            f"compile={result.get('candidate_compile_s')} warm={result.get('warm_chip_s')}",
            flush=True,
        )
        try:
            _http_json(
                f"{base}/result",
                {
                    "lease_id": lease_id,
                    "work_id": item["work_id"],
                    "result": result,
                    "terminal": bool(isinstance(result, dict) and result.get("terminal")),
                },
            )
        except Exception:
            pass  # lease will expire; someone else regrades (idempotent)
        if attach_terminal_poster is not None:
            attach_terminal_poster(None)
        done += 1
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--problem",
        required=True,
        help="problem name, or a ';'-separated list of 'name[:case1,case2,...]' to serve "
        "several problems from ONE chip (dispatch is on the payload's `problem` field). "
        "The arena's rule that a judge only answers for the problem it was launched with "
        "is unchanged -- the launch flag just names more than one.",
    )
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--sim-mode", choices=["real", "mock"], default="real")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", default=None, help="comma-separated case names")
    ap.add_argument(
        "--no-adversarial",
        action="store_true",
        help="skip the adversarial vector library. Its shapes are exported alongside the "
        "scored ones, so on a non-default case set it would require candidates to trace at "
        "shapes no prompt declared. Anti-cheat relevance is a trained policy gaming the "
        "judge; a base-model probe has no policy.",
    )
    ap.add_argument("--adversarial-case", default=None, help="shape case the adversarial library is built on")
    ap.add_argument("--timing-pairs", type=int, default=20)
    ap.add_argument("--grade-budget-s", type=float, default=900.0, help="0 disables the per-grade wall budget")
    ap.add_argument(
        "--compile-budget-s",
        type=float,
        default=DEFAULT_COMPILE_BUDGET_S,
        help="deadline for candidate compile-warm alone; 0 disables (NOT recommended on a fleet)",
    )
    ap.add_argument("--mock-grade-s", type=float, default=1.0)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--boot-report", default=None, help="write boot report JSON here")
    args = ap.parse_args()

    worker_id = args.worker_id or f"worker-{secrets.token_hex(3)}"
    attach = None

    if args.sim_mode == "mock":
        # The mock honours --cache through the SAME hash->reward path the real
        # worker uses. Without it a simulated fleet cannot exercise the
        # byte-identical-repeat property at all: every duplicate would get a
        # fresh mock grade with fresh per-judge jitter, and the driver's cache
        # check would be testing the mock rather than the cache.
        mock_cache = None
        if args.cache:
            from pallas_arena.judge.cache import RewardCache

            mock_cache = RewardCache(args.cache)

        def grade_fn(payload):
            key = grader.cache_key(args.problem, "mock", "mock_full", payload.get("code", ""), args.smoke)
            if mock_cache is not None:
                hit = mock_cache.get(key)
                if hit is not None:
                    hit = dict(hit)
                    hit["cache_hit"] = True
                    return hit
            res = mock_grade(payload, args.mock_grade_s, worker_id)
            res["cache_hit"] = False
            if mock_cache is not None:
                mock_cache.put(key, {k: v for k, v in res.items() if k != "cache_hit"})
            return res

    else:
        cache = None
        if args.cache:
            from pallas_arena.judge.cache import RewardCache

            cache = RewardCache(args.cache)
        workers: dict[str, PersistentWorker] = {}
        reports: dict[str, dict] = {}
        for spec in args.problem.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            name, _, case_str = spec.partition(":")
            case_list = (case_str or args.cases or "").split(",") if (case_str or args.cases) else None
            w = PersistentWorker(
                name,
                smoke=args.smoke,
                cases=[c for c in case_list if c] if case_list else None,
                use_adversarial=not args.no_adversarial,
                adversarial_case=args.adversarial_case,
                timing_pairs=args.timing_pairs,
                grade_budget_s=args.grade_budget_s or None,
                compile_budget_s=args.compile_budget_s or None,
                cache=cache,
                worker_id=worker_id,
            )
            reports[name] = w.boot()
            print(f"[worker {worker_id}] boot {name}: {json.dumps(reports[name], default=str)}", flush=True)
            if not reports[name].get("ok"):
                # One dead problem must not silently take the others with it,
                # but it must be loud: record it and keep serving the rest.
                print(f"[worker {worker_id}] BOOT FAILED for {name}; it will not be served", flush=True)
                continue
            workers[name] = w
        if args.boot_report:
            Path(args.boot_report).write_text(json.dumps(reports, indent=1, default=str))
        if not workers:
            raise SystemExit(1)
        default_name = next(iter(workers))
        # The terminal poster belongs to whichever worker is mid-grade.
        holder = {"w": workers[default_name]}
        attach = lambda poster: setattr(holder["w"], "terminal_poster", poster)  # noqa: E731

        def grade_fn(payload):
            name = payload.get("problem") or default_name
            w = workers.get(name)
            if w is None:
                return {
                    "ok": True,
                    "passed": False,
                    "gate": "harness",
                    "reward": 0.0,
                    "problem": name,
                    "violations": [f"this judge does not serve problem {name!r}"],
                }
            prev = holder["w"]
            holder["w"] = w
            w.terminal_poster = prev.terminal_poster
            return w.grade_code(
                payload["code"],
                enforce_pallas=payload.get("enforce_pallas"),
                tag=payload.get("tag"),
            )

    n = poll_loop(
        args.queue,
        grade_fn,
        worker_id=worker_id,
        poll_s=args.poll_s,
        max_items=args.max_items,
        attach_terminal_poster=attach,
    )
    print(f"[worker {worker_id}] done after {n} items", flush=True)


if __name__ == "__main__":
    main()
