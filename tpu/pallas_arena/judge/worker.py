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
import secrets
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from pallas_arena.judge import grader
from pallas_arena.judge import timing as timing_mod


class PersistentWorker:
    def __init__(
        self,
        problem_name: str,
        *,
        smoke: bool = False,
        cases: list[str] | None = None,
        timing_pairs: int = 20,
        timing_warmup: int = 3,
        determinism_runs: int = 5,
        correctness_seeds: int = 2,
        export_timeout_s: float = 240.0,
        export_rlimit_gb: float | None = None,
        cache=None,
        worker_id: str = "worker-0",
    ):
        import jax

        from pallas_arena.judge.problems import get_problem

        self.jax = jax
        self.problem = get_problem(problem_name)
        self.problem_name = problem_name
        self.smoke = smoke
        self.timing_pairs = timing_pairs
        self.timing_warmup = timing_warmup
        self.determinism_runs = determinism_runs
        self.correctness_seeds = correctness_seeds
        self.export_timeout_s = export_timeout_s
        self.export_rlimit_gb = export_rlimit_gb
        self.cache = cache
        self.worker_id = worker_id

        self.perf = time.perf_counter
        self.block = jax.block_until_ready
        self.device = jax.local_devices()[0]
        self.platform = jax.default_backend()

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
            "boot_s": self.perf() - t0,
        }
        return self.boot_report

    # ------------------------------------------------------------ signatures
    def _signatures(self):
        """Deduped input signatures the child must export: shape cases +
        adversarial shapes + the gradient functional. Shapes are public."""
        jax = self.jax
        sigs, seen = [], {}

        def sig_of(abstract, kind, label):
            args = [{"shape": list(a.shape), "dtype": str(a.dtype)} for a in abstract]
            key = (kind, json.dumps(args))
            if key in seen:
                return seen[key]
            name = f"{kind}{len(sigs):02d}"
            sigs.append({"name": name, "kind": kind, "args": args, "label": label})
            seen[key] = name
            return name

        case_sig = {}
        for case in self.scored_cases + self.holdout_cases:
            case_sig[case.name] = sig_of(self.problem.abstract_inputs(case), "fwd", case.name)
        adv_sig = {}
        for i, adv in enumerate(self.problem.adversarial_cases()):
            abstract = jax.eval_shape(adv.make_inputs, jax.ShapeDtypeStruct((2,), "uint32"))
            adv_sig[adv.name] = sig_of(abstract, "fwd", f"adv:{adv.name}")
        grad_sig = None
        if self.problem.has_bwd and self.scored_cases:
            grad_sig = sig_of(self.problem.abstract_inputs(self.scored_cases[0]), "grad", "grad")
        return sigs, case_sig, adv_sig, grad_sig

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
            key = grader.cache_key(self.problem_name, problem.version, "worker_full", code, self.smoke)
            hit = self.cache.get(key)
            if hit is not None:
                hit = dict(hit)
                hit["cache_hit"] = True
                return hit

        def fail(gate, why):
            result.update(passed=False, gate=gate, violations=[why], reward=0.0)
            self._store(code, result)
            return result

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

        t_chip = self.perf()
        try:
            # ---- 3. correctness on fresh hidden seeds + adversarial vectors
            fixtures = []
            for case in self.scored_cases:
                for g in range(self.correctness_seeds):
                    inputs = problem.make_inputs(fold_in(fold_in(k_corr, g), hash_stable(case.name)), case)
                    block(inputs)
                    fixtures.append((f"{case.name}#seed{g}", case_sig[case.name], inputs))
            for i, adv in enumerate(problem.adversarial_cases()):
                inputs = adv.make_inputs(fold_in(k_adv, i))
                block(inputs)
                fixtures.append((f"adv:{adv.name}", adv_sig[adv.name], inputs))

            from pallas_arena.judge.problems.base import check_tolerance, error_stats

            for label, sig_name, inputs in fixtures:
                ref32 = problem.reference(*inputs)
                tol = problem.calibrated_tolerance(inputs, ref32)
                try:
                    out = fns[sig_name](*inputs)
                    block(out)
                except Exception as e:
                    return fail("correctness", f"{label}: runtime error {type(e).__name__}: {e}")
                okay, why = check_tolerance(error_stats(out, ref32), tol)
                if not okay:
                    return fail("correctness", f"{label}: {why}")

            # ---- 4. determinism: N bitwise-identical runs
            import numpy as np

            _det_label, det_sig, det_inputs = fixtures[0]
            outs = []
            for _ in range(self.determinism_runs):
                o = fns[det_sig](*det_inputs)
                block(o)
                leaves = o if isinstance(o, (tuple, list)) else (o,)
                outs.append(b"".join(np.asarray(x).tobytes() for x in leaves))
            if len(set(outs)) != 1:
                return fail("determinism", f"outputs not bitwise identical across {self.determinism_runs} runs")

            # ---- 5. gradient contract via the exported grad artifact
            if grad_sig is not None:
                from pallas_arena.judge.problems.base import tolerance_from_reference

                g_inputs = problem.make_inputs(fold_in(k_corr, 999), self.scored_cases[0])
                block(g_inputs)
                ref_g = problem.grad_outputs(lambda *i: problem.reference(*i), *g_inputs)
                cal_g = problem.grad_outputs(lambda *i: problem.reference_bf16(*i), *g_inputs)
                g_tol = tolerance_from_reference(ref_g, cal_g)
                try:
                    cand_g = fns[grad_sig](*g_inputs)
                    block(cand_g)
                except Exception as e:
                    return fail("gradient", f"grad failed: {type(e).__name__}: {e}")
                okay, why = check_tolerance(error_stats(cand_g, ref_g), g_tol)
                if not okay:
                    return fail("gradient", why)

            # ---- 6. counterbalanced interleaved timing, fresh inputs per
            # ---- iteration, correctness verified on TIMED outputs
            case_timings, sol_fracs = [], {}
            for case in self.scored_cases + self.holdout_cases:
                pairs = []
                check_iters = sorted({0, self.timing_pairs // 2, self.timing_pairs - 1})
                checks = {}
                cand_fn = fns[case_sig[case.name]]
                for i in range(self.timing_warmup + self.timing_pairs):
                    inputs = problem.make_inputs(fold_in(fold_in(k_time, i), hash_stable(case.name)), case)
                    block(inputs)
                    pair, _r, c_out = timing_mod.counterbalanced_pair(
                        i, lambda: self._baseline_fn(*inputs), lambda: cand_fn(*inputs), perf, block
                    )
                    if i >= self.timing_warmup:
                        it = i - self.timing_warmup
                        pairs.append(pair)
                        if it in check_iters:
                            checks[it] = (inputs, c_out)
                for it, (inputs, c_out) in checks.items():
                    ref32 = problem.reference(*inputs)
                    tol = problem.calibrated_tolerance(inputs, ref32)
                    okay, why = check_tolerance(error_stats(c_out, ref32), tol)
                    if not okay:
                        return fail("timed_output_correctness", f"{case.name} timed iter {it}: {why}")
                ct = timing_mod.CaseTiming(case=case.name, pairs=pairs, holdout=case.holdout)
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
        reward_frame = timing_mod.final_reward(case_timings, self.noise_floor or 0.0)
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
            latencies={
                t.case: {"ref_median_s": t.ref_median_s, "cand_median_s": t.cand_median_s} for t in case_timings
            },
            export_child_s=child.get("export_s"),
            cache_hit=False,
        )
        self._store(code, result)
        return result

    def _store(self, code: str, result: dict) -> None:
        if self.cache is None:
            return
        try:
            key = grader.cache_key(self.problem_name, self.problem.version, "worker_full", code, self.smoke)
            self.cache.put(key, {k: v for k, v in result.items() if k != "cache_hit"})
        except Exception:
            pass


def hash_stable(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:4], 16)


# ------------------------------------------------------------- mock grading
def mock_grade(payload: dict, grade_s: float, worker_id: str) -> dict:
    """Deterministic pseudo-grade for queue/chaos simulation: same code ->
    same reward, wall time controlled by --mock-grade-s."""
    time.sleep(grade_s)
    code = payload.get("code", "")
    h = hashlib.sha256(code.encode()).hexdigest()
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
) -> int:
    """GET /work -> grade (heartbeating) -> POST /result, forever (or until
    max_items). Every network error is survivable: sleep and re-poll."""
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
        try:
            result = grade_fn(item["payload"])
        except Exception as e:
            result = {"ok": False, "passed": False, "gate": "worker", "reward": 0.0, "violations": [repr(e)]}
        finally:
            stop.set()
            hb.join(timeout=2.0)
        try:
            _http_json(f"{base}/result", {"lease_id": lease_id, "work_id": item["work_id"], "result": result})
        except Exception:
            pass  # lease will expire; someone else regrades (idempotent)
        done += 1
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--sim-mode", choices=["real", "mock"], default="real")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", default=None, help="comma-separated case names")
    ap.add_argument("--timing-pairs", type=int, default=20)
    ap.add_argument("--mock-grade-s", type=float, default=1.0)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--boot-report", default=None, help="write boot report JSON here")
    args = ap.parse_args()

    worker_id = args.worker_id or f"worker-{secrets.token_hex(3)}"

    if args.sim_mode == "mock":
        grade_fn = lambda payload: mock_grade(payload, args.mock_grade_s, worker_id)  # noqa: E731
    else:
        cache = None
        if args.cache:
            from pallas_arena.judge.cache import RewardCache

            cache = RewardCache(args.cache)
        w = PersistentWorker(
            args.problem,
            smoke=args.smoke,
            cases=args.cases.split(",") if args.cases else None,
            timing_pairs=args.timing_pairs,
            cache=cache,
            worker_id=worker_id,
        )
        report = w.boot()
        print(f"[worker {worker_id}] boot: {json.dumps(report, default=str)}", flush=True)
        if args.boot_report:
            Path(args.boot_report).write_text(json.dumps(report, indent=1, default=str))
        if not report.get("ok"):
            raise SystemExit(1)

        def grade_fn(payload):
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
    )
    print(f"[worker {worker_id}] done after {n} items", flush=True)


if __name__ == "__main__":
    main()
