"""Phase-4 driver: submits the RMSNorm acceptance battery through the
pull-queue and asserts that EVERY phase-2 red is now green on silicon:

  [inv] worker-polls-over-wire   a real v6e worker leased from this queue
  [inv] ref-vs-ref-1.00±2%       counterbalanced boot floor (phase-2 red 1)
  [inv] warm-cost                warm chip-time per candidate ~4s target
                                 (phase-2 red 2: 112s cold forks)
  [inv] recalibrated-goldens     pallas br16/br256/br512 + unjitted-robust
                                 + f32-out honest all PASS (phase-2 red 3)
  [inv] verdicts                 whole cheater battery at the right gates
                                 (determinism N>=5 enforced inside grades)
  [inv] regrade-±3%              honest-xla vs honest-xla-b
  [inv] peak-hbm                 reported per grade

Runs on the compute node next to the queue; the worker reaches the queue
through the reverse ssh tunnel the orchestrator holds open.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ARENA_IMPORT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARENA_IMPORT_ROOT))

from pallas_arena.judge.queue import ArenaQueueClient  # noqa: E402
from pallas_arena.phase2.variants import (  # noqa: E402
    WORKER_EXPECT_FAIL_GATE,
    WORKER_EXPECT_PASS,
    WORKER_EXPECT_SLOW,
    shakedown_variants,
)

WARM_TARGET_S = 4.0
WARM_HARD_CAP_S = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="http://127.0.0.1:8770")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker-wait-s", type=float, default=1800.0)
    ap.add_argument("--grade-wait-s", type=float, default=3600.0)
    ap.add_argument("--chaos-qr", default=None, help="QR name; enables the mid-lease worker-kill test")
    ap.add_argument("--chaos-zone", default="us-east5-b")
    ap.add_argument("--chaos-project", default="vision-mix")
    ap.add_argument("--chaos-wait-s", type=float, default=1200.0)
    args = ap.parse_args()

    client = ArenaQueueClient(args.queue)
    results: dict = {"invariants": {}, "grades": {}, "meta": {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    hard_fail: list[str] = []

    def invariant(name: str, ok: bool, detail):
        results["invariants"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail, default=str)[:400]}", flush=True)
        if not ok:
            hard_fail.append(name)

    def measure(name: str, detail):
        """Recorded + printed, never a gate: numbers the phase report quotes
        but that no acceptance item is defined against."""
        results.setdefault("measurements", {})[name] = detail
        print(f"[measure] {name}: {json.dumps(detail, default=str)[:400]}", flush=True)

    # ---- wait for the (tunneled) worker to start polling
    deadline = time.time() + args.worker_wait_s
    worker_seen = None
    while time.time() < deadline:
        try:
            st = client.status()
            if st.get("workers_seen"):
                worker_seen = list(st["workers_seen"])
                break
        except Exception:
            pass
        time.sleep(5)
    invariant("worker-polls-over-wire", worker_seen is not None, worker_seen)
    if worker_seen is None:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
        return 1

    # ---- submit the whole battery FIFO
    variants = shakedown_variants()
    ids = {name: client.submit("rmsnorm", code, tag=name) for name, code in variants.items()}
    print(f"[driver] submitted {len(ids)} candidates", flush=True)
    got = client.wait(list(ids.values()), timeout_s=args.grade_wait_s, poll_s=2.0)
    id2name = {wid: name for name, wid in ids.items()}
    grades = {id2name[wid]: r for wid, r in got.items()}
    results["grades"] = grades
    invariant("all-graded", len(grades) == len(ids), {"graded": len(grades), "submitted": len(ids)})

    # ---- verdicts
    for name in variants:
        g = grades.get(name)
        if g is None:
            invariant(f"verdict:{name}", False, "no result")
            continue
        want_pass = name in WORKER_EXPECT_PASS
        okay = bool(g.get("passed")) == want_pass
        if okay and not want_pass:
            allowed = WORKER_EXPECT_FAIL_GATE.get(name)
            if allowed and g.get("gate") not in allowed:
                okay = False
        invariant(
            f"verdict:{name}",
            okay,
            {"expected_pass": want_pass, "gate": g.get("gate"), "score": g.get("score"), "reward": g.get("reward")},
        )
        if name in WORKER_EXPECT_SLOW and g.get("passed"):
            invariant(f"slowdown-measured:{name}", (g.get("score") or 9) < 0.95, {"score": g.get("score")})

    passing = {n: g for n, g in grades.items() if g and g.get("passed")}

    # ---- phase-2 red 1: ref-vs-ref 1.00±2% (counterbalanced)
    boot = next((g.get("worker_boot") for g in grades.values() if g and g.get("worker_boot")), None)
    if boot and boot.get("ref_vs_ref_scores"):
        bad = {c: s for c, s in boot["ref_vs_ref_scores"].items() if abs(s - 1.0) > 0.02}
        invariant("ref-vs-ref-1.00±2%", not bad, {"scores": boot["ref_vs_ref_scores"], "violations": bad})
        results["boot"] = boot
    else:
        invariant("ref-vs-ref-1.00±2%", False, "no boot report in results")

    # ---- phase-2 red 2: warm per-candidate chip cost
    warm = [g["warm_chip_s"] for g in passing.values() if g.get("warm_chip_s")]
    if warm:
        med = sorted(warm)[len(warm) // 2]
        invariant(
            "warm-cost",
            med <= WARM_HARD_CAP_S,
            {"median_s": med, "target_s": WARM_TARGET_S, "hard_cap_s": WARM_HARD_CAP_S, "all": warm},
        )
    else:
        invariant("warm-cost", False, "no warm_chip_s measurements")

    # end-to-end lease wall + its breakdown: what a judge lane actually costs
    # per candidate (the fleet-throughput number), reported not gated.
    def _stat(key, only_passing=True):
        src = passing.values() if only_passing else [g for g in grades.values() if g]
        vals = sorted(v for v in (g.get(key) for g in src) if isinstance(v, (int, float)))
        if not vals:
            return None
        return {"n": len(vals), "median": vals[len(vals) // 2], "min": vals[0], "max": vals[-1]}

    measure(
        "per-candidate-cost-breakdown-s",
        {
            "item_wall": _stat("item_wall_s", only_passing=False),
            "item_wall_passing": _stat("item_wall_s"),
            "export_child": _stat("export_s", only_passing=False),
            "artifact_load": _stat("load_s"),
            "candidate_compile": _stat("candidate_compile_s"),
            "warm_chip": _stat("warm_chip_s"),
        },
    )

    # ---- phase-2 red 3: recalibrated goldens
    goldens = ["pallas-br16", "pallas-br256", "pallas-br512", "unjitted-robust", "honest-xla-f32out"]
    bad_g = [n for n in goldens if not (grades.get(n) or {}).get("passed")]
    invariant("recalibrated-goldens", not bad_g, {"failed": bad_g})

    # ---- regrade stability ±3%
    a = (grades.get("honest-xla") or {}).get("score")
    b = (grades.get("honest-xla-b") or {}).get("score")
    if a and b:
        invariant("regrade-±3%", abs(a / b - 1.0) < 0.03, {"a": a, "b": b, "ratio": a / b})
    else:
        invariant("regrade-±3%", False, {"a": a, "b": b})

    # honest bf16-out == baseline: the tie must gate to EXACTLY 1.0
    if a:
        invariant("honest-ties-to-1.0", (grades["honest-xla"].get("reward") == 1.0) and 0.9 < a < 1.1, {"score": a})

    # ---- peak HBM + on-tpu
    hbm = [g.get("peak_hbm_bytes") for g in passing.values() if g.get("peak_hbm_bytes")]
    invariant("peak-hbm-reported", bool(hbm), {"max_gb": max(hbm) / 2**30 if hbm else None})
    backends = {g.get("backend") for g in grades.values() if g and g.get("backend")}
    invariant("children-on-tpu", backends == {"tpu"}, sorted(backends))

    # ---- chaos: SIGKILL the worker MID-LEASE over the wire. Phase 3 proved
    # requeue-on-lease-expiry in local simulation; this is the same property
    # against a real preempted-judge stand-in (the orchestrator's ssh restart
    # loop brings the worker back, exactly like the fleet keeper would).
    if args.chaos_qr:
        import subprocess

        # Payload = the costliest RELIABLY-COMPLETING variant (phase 4
        # measured `nondeterministic` at ~14 s export + ~46 s compile), so
        # the ssh round-trip (~10-30 s) lands while the grade is in flight.
        # The test only counts if we can PROVE the kill was mid-lease, so we
        # re-read the item straight after the kill and retry if the worker
        # had already posted its result.
        chaos_detail: dict = {}
        chaos_ok = False
        for attempt in range(3):
            wid = client.submit("rmsnorm", variants["nondeterministic"], tag=f"chaos-requeue-{attempt}")
            leased, t0 = False, time.time()
            while time.time() - t0 < 600:
                try:
                    if client._get(f"/result/{wid}").get("state") == "leased":
                        leased = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            if not leased:
                chaos_detail = {"attempt": attempt, "why": "item never leased within 600 s"}
                continue
            # bracket the pattern so pkill never matches its own command line
            kill = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "ssh",
                    args.chaos_qr,
                    f"--zone={args.chaos_zone}",
                    f"--project={args.chaos_project}",
                    "--worker=0",
                    "--command=pkill -9 -f '[p]allas_arena.judge.worker'",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            post = {}
            try:
                post = client._get(f"/result/{wid}")
            except Exception:
                pass
            midlease = not post.get("done")
            print(
                f"[driver] chaos attempt={attempt} kill rc={kill.returncode} "
                f"midlease={midlease} state={post.get('state')}",
                flush=True,
            )
            if not midlease:
                chaos_detail = {"attempt": attempt, "why": "grade completed before the kill landed; retrying"}
                continue
            got_c = client.wait([wid], timeout_s=args.chaos_wait_s, poll_s=3.0)
            raw = {}
            try:
                raw = client._get(f"/result/{wid}")
            except Exception:
                pass
            res_c = got_c.get(wid) or {}
            chaos_ok = bool(res_c.get("passed") and raw.get("attempts", 0) >= 2)
            chaos_detail = {
                "attempt": attempt,
                "killed_mid_lease": True,
                "kill_rc": kill.returncode,
                "attempts": raw.get("attempts"),
                "regraded_passed": res_c.get("passed"),
                "score": res_c.get("score"),
            }
            results["chaos"] = {"result": res_c, **chaos_detail}
            break
        invariant("chaos-midlease-kill-requeued", chaos_ok, chaos_detail)

    # ---- queue accounting after the run
    st = client.status()
    results["queue_status"] = st
    invariant("queue-drained", st["queue_depth"] == 0 and st["leased"] == 0, st)

    results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["hard_failures"] = hard_fail
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print(f"[driver] done; {len(hard_fail)} hard failures -> {args.out}", flush=True)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
