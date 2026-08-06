"""Phase-5 fleet driver: the grading server at fleet scale.

Generalizes phase4/driver.py from one judge to N. Everything phase 4 proved
on a single chip is re-asserted here (verdicts at the right gate, ref-vs-ref,
peak HBM), plus the properties that only exist at fleet scale and therefore
could not be tested before:

  [inv] fleet-size                      N judges leasing from ONE queue
  [inv] every-verdict-correct           whole battery, every copy, its gate
  [inv] compile-bomb-rejected           the new poison-pill fixture
  [inv] compile-bomb-never-requeued     terminal: graded once, fleet-wide
  [inv] regrade-spread-within-3%        ~100 salted copies of one kernel
  [inv] ref-vs-ref-1.00+-2%-every-judge not just the one judge phase 4 had
  [inv] cross-judge-agreement-3%        same kernel, different chips
  [inv] cache-hit-identical             shared GCS cache, byte-identical
  [inv] chaos-*                         judge killed mid-lease -> requeue
  [inv] no-duplicate-results            exactly-once under requeue
  [measure] throughput                  candidates/min + per-judge lanes

No RL, no training, no policy: a fixed corpus in, verdicts out. Runs on the
compute node next to the queue; judges reach it through reverse ssh tunnels.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ARENA_IMPORT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARENA_IMPORT_ROOT))

from pallas_arena.judge.queue import ArenaQueueClient  # noqa: E402
from pallas_arena.phase5 import corpus as corpus_mod  # noqa: E402

REGRADE_BAND = 0.03  # +-3%: the phase-1 timing-invariant contract
REF_BAND = 0.02  # +-2%: ref-vs-ref

DEFAULT_KILL_CMD = (
    "gcloud compute tpus tpu-vm ssh {judge} --zone={zone} --project={project} "
    "--worker=0 --command=\"pkill -9 -f '[p]allas_arena.judge.worker'\""
)


def _stat(vals):
    v = sorted(x for x in vals if isinstance(x, (int, float)))
    if not v:
        return None
    return {
        "n": len(v),
        "median": v[len(v) // 2],
        "mean": sum(v) / len(v),
        "p95": v[min(int(0.95 * len(v)), len(v) - 1)],
        "min": v[0],
        "max": v[-1],
    }


class Driver:
    def __init__(self, args):
        self.args = args
        self.client = ArenaQueueClient(args.queue, timeout_s=120.0)
        self.results: dict = {
            "invariants": {},
            "measurements": {},
            "meta": {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "queue": args.queue},
        }
        self.hard_fail: list[str] = []
        self.judges = [j for j in (args.judges or "").split(",") if j]
        self._by_variant: dict[str, list[dict]] = {}

    # ------------------------------------------------------------- reporting
    def invariant(self, name: str, ok: bool, detail):
        self.results["invariants"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail, default=str)[:600]}", flush=True)
        if not ok:
            self.hard_fail.append(name)

    def measure(self, name: str, detail):
        self.results["measurements"][name] = detail
        print(f"[measure] {name}: {json.dumps(detail, default=str)[:600]}", flush=True)

    def save(self):
        self.results["hard_failures"] = self.hard_fail
        self.results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        Path(self.args.out).write_text(json.dumps(self.results, indent=1, default=str))

    # ------------------------------------------------------------------ boot
    def await_fleet(self) -> list[str]:
        """Wait for judges to start polling. A degraded fleet is legal: the
        demo reports the N that landed rather than failing, because every
        acceptance item except raw throughput is N-independent."""
        want, floor = self.args.judges_expected, self.args.judges_min
        deadline = time.time() + self.args.fleet_wait_s
        seen: list[str] = []
        while time.time() < deadline:
            try:
                seen = sorted((self.client.status().get("workers_seen") or {}))
            except Exception:
                seen = []
            if len(seen) >= want:
                break
            time.sleep(5)
        self.invariant("fleet-size", len(seen) >= floor, {"polling": seen, "want": want, "min": floor})
        self.measure("fleet-size-actual", {"n": len(seen), "judges": seen})
        return seen

    # ------------------------------------------------------------ the corpus
    def run_corpus(self) -> dict:
        items = corpus_mod.build(self.args.target)
        plan = corpus_mod.plan(self.args.target)
        print(f"[driver] corpus {len(items)} items: {json.dumps(plan)}", flush=True)
        self.measure("corpus-plan", {"total": len(items), "counts": plan})

        t0 = time.time()
        ids: dict[str, dict] = {}
        for it in items:
            ids[self.client.submit("rmsnorm", it["code"], tag=it["tag"])] = it
        submit_s = time.time() - t0
        print(f"[driver] submitted {len(ids)} in {submit_s:.1f}s", flush=True)

        pending, done = list(ids), {}
        deadline = t0 + self.args.grade_wait_s
        chaos_pending = self.args.chaos_after > 0 and bool(self.judges)
        last_log, last_progress, stall_from = 0.0, 0, time.time()
        while pending and time.time() < deadline:
            got = self.client.poll_bulk(pending)
            if got:
                done.update(got)
                pending = [w for w in pending if w not in done]
            frac = len(done) / max(len(ids), 1)
            now = time.time()
            if len(done) > last_progress:
                last_progress, stall_from = len(done), now
            if now - last_log > 30:
                last_log = now
                print(
                    f"[driver] {len(done)}/{len(ids)} graded ({100 * frac:.1f}%) "
                    f"{len(done) / max(now - t0, 1e-9) * 60:.1f}/min elapsed={now - t0:.0f}s "
                    f"stalled={now - stall_from:.0f}s",
                    flush=True,
                )
            if chaos_pending and frac >= self.args.chaos_after:
                chaos_pending = False
                self.chaos_kill_mid_lease()
            if pending:
                time.sleep(self.args.poll_s)
        wall = time.time() - t0

        self.invariant(
            "all-graded", not pending, {"graded": len(done), "submitted": len(ids), "still_pending": len(pending)}
        )
        return {"ids": ids, "done": done, "wall_s": wall, "submit_s": submit_s, "t0": t0}

    # ------------------------------------------------------------ acceptance
    def check_verdicts(self, run: dict):
        by_variant: dict[str, list[dict]] = defaultdict(list)
        for wid, it in run["ids"].items():
            rec = run["done"].get(wid)
            if rec:
                by_variant[it["variant"]].append(rec)
        self._by_variant = dict(by_variant)

        bad, gate_hist = {}, {}
        for variant, recs in sorted(by_variant.items()):
            hist: dict[str, int] = defaultdict(int)
            wrong: list[str] = []
            for rec in recs:
                r = rec.get("result") or {}
                hist[str(r.get("gate"))] += 1
                ok, why = corpus_mod.verdict_ok(variant, r)
                if not ok:
                    wrong.append(why)
            gate_hist[variant] = dict(hist)
            if wrong:
                bad[variant] = {
                    "wrong": len(wrong),
                    "of": len(recs),
                    "gates": dict(hist),
                    "why": sorted(set(wrong))[:3],
                    "expected": corpus_mod.expectation(variant),
                }
        self.invariant(
            "every-verdict-correct",
            not bad,
            bad or {"variants": len(by_variant), "graded": sum(len(v) for v in by_variant.values())},
        )
        self.measure("gate-histogram", gate_hist)

        # The new fixture gets its own line: it is the reason phase 5 was
        # gated on a compile deadline at all.
        bombs = by_variant.get("compile-bomb", [])
        allowed = set(corpus_mod.expectation("compile-bomb")["gates"])
        gates = [str((r.get("result") or {}).get("gate")) for r in bombs]
        self.invariant(
            "compile-bomb-rejected",
            bool(bombs) and set(gates) <= allowed and all(not (r.get("result") or {}).get("passed") for r in bombs),
            {"n": len(bombs), "gates": sorted(set(gates)), "allowed": sorted(allowed)},
        )
        # Terminal is required exactly where the judge-side budget fired; a
        # rejection in the killable export child is one gate earlier and is
        # already un-requeuable by being done. Either way: graded ONCE.
        need_terminal = [r for r in bombs if (r.get("result") or {}).get("gate") == "compile_budget"]
        self.invariant(
            "compile-bomb-never-requeued",
            bool(bombs)
            and all(r.get("attempts") == 1 for r in bombs)
            and all(r.get("terminal") for r in need_terminal),
            {
                "n": len(bombs),
                "attempts": sorted({r.get("attempts") for r in bombs}),
                "at_compile_budget": len(need_terminal),
                "flagged_terminal": sum(1 for r in bombs if r.get("terminal")),
            },
        )
        self.measure(
            "compile-bomb-cost",
            {
                "budget_s": next(((r.get("result") or {}).get("compile_budget_s") for r in bombs), None),
                "candidate_compile_s": [(r.get("result") or {}).get("candidate_compile_s") for r in bombs],
                "item_wall_s": [(r.get("result") or {}).get("item_wall_s") for r in bombs],
                "judge_restarted": [bool((r.get("result") or {}).get("judge_restarted")) for r in bombs],
            },
        )

    def check_reproducibility(self):
        """Same kernel, ~100 independent grades (salted -> real regrades, not
        cache hits). Two different questions: spread WITHIN the population
        (regrade stability) and agreement BETWEEN judges — the genuinely new
        fleet property, which a single-judge run cannot show."""
        spread, cross = {}, {}
        worst_spread, worst_cross = 0.0, 0.0
        for variant, recs in sorted(self._by_variant.items()):
            pairs = []
            for r in recs:
                res = r.get("result") or {}
                s, w = res.get("score"), res.get("worker_id") or r.get("worker_id")
                if isinstance(s, (int, float)):
                    pairs.append((w, s))
            scores = [s for _w, s in pairs]
            if len(scores) < 8:
                continue
            med = statistics.median(scores)
            devs = sorted(abs(s / med - 1.0) for s in scores)
            spread[variant] = {
                "n": len(scores),
                "median": med,
                "max_dev": devs[-1],
                "p95_dev": devs[min(int(0.95 * len(devs)), len(devs) - 1)],
                "min": min(scores),
                "max": max(scores),
            }
            worst_spread = max(worst_spread, devs[-1])

            per_judge: dict[str, list[float]] = defaultdict(list)
            for w, s in pairs:
                if w:
                    per_judge[w].append(s)
            meds = {w: statistics.median(v) for w, v in per_judge.items() if len(v) >= 3}
            if len(meds) >= 2:
                d = max(meds.values()) / min(meds.values()) - 1.0
                cross[variant] = {"per_judge_median": meds, "max_disagreement": d}
                worst_cross = max(worst_cross, d)

        # Gate on the p95 deviation, report the max. With ~120 independent
        # grades per honest variant the max is an extreme-value statistic --
        # one first-touch-compile outlier would fail a max-based gate while
        # saying nothing about reproducibility. The max is still reported per
        # variant so a genuinely bimodal population is visible.
        worst_p95 = max((v["p95_dev"] for v in spread.values()), default=1.0)
        self.invariant(
            "regrade-spread-within-3%",
            bool(spread) and worst_p95 < REGRADE_BAND,
            {"worst_p95_dev": worst_p95, "worst_max_dev": worst_spread, "band": REGRADE_BAND, "per_variant": spread},
        )
        self.invariant(
            "cross-judge-agreement-3%",
            bool(cross) and worst_cross < REGRADE_BAND,
            {"worst_disagreement": worst_cross, "band": REGRADE_BAND, "per_variant": cross},
        )

    def check_boot_invariants(self):
        """ref-vs-ref on EVERY judge. Phase 4 could only show it on one; a
        single mis-calibrated chip in a fleet silently biases every reward
        that lands on it, and nothing downstream would notice."""
        per_judge: dict[str, dict] = {}
        hbm, backends = [], set()
        for recs in self._by_variant.values():
            for rec in recs:
                r = rec.get("result") or {}
                w = r.get("worker_id") or rec.get("worker_id")
                boot = r.get("worker_boot") or {}
                if w and boot.get("ref_vs_ref_scores") and w not in per_judge:
                    per_judge[w] = boot["ref_vs_ref_scores"]
                if r.get("peak_hbm_bytes"):
                    hbm.append(r["peak_hbm_bytes"])
                if r.get("backend"):
                    backends.add(r["backend"])
        bad = {
            w: {c: s for c, s in sc.items() if abs(s - 1.0) > REF_BAND}
            for w, sc in per_judge.items()
            if any(abs(s - 1.0) > REF_BAND for s in sc.values())
        }
        self.invariant(
            "ref-vs-ref-1.00±2%-every-judge",
            bool(per_judge) and not bad,
            {"judges": len(per_judge), "scores": per_judge, "violations": bad},
        )
        self.invariant("judges-on-tpu", backends == {"tpu"}, sorted(backends))
        self.invariant(
            "peak-hbm-reported",
            bool(hbm),
            {"max_gb": max(hbm) / 2**30 if hbm else None, "min_gb": min(hbm) / 2**30 if hbm else None},
        )

    def check_cache(self, run: dict):
        """Resubmit the same text: the shared hash->reward cache must return
        a byte-identical verdict. This is what makes a stateless preemptible
        fleet safe for RL — a kernel's reward must not depend on which judge
        (or which incarnation of it) happened to draw the item."""
        dups = corpus_mod.duplicate_set(list(run["ids"].values()), self.args.dups)
        first = {it["tag"]: (run["done"].get(w) or {}).get("result") or {} for w, it in run["ids"].items()}
        t0 = time.time()
        dup_ids = {self.client.submit("rmsnorm", d["code"], tag=d["tag"]): d for d in dups}
        pending, got = list(dup_ids), {}
        deadline = time.time() + self.args.dup_wait_s
        while pending and time.time() < deadline:
            got.update(self.client.poll_bulk(pending))
            pending = [w for w in pending if w not in got]
            if pending:
                time.sleep(1.0)
        wall = time.time() - t0

        compared = ("reward", "score", "gate", "passed", "problem_version")
        hits, identical, mismatched = 0, 0, []
        for wid, d in dup_ids.items():
            rec = got.get(wid)
            if not rec:
                mismatched.append({"tag": d["tag"], "why": "no result"})
                continue
            r = rec.get("result") or {}
            hits += bool(r.get("cache_hit"))
            base = first.get(d["dup_of"]) or {}
            if all(r.get(k) == base.get(k) for k in compared):
                identical += 1
            else:
                mismatched.append(
                    {"tag": d["tag"], "got": {k: r.get(k) for k in compared}, "want": {k: base.get(k) for k in compared}}
                )
        self.invariant(
            "cache-hit-identical",
            bool(dup_ids) and identical == len(dup_ids) and not mismatched,
            {"n": len(dup_ids), "identical": identical, "cache_hit_flag": hits, "mismatched": mismatched[:5]},
        )
        self.measure(
            "cache-hit-cost",
            {
                "n": len(dup_ids),
                "resolved": len(got),
                "wall_s": wall,
                "per_item_wall_s": _stat([(r.get("result") or {}).get("item_wall_s") for r in got.values()]),
            },
        )

    # ----------------------------------------------------------------- chaos
    def chaos_kill_mid_lease(self):
        """SIGKILL a judge while it holds a lease. Phase 4 proved one judge
        coming back; at fleet scale the property is different and stronger:
        the work must complete on a DIFFERENT judge, with no lost and no
        double-counted result, while the survivors keep grading."""
        detail: dict = {"tries": []}
        ok = False
        for attempt in range(3):
            leases = []
            t_wait = time.time()
            while time.time() - t_wait < 180:
                try:
                    leases = [le for le in self.client.leases() if le.get("worker_id") in self.judges]
                except Exception:
                    leases = []
                if leases:
                    break
                time.sleep(1)
            if not leases:
                detail["tries"].append({"attempt": attempt, "why": "no known judge held a lease"})
                continue
            target = leases[0]
            victim, wid = target["worker_id"], target["work_id"]
            cmd = self.args.chaos_kill_cmd.format(judge=victim, zone=self.args.zone, project=self.args.project)
            t_k = time.time()
            kill = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            mid = wid not in self.client.poll_bulk([wid])
            print(
                f"[driver] chaos attempt={attempt} victim={victim} work={wid} "
                f"kill_rc={kill.returncode} midlease={mid} ({time.time() - t_k:.0f}s)",
                flush=True,
            )
            if not mid:
                detail["tries"].append({"attempt": attempt, "why": "grade completed before the kill landed"})
                continue
            rec = None
            while time.time() - t_k < self.args.chaos_wait_s:
                p = self.client.poll_bulk([wid])
                if wid in p:
                    rec = p[wid]
                    break
                time.sleep(3)
            r = (rec or {}).get("result") or {}
            survivor = r.get("worker_id") or (rec or {}).get("worker_id")
            ok = bool(rec) and rec.get("attempts", 0) >= 2
            detail.update(
                killed_judge=victim,
                work_id=wid,
                tag=target.get("tag"),
                kill_rc=kill.returncode,
                killed_mid_lease=True,
                attempts=rec.get("attempts") if rec else None,
                completed_on=survivor,
                completed_on_survivor=bool(survivor and survivor != victim),
                requeue_to_completion_s=time.time() - t_k,
                regraded_passed=r.get("passed"),
                regraded_gate=r.get("gate"),
            )
            break
        self.results["chaos"] = detail
        self.invariant("chaos-midlease-kill-requeued", ok, detail)
        self.invariant(
            "chaos-completed-on-a-survivor",
            bool(ok and detail.get("completed_on_survivor")),
            {"killed": detail.get("killed_judge"), "completed_on": detail.get("completed_on")},
        )

    # ------------------------------------------------------------ throughput
    def check_throughput(self, run: dict):
        done = run["done"]
        walls, per_judge, completed_at = [], defaultdict(list), []
        for rec in done.values():
            r = rec.get("result") or {}
            w = r.get("item_wall_s")
            if isinstance(w, (int, float)):
                walls.append(w)
                per_judge[r.get("worker_id") or rec.get("worker_id")].append(w)
            if rec.get("completed_at"):
                completed_at.append(rec["completed_at"])
        span = max(completed_at) - run["t0"] if completed_at else run["wall_s"]
        self.measure(
            "throughput",
            {
                "graded": len(done),
                "submitted": len(run["ids"]),
                "submit_s": run["submit_s"],
                "wall_s": run["wall_s"],
                "first_submit_to_last_result_s": span,
                "candidates_per_min": len(done) / max(span, 1e-9) * 60.0,
                "candidates_per_min_per_judge": len(done) / max(span, 1e-9) * 60.0 / max(len(per_judge), 1),
                "judges": len(per_judge),
                "per_judge_items": {k: len(v) for k, v in per_judge.items()},
                "per_judge_item_wall_median": {k: sorted(v)[len(v) // 2] for k, v in per_judge.items() if v},
                "item_wall_s": _stat(walls),
            },
        )
        self.measure(
            "per-candidate-cost-breakdown-s",
            {
                k: _stat([(rec.get("result") or {}).get(k) for rec in done.values()])
                for k in ("export_s", "load_s", "candidate_compile_s", "warm_chip_s", "item_wall_s")
            },
        )

    def check_accounting(self):
        st = self.client.status()
        self.results["queue_status"] = st
        keys = ("submitted", "completed", "duplicates", "requeues", "expired_leases", "terminal_rejections")
        self.invariant("no-duplicate-results", st.get("duplicates", 0) == 0, {k: st.get(k) for k in keys})
        self.invariant("queue-drained", st["queue_depth"] == 0, {"depth": st["queue_depth"], "leased": st["leased"]})

    # ------------------------------------------------------------------- run
    def main(self) -> int:
        seen = self.await_fleet()
        if len(seen) < self.args.judges_min:
            self.save()
            return 1
        if not self.judges:
            self.judges = seen  # simulation / self-identifying workers
        run = self.run_corpus()
        self.check_verdicts(run)
        self.check_reproducibility()
        if self.args.expect_tpu:
            self.check_boot_invariants()
        self.check_throughput(run)
        self.check_cache(run)
        self.check_accounting()
        self.save()
        print(f"[driver] done; {len(self.hard_fail)} hard failures -> {self.args.out}", flush=True)
        return 1 if self.hard_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="http://127.0.0.1:8770")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", default="", help="comma-separated worker ids (== QR names on the fleet)")
    ap.add_argument("--judges-expected", type=int, default=5)
    ap.add_argument("--judges-min", type=int, default=2)
    ap.add_argument("--fleet-wait-s", type=float, default=1800.0)
    ap.add_argument("--target", type=int, default=1600)
    ap.add_argument("--dups", type=int, default=100)
    ap.add_argument("--grade-wait-s", type=float, default=4800.0)
    ap.add_argument("--dup-wait-s", type=float, default=900.0)
    ap.add_argument("--poll-s", type=float, default=3.0)
    ap.add_argument("--chaos-after", type=float, default=0.35, help="fraction graded before the kill; 0 disables")
    ap.add_argument("--chaos-wait-s", type=float, default=1200.0)
    ap.add_argument("--chaos-kill-cmd", default=DEFAULT_KILL_CMD)
    ap.add_argument("--zone", default="us-east5-b")
    ap.add_argument("--project", default="vision-mix")
    ap.add_argument("--expect-tpu", action="store_true", default=False)
    return Driver(ap.parse_args()).main()


if __name__ == "__main__":
    sys.exit(main())
