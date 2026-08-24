"""Re-grade executor outputs directly from the cached workdirs.

run_executor.py discards solution.py on codex TIMEOUT (returns None without reading
the file), so its in-process got_code/c5 undercounts badly. But every workdir +
solution.py is preserved on disk (memory: cache-raw-model-generations). This walks
`_exec/<tag>/wd_<base8>_<plan>_<hex>/solution.py`, grades each against its base, and
prints the TRUE per-executor table. Grading is local (free), so this is authoritative.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common

WD_RE = re.compile(r"wd_([0-9a-f]{8})_(\d+)_[0-9a-f]+$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. van_luna (scratch dir under _exec)")
    ap.add_argument("--executor", default="", help="label for the printout")
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--worse-set", default="tpu/distill_ablation/corpora/worse_set.json")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    common.load_dotenv_key()
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.state import state_from_dict

    ST = ErdosMinOverlapEnv.state_type
    worse = common.read_json(args.worse_set)["worse"]
    bases = {w["id"]: state_from_dict(w, state_type=ST) for w in worse}
    by8 = {}
    for bid, b in bases.items():
        by8.setdefault(bid[:8], b)  # first 8 hex -> base (16 bases, no collision)

    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_exec/{args.tag}"
    wds = sorted(Path(scratch).glob("wd_*"))
    jobs = []  # (base, plan_idx, code_text)
    missing = 0
    for wd in wds:
        m = WD_RE.match(wd.name)
        if not m:
            continue
        b8, j = m.group(1), int(m.group(2))
        base = by8.get(b8)
        sol = wd / "solution.py"
        if base is None or not sol.exists():
            missing += 1
            continue
        txt = sol.read_text()
        if txt.strip():
            jobs.append((base, j, txt))

    def grade(code_text, base):
        ev = ErdosMinOverlapRewardEvaluator(problem_type="", log_dir=scratch, num_cpus_per_task=1,
                                            eval_timeout=args.eval_timeout, eval_backend="local")
        try:
            out = ev.get_reward(f"```python\n{code_text}\n```", base)
            return out.get("raw_score") if out.get("correctness", 0) > 0 else None
        except Exception:
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        c5s = list(ex.map(lambda t: grade(t[2], t[0]), jobs))

    records = []
    per_base = {}
    for (base, j, txt), c5 in zip(jobs, c5s):
        base_c5 = -base.value
        rec = {"base_id": base.id, "base_c5": base_c5, "plan_idx": j,
               "c5": c5, "code_len": len(txt),
               "improved": c5 is not None and (base_c5 - c5) > 1e-4,
               "delta": (base_c5 - c5) if c5 is not None else None}
        records.append(rec)
        per_base.setdefault(base.id, []).append((base_c5, c5))

    n = len(records)
    valid = [r for r in records if r["c5"] is not None]
    imp = [r for r in records if r["improved"]]
    bases_seen = len(per_base)
    bases_imp = sum(1 for v in per_base.values() if any(c is not None and bc - c > 1e-4 for bc, c in v))
    label = args.executor or args.tag
    print(f"\n===== REGRADE {label} / tag={args.tag}  ({time.time()-t0:.0f}s) =====")
    print(f"  workdirs={len(wds)} graded={n} missing_or_empty={missing + (len(wds)-n)}")
    print(f"  got_code(on disk) {n} | VALID {len(valid)}/{n} ({100*len(valid)/max(n,1):.0f}%) | "
          f"improved(plan) {len(imp)}/{n} | bases-improved(best-of) {bases_imp}/{bases_seen}")
    if valid:
        c5s2 = sorted(r["c5"] for r in valid)
        print(f"  c5: best={c5s2[0]:.6f} median={c5s2[len(c5s2)//2]:.6f} worst={c5s2[-1]:.6f}")
    if args.out:
        common.write_json(args.out, {"executor": label, "tag": args.tag,
                                     "n": n, "valid": len(valid), "results": records})
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
