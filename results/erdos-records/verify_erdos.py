"""Independently re-score an Erdos min-overlap record artifact.

Never trusts the logged value: recomputes C5 from the stored construction, and
reports the SIGNED delta rather than a binary verdict -- a -5e-13 float round-trip
difference and a +5.7e-5 constraint-violation overclaim are not the same finding,
and the old exact-repr comparison called both "INVALID".

Also gates on the mass constraint sum(h) == n/2 directly. The in-run grader used
to score an un-normalized density and return the model's claim, so a program
could smuggle in ~0.02% excess mass for a ~6e-5 apparent gain (fixed upstream in
examples/erdos_min_overlap/env.py, 2026-08-20). Checking sum(h) catches that
class without re-grading.
"""
import json, sys, numpy as np

TOL_FLOAT = 1e-9   # below this, a mismatch is float round-trip noise


def score(h_list):
    h = np.array(h_list, dtype=float); n = len(h); dx = 2.0 / n; t = n / 2.0
    if not np.all(np.isfinite(h)):            return None, "non-finite", None
    if np.any(h < 0) or np.any(h > 1):        return None, "out of [0,1]", None
    mass = float(h.sum())
    if mass != t:
        h = h * (t / mass)
        if np.any(h < 0) or np.any(h > 1):    return None, "out of [0,1] post-norm", mass
    return float((np.correlate(h, 1.0 - h, mode="full") * dx).max()), None, mass


for path in sys.argv[1:]:
    rec = json.load(open(path))
    logged = float(rec["value_full_precision"])
    c, err, mass = score(rec["construction"])
    name = path.split("/")[-1]
    if err:
        print(f"{name:<32} ERROR {err}")
        continue
    n = len(rec["construction"]); target = n / 2.0
    delta = c - logged
    excess = (mass - target) / target
    if delta > TOL_FLOAT:
        verdict = f"OVERCLAIM {delta:+.3e}"
    elif abs(delta) <= TOL_FLOAT:
        verdict = "OK"
    else:
        verdict = f"conservative {delta:+.3e}"
    # A construction round-trips through JSON at ~1e-16 relative, so tiny mass
    # error is noise. Real smuggling was +2.4e-4 relative (tsw-n) -- four orders up.
    massnote = "" if abs(excess) < 1e-9 else f"  [sum(h) off by {100*excess:+.4f}% -> INFEASIBLE as stored]"
    print(f"{name:<32} logged={logged:.13f} true={c:.13f}  {verdict}{massnote}")
