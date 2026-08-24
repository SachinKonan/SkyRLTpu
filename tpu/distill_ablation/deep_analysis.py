"""Mechanistic deep-dive on the Stage-2 2x2 (all from cached data, no re-grading).

Probes:
  (1) TECHNIQUE FINGERPRINT per arm — does foreign context (or the executor's prior)
      change WHAT technique gets written? Especially smooth-max (the high-scoring technique
      gpt-oss underuses, that the foreign refs are meant to inject) and CMA-ES (terra's prior).
  (2) FULL c5 DISTRIBUTION per arm — median / p25 / failure count, not just best-of-N, to
      see any real shift under the best-of-3 noise floor.
  (3) PER-BASE CEILING — best c5 any arm reached per base; is there a base-specific floor
      every method hits (=> exploration ceiling, not execution/ideation)?
"""
from __future__ import annotations
import glob, json, os, re, statistics
from pathlib import Path

AB = os.path.dirname(os.path.abspath(__file__))
EXEC = "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/distill_ablation/_exec"

TECH = {
    "smoothmax/LSE": r"logsumexp|log_sum_exp|np\.log\(\s*np\.sum\(\s*np\.exp|softmax|smooth.?max|/\s*alpha|\*\s*alpha",
    "CMA-ES":        r"\bcma\b|CMAEvolution|purecma|cma_es|import cma",
    "L-BFGS":        r"[lL]-?BFGS|fmin_l_bfgs|lbfgs",
    "SLSQP":         r"SLSQP|slsqp",
    "annealing":     r"anneal|dual_annealing|basinhopping|temperature|\bcooling\b",
    "projection":    r"project|waterfill|simplex|renormal|/ *np\.sum|sum\(\) *\* *dx",
    "symmetry":      r"symmetr|mirror|\[::-1\]|np\.flip|reverse",
    "gradient/jac":  r"\bjac\b|jac=|gradient|grad\b|\.grad",
}

def code_from_gen(text: str) -> str:
    # grade uses the LAST ```python (final answer, not the <think> draft)
    ms = re.findall(r"```python\s+([\s\S]*?)\s*```", text)
    return ms[-1] if ms else text

def direct_arm(path):
    d = json.load(open(path)); res = d["results"]
    progs, c5map = [], {}
    for r in res:
        for t in r.get("gens", []):
            progs.append(code_from_gen(t))
        c5map[r["base_id"]] = [c for c in r["results"] if c is not None]
    allc5 = [c for v in c5map.values() for c in v]
    return progs, c5map, allc5

def decoupled_arm(tag, reg_path):
    progs = [Path(p).read_text() for p in glob.glob(f"{EXEC}/{tag}/wd_*/solution.py")]
    d = json.load(open(reg_path)); res = d["results"]
    c5map = {}
    for r in res:
        if r["c5"] is not None:
            c5map.setdefault(r["base_id"], []).append(r["c5"])
    allc5 = [r["c5"] for r in res if r["c5"] is not None]
    return progs, c5map, allc5

def tech_counts(progs):
    n = len(progs) or 1
    out = {}
    for name, pat in TECH.items():
        rx = re.compile(pat, re.I)
        out[name] = 100.0 * sum(1 for p in progs if rx.search(p)) / n
    return out

def dist(allc5):
    if not allc5: return None
    s = sorted(allc5)
    return dict(n=len(s), best=s[0], p25=s[len(s)//4], med=s[len(s)//2],
                worst=s[-1], mean=statistics.mean(s))

ARMS = {
 "direct-van":  lambda: direct_arm(f"{AB}/corpora/s2_direct_vanilla.json"),
 "direct-for":  lambda: direct_arm(f"{AB}/corpora/s2_direct_foreign.json"),
 "terra-van":   lambda: decoupled_arm("s2_dec_van", f"{AB}/corpora/regrade_s2_dec_van.json"),
 "terra-for":   lambda: decoupled_arm("s2_dec_for", f"{AB}/corpora/regrade_s2_dec_for.json"),
}

data = {k: f() for k, f in ARMS.items()}

print("="*78)
print("(1) TECHNIQUE FINGERPRINT  (% of programs in arm using each; n programs in parens)")
print("="*78)
techs = list(TECH.keys())
hdr = "technique".ljust(16) + "".join(k.rjust(12) for k in ARMS)
print(hdr); print("-"*len(hdr))
counts = {k: tech_counts(data[k][0]) for k in ARMS}
npro = {k: len(data[k][0]) for k in ARMS}
print("(n programs)".ljust(16) + "".join(f"{npro[k]:>12}" for k in ARMS))
for t in techs:
    print(t.ljust(16) + "".join(f"{counts[k][t]:>11.0f}%" for k in ARMS))

print("\n" + "="*78)
print("(2) FULL c5 DISTRIBUTION per arm (lower=better; best-of-N hides the bulk)")
print("="*78)
print("arm".ljust(12) + "  n   best      p25       median    mean      worst")
for k in ARMS:
    dd = dist(data[k][2])
    if dd:
        print(f"{k:<12}{dd['n']:>3}  {dd['best']:.6f}  {dd['p25']:.6f}  "
              f"{dd['med']:.6f}  {dd['mean']:.6f}  {dd['worst']:.6f}")

print("\n" + "="*78)
print("(3) PER-BASE CEILING  (best c5 each arm reached on each base; '.' = no valid gen)")
print("="*78)
# base_c5 per base (from direct-van json records)
base_c5 = {}
for path in (f"{AB}/corpora/s2_direct_vanilla.json", f"{AB}/corpora/s2_direct_foreign.json"):
    for r in json.load(open(path))["results"]:
        base_c5.setdefault(r["base_id"], r["base_c5"])

def armmin(k, b):
    v = data[k][1].get(b, [])
    return min(v) if v else None

bases = sorted({b for k in ARMS for b in data[k][1]},
               key=lambda b: base_c5.get(b, 9))
print("base(c5)".ljust(20) + "".join(k.rjust(11) for k in ARMS) + "    floor   Δbase")
for b in bases:
    mins = [armmin(k, b) for k in ARMS]
    valid = [m for m in mins if m is not None]
    floor = min(valid) if valid else None
    row = "".join((f"{m:>11.6f}" if m is not None else f"{'.':>11}") for m in mins)
    bc = base_c5.get(b)
    head = f"{b[:6]}({bc:.5f})" if bc else b[:6]
    dfl = f"{bc-floor:+.2e}" if (floor is not None and bc is not None) else "   ."
    print(f"{head:<20}{row}   {floor:.6f} {dfl}" if floor else f"{head:<20}{row}")
