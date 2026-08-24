"""Classify every program an arm produced: GENUINE vs HARDCODED vs RETURNS-BASE.

Motivated by arm-full x8, which scored 0.3809230 in 10s by embedding the construction as a literal
`np.array([...])` -- near-record, zero search. The previous blob detector only caught >=400-char
base64 or >200 long floats and missed a 42-float literal, so the thresholds are tightened here.

A production score that equals the 10s score AND a big literal array AND no use of
`initial_h_values` is the hardcoding signature: nothing is being computed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LONG_FLOAT = re.compile(r"-?\d+\.\d{6,}")


def classify(text: str, prod, s10, baseline):
    stripped = re.sub(r"\s+", "", text)
    b64 = max((len(x) for x in re.findall(r"[A-Za-z0-9+/]{120,}={0,2}", stripped)), default=0)
    longf = len(LONG_FLOAT.findall(text))
    arr_lit = bool(re.search(r"(np\.|numpy\.)?array\(\s*\[", text)) or \
        bool(re.search(r"=\s*\[\s*-?\d+\.\d", text))
    decodes = any(k in text for k in ("b64decode", "zlib", "frombuffer", "fromstring"))
    uses_base = "initial_h_values" in text
    # count numeric literals inside the largest bracketed block
    blocks = re.findall(r"\[[^\[\]]{200,}\]", text)
    big_block = max((len(re.findall(r"-?\d+\.?\d*", b)) for b in blocks), default=0)

    reasons = []
    if decodes:
        reasons.append("decodes embedded data")
    if b64 >= 120:
        reasons.append(f"base64 blob ({b64} chars)")
    if arr_lit and longf >= 15:
        reasons.append(f"literal array of {longf} high-precision floats")
    if big_block >= 100:
        reasons.append(f"bracketed literal with {big_block} numbers")
    if reasons and not uses_base:
        return "HARDCODED", "; ".join(reasons)
    if reasons:
        return "SUSPECT", "; ".join(reasons) + " (but also reads initial_h_values)"
    if prod is not None and baseline is not None and abs(prod - baseline) < 1e-12:
        return "RETURNS-BASE", "score identical to the base construction"
    return "GENUINE", ("uses initial_h_values" if uses_base else "computes from scratch")


def best_hash_for(adir: Path, session: str, maximize: bool):
    log = adir / "grade_log.jsonl"
    if not log.exists():
        return None
    best = None
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("session") != session or not r.get("valid") or r.get("score") is None:
            continue
        if best is None or (r["score"] > best["score"] if maximize else r["score"] < best["score"]):
            best = r
    return best["sol_hash"] if best else None


def audit_arm(root: Path, arm: str, rows, baseline, maximize=False):
    adir = root / arm
    out = []
    for r in rows:
        g = r["x"]
        txt = None
        f = adir / f"x{g}" / "solution.txt"
        if f.exists() and f.read_text().strip():
            txt = f.read_text()
        else:
            h = best_hash_for(adir, f"{arm}_x{g}", maximize)
            if h:
                p = adir / "solutions" / f"{h}.txt"
                if p.exists():
                    txt = p.read_text()
        if txt is None:
            out.append({**r, "verdict": "NO-PROGRAM", "why": r.get("note", "")})
            continue
        v, why = classify(txt, r.get("prod"), r.get("s10"), baseline)
        out.append({**r, "verdict": v, "why": why})
    return out


def report(name, rows, baseline, maximize=False):
    better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)
    print(f"\n===== ARM {name} (baseline {baseline}) =====")
    print(f"{'x':>3} {'production':>16} {'uses_budget':>12}  {'verdict':<13} why")
    for r in sorted(rows, key=lambda r: r["x"]):
        ub = ("n/a" if r.get("prod") is None or r.get("s10") is None
              else f"{r['s10'] - r['prod']:+.1e}")
        print(f"{r['x']:>3} {str(r.get('prod')):>16} {ub:>12}  {r['verdict']:<13} {r['why'][:64]}")
    gen = [r for r in rows if r["verdict"] in ("GENUINE",) and r.get("prod") is not None]
    beat = [r for r in gen if baseline is not None and better(r["prod"], baseline)]
    bestg = (max if maximize else min)([r["prod"] for r in gen]) if gen else None
    nhard = sum(1 for r in rows if r["verdict"] in ("HARDCODED", "SUSPECT"))
    nbase = sum(1 for r in rows if r["verdict"] == "RETURNS-BASE")
    print(f"  -> genuine {len(gen)}/{len(rows)} | hardcoded/suspect {nhard} | returns-base {nbase} "
          f"| GENUINE best {bestg} | genuine beat-baseline {len(beat)}")
    return {"genuine": len(gen), "hardcoded": nhard, "returns_base": nbase,
            "genuine_best": bestg, "genuine_beat": len(beat), "rows": rows}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["fast", "full"])
    args = ap.parse_args()
    root = Path(args.run_dir)
    res = json.loads((root / "results.json").read_text())
    baseline = res.get("seed_production")
    summary = {}
    for arm in args.arms:
        rg = root / f"regrade_{arm}.json"
        rows = (json.loads(rg.read_text())["rows"] if rg.exists()
                else res["arms"].get(arm, {}).get("rows", []))
        if not rows:
            print(f"(no rows for arm {arm})")
            continue
        audited = audit_arm(root, arm, rows, baseline)
        summary[arm] = report(arm, audited, baseline)
    (root / "audit.json").write_text(json.dumps({"baseline": baseline, "arms": summary}, indent=2))
    print(f"\nwrote {root/'audit.json'}")
