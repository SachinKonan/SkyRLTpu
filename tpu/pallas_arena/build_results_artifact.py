"""Build the re-evaluated results page: original -> improvement -> per-case ->
aggregated -> golden truth.

Usage: python build_results_artifact.py <base_jobid> <improve_jobid>

Sources
  runs/pallas_arena/baselines-tpu-<job>.json   regraded rewards + golden truth
  runs/pallas_arena/baselines-tpu-<job>.log    boot elections (both candidates)
  runs/pallas_arena/sd-results-3687904.jsonl   the ORIGINAL cold-start run
  runs/pallas_arena/repair-improve-3695727.jsonl  the IMPROVE run
"""
from __future__ import annotations
import collections, html, json, re, sys, pathlib

base_job, imp_job = sys.argv[1], sys.argv[2]
R = pathlib.Path("runs/pallas_arena")
esc = html.escape


def load_json(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return {}


def load_jsonl(p):
    try:
        return [json.loads(l) for l in open(p)]
    except Exception:
        return []


def elections(job):
    """Golden truth from the boot log: EVERY candidate's time per shape."""
    out = {}
    try:
        log = (R / f"baselines-tpu-{job}.log").read_text(errors="ignore")
    except Exception:
        return out
    for m in re.finditer(r"\[boot\] (\S+): baseline=(\S+) \(([^)]*)\)", log):
        case, winner, times = m.group(1), m.group(2), m.group(3)
        cand = {}
        for part in times.split(","):
            if "=" in part:
                k, v = part.strip().split("=")
                cand[k] = float(v.rstrip("ms"))
        out[case] = {"elected": winner, "candidates": cand}
    return out


base, imp = load_json(R / f"baselines-tpu-{base_job}.json"), load_json(R / f"baselines-tpu-{imp_job}.json")
gold = {**elections(base_job), **elections(imp_job)}
orig_rows = load_jsonl(R / "sd-results-3687904.jsonl")
impr_rows = load_jsonl(R / "repair-improve-3695727.jsonl")

TASKS = ["splash_attention", "ragged_paged_attention", "megablox_gmm", "rg_lru"]
NICE = {"splash_attention": "splash attention", "ragged_paged_attention": "ragged paged attention",
        "megablox_gmm": "megablox GMM", "rg_lru": "RG-LRU"}
GOOGLE = {"splash_attention": "jax splash-attention (tuned 1024 blocks)",
          "ragged_paged_attention": "jax ragged-paged-attention",
          "megablox_gmm": "jax megablox GMM (tuned tiling)",
          "rg_lru": "DeepMind recurrentgemma Pallas scan"}


def best_old(rows, task, key="reward"):
    v = [r.get(key) or 0 for r in rows if r.get("task") == task and r.get("gate") == "all"]
    return max(v) if v else None


def summarize(rep, task):
    t = (rep.get("tasks") or {}).get(task) or {}
    res = t.get("results") or {}
    graded = {k: v for k, v in res.items() if "new_reward" in v}
    if not graded:
        return None
    best_k = max(graded, key=lambda k: graded[k].get("new_reward") or 0)
    return {"n": len(res), "graded": len(graded),
            "passed": sum(1 for v in graded.values() if v.get("passed")),
            "above1": sum(1 for v in graded.values() if (v.get("new_reward") or 0) > 1.0),
            "best": graded[best_k].get("new_reward"), "best_name": best_k,
            "best_row": graded[best_k], "noise": t.get("noise_floor"),
            "device_timing": t.get("device_timing"), "golden": t.get("golden_truth") or {}}


payload = {"base_job": base_job, "improve_job": imp_job, "tasks": {}}
for task in TASKS:
    payload["tasks"][task] = {
        "orig_best": best_old(orig_rows, task),
        "improve_best": best_old(impr_rows, task),
        "base": summarize(base, task),
        "improve": summarize(imp, task),
    }
payload["gold"] = gold
print(json.dumps(payload, indent=1, default=str)[:3000])
pathlib.Path("/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/29bedbbc-f11f-4951-aab4-a75db613049f/scratchpad/results_payload.json").write_text(json.dumps(payload, default=str))
print("\npayload written")
