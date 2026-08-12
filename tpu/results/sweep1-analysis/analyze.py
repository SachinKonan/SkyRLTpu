#!/usr/bin/env python3
"""Aggregate sweep1 metrics + PUCT pools into a single JSON series file."""
import json, os, math, statistics as st
from collections import Counter, defaultdict

D = "/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/db8bd534-7f0f-4b20-ada3-76d5c685d92a/scratchpad/data"
OUT = "/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/db8bd534-7f0f-4b20-ada3-76d5c685d92a/scratchpad/sweep1_series.json"

# value in the pool is ALWAYS "higher is better".
# score = the problem's native quantity.
PROBLEMS = {
    "erdos": dict(
        label="Erdős minimum-overlap",
        problem="erdos_min_overlap",
        metric="C5", direction="lower",   # score = -value
        sign=-1,
        target=0.380875, target_label="authors' record C5 = 0.380875",
        note="reward = 1/C5",
        arms=["ttd", "grpo"], status="complete",
    ),
    "jssp": dict(
        label="Job-shop scheduling",
        problem="frontier_algo #46 (JSSP)",
        metric="reward", direction="higher",
        sign=1,
        target=0.165570, target_label="best observed across both arms (0.165570); reference solutions score 0.051–0.083",
        note="env reward, higher is better",
        arms=["ttd", "grpo"], status="complete",
    ),
    "acineq": dict(
        label="Andrews–Curtis inequality ac1",
        problem="ac_inequalities ac1",
        metric="value", direction="lower",
        sign=-1,
        target=1.5030, target_label="target value 1.5030",
        note="reward = 1/value",
        arms=["ttd", "grpo"], status="complete",
    ),
    "circle": dict(
        label="Circle packing",
        problem="circle_packing",
        metric="sum of radii", direction="higher",
        sign=1,
        target=2.635983, target_label="best-known packing 2.635983",
        note="retired early: ttd hit the best-known packing exactly",
        arms=["ttd", "grpo"], status="partial",
    ),
    "ud": dict(
        label="Erdős unit-distance",
        problem="frontier_erdos_ud (n=65536)",
        metric="pairs per point", direction="higher",
        sign=1,
        target=11.3516845703125, target_label="best observed across both arms (11.3516846)",
        note="stopped after 2–3 steps (~8 h/step vs ~1 h spot lifetimes)",
        arms=["ttd", "grpo"], status="partial",
    ),
}

METRIC_KEYS = [
    "env/all/reward/mean", "env/all/reward/max", "env/all/reward/min",
    "env/all/raw_score", "env/all/raw_score/max", "env/all/raw_score/min",
    "env/all/correctness", "env/all/format", "env/all/parsed_code",
    "env/all/by_group/frac_mixed", "env/all/by_group/frac_all_bad", "env/all/by_group/frac_all_good",
    "advantage/mean", "advantage/max", "advantage/min",
    "puct/buffer_size", "puct/sampled_size", "puct/T",
    "puct/buffer_value/mean", "puct/buffer_value/std", "puct/buffer_value/max", "puct/buffer_value/min",
    "puct/sampled_value/mean", "puct/sampled_value/std", "puct/sampled_value/max", "puct/sampled_value/min",
    "puct/buffer_timestep/mean", "puct/buffer_timestep/std", "puct/buffer_timestep/max", "puct/buffer_timestep/min",
    "puct/sampled_timestep/mean", "puct/sampled_timestep/std", "puct/sampled_timestep/max", "puct/sampled_timestep/min",
    "puct/buffer_construction_len/mean", "puct/sampled_construction_len/mean",
    "reject_truncated/dropped_trajs", "reject_truncated/groups_touched",
    "train/dropped_too_long",
    "time/total", "time/sampling", "time/train",
    "env/all/total_ac_tokens", "env/all/ac_tokens_per_turn",
]


def load_metrics(run):
    p = f"{D}/{run}/metrics.jsonl"
    if not os.path.exists(p):
        return None
    by_step = {}
    dup = Counter()
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        s = int(d["step"])
        dup[s] += 1
        by_step[s] = d          # last write wins (resume overwrites)
    steps = sorted(by_step)
    out = {"steps": steps,
           "duplicated_steps": sorted(k for k, v in dup.items() if v > 1),
           "missing_steps": [s for s in range(steps[0], steps[-1] + 1) if s not in by_step]}
    for k in METRIC_KEYS:
        out[k] = [by_step[s].get(k) for s in steps]
    out["advantage/spread"] = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(out["advantage/max"], out["advantage/min"])]
    return out


def load_nodes(run):
    rows = [json.loads(l) for l in open(f"{D}/{run}/nodes.jsonl")]
    seen, nodes = set(), []
    for r in rows:                       # initial_states duplicate the roots inside .states
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        nodes.append(r)
    return nodes


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = p * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def analyse_pool(run, pcfg):
    nodes = load_nodes(run)
    sign = pcfg["sign"]
    tgt = pcfg["target"]
    tgt_v = None if tgt is None else sign * tgt   # target in value space

    by_id = {n["id"]: n for n in nodes}
    roots = [n for n in nodes if n["ts"] == -1]
    gen = [n for n in nodes if n["ts"] >= 0]
    steps = sorted({n["ts"] for n in gen})

    # children map
    kids = defaultdict(list)
    for n in gen:
        if n["p"]:
            kids[n["p"]].append(n)

    # running champion (roots count as the step -1 baseline)
    champ = max((n["v"] for n in roots), default=-math.inf)
    champ_hist = {-1: champ}
    best_so_far = {}
    per_step = []
    ties_eps = 0.0
    # fixed headroom reference: the gap the run faced at the very start
    root_best = champ
    fixed_headroom = None if tgt_v is None else (tgt_v - root_best)
    for t in steps:
        cur = [n for n in gen if n["ts"] == t]
        prev_champ = champ
        deltas, pcts, hns, gaps, fhs = [], [], [], [], []
        n_imp = n_tie = n_worse = 0
        n_beat_champ = 0
        n_tie_zero = n_zero = 0
        for n in cur:
            if n["v"] == 0.0:
                n_zero += 1
            if n["pv"] is None:
                continue
            d = n["v"] - n["pv"]
            deltas.append(d)
            if abs(n["pv"]) > 1e-12:
                pcts.append(100.0 * d / abs(n["pv"]))
            if d > ties_eps:
                n_imp += 1
            elif d == 0.0:
                n_tie += 1
                if n["v"] == 0.0:
                    n_tie_zero += 1
            else:
                n_worse += 1
            if tgt_v is not None and tgt_v > n["pv"]:
                hns.append(100.0 * d / (tgt_v - n["pv"]))
            if fixed_headroom:
                fhs.append(100.0 * d / fixed_headroom)
            # distance from the champion standing at the START of this step
            gaps.append(prev_champ - n["v"])
            if n["v"] > prev_champ:
                n_beat_champ += 1
        vals = [n["v"] for n in cur]
        if vals:
            champ = max(champ, max(vals))
        champ_hist[t] = champ
        best_so_far[t] = champ

        imp_d = [d for d in deltas if d > 0]
        imp_p = [p for p in pcts if p > 0]
        imp_h = [h for h in hns if h > 0]
        imp_f = [h for h in fhs if h > 0]

        # tree shape for this generation
        depths = [n["d"] for n in cur]
        parents_used = Counter(n["p"] for n in cur if n["p"])
        ages = [t - n["pts"] for n in cur if n["pts"] is not None and n["pts"] >= 0]
        fams = len({n["root"] for n in cur if n["root"]})

        per_step.append(dict(
            step=t, n_new=len(cur),
            pool_size=sum(1 for n in gen if n["ts"] <= t) + len(roots),
            champ_value=champ, champ_score=sign * champ,
            prev_champ_score=sign * prev_champ,
            champ_improved=bool(champ > prev_champ),
            champ_gain_score=(sign * champ) - (sign * prev_champ),
            n_children=len(deltas),
            frac_improve=(n_imp / len(deltas)) if deltas else None,
            frac_tie=(n_tie / len(deltas)) if deltas else None,
            frac_tie_at_zero=(n_tie_zero / len(deltas)) if deltas else None,
            frac_zero_value=(n_zero / len(cur)) if cur else None,
            frac_worse=(n_worse / len(deltas)) if deltas else None,
            frac_beat_champ=(n_beat_champ / len(deltas)) if deltas else None,
            mean_delta=(sum(deltas) / len(deltas)) if deltas else None,
            median_delta=q(deltas, .5),
            mean_delta_improvers=(sum(imp_d) / len(imp_d)) if imp_d else None,
            max_delta=max(deltas) if deltas else None,
            mean_pct=(sum(pcts) / len(pcts)) if pcts else None,
            mean_pct_improvers=(sum(imp_p) / len(imp_p)) if imp_p else None,
            median_pct_improvers=q(imp_p, .5),
            mean_headroom_pct=(sum(hns) / len(hns)) if hns else None,
            mean_headroom_pct_improvers=(sum(imp_h) / len(imp_h)) if imp_h else None,
            median_headroom_pct_improvers=q(imp_h, .5),
            mean_fixed_headroom_pct_improvers=(sum(imp_f) / len(imp_f)) if imp_f else None,
            median_fixed_headroom_pct_improvers=q(imp_f, .5),
            p90_delta=q(deltas, .9), p10_delta=q(deltas, .1),
            mean_gap_to_champ=(sum(gaps) / len(gaps)) if gaps else None,
            median_gap_to_champ=q(gaps, .5),
            min_gap_to_champ=min(gaps) if gaps else None,
            mean_depth=(sum(depths) / len(depths)) if depths else None,
            max_depth=max(depths) if depths else None,
            n_distinct_parents=len(parents_used),
            mean_children_per_used_parent=(len(cur) / len(parents_used)) if parents_used else None,
            max_children_per_parent=max(parents_used.values()) if parents_used else None,
            n_families=fams,
            mean_parent_age=(sum(ages) / len(ages)) if ages else None,
            max_parent_age=max(ages) if ages else None,
            frac_parent_from_prev_step=(sum(1 for a in ages if a == 1) / len(ages)) if ages else None,
            best_value_this_step=max(vals) if vals else None,
            best_score_this_step=sign * max(vals) if vals else None,
            mean_value_this_step=(sum(vals) / len(vals)) if vals else None,
        ))

    # convergence.  A "material" improvement moves the champion by more than a
    # relative 1e-9 -- this filters float-noise gains (e.g. 4e-16) that would
    # otherwise count as progress.
    REL_EPS = 1e-9
    improved = [r["step"] for r in per_step if r["champ_improved"]]
    material = [r["step"] for r in per_step
                if r["champ_improved"] and abs(r["champ_gain_score"]) > REL_EPS * max(abs(r["champ_score"]), 1e-12)]
    for r in per_step:
        r["champ_improved_material"] = bool(
            r["champ_improved"] and abs(r["champ_gain_score"]) > REL_EPS * max(abs(r["champ_score"]), 1e-12))
    last_step = steps[-1] if steps else None
    conv = dict(
        last_improving_step=material[-1] if material else None,
        last_improving_step_any=improved[-1] if improved else None,
        n_improving_steps=len(material),
        n_improving_steps_any=len(improved),
        steps_since_last_improvement=(last_step - material[-1]) if material else None,
        final_step=last_step,
        improving_steps=material,
        improving_steps_any=improved,
        rel_eps=REL_EPS,
    )

    # whole-pool tree summary
    all_depths = [n["d"] for n in gen]
    kid_counts = [len(v) for v in kids.values()]
    fam_alive = {}
    for t in steps:
        fam_alive[t] = len({n["root"] for n in gen if n["ts"] == t and n["root"]})
    tree = dict(
        n_nodes=len(nodes), n_roots=len(roots), n_generated=len(gen),
        max_depth=max(all_depths) if all_depths else 0,
        mean_depth=(sum(all_depths) / len(all_depths)) if all_depths else 0,
        n_parents_with_children=len(kids),
        mean_branching=(sum(kid_counts) / len(kid_counts)) if kid_counts else 0,
        max_branching=max(kid_counts) if kid_counts else 0,
        leaf_frac=1 - (len(kids) / len(gen)) if gen else None,
        depth_hist={str(k): v for k, v in sorted(Counter(all_depths).items())},
        branching_hist={str(k): v for k, v in sorted(Counter(kid_counts).items())},
        families_total=len({n["root"] for n in gen if n["root"]}),
    )

    champ_node = max(gen, key=lambda n: n["v"]) if gen else None
    # champion lineage: walk up
    lineage = []
    cn = champ_node
    while cn is not None:
        lineage.append(dict(id=cn["id"], ts=cn["ts"], value=cn["v"], score=sign * cn["v"]))
        cn = by_id.get(cn["p"]) if cn["p"] else None
    lineage.reverse()

    # step-to-step volatility of the per-step best (a within-run noise proxy)
    bs = [r["best_score_this_step"] for r in per_step if r["best_score_this_step"] is not None]
    diffs = [abs(bs[i + 1] - bs[i]) for i in range(len(bs) - 1)]
    tail = bs[-5:]
    noise = dict(
        mean_abs_step_to_step_change=(sum(diffs) / len(diffs)) if diffs else None,
        max_abs_step_to_step_change=max(diffs) if diffs else None,
        tail5_best_scores=tail,
        tail5_std=(st.pstdev(tail) if len(tail) > 1 else None),
        tail5_range=(max(tail) - min(tail)) if tail else None,
    )
    conv["noise"] = noise

    return dict(per_step=per_step, convergence=conv, tree=tree,
                root_values=sorted(sign * n["v"] for n in roots),
                champion=dict(id=champ_node["id"], step=champ_node["ts"],
                              depth=champ_node["d"], value=champ_node["v"],
                              score=sign * champ_node["v"]) if champ_node else None,
                champion_lineage=lineage)


out = {"problems": {}, "generated": "sweep1 analysis"}
for pk, pcfg in PROBLEMS.items():
    entry = {k: v for k, v in pcfg.items() if k != "sign"}
    entry["sign"] = pcfg["sign"]
    entry["arms"] = {}
    for arm in ("ttd", "grpo"):
        run = f"{arm}-{pk}"
        if not os.path.exists(f"{D}/{run}/nodes.jsonl"):
            continue
        a = analyse_pool(run, pcfg)
        a["metrics"] = load_metrics(run)
        a["run"] = f"sweep1-{run}"
        entry["arms"][arm] = a
    out["problems"][pk] = entry

with open(OUT, "w") as f:
    json.dump(out, f)
print("wrote", OUT, os.path.getsize(OUT))

# ---- console summary ----
for pk, p in out["problems"].items():
    print(f"\n===== {p['label']}  ({p['metric']}, {p['direction']} better)  target={p['target']}")
    for arm, a in p["arms"].items():
        c = a["champion"]
        cv = a["convergence"]
        print(f"  {arm:5s} champ score={c['score']:.9g} @step {c['step']} depth {c['depth']} | "
              f"last improve step {cv['last_improving_step']} / final {cv['final_step']} "
              f"(idle {cv['steps_since_last_improvement']}) | improving steps {cv['n_improving_steps']}")
        m = a["metrics"]
        if m:
            print(f"        metrics steps {m['steps'][0]}..{m['steps'][-1]} missing={m['missing_steps']} dup={m['duplicated_steps']}")
        t = a["tree"]
        print(f"        tree: nodes={t['n_generated']} maxdepth={t['max_depth']} meanbranch={t['mean_branching']:.2f} "
              f"maxbranch={t['max_branching']} families={t['families_total']}")
        ps = a["per_step"]
        print("        step  n   f_imp f_tie  meanpct%   impPct%   headroom%   champ")
        for r in ps:
            print("        {:>4} {:>3}  {:>5} {:>5}  {:>9} {:>9} {:>10}  {:.9g}".format(
                r["step"], r["n_new"],
                "%.2f" % r["frac_improve"] if r["frac_improve"] is not None else "-",
                "%.2f" % r["frac_tie"] if r["frac_tie"] is not None else "-",
                "%.3g" % r["mean_pct"] if r["mean_pct"] is not None else "-",
                "%.3g" % r["mean_pct_improvers"] if r["mean_pct_improvers"] is not None else "-",
                "%.3g" % r["mean_headroom_pct_improvers"] if r["mean_headroom_pct_improvers"] is not None else "-",
                r["champ_score"]))
