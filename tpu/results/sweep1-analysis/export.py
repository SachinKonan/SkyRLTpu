#!/usr/bin/env python3
"""Compact the full series file into the bundle the artifact embeds."""
import json, os

SC = "/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/db8bd534-7f0f-4b20-ada3-76d5c685d92a/scratchpad"
S = json.load(open(f"{SC}/sweep1_series.json"))

POOL_FIELDS = [
    "step", "n_new", "pool_size", "champ_score", "champ_improved_material",
    "champ_gain_score", "best_score_this_step", "n_children",
    "frac_improve", "frac_tie", "frac_tie_at_zero", "frac_worse", "frac_zero_value",
    "frac_beat_champ", "mean_delta", "mean_delta_improvers", "mean_pct",
    "mean_pct_improvers", "median_pct_improvers", "mean_headroom_pct_improvers",
    "mean_fixed_headroom_pct_improvers", "mean_gap_to_champ", "median_gap_to_champ",
    "mean_depth", "max_depth", "n_distinct_parents", "mean_children_per_used_parent",
    "n_families", "mean_parent_age", "frac_parent_from_prev_step",
]
MET_FIELDS = [
    "env/all/reward/mean", "env/all/reward/max", "env/all/format", "env/all/correctness",
    "env/all/by_group/frac_mixed", "env/all/by_group/frac_all_bad",
    "advantage/mean", "advantage/max", "advantage/min", "advantage/spread",
    "puct/buffer_size", "puct/sampled_timestep/mean", "puct/buffer_timestep/mean",
    "puct/buffer_timestep/std", "puct/buffer_value/std",
    "env/all/total_ac_tokens", "time/total",
]

TECH = {
    "ttd-erdos": ("Symmetry-reduced smooth-max ladder",
                  "Optimises the half-vector h[0..n/2] (mirror symmetry assumed), scores it with a "
                  "log-sum-exp softening of maxₖ (h ⋆ (1−h)) so the true max is differentiable, and walks a "
                  "resolution ladder n = 192 → 256 → 384 → 512, interpolating the previous winner into each "
                  "new grid. Box+sum projection keeps the iterate feasible; SLSQP does the local work."),
    "grpo-erdos": ("Smooth-max with an explicit β anneal",
                   "Same symmetry reduction and box/sum projection (via a bisection on the shift t), but the "
                   "sharpening is an explicit schedule β = 5, 10, 20, 40, 80, 160, 320 with an SLSQP solve at "
                   "each β and an equality constraint on the sum. Single resolution rather than a ladder."),
    "ttd-jssp": ("Allocation-free SA on the disjunctive graph",
                 "C++ simulated annealing over machine permutations. The makespan evaluator is rewritten to do "
                 "zero allocation in the inner loop: static buffers, precomputed job↔operation lookup tables, "
                 "successors computed by O(1) index arithmetic, Kahn topological sort into a fixed queue. "
                 "Adjacent-swap moves with an adaptive cooling schedule."),
    "grpo-jssp": ("Iterated local search with critical-path moves",
                  "C++ ILS + SA seeded by a most-remaining-operations dispatch rule instead of SPT, with "
                  "critical-path-guided swaps, explicit cycle rejection, cache-friendly static edge arrays, "
                  "adaptive cooling and a light anti-oscillation memory. Restarts when a plateau is detected."),
    "ttd-acineq": ("FFT autoconvolution + peak subgradient",
                   "Evaluates 2n·max(a⋆a)/(Σa)² through an FFT autoconvolution (O(n log n)) so long "
                   "sequences are affordable, then drives the active peak index with a subgradient step. "
                   "Reached its final value at step 5 and never moved again."),
    "grpo-acineq": ("Linear-programming inner solve",
                    "Replaces heuristic descent with an LP: build the (2n−1)×n convolution matrix of the current "
                    "sequence f, then maximise Σg subject to f⋆g ≤ M, g ≥ 0 via HiGHS, and iterate that map with a "
                    "decaying trust parameter. The only champion in the sweep still improving at step 14."),
    "ttd-circle": ("Structured seed + SLSQP polish",
                   "Seeds a corner/edge/interior decomposition (four large corner circles, four edge circles, a "
                   "hexagonal interior patch) and polishes centres and radii jointly with SLSQP under explicit "
                   "non-overlap and containment constraints. Matches the best-known sum of radii exactly."),
    "grpo-circle": ("Same family, fewer steps",
                    "Structured initial layout plus constrained local polish; the run was stopped at step 3 and "
                    "never reached the best-known packing."),
    "ttd-ud": ("Eisenstein lattice with a searched modulus",
               "Places 256×256 points on the triangular (Eisenstein) lattice and scales by 1/√k so that lattice "
               "vectors of norm k become unit distance. Brute-forces k ≤ 10000 to maximise the number of "
               "representations of the form x²+xy+y² = k. Found at the first generation."),
    "grpo-ud": ("Eisenstein lattice, k = 91 argued in closed form",
                "Same construction, but k = 7·13 = 91 is chosen analytically: both primes are 1 mod 3, so "
                "r′(91) = 6·(1+1)(1+1) = 24 representations, and the boundary strip is √91 ≈ 9.5 wide instead of "
                "√325 ≈ 18 for the square lattice. Same score as ttd, reached two steps later."),
}

out = {"problems": {}, "tech": {k: {"title": v[0], "body": v[1]} for k, v in TECH.items()}}
for pk, p in S["problems"].items():
    e = {k: p[k] for k in ("label", "problem", "metric", "direction", "target",
                           "target_label", "note", "status", "sign")}
    e["arms"] = {}
    for arm, a in p["arms"].items():
        rec = {
            "run": a["run"],
            "champion": a["champion"],
            "convergence": a["convergence"],
            "tree": {k: a["tree"][k] for k in ("n_generated", "max_depth", "mean_branching",
                                               "max_branching", "families_total",
                                               "n_parents_with_children", "leaf_frac",
                                               "branching_hist")},
            "root_best": max(a["root_values"]) if p["sign"] > 0 else min(a["root_values"]),
            "pool": {f: [r.get(f) for r in a["per_step"]] for f in POOL_FIELDS},
        }
        m = a["metrics"]
        if m:
            rec["metrics"] = {"steps": m["steps"], "missing": m["missing_steps"],
                              "dup": m["duplicated_steps"]}
            rec["metrics"].update({f: m[f] for f in MET_FIELDS if f in m})
        else:
            rec["metrics"] = None
        e["arms"][arm] = rec
    out["problems"][pk] = e

dest = f"{SC}/bundle.json"
json.dump(out, open(dest, "w"), separators=(",", ":"))
print("wrote", dest, os.path.getsize(dest), "bytes")
