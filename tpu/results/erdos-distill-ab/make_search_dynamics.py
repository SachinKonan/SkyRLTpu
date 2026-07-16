#!/usr/bin/env python3
"""Figure: the base method's search dynamics — early exploration, collapse,
micro-capped optimization. Data: ttt-discover Erdos runs on neuronic."""
import json, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "/n/fs/vision-mix/sk7524/SkyRLTpu/runs"
OUT = "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/results/erdos-distill-ab/search_dynamics.png"

# --- palette (validated: all six checks pass, light mode, surface #fcfcfb) ---
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
BASE_HUE  = "#2a78d6"   # categorical slot 1 — the base method (ctrl15)
FIX_HUE   = "#008300"   # categorical slot 2 — distill + elite (distelite15)

def tables(tag, run):
    """per-step rollout records: (step, seed_c5, achieved_c5 or None)"""
    out = {}
    for s in range(20):
        fs = glob.glob(f"{BASE}/{tag}/tinker_log/{run}/wandb/run-*/files/media/table/"
                       f"gen&score_train_{s}_*.table.json")
        if not fs:
            continue
        d = json.load(open(max(fs)))
        gi = {c: i for i, c in enumerate(d["columns"])}
        recs = []
        for row in d["data"]:
            r = float(row[gi["Reward"]])
            seed = abs(float(row[gi["Initial Raw Score"]]))
            recs.append((seed, (1.0 / r) if r > 0 else None))
        out[s] = recs
    return out

def pool(tag, run):
    fs = sorted(glob.glob(f"{BASE}/{tag}/tinker_log/{run}/puct_sampler_step_*.json"))
    d = json.load(open(fs[-1]))
    return [(s["timestep"], -s["value"]) for s in d["states"]
            if s.get("value") is not None and s.get("timestep") is not None
            and (s.get("code") or "").strip()]

def improve_rate(recs_by_step):
    xs, ys = [], []
    for s, recs in sorted(recs_by_step.items()):
        ok = [(seed, c5) for seed, c5 in recs if c5 is not None]
        if not ok:
            continue
        imp = sum(1 for seed, c5 in ok if seed - c5 > 1e-4)
        xs.append(s); ys.append(100.0 * imp / len(ok))
    return xs, ys

print("loading…", flush=True)
ctrl_t = tables("ttd_gptoss20b_ctrl15", "erdos-gptoss20b-ctrl15")
delite_t = tables("ttd_gptoss20b_distelite15", "erdos-gptoss20b-distelite15")
ctrl_pool = pool("ttd_gptoss20b_ctrl15", "erdos-gptoss20b-ctrl15")

fig = plt.figure(figsize=(13.0, 8.4), facecolor=SURFACE, dpi=170)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.06], hspace=0.42, wspace=0.20,
                      left=0.072, right=0.975, top=0.845, bottom=0.085)

def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)

# ---------------- Panel A: where the search actually looked ----------------
# Every VALID rollout's own construction, as distance above the known record, log
# scale. Uses the complete rollout record (wandb tables) rather than the final
# PUCT pool — that pool is pruned to 1000 survivors, so "states by discovery
# step" would be survivor-biased. A linear C₅ axis squashes this to a smear.
RECORD = 0.380875323
axA = fig.add_subplot(gs[0, 0]); style(axA)
ax_x, ax_y = [], []
fr_x, fr_y, best = [], [], np.inf
for s, recs in sorted(ctrl_t.items()):
    for _, c5 in recs:
        if c5 is None:
            continue
        ax_x.append(s + np.random.uniform(-.2, .2)); ax_y.append(max(c5 - RECORD, 1e-9))
        best = min(best, c5 - RECORD)
    fr_x.append(s); fr_y.append(best)
axA.scatter(ax_x, ax_y, s=13, c=BASE_HUE, alpha=.30, linewidths=0, zorder=3)
axA.plot(fr_x, fr_y, color=INK, linewidth=2.0, zorder=5, solid_capstyle="round")
axA.set_yscale("log"); axA.set_ylim(4e-5, 1.4); axA.set_xlim(-0.7, 15.4)
axA.set_ylabel("C₅ above the known record  (log)", color=INK_2, fontsize=9.5)
axA.set_xlabel("training step", color=INK_2, fontsize=9.5)
axA.set_title("Rollouts concentrate onto one sliver", color=INK, fontsize=11.5,
              fontweight="bold", loc="left", pad=8)
axA.annotate("best so far (as scored in-run)", xy=(10.0, fr_y[10]), xytext=(8.0, 1.6e-3), color=INK,
             fontsize=9, fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="-", color=INK, lw=1.1, shrinkA=2, shrinkB=4))
axA.text(0.985, 0.955, "each dot = one valid construction a rollout produced",
         transform=axA.transAxes, ha="right", va="top", color=MUTED, fontsize=8.5)

# ---------------- Panel B: improve-rate collapse ----------------
axB = fig.add_subplot(gs[0, 1]); style(axB)
cx, cy = improve_rate(ctrl_t)
dx, dy = improve_rate(delite_t)
axB.plot(cx, cy, color=BASE_HUE, linewidth=2.0, marker="o", markersize=4.5,
         markeredgewidth=0, zorder=4, solid_capstyle="round")
axB.plot(dx, dy, color=FIX_HUE, linewidth=2.0, marker="o", markersize=4.5,
         markeredgewidth=0, zorder=3, solid_capstyle="round")
axB.set_ylim(-4, 104); axB.set_xlim(-0.4, 14.9)
axB.set_ylabel("rollouts that improved their seed  (%)", color=INK_2, fontsize=9.5)
axB.set_xlabel("training step", color=INK_2, fontsize=9.5)
axB.set_title("…then stops improving at all", color=INK, fontsize=11.5,
              fontweight="bold", loc="left", pad=8)
axB.annotate("+ distillation + elite slots", xy=(dx[5], dy[5]), xytext=(6.6, 56),
             color=FIX_HUE, fontsize=9, fontweight="bold", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=FIX_HUE, lw=1.1, shrinkA=3, shrinkB=4))
axB.annotate("base method (RL only)", xy=(cx[4], cy[4]), xytext=(6.6, 26),
             color=BASE_HUE, fontsize=9, fontweight="bold", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=BASE_HUE, lw=1.1, shrinkA=3, shrinkB=4))

# ---------------- Panel C: improvement magnitudes (the micro-cap) ----------------
axC = fig.add_subplot(gs[1, :]); style(axC)
mx, my = [], []
mean_x, mean_y = [], []
for s, recs in sorted(ctrl_t.items()):
    at = []
    for seed, c5 in recs:
        if c5 is None:
            continue
        d = seed - c5
        if d > 1e-6:
            mx.append(s + np.random.uniform(-.22, .22)); my.append(d); at.append(d)
    if at:
        mean_x.append(s); mean_y.append(float(np.mean(at)))
axC.axhspan(1e-2, 1.0, color=GRID, alpha=.45, zorder=1)
axC.scatter(mx, my, s=17, c=BASE_HUE, alpha=.34, linewidths=0, zorder=3)
axC.axhline(1e-2, color=MUTED, linewidth=1.0, linestyle=(0, (5, 4)), zorder=2)
axC.axhline(1e-3, color=MUTED, linewidth=1.0, linestyle=(0, (5, 4)), zorder=2)
# mean improvement per step — the descending ceiling, in ink so it reads on the cloud
axC.plot(mean_x, mean_y, color=INK, linewidth=2.0, marker="o", markersize=5.5,
         markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6, solid_capstyle="round")
axC.annotate("mean improvement, when it improves at all",
             xy=(mean_x[8], mean_y[8]), xytext=(8.4, 4.0e-3),
             color=INK, fontsize=9, fontweight="bold", ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", color=INK, lw=1.1, shrinkA=3, shrinkB=5))
axC.set_yscale("log"); axC.set_ylim(2e-7, 0.9); axC.set_xlim(-0.6, 14.9)
axC.set_ylabel("size of improvement over seed  (log)", color=INK_2, fontsize=9.5)
axC.set_xlabel("training step", color=INK_2, fontsize=9.5)
axC.set_title("Every improvement it ever made — mean gain shrinks 8,000× (6.3e⁻² → 7.6e⁻⁶)",
              color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
axC.text(14.7, 4.5e-2, "structural leaps — none after step 3", color=INK_2, fontsize=9,
         ha="right", va="center", fontweight="bold")
axC.text(0.985, 0.045, "each dot = one rollout that beat its seed",
         transform=axC.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8.5)

fig.text(0.072, 0.945, "The base method discovers early, then only polishes",
         color=INK, fontsize=16, fontweight="bold", ha="left")
fig.text(0.072, 0.905,
         "gpt-oss-20b · Erdős minimum-overlap · 512 rollouts/step · ctrl15 (RL only) unless labelled otherwise",
         color=INK_2, fontsize=10, ha="left")
fig.text(0.975, 0.906,
         "C₅ as scored in-run (self-reported, ±1e-4 — discover#19)",
         color=MUTED, fontsize=8.5, ha="right")

fig.savefig(OUT, facecolor=SURFACE)
print("wrote", OUT)
print(f"panelA valid rollouts={len(ax_x)}  panelC improving={len(mx)}")
print("frontier gap:", " ".join(f"s{s}={v:.1e}" for s,v in zip(fr_x,fr_y)))
print(f"improve-rate ctrl {cy[0]:.0f}%→{cy[-1]:.0f}%   delite {dy[0]:.0f}%→{dy[-1]:.0f}%")
print(f"max improvement={max(my):.4f}  after step 3: {max([d for x,d in zip(mx,my) if x>3.5]):.2e}")
print("mean improvement per step:", " ".join(f"s{s}={v:.2e}" for s, v in zip(mean_x, mean_y)))
