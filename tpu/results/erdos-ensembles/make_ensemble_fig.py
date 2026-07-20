#!/usr/bin/env python3
"""Ensemble figure: frontier trajectories + family-diversity (the exploration story)."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/results/erdos-ensembles/ensemble_dynamics.png"
SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; AXIS="#c3c2b7"
# 3 categorical slots (validated order): blue, green, magenta
C = {"20b+20b":"#2a78d6", "20b+120b":"#008300", "nemotron+gptoss":"#e87ba4"}
RUNS = {
 "20b+20b":"/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_ens_20b20b/tinker_log/erdos-ens-20b20b",
 "20b+120b":"/n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_ens_20b120b/tinker_log/erdos-ens-20b120b",
 "nemotron+gptoss":"/n/fs/vision-mix/sk7524/SkyRLTpu-ensemble/runs/ttd_ensemble15/tinker_log/erdos-ensemble15",
}
RECORD=0.380875323

def canon(h,m=256):
    h=np.asarray(h,dtype=np.float64);x=np.linspace(0,1,len(h));xi=np.linspace(0,1,m)
    v=np.interp(xi,x,h);v=v/max(v.sum(),1e-9)*(m/2);r=v[::-1]
    return v if tuple(v)<=tuple(r) else r
def vf(h):
    h=np.asarray(h,dtype=np.float64);n=len(h);t=n/2.0
    h2=h*(t/h.sum()) if h.sum()!=t else h
    return float(np.max(np.correlate(h2,1.0-h2,mode="full")*(2.0/n)))
def nfam(vecs,eps=0.03):
    reps=[]
    for c in vecs:
        if not any(float(np.sqrt(np.mean((c-r)**2)))<eps for r in reps): reps.append(c)
    return len(reps)

fig=plt.figure(figsize=(12.4,4.9),facecolor=SURFACE,dpi=175)
gs=fig.add_gridspec(1,2,wspace=0.24,left=0.075,right=0.98,top=0.80,bottom=0.14)
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(AXIS);ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED,labelsize=9,length=3); ax.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)

# Panel 1: frontier (best-so-far c5) by step
ax1=fig.add_subplot(gs[0,0]); style(ax1)
for name,D in RUNS.items():
    fin=json.load(open(sorted(glob.glob(f"{D}/puct_sampler_step_*.json"))[-1]))
    allst=[s for s in fin["states"] if s.get("value") is not None and s.get("construction")]
    best=-9; xs=[]; ys=[]
    for k in range(15):
        at=[s for s in allst if s.get("timestep")==k]
        for s in at: best=max(best,s["value"])
        if best>-9: xs.append(k); ys.append(-best - RECORD)
    ax1.plot(xs,ys,color=C[name],lw=2.2,marker="o",ms=4,markeredgewidth=0,solid_capstyle="round")
ax1.axhline(0.380928-RECORD,color=MUTED,lw=1,ls=(0,(4,3)))
ax1.text(14.3,0.380928-RECORD,"  solo distelite\n  20b (0.380925)",color=MUTED,fontsize=8,va="center")
ax1.set_yscale("log"); ax1.set_xlim(-0.4,17.6); ax1.set_ylim(4e-5,8e-4)
ax1.set_ylabel("best C₅ above the record  (log)",color=INK2,fontsize=9.5)
ax1.set_xlabel("training step",color=INK2,fontsize=9.5)
ax1.set_title("Frontier: 20b+20b keeps descending; 20b+120b plateaus at step 9",
              color=INK,fontsize=10.5,fontweight="bold",loc="left",pad=7)
ax1.annotate("20b+20b",xy=(13,0.380944-RECORD),xytext=(9.5,1.4e-4),color=C["20b+20b"],
             fontsize=9,fontweight="bold",arrowprops=dict(arrowstyle="-",color=C["20b+20b"],lw=1))
ax1.annotate("20b+120b",xy=(11,0.380953-RECORD),xytext=(11.4,2.6e-4),color=C["20b+120b"],
             fontsize=9,fontweight="bold",ha="left")
ax1.annotate("nemotron+gptoss",xy=(12,0.381013-RECORD),xytext=(6.6,4.3e-4),color=C["nemotron+gptoss"],
             fontsize=9,fontweight="bold",ha="left")

# Panel 2: distinct families discovered (running, in top pool) by step
ax2=fig.add_subplot(gs[0,1]); style(ax2)
for name,D in RUNS.items():
    xs=[]; ys=[]
    for k in range(1,15,1):
        try: d=json.load(open(f"{D}/puct_sampler_step_{k:06d}.json"))
        except FileNotFoundError: continue
        st=sorted((s for s in d["states"] if s.get("value") is not None and s.get("construction")),
                  key=lambda s:-s["value"])[:120]
        xs.append(k); ys.append(nfam([canon(s["construction"]) for s in st]))
    ax2.plot(xs,ys,color=C[name],lw=2.2,marker="o",ms=4,markeredgewidth=0,solid_capstyle="round")
ax2.set_xlim(0.4,14.6); ax2.set_ylim(0,None)
ax2.set_ylabel("distinct construction families\n(in the top ~120 states)",color=INK2,fontsize=9.5)
ax2.set_xlabel("training step",color=INK2,fontsize=9.5)
ax2.set_title("Diversity: fewer families discovered with a 120B in the pool",
              color=INK,fontsize=10.5,fontweight="bold",loc="left",pad=7)
ax2.text(0.985,0.05,"more families = wider exploration = more chances at a low-floored basin",
         transform=ax2.transAxes,ha="right",va="bottom",color=MUTED,fontsize=8)

fig.text(0.075,0.945,"Shared-pool ensembles: the win comes from exploration diversity, not the bigger model",
         color=INK,fontsize=13.5,fontweight="bold",ha="left")
fig.text(0.075,0.885,"ttt-discover · Erdős minimum-overlap · 15 steps · shared PUCT pool + symmetric cross-model distillation (β=0.1) + elite slots  ·  C₅ as scored in-run",
         color=INK2,fontsize=8.8,ha="left")
fig.savefig(OUT,facecolor=SURFACE); print("wrote",OUT)
