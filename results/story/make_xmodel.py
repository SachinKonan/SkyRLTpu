"""Cross-model comparison figures for the walkthrough's Cross-model tab.
Reads xmodel_data.json (built from run metrics.jsonl + trajectory archives +
the 120b final tree snapshot). Produces figx1..figx6.png."""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open(os.path.join(os.path.dirname(__file__), "xmodel_data.json")))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
MC = {"full120b": "#8659c9", "lrn": "#d94f2b", "ggn": "#1a7f6b", "mgn": "#b8860b"}
ML = {"full120b": "gpt-oss-120b (entropic 64x8)", "lrn": "qwen (GRPO 1.5e-4)",
      "ggn": "gemma (GRPO 4e-5)", "mgn": "muse (GRPO 4e-5)"}
PCOL = {"agn":"#d94f2b","atn":"#d94f2b","ggn":"#1a7f6b","gtn":"#1a7f6b",
        "mgn":"#b8860b","mtn":"#b8860b","objg":"#8659c9","objt":"#8659c9",
        "lrn":"#d94f2b","full120b":"#8659c9","ctrl15":"#8659c9","a8x64c":"#8659c9"}
CAT = {"fail": "#c0392b", "reg": "#e67e22", "noop": "#95a5a6", "imp": "#27ae60"}

def ser(tag, key):
    return [s[key] for s in D[tag]["series"]]

# figx1: flagship metric grid (continuous step index)
fig, ax = plt.subplots(2, 3, figsize=(11.6, 5.6))
panels = [("best", "best C5 so far (lower better)"), ("rmean", "mean reward"),
          ("fmt", "format rate"), ("corr", "correctness rate"),
          ("amax", "advantage max (log)"), ("aspread", "advantage spread max-min (log)")]
X1 = dict(MC); X1["a8x64c"] = "#8659c9"   # 120b 8x64 entropic, dashed
X1L = dict(ML); X1L["a8x64c"] = "gpt-oss-120b (8x64 = TTT-Discover cfg)"
for i, (key, title) in enumerate(panels):
    a = ax[i // 3][i % 3]
    for tag in X1:
        if key == "aspread":
            y = [(s["amax"] - s["amin"]) if s["amax"] is not None else None for s in D[tag]["series"]]
        else:
            y = ser(tag, key)
        x = list(range(len(y)))
        a.plot(x, y, color=X1[tag], lw=1.6, ls="--" if tag == "a8x64c" else "-", label=X1L[tag])
    a.set_title(title)
    if key in ("amax", "aspread"): a.set_yscale("log")
    if key == "best": a.ticklabel_format(useOffset=False, style="plain"); a.set_ylim(0.38084, 0.38115)
    a.set_xlabel("step index")
ax[0][0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figx1.png", bbox_inches="tight"); plt.close(fig)

# figx2: category stacks (archived steps; 120b = committed lineage, biased, labeled)
fig, ax = plt.subplots(1, 4, figsize=(11.6, 2.9))
for i, tag in enumerate(["lrn", "ggn", "mgn"]):
    cats = {int(k): v for k, v in D[tag]["cats"].items()}
    ks = sorted(cats)
    bottom = np.zeros(len(ks))
    for c in ("fail", "reg", "noop", "imp"):
        v = np.array([cats[k][c] / cats[k]["n"] for k in ks])
        ax[i].bar(range(len(ks)), v, bottom=bottom, color=CAT[c], width=0.85, label=c)
        bottom += v
    ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels(ks, fontsize=7)
    ax[i].set_title(ML[tag] + f"\n(archived steps)")
    ax[i].set_ylim(0, 1)
L = {int(k): v for k, v in D["full120b"]["lineage"].items()}
ks = sorted(L)
bottom = np.zeros(len(ks))
for c in ("reg", "noop", "imp"):
    v = np.array([L[k][c] / L[k]["n"] for k in ks])
    ax[3].bar(ks, v, bottom=bottom, color=CAT[c], width=0.85)
    bottom += v
ax[3].set_title("gpt-oss-120b\nCOMMITTED tree states only")
ax[3].set_ylim(0, 1)
ax[0].legend(fontsize=7, frameon=False, ncol=2)
ax[0].set_ylabel("fraction of rollouts")
fig.tight_layout(); fig.savefig("figx2.png", bbox_inches="tight"); plt.close(fig)

# figx2b: companion stacks — the TTD/entropic twins that have archives
EL = {"tlr": "qwen entropic 1.5e-4\n(same LR as lr-n)", "gtlr": "gemma entropic 1.5e-4\n(2 steps so far)",
      "mttd": "muse TTD 1.5e-4\n(vs GRPO at 4e-5)"}
ECOL = {"tlr": "#d94f2b", "gtlr": "#1a7f6b", "mttd": "#b8860b"}
fig, ax = plt.subplots(1, 3, figsize=(11.6, 2.9))
for i, tag in enumerate(["tlr", "gtlr", "mttd"]):
    cats = {int(k): v for k, v in D[tag]["cats"].items()}
    ks = sorted(cats)
    bottom = np.zeros(len(ks))
    for c in ("fail", "reg", "noop", "imp"):
        v = np.array([cats[k][c] / cats[k]["n"] for k in ks])
        ax[i].bar(range(len(ks)), v, bottom=bottom, color=CAT[c], width=0.85, label=c)
        bottom += v
    ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels(ks, fontsize=7)
    ax[i].set_title(EL[tag]); ax[i].set_ylim(0, 1)
ax[0].legend(fontsize=7, frameon=False, ncol=2)
ax[0].set_ylabel("fraction of rollouts")
fig.tight_layout(); fig.savefig("figx2b.png", bbox_inches="tight"); plt.close(fig)

# figx3: delta-distribution heatmaps (signed log-magnitude bins per step)
# display order bottom->top: regress big..tiny, no-op, improve tiny..big
BINS = [("-", e) for e in range(-1, -11, -1)] + ["0"] + [("+", e) for e in range(-10, 0)]
def binkey(b):  # data key: sign char + str(exponent), e.g. "+-8", "--3", "0"
    return "0" if b == "0" else b[0] + str(b[1])
fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.4), sharey=True)
for i, tag in enumerate(["lrn", "ggn", "mgn"]):
    cats = {int(k): v for k, v in D[tag]["cats"].items()}
    ks = sorted(cats)
    M = np.zeros((len(BINS), len(ks)))
    for j, k in enumerate(ks):
        h = cats[k]["hist"]; nvalid = max(1, cats[k]["n"] - cats[k]["fail"])
        for bi, b in enumerate(BINS):
            M[bi, j] = h.get(binkey(b), 0) / nvalid
    im = ax[i].imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=0.7)
    ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels(ks, fontsize=7)
    ax[i].set_title(ML[tag]); ax[i].set_xlabel("step")
ax[0].set_yticks(range(len(BINS)))
ax[0].set_yticklabels(["regress 1e-1","1e-2","1e-3","1e-4","1e-5","1e-6","1e-7","1e-8","1e-9","1e-10",
                       "exact no-op","improve 1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1"], fontsize=6.5)
fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of valid rollouts")
fig.savefig("figx3.png", bbox_inches="tight"); plt.close(fig)

# figx3b: companion heatmaps
fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.4), sharey=True)
for i, tag in enumerate(["tlr", "gtlr", "mttd"]):
    cats = {int(k): v for k, v in D[tag]["cats"].items()}
    ks = sorted(cats)
    M = np.zeros((len(BINS), len(ks)))
    for j, k in enumerate(ks):
        h = cats[k]["hist"]; nvalid = max(1, cats[k]["n"] - cats[k]["fail"])
        for bi, b in enumerate(BINS):
            M[bi, j] = h.get(binkey(b), 0) / nvalid
    im = ax[i].imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=0.7)
    ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels(ks, fontsize=7)
    ax[i].set_title(EL[tag].split("\n")[0]); ax[i].set_xlabel("step")
ax[0].set_yticks(range(len(BINS)))
ax[0].set_yticklabels(["regress 1e-1","1e-2","1e-3","1e-4","1e-5","1e-6","1e-7","1e-8","1e-9","1e-10",
                       "exact no-op","improve 1e-10","1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3","1e-2","1e-1"], fontsize=6.5)
fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of valid rollouts")
fig.savefig("figx3b.png", bbox_inches="tight"); plt.close(fig)

# figx4: estimator pairs per model
PAIRS = [("qwen", "agn", "atn"), ("gemma", "ggn", "gtn"),
         ("muse (LR-confounded)", "mgn", "mtn"), ("gpt-oss-20b 16x32", "objg", "objt")]
fig, ax = plt.subplots(1, 4, figsize=(11.6, 2.9), sharey=True)
for i, (name, gt, tt) in enumerate(PAIRS):
    for tag, style, lab in ((gt, "-", "GRPO"), (tt, "--", "TTD/entropic")):
        y = ser(tag, "best"); ax[i].plot(range(len(y)), y, style, color=PCOL[tag], lw=1.7, label=lab)
        ax[i].annotate(f"{y[-1]:.9f}", (len(y) - 1, y[-1]), fontsize=6.5, xytext=(-62, -8 if style == "-" else 5),
                       textcoords="offset points")
    ax[i].set_title(name); ax[i].legend(fontsize=7, frameon=False)
    ax[i].ticklabel_format(useOffset=False, style="plain")
ax[0].set_ylim(0.38084, 0.38116); ax[0].set_ylabel("best C5 so far")
fig.tight_layout(); fig.savefig("figx4.png", bbox_inches="tight"); plt.close(fig)

# figx5: the interaction — initial correctness vs GRPO edge; correctness repair vs decay
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.6))
pts = []
for name, gt, tt in PAIRS:
    c0 = (D[gt]["series"][0]["corr"] + D[tt]["series"][0]["corr"]) / 2
    edge = D[tt]["series"][-1]["best"] - D[gt]["series"][-1]["best"]  # >0: GRPO better
    pts.append((c0, edge, name))
    a1.scatter(c0, edge * 1e5, s=70, color=PCOL[gt], zorder=3)
    a1.annotate(name.split(" ")[0], (c0, edge * 1e5), xytext=(6, 4), textcoords="offset points", fontsize=8)
a1.axhline(0, color="#999", lw=0.8)
a1.set_xlabel("initial correctness (pair mean, step 0)")
a1.set_ylabel("GRPO edge over TTD, x1e-5 C5 (>0 = GRPO wins)")
a1.set_title("Does starting validity predict the winning estimator?")
for name, gt, tt in PAIRS:
    cg, ct = ser(gt, "corr"), ser(tt, "corr")
    col = PCOL[gt]
    a2.plot(range(len(cg)), cg, "-", color=col, lw=1.6)
    a2.plot(range(len(ct)), ct, "--", color=col, lw=1.6)
a2.set_title("correctness over training: GRPO (solid) repairs, TTD (dashed) drifts")
a2.set_xlabel("step index"); a2.set_ylabel("correctness rate"); a2.set_ylim(0, 1)
fig.tight_layout(); fig.savefig("figx5.png", bbox_inches="tight"); plt.close(fig)

# figx6: batch-shape contrast, entropic no-CE
SH = [("objt", "20b 16x32", "#2c7fb8", "-"), ("ctrl15", "20b 64x8", "#2c7fb8", "--"),
      ("full120b", "120b 64x8", "#8659c9", "--"), ("a8x64c", "120b 8x64 = TTT-Discover cfg (elite2)", "#8659c9", "-")]
fig, a = plt.subplots(figsize=(6.4, 3.6))
for tag, lab, col, sty in SH:
    y = ser(tag, "best"); a.plot(range(len(y)), y, sty, color=col, lw=1.7, label=f"{lab}  {y[-1]:.9f}")
a.legend(fontsize=8, frameon=False)
a.ticklabel_format(useOffset=False, style="plain")
a.set_ylim(0.38084, 0.38120)
a.set_xlabel("step index"); a.set_ylabel("best C5 so far")
a.set_title("Entropic, no CE distillation: batch shape and scale")
fig.tight_layout(); fig.savefig("figx6.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
