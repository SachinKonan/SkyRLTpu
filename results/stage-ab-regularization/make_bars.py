"""Stage A (regularization) + Stage B (base model) bar charts, matplotlib."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

S = os.path.dirname(os.path.abspath(__file__))
CN, CK, CR, CB = "#14867c", "#d99a26", "#8b5cf6", "#94a3b8"
CG = "#c2410c"          # gemma
RECORD = 0.3808616312089098

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 3, figsize=(22, 7.2))
ax1, ax2, ax3 = axes
fig.patch.set_facecolor("white")


def group_plot(ax, groups, ylim, ylabel, title, fmt, label_dy, group_dy):
    x = 0
    ticks, ticklabs, centers = [], [], []
    for gname, bars in groups:
        x0 = x
        for name, v, c, lab in bars:
            ax.bar(x, max(v, ylim[1] * 0.004), width=0.72, color=c, zorder=3)
            ax.text(x, max(v, 0) + label_dy, lab, ha="center", fontsize=8.5)
            ticks.append(x); ticklabs.append(name); x += 1
        centers.append(((x0 + x - 1) / 2, gname)); x += 0.75
    for gc, gname in centers:
        ax.text(gc, group_dy, gname, ha="center", fontsize=12.5, fontweight="bold")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabs, fontsize=8.5, rotation=16, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.25, zorder=0); ax.margins(x=0.02)


# ---- Erdős: distance above the record, x1e-5 (lower better) ---------------
erdos = [
    ("qwen GRPO", [("N", 0.0, CN, "0.3808616\n★ record"),
                   ("K", None, CK, ""),
                   ("R", 0.880, CR, "0.3808704")]),
    ("qwen TTD", [("N", 0.627, CN, "0.3808679"),
                  ("K", 3.007, CK, "0.3808917"),
                  ("R", 4.563, CR, "0.3809073")]),
    ("gemma", [("GRPO-N", 5.592, CG, "0.3809175\ncorrected"),
               ("TTD-N", 0.156, CG, "0.3808632\n2nd overall")]),
    ("published", [("SimpleTES", 0.697, CB, "0.3808686"),
                   ("TTT-Disc", 1.337, CB, "0.380875"),
                   ("AlphaEvolve", 6.237, CB, "0.380924")]),
]
x = 0
ticks, ticklabs, centers = [], [], []
for gname, bars in erdos:
    x0 = x
    for name, v, c, lab in bars:
        if v is None:
            ax1.bar(x, 6.8, width=0.72, color="none", edgecolor="#999",
                    ls="--", lw=1.2, hatch="//", alpha=0.5, zorder=3)
            ax1.text(x, 3.4, "did not\nfit (HBM)", ha="center", va="center",
                     fontsize=8.5, color="#555")
        else:
            ax1.bar(x, max(v, 0.06), width=0.72, color=c, zorder=3)
            ax1.text(x, max(v, 0.06) + 0.1, lab, ha="center", fontsize=8.5)
        ticks.append(x); ticklabs.append(name); x += 1
    centers.append(((x0 + x - 1) / 2, gname)); x += 0.75
for gc, gname in centers:
    ax1.text(gc, -1.15, gname, ha="center", fontsize=12.5, fontweight="bold")
ax1.set_xticks(ticks); ax1.set_xticklabels(ticklabs, fontsize=8.5, rotation=16, ha="right")
ax1.tick_params(axis="x", length=0); ax1.set_ylim(0, 7.4)
ax1.set_ylabel("distance above our record, ×10⁻⁵\n(lower is better)")
ax1.set_title("Erdős min-overlap (C₅)", fontsize=14)
ax1.grid(axis="y", alpha=0.25, zorder=0); ax1.margins(x=0.02)

# ---- JSSP (higher better) --------------------------------------------------
group_plot(ax2, [
    ("qwen GRPO", [("N", 0.1501, CN, "0.1501"), ("K", 0.1627, CK, "0.1627"),
                   ("R", 0.1819, CR, "0.1819")]),
    ("qwen TTD", [("N", 0.1635, CN, "0.1635"), ("K", 0.1357, CK, "0.1357"),
                  ("R", 0.1538, CR, "0.1538")]),
    ("gemma", [("GRPO-N", 0.2274, CG, "0.2274\n★ best"),
               ("TTD-N", 0.2230, CG, "0.2230")]),
], (0, 0.265), "best score (higher is better)", "JSSP (frontier_algo 46)",
   None, 0.005, -0.036)

# ---- ac1 (higher better; values negative) ---------------------------------
group_plot(ax3, [
    ("qwen", [("GRPO-N", 0.0062, CN, "-1.5138\n11/15"),
              ("TTD-N", 0.0134, CN, "-1.5066\n13/15")]),
    ("gemma", [("GRPO-N", 0.0145, CG, "-1.5055\n12/15"),
               ("TTD-N", 0.0127, CG, "-1.5073\n✓ 15/15")]),
], (0, 0.021), "best score, offset from −1.52\n(higher is better)",
   "ac_inequalities  (in progress)", None, 0.0006, -0.0029)

legend = [
    Patch(color=CN, label="N — control: no regularizer"),
    Patch(color=CK, label="K — KL penalty 0.1 toward base"),
    Patch(color=CR, label="R — fresh-weights restart when gain < 1% of peak"),
    Patch(color=CG, label="gemma-4-31B (N arms only; ctx 10240 vs qwen 18432)"),
    Patch(color=CB, label="published baselines (Erdős): SimpleTES / TTT-Discover = gpt-oss-120b, 50 steps"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Regularization (N/K/R) and base model — 15 steps per cell, every banked step clean",
             fontsize=15.5, y=0.99)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
png = f"{S}/nkr_bars.png"
fig.savefig(png, dpi=140, bbox_inches="tight")
print("saved", png, os.path.getsize(png) // 1024, "KB")
