"""Erdos C5: our two best arms against the three verified published baselines.

C5 is minimized and the whole field lives inside 6e-5, so absolute bars would be
five identical rectangles. Both panels plot distance above the best result; the
right panel is the same quantity at 1000x finer scale, because the gap between
our two arms (4.1e-8) is 1/1500th of the gap to the nearest baseline and would
otherwise be a hairline.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

QWEN, GEMMA, BASE = "#14867c", "#c2410c", "#94a3b8"
BEST = 0.38086159053056806

rows = [
    ("GRPO-Qwen-\nthen-Gemma", 0.38086159053056806, GEMMA, "ours"),
    ("GRPO-Qwen",              0.3808616312089098, QWEN,  "ours"),
    ("SimpleTES",              0.3808686,          BASE,  "published"),
    ("TTT-Discover",           0.380875,           BASE,  "published"),
    ("AlphaEvolve",            0.380924,           BASE,  "published"),
]

plt.rcParams.update({"font.size": 12})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                               gridspec_kw={"width_ratios": [2.05, 1]})
fig.patch.set_facecolor("white")

# ---- left: the whole field, x1e-5 ---------------------------------------
# Our two bars round to the same 7 decimals and are ~0 tall here, so they get a
# floor height to stay visible as colour and are labelled at full precision.
FLOOR = 0.075
xs = range(len(rows))
for x, (name, v, c, kind) in zip(xs, rows):
    h = (v - BEST) * 1e5
    axL.bar(x, max(h, FLOOR), width=0.66, color=c, zorder=3,
            alpha=1.0 if kind == "ours" else 0.85)
    lab = f"{v:.10f}" if kind == "ours" else f"{v:.7f}".rstrip("0")
    axL.text(x, max(h, FLOOR) + 0.12, lab, ha="center", fontsize=9.5)
axL.annotate("", xy=(-0.32, 0.60), xytext=(1.32, 0.60),
             arrowprops=dict(arrowstyle="<->", color="#999", lw=1.1))
axL.text(0.5, 0.68, "both ours — 4.1×10⁻⁸ apart, see right panel",
         ha="center", fontsize=9.5, style="italic", color="#555")
axL.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=GEMMA, label="ours — tree swapped to gemma"),
    plt.Rectangle((0, 0), 1, 1, color=QWEN, label="ours — qwen throughout"),
    plt.Rectangle((0, 0), 1, 1, color=BASE, alpha=0.85, label="published baseline"),
], loc="upper left", frameon=False, fontsize=10)
axL.set_xticks(list(xs))
axL.set_xticklabels([r[0] for r in rows], fontsize=10.5)
axL.tick_params(axis="x", length=0)
axL.set_ylim(0, 7.0)
axL.set_ylabel("distance above best, ×10⁻⁵   (lower is better)")
axL.set_title("Erdős min-overlap (C₅) — our arms vs verified baselines", fontsize=13.5)
axL.grid(axis="y", alpha=0.25, zorder=0); axL.margins(x=0.06)

# ---- right: our two arms, x1e-8 -----------------------------------------
sub = rows[:2]
for x, (name, v, c, _) in zip(range(len(sub)), sub):
    h = (v - BEST) * 1e8
    axR.bar(x, max(h, 0.055), width=0.5, color=c, zorder=3)
    axR.text(x, max(h, 0.055) + 0.10, f"{v:.13f}", ha="center", fontsize=10)
axR.set_xticks([0, 1]); axR.set_xticklabels([r[0] for r in sub], fontsize=10.5)
axR.tick_params(axis="x", length=0)
axR.set_ylim(0, 5.6)
axR.set_ylabel("distance above best, ×10⁻⁸")
axR.set_title("the same two bars, 1000× finer", fontsize=13.5)
axR.grid(axis="y", alpha=0.25, zorder=0); axR.margins(x=0.18)
axR.annotate("4.1×10⁻⁸", xy=(1, 4.07), xytext=(0.42, 4.9),
             fontsize=10.5, color="#333",
             arrowprops=dict(arrowstyle="->", color="#777", lw=1.1))

fig.suptitle("The best verified C₅ moved between two of our own arms — "
             "the baseline standing did not change", fontsize=14.5, y=0.985)
fig.text(0.5, 0.012,
         "Left panel: our two bars sit at 0 and 4×10⁻⁸ on a 10⁻⁵ axis and are drawn at a fixed minimum "
         "height so their colour is visible — their real heights are the right panel. "
         "Both recomputed independently (sum(h)=n/2, agreement to ~1e-15). "
         "Baselines: SimpleTES and TTT-Discover are gpt-oss-120b at 50 steps; AlphaEvolve as published.",
         ha="center", fontsize=9.5, color="#555")
fig.tight_layout(rect=[0, 0.035, 1, 0.955])
out = "/n/fs/vision-mix/sk7524/SkyRLTpu-league/results/erdos-records/record_standing.png"
fig.savefig(out, dpi=140, facecolor="white")
print("saved", out)
