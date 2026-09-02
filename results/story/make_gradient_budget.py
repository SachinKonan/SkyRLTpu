"""figx10/figx11: usable rollouts and effective gradient mass per step, GRPO vs TTD, across models.
Inputs (scratchpad dir as argv[1]): <tag>_metrics.jsonl (validity every step, every run),
gradmass.json (gpt-oss runs: APPLIED advantages from wandb gen tables),
league_ess.json (league runs: advantages recomputed from trajectory archives)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = sys.argv[1]
def metrics(tag):
    f = f"{S}/{tag}_metrics.jsonl"
    if not os.path.exists(f): return []
    rows = [json.loads(l) for l in open(f)]
    out = []
    for d in rows:
        k = [x for x in d if x.endswith("env/all/correctness") and "/max" not in x and "/min" not in x]
        out.append(d[k[0]] * 512 if k else None)
    return out

# Panels: (title, [(tag, label, color, ls)])
PAIRS = [
 ("qwen 4e-5 (stage A)",  [("agn","GRPO","#d94f2b","-"),("atn","TTD","#d94f2b","--")]),
 ("qwen 1.5e-4",          [("lrn","GRPO","#d94f2b","-"),("tlr","entropic","#d94f2b","--")]),
 ("gemma 4e-5",           [("ggn","GRPO","#1a7f6b","-"),("gtn","TTD","#1a7f6b","--")]),
 ("muse (GRPO 4e-5 / TTD 1.5e-4)", [("mgn","GRPO","#b8860b","-"),("mtn","TTD","#b8860b","--")]),
 ("gpt-oss-20b 16x32 elite2", [("obj_grpo_16x32","GRPO","#8659c9","-"),("obj_ttt_16x32","entropic","#8659c9","--")]),
 ("gpt-oss entropic, other shapes", [("ctrl15","20b 64x8 e0","#2c7fb8","--"),("full120b","120b 64x8 e0","#8659c9","--"),
                                     ("a8x64c","120b 8x64 e2","#8659c9",":")]),
]
fig, ax = plt.subplots(2, 3, figsize=(13.2, 6.2))
for i, (title, runs) in enumerate(PAIRS):
    a = ax[i // 3][i % 3]
    for tag, lab, col, ls in runs:
        v = metrics(tag)
        if v: a.plot(range(len(v)), v, ls, color=col, lw=1.7, label=f"{lab}  (end {v[-1]:.0f})")
    a.set_title(title); a.set_ylim(0, 512); a.axhline(256, color="#bbb", lw=0.6)
    a.legend(fontsize=7, frameon=False); a.set_xlabel("step")
ax[0][0].set_ylabel("valid rollouts / step (of 512)"); ax[1][0].set_ylabel("valid rollouts / step (of 512)")
fig.suptitle("Usable rollouts per step: GRPO (solid) vs TTD/entropic (dashed)", fontsize=11)
fig.tight_layout(); fig.savefig("figx10.png", bbox_inches="tight"); plt.close(fig)

# figx11: where the gradient mass actually goes
gm = json.load(open(f"{S}/gradmass.json")); le = json.load(open(f"{S}/league_ess.json"))
def series(src, tag, key):
    d = src.get(tag, {}); ks = sorted(d, key=int)
    return [int(k) for k in ks], [d[k][key] for k in ks]
SETS = [
 ("effective sample size of |advantage| (of 512)", "ess"),
 ("share of POSITIVE advantage mass on the top 8 rollouts", "top8_share"),
 ("rollouts with positive advantage", "n_pos"),
]
LINES = [("objg","gm","20b GRPO 16x32","#8659c9","-"),("objt","gm","20b entropic 16x32","#8659c9","--"),
         ("full120b","gm","120b entropic 64x8","#5b3fa0","--"),("a8x64c","gm","120b entropic 8x64","#5b3fa0",":"),
         ("lrn","le","qwen GRPO 1.5e-4","#d94f2b","-"),("tlr","le","qwen entropic 1.5e-4","#d94f2b","--"),
         ("ggn","le","gemma GRPO","#1a7f6b","-"),("gtlr","le","gemma entropic","#1a7f6b","--"),
         ("mgn","le","muse GRPO","#b8860b","-"),("mttd","le","muse TTD","#b8860b","--")]
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
for i, (title, key) in enumerate(SETS):
    for tag, src, lab, col, ls in LINES:
        x, y = series(gm if src == "gm" else le, tag, key)
        if x: ax[i].plot(x, y, ls, color=col, lw=1.5, label=lab)
    ax[i].set_title(title, fontsize=9); ax[i].set_xlabel("step")
    if key == "top8_share": ax[i].set_ylim(0, 1)
ax[0].legend(fontsize=6.5, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig("figx11.png", bbox_inches="tight"); plt.close(fig)

# console summary: late-run (last 5 steps) means
print(f"{'run':10s} {'valid(last5)':>12s} {'ESS(last5)':>10s} {'top8share':>9s} {'n_pos':>6s}")
for tag, src, lab, col, ls in LINES:
    d = (gm if src == "gm" else le).get(tag, {}); ks = sorted(d, key=int)[-5:]
    if not ks: continue
    m = lambda k: sum(d[s][k] for s in ks) / len(ks)
    print(f"{tag:10s} {m('valid'):>12.0f} {m('ess'):>10.1f} {m('top8_share'):>9.2f} {m('n_pos'):>6.0f}")
