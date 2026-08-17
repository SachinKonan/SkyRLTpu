"""ctrl vs main (league qwen+gemma shared-tree on Erdős): four panels, wall-clock.

Evidence merged per arm:
  * first life  : wandb history (each relaunch made a NEW wandb run; only the
                  first life's run captured rows before node-side wandb broke)
  * later lives : per-step PUCT snapshot files -- later lives overwrite
                  same-numbered files, so surviving (mtime, step, tree-best)
                  triples are the chronology of last writes
  * restarts    : a step-0 snapshot write is a restart moment; step-number
                  drops in mtime order are boundaries; one main boundary sits
                  inside an overwritten gap (lineage analysis: 4 total) and is
                  drawn hatched/inferred
"""
import glob, json, os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = os.path.dirname(os.path.abspath(__file__))
ARMS = {"ctrl": ("#14867c", "ctrl — shared tree only"),
        "main": ("#d97726", "ours — tree + cross-CE (λ=0.1)")}
RECORD = 0.3808616492829141
SIMPLETES = 0.380868561


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()

T0 = {"main": ts("2026-08-04T04:08:44Z"), "ctrl": ts("2026-08-05T17:39:10Z")}
RESTARTS = {  # (epoch seconds, measured|inferred)
    "ctrl": [(ts("2026-08-06T09:30:21Z"), "measured")],
    "main": [(ts("2026-08-05T10:47:11Z"), "measured"),
             (ts("2026-08-06T18:00:00Z"), "inferred"),
             (ts("2026-08-08T22:00:00Z"), "measured"),
             (ts("2026-08-09T09:30:00Z"), "measured")],
}

# ---- merged best-over-time -------------------------------------------------
lives_json = json.load(open(f"{S}/league_lives.json"))
series = {}
for arm in ARMS:
    pts = []
    for r in lives_json[arm][0]["rows"]:
        if r.get("pool/best_value") is not None:
            pts.append((r["_timestamp"], -r["pool/best_value"], r["_step"], "w"))
    for line in open(f"{S}/snap_index.txt"):
        a, mt, url = line.split()
        if a != arm:
            continue
        step = int(os.path.basename(url).split("_")[-1].split(".")[0])
        d = json.load(open(f"{S}/snaps/{arm}_{os.path.basename(url)}"))
        allvals = [s["value"] for s in d["states"]] or [-10.0]
        pts.append((ts(mt), -max(allvals), step, "s"))
    pts.sort()
    xs, best, steps = [], [], []
    b = 10.0
    for t, v, st, src in pts:
        b = min(b, v)
        xs.append((t - T0[arm]) / 3600); best.append(b); steps.append(st)
    imp_x, imp_y = [], []
    for i in range(1, len(pts)):
        db = max(0.0, best[i - 1] - best[i])
        dsteps = max(1, abs(steps[i] - steps[i - 1]))
        imp_x.append(xs[i]); imp_y.append(db / dsteps)
    series[arm] = dict(xs=xs, best=best, imp_x=imp_x, imp_y=imp_y)

# ---- format telemetry ------------------------------------------------------
fmt = {a: {"qwen": [], "gemma": []} for a in ARMS}
for arm in ARMS:
    for lf in lives_json.get(arm, []):
        for r in lf["rows"]:
            st = r.get("_step")
            for m in ("qwen", "gemma"):
                v = r.get(f"{m}/env/all/format")
                if v is not None and st is not None:
                    fmt[arm][m].append(("first", st, v))
    with open(f"{S}/{arm}.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            st = r.get("step")
            for m in ("qwen", "gemma"):
                v = r.get(f"{m}/env/all/format")
                if v is not None and st is not None:
                    fmt[arm][m].append(("last", st, v))

fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
fig.patch.set_facecolor("white")


def draw_restarts(ax, arm, c):
    for t, kind in RESTARTS[arm]:
        x = (t - T0[arm]) / 3600
        ax.axvline(x, color=c, ls=":" if kind == "measured" else (0, (1, 3)),
                   lw=1.4 if kind == "measured" else 1.0,
                   alpha=0.75 if kind == "measured" else 0.45)

ax = axes[0][0]
for arm, (c, lab) in ARMS.items():
    s = series[arm]
    ax.step(s["xs"], s["best"], where="post", color=c, lw=1.9, label=lab,
            marker="o", ms=2.8)
    draw_restarts(ax, arm, c)
for y, name, col in [(0.380924, "AlphaEvolve 0.380924", "#64748b"),
                     (0.380875, "TTT-Discover 0.380875", "#64748b"),
                     (SIMPLETES, "SimpleTES 0.3808686", "#b91c1c"),
                     (RECORD, "our record 0.3808616", "#8b5cf6")]:
    ax.axhline(y, color=col, ls="--", lw=1.0)
    ax.text(0.995, y, f"{name} ", va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=7.2, color=col)
ax.set_ylim(0.380855, 0.38102)
ax.set_title("Best C₅ over wall-clock (dotted verticals = fresh-weights restarts)",
             fontsize=10)
ax.set_xlabel("hours since arm start"); ax.set_ylabel("best C₅ (lower is better)")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[0][1]
for arm, (c, lab) in ARMS.items():
    rx = sorted((t - T0[arm]) / 3600 for t, _ in RESTARTS[arm])
    xmax = max(series[arm]["xs"])
    cx = [0] + [x for x in rx for _ in (0, 1)] + [xmax]
    cy = [0] + [y for y0 in range(len(rx)) for y in (y0, y0 + 1)] + [len(rx)]
    ax.plot(cx, cy, color=c, lw=1.9,
            label=f"{lab} — {len(rx)} restarts")
ax.set_title("Cumulative fresh-weights restarts (weights+optimizer reset, tree kept)",
             fontsize=10)
ax.set_xlabel("hours since arm start"); ax.set_ylabel("restarts")
ax.set_ylim(-0.2, 4.8)
ax.annotate("one main boundary inferred\n(files overwritten)", xy=(0.55, 0.42),
            xycoords="axes fraction", fontsize=7.5, color="#8a6d3b")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

ax = axes[1][0]
# per-PROGRAM improvement: each committed state vs its best parent, timed by the
# first surviving snapshot that contains it (mtime); 6h buckets, mean +/- 1 std
for arm, (c, lab) in ARMS.items():
    files = []
    for line in open(f"{S}/snap_index.txt"):
        a, mt, url = line.split()
        if a == arm:
            files.append((ts(mt), f"{S}/snaps/{arm}_{os.path.basename(url)}"))
    files.sort()
    seen = set()
    obs = []
    for t, f in files:
        d = json.load(open(f))
        for st in d["states"]:
            if st.get("timestep", -1) < 0 or st["id"] in seen:
                continue
            seen.add(st["id"])
            pv = st.get("parent_values") or []
            if pv:
                obs.append(((t - T0[arm]) / 3600, st["value"] - max(pv)))
    bins = {}
    for t, v in obs:
        bins.setdefault(int(t // 6), []).append(v)
    xs, mu, sd = [], [], []
    for b in sorted(bins):
        vs = bins[b]
        m = sum(vs) / len(vs)
        xs.append(b * 6 + 3); mu.append(m)
        sd.append((sum((v - m) ** 2 for v in vs) / max(1, len(vs) - 1)) ** 0.5)
    ax.errorbar(xs, mu, yerr=sd, color=c, lw=1.8, marker="o", ms=4,
                capsize=3, elinewidth=1.1, label=lab, alpha=0.9)
ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.6)
ax.set_yscale("symlog", linthresh=1e-6)
ax.set_title("Per-program improvement over its parent (6h mean ± 1 std)", fontsize=10)
ax.set_xlabel("hours since arm start"); ax.set_ylabel("Δ C₅ vs parent (symlog)")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[1][1]
snap_step_t = {a: {} for a in ARMS}
for line in open(f"{S}/snap_index.txt"):
    a, mt, url = line.split()
    st = int(os.path.basename(url).split("_")[-1].split(".")[0])
    snap_step_t[a][st] = ts(mt)
for arm, (c, lab) in ARMS.items():
    obs = []
    for r in lives_json[arm][0]["rows"]:
        vs = [r.get(f"{m}/env/all/format") for m in ("qwen", "gemma")]
        vs = [v for v in vs if v is not None]
        if vs and r.get("_timestamp"):
            obs.append(((r["_timestamp"] - T0[arm]) / 3600, sum(vs) / len(vs)))
    with open(f"{S}/{arm}.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            st = r.get("step")
            vs = [r.get(f"{m}/env/all/format") for m in ("qwen", "gemma")]
            vs = [v for v in vs if v is not None]
            if vs and st in snap_step_t[arm]:
                obs.append(((snap_step_t[arm][st] - T0[arm]) / 3600,
                            sum(vs) / len(vs)))
    # bucket to 6h bins, mean per bin -> one clean line
    bins = {}
    for t, v in obs:
        bins.setdefault(int(t // 6), []).append(v)
    xs = [b * 6 + 3 for b in sorted(bins)]
    ys = [sum(bins[b]) / len(bins[b]) for b in sorted(bins)]
    ax.plot(xs, ys, color=c, lw=2.0, marker="o", ms=4, label=lab)
ax.set_ylim(0.82, 1.0)
ax.set_title("Format-valid rate over time (6h means, qwen+gemma averaged)", fontsize=10)
ax.set_xlabel("hours since arm start"); ax.set_ylabel("format rate")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)

fig.tight_layout()
png = f"{S}/league_ctrl_vs_main.png"
fig.savefig(png, dpi=140)
for arm in ARMS:
    print(arm, "final best %.10f" % series[arm]["best"][-1],
          "| span %.0fh" % max(series[arm]["xs"]))
print("saved", png)
