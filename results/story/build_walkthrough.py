import json, os
S="/tmp/claude-374192/-n-fs-vision-mix-sk7524-SkyRLTpu/7702cd20-fe25-42c0-871b-c38f00057807/scratchpad"
D=json.load(open(f"{S}/blog_data.json"))

def fnum(x, sig=3):
    if x==0: return "0"
    a=abs(x)
    if a>=1e4 or a<1e-3:
        m,e=f"{x:.{sig-1}e}".split("e"); return f"{m}×10<sup>{int(e)}</sup>"
    return f"{x:.{sig}g}"

KIND={"fail":("failure","k-fail"),"noop":("no-op","k-noop"),
      "improve":("improve","k-imp"),"regress":("regress","k-reg")}

def extable(ex, want, cap=None):
    """Pick rows SPREAD across each category's range, so the table shows the
    extremes an estimator produces, not just the top of the group."""
    by={}
    for r in ex["rows"]: by.setdefault(r["kind"],[]).append(r)
    out=[]
    for kind,n in want.items():
        rows=sorted(by.get(kind,[]), key=lambda r:-r["delta"])
        if not rows: continue
        if len(rows)<=n: pick=rows
        else:
            idx=[round(i*(len(rows)-1)/(n-1)) for i in range(n)] if n>1 else [0]
            pick=[rows[i] for i in sorted(set(idx))]
        out+=pick
    order={"improve":0,"noop":1,"regress":2,"fail":3}
    out.sort(key=lambda r:(order[r["kind"]], -r["delta"]))
    h=['<div class="tw"><table><tr><th>rollout</th><th class="num">\u0394 vs parent</th>'
       '<th class="num">GRPO</th><th class="num">TTT-D</th>'
       '<th class="num">v2 parent baseline</th>'
       '<th class="num">v3 parent + remembered scale</th></tr>']
    for r in out:
        lab,cls=KIND[r["kind"]]
        d = "\u22122.63 (whole cliff)" if r["kind"]=="fail" else fnum(r["delta"])
        def cell(v):
            neg = ' style="color:var(--bad);font-weight:600"' if v<0 else ""
            return f'<td class="num"{neg}>{v:+.3f}</td>'
        h.append(f'<tr class="{cls}"><td>{lab}</td><td class="num">{d}</td>'
                 f'{cell(r["grpo"])}{cell(r["ttd"])}{cell(r["v2"])}{cell(r["v3"])}</tr>')
    h.append("</table></div>")
    return "\n".join(h)

E={e["label"]:e for e in D["examples"]}
qw = E["Qwen · step 12 · converged"]
gm = E["Gemma · step 12"]
gh = E["Gemma · step 12 · half the group failed"]
q3 = E["Qwen · step 3 · still exploring"]

trace_rows="".join(
  f'<tr><td class="tag">{t["model"]}</td><td class="num">{t["step"]}</td>'
  f'<td class="num">{t["fail"]:.1f}%</td><td class="num">{t["noop"]:.1f}%</td>'
  f'<td class="num">{t["imp"]:.1f}%</td><td class="num">{t["reg"]:.1f}%</td>'
  f'<td class="num">{fnum(t["med_imp"])}</td><td class="num">{fnum(t["med_beta"])}</td>'
  f'<td class="num">{t["cap_frac"]:.0f}%</td></tr>' for t in D["trace"])

beta_rows="".join(
  f'<tr><td>{s["name"]}</td><td class="num">{s["nfail"]}</td>'
  f'<td class="num">{fnum(s["beta_all"])}{" <span class=cap>cap</span>" if s["capped_all"] else ""}</td>'
  f'<td class="num">{fnum(s["beta_valid"])}{" <span class=cap>cap</span>" if s["capped_valid"] else ""}</td></tr>'
  for s in D["beta_scenarios"])

HTML = r"""<title>What the Advantage Actually Says</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap">
<style>
:root{--paper:#fbfbfa;--panel:#fff;--sunk:#f4f5f4;--ink:#14181c;--ink2:#3d4852;--muted:#6b7885;
  --rule:#e2e5e3;--teal:#0f766e;--rust:#c2410c;--indigo:#4338ca;--amber:#b45309;--crimson:#9f1239;
  --good:#0f766e;--bad:#9f1239;--figpaper:#fff;--figrule:#e2e5e3}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#101315;--panel:#171b1e;--sunk:#1b2023;--ink:#e8ebe9;--ink2:#b3bcc2;--muted:#8b979f;
  --rule:#262c30;--teal:#2dd4bf;--rust:#fb923c;--indigo:#a5b4fc;--amber:#fbbf24;--crimson:#fb7185;
  --good:#2dd4bf;--bad:#fb7185;--figpaper:#171b1e;--figrule:#2c3237}}
:root[data-theme="dark"]{--paper:#101315;--panel:#171b1e;--sunk:#1b2023;--ink:#e8ebe9;--ink2:#b3bcc2;
  --muted:#8b979f;--rule:#262c30;--teal:#2dd4bf;--rust:#fb923c;--indigo:#a5b4fc;--amber:#fbbf24;
  --crimson:#fb7185;--good:#2dd4bf;--bad:#fb7185;--figpaper:#171b1e;--figrule:#2c3237}
:root[data-theme="light"]{--paper:#fbfbfa;--panel:#fff;--sunk:#f4f5f4;--ink:#14181c;--ink2:#3d4852;
  --muted:#6b7885;--rule:#e2e5e3;--teal:#0f766e;--rust:#c2410c;--indigo:#4338ca;--amber:#b45309;
  --crimson:#9f1239;--good:#0f766e;--bad:#9f1239;--figpaper:#fff;--figrule:#e2e5e3}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 16.5px/1.68 "IBM Plex Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 26px 90px}
.col{max-width:70ch}
h1,h2,h3{font-family:Newsreader,Georgia,serif;font-weight:600;text-wrap:balance;margin:0}
h1{font-size:clamp(2.2rem,5vw,3.3rem);line-height:1.07;letter-spacing:-.012em}
h2{font-size:clamp(1.45rem,2.8vw,1.95rem);line-height:1.17;margin:0 0 .35em}
h3{font-size:1.14rem;margin:2em 0 .5em}
p{margin:0 0 1.05em;color:var(--ink2)}
p.lead{font-size:1.14rem;color:var(--ink)}
strong{color:var(--ink);font-weight:600}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
code{font-size:.87em;background:var(--sunk);padding:.12em .38em;border-radius:3px;border:1px solid var(--rule)}
header.mast{padding:74px 0 32px;border-bottom:1px solid var(--rule);margin-bottom:40px}
.kicker{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.17em;
  text-transform:uppercase;color:var(--muted);margin-bottom:20px}
.sub{font-size:1.2rem;color:var(--ink2);max-width:64ch;margin:22px 0 0}
section{padding:44px 0;border-top:1px solid var(--rule)}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--teal);margin-bottom:12px}
.tw{overflow-x:auto;margin:22px 0}
table{border-collapse:collapse;font-size:.88rem;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--rule);padding:8px 12px;text-align:left;vertical-align:top}
th{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);font-weight:500;border-bottom:1px solid var(--ink2)}
td.num{text-align:right;font-family:"IBM Plex Mono",monospace}
.tag{font-family:"IBM Plex Mono",monospace;font-size:.82rem}
tr.k-fail td{color:var(--muted)}
tr.k-noop td{background:color-mix(in srgb,var(--amber) 9%,transparent)}
tr.k-imp  td{background:color-mix(in srgb,var(--teal) 8%,transparent)}
tr.k-reg  td{background:color-mix(in srgb,var(--crimson) 7%,transparent)}
.cap{font-family:"IBM Plex Mono",monospace;font-size:.62rem;color:var(--crimson);
  border:1px solid currentColor;border-radius:3px;padding:0 4px;vertical-align:middle}
.note{font-size:.9rem;color:var(--muted);border-top:1px solid var(--rule);padding-top:14px;margin-top:24px}
.callout{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--teal);
  border-radius:0 5px 5px 0;padding:17px 20px;margin:24px 0}
.callout.bad{border-left-color:var(--crimson)}
.callout.warn{border-left-color:var(--amber)}
.callout p:last-child{margin-bottom:0}
.callout .lab{font-family:"IBM Plex Mono",monospace;font-size:.68rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--muted);display:block;margin-bottom:7px}
.eq{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:15px 19px;margin:18px 0;
  font-family:"IBM Plex Mono",monospace;font-size:.87rem;line-height:1.85;overflow-x:auto;color:var(--ink)}
.eq .nm{color:var(--muted);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
  display:block;margin-bottom:6px}
figure{margin:28px 0 10px;padding:20px 18px;background:var(--figpaper);border:1px solid var(--figrule);
  border-radius:6px;overflow-x:auto}
.dgm{display:block;width:100%;max-width:100%;height:auto;color:var(--ink2)}\nfigure img{display:block;width:100%;max-width:100%;height:auto}
figcaption{font-size:.86rem;color:var(--muted);margin:12px 2px 0;max-width:88ch}
ul{padding-left:20px;color:var(--ink2)}li{margin:.42em 0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin:22px 0}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:17px 19px}
.card h4{margin:0 0 .5em;font-family:Newsreader,serif;font-size:1.05rem;font-weight:600}
.card p:last-child{margin-bottom:0}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:.83rem;color:var(--muted);margin:-8px 0 18px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
</style>
<div class="wrap">

<header class="mast">
  <div class="kicker">Erdős discovery fleet · advantage design · Aug 2026</div>
  <h1>What the Advantage Actually Says</h1>
  <p class="sub">Four ways to turn 32 rollouts into a gradient, walked through on real
  training groups. Every reward below is measured, not illustrative — and two of the four
  estimators pay a model for making its answer worse.</p>
</header>

<section style="border-top:none;padding-top:0"><div class="col">
  <div class="eyebrow">The setting</div>
  <h2>One group, thirty-two tries</h2>
  <p>Each training step the search picks 16 parent states and asks the model for
  <strong>32 rollouts on each</strong>. A rollout is a Python program that outputs a
  construction; it scores C₅ (lower is better), converted to reward
  <code>r = 1/(1e-8 + C₅) ≈ 2.6256</code> near the record. A rollout that fails —
  bad format, crash, invalid output — scores <strong>0</strong>.</p>
  <p>The tree keeps the best child and discards the other 31. So the question every
  estimator answers is: <em>given these 32 rewards, how hard do we push each one?</em></p>
</div>

<figure>__DISTSVG__</figure>
<figcaption>The shape of a converged group. Three regions with wildly different scales: a
failure spike at zero, a tall spike of exact copies at the parent's own reward, and — carrying
all the discovery signal — two hair-thin tails about 10⁻⁸ wide. The gap between failing and
succeeding is <strong>a hundred million times larger</strong> than the gap between a good
construction and a better one.</figcaption>

<div class="col">
  <h3>How that shape changes over training</h3>
  <p>Measured from per-rollout archives (512 rollouts per step). "No-op" means a valid rollout whose construction scores <em>exactly</em> the parent's value.</p>
</div>
<div class="tw"><table>
  <tr><th>model</th><th class="num">step</th><th class="num">fail</th><th class="num">no-op</th>
      <th class="num">improve</th><th class="num">regress</th><th class="num">median improvement</th>
      <th class="num">median β</th><th class="num">β at solver cap</th></tr>
  __TRACE__
</table></div>
<figure><img alt="Six metrics over training steps, qwen on top and gemma below" src="data:image/png;base64,__TRACEIMG__"></figure>
<figcaption>The same statistics as the table, per step. Qwen appears as two runs because the
per-rollout archive only began mid-flight: circles are steps 0&ndash;3 at LR 4e-4, squares are
steps 11&ndash;14 at LR 1.5e-4. Gemma is one continuous run.</figcaption>

<div class="col">
  <h3>Why the two models diverge under the same objective</h3>
  <p>GRPO's advantage has exactly one axis. With a fraction <em>f</em> of the group failing,
  every valid rollout receives <code>+2.6256 &times; f</code> and every failure receives
  <code>&minus;2.6256 &times; (1&minus;f)</code>. That is the whole signal: <strong>be valid.</strong>
  Nothing in it distinguishes a better construction from a worse one. The two trajectories above
  are two different ways of responding to that single instruction.</p>

  <p><strong>Qwen had somewhere to go.</strong> It began at 78% failures, so the validity
  gradient was enormous, and it converted almost all of it: failures fell 78% &rarr; 5% while
  no-ops rose 1% &rarr; 57%. It learned to write programs that reliably land on the parent's
  optimum &mdash; the safest possible way to be valid.</p>

  <p><strong>Gemma had nowhere to go.</strong> Its failure rate is flat across ten steps
  &mdash; 29, 29, 27, 31, 26, 31, 30, 24, 23, 25 &mdash; with no trend at all. That is the
  signature of a <em>strategy cost</em> rather than a skill gap: gemma always writes a genuinely
  new construction, and some fraction of novel constructions simply fail to satisfy the
  constraints. A validity gradient can fix ignorance; it cannot fix the price of exploring.</p>

  <div class="callout warn"><span class="lab">A correction: no-ops are not copies</span>
  <p>I assumed a no-op meant the model reproducing the parent's program. It does not. In qwen's
  step-14 archive, <strong>291 no-ops came from 284 distinct programs</strong> &mdash; almost no
  literal duplication. These are <em>different</em> programs whose numerical optimizers converge
  to the same fixed point. Qwen didn't learn to copy; it learned to write varied code that
  reliably re-derives the same optimum. The incentive story is unchanged &mdash; that is still
  the safe play &mdash; but "copying" was the wrong word for it, here and in the earlier
  discussion.</p></div>

  <p><strong>And gemma's no-ops became regressions.</strong> Watch the two columns together:
  no-ops fall 41% &rarr; 2% while regressions climb 15% &rarr; 51%, and their <em>sum stays
  nearly constant</em> &mdash; 55.7, 61.9, 55.7, 52.9. It is one population being reclassified,
  not two independent trends. The cause is geometric: as the tree's parents approach a local
  optimum, the set of programs that land exactly on it shrinks, and almost everything nearby
  lands slightly <em>below</em> it. Gemma perturbs hardest, so it falls off the exact point
  first.</p>

  <p>Nothing pushes back, either. GRPO pays a regression <code>+0.656</code> &mdash; the same as
  an improvement, since both are simply "valid." Half of gemma's rollouts drift downhill and the
  objective is indifferent to all of them.</p>

  <div class="callout"><span class="lab">The through-line</span>
  <p>Same objective, two models, opposite-looking outcomes &mdash; but one explanation. GRPO
  optimizes validity and nothing else. Qwen had a validity deficit, so it improved and then
  parked on the optimum. Gemma had none to fix, so its rollouts wandered off the optimum with
  no gradient to stop them. Neither model was ever told what "better" means.</p></div>

  <h3>Three things to carry forward</h3>
  <p> The improvement signal <strong>collapses by seven orders
  of magnitude</strong> (10⁻¹ → 10⁻⁸) as runs converge. Qwen ends with a majority of <strong>no-ops</strong>
  — 57–61% exact no-ops. Gemma becomes a <strong>perturber</strong> — it barely ever copies,
  but <strong>half its valid rollouts make the construction worse</strong>. And the last
  column is a preview of trouble.</p>
</div>
</section>

<section><div class="col">
  <div class="eyebrow">Estimator one</div>
  <h2>GRPO: subtract the group average</h2>
  <div class="eq"><span class="nm">GRPO</span>advantage = reward − mean(rewards in group)</div>
  <p>Here it is on a real converged qwen group — 3 failures, 17 exact copies, 12 improvements,
  no regressions. Group mean 2.379.</p>
</div>
<div class="col"><p class="note" style="margin-top:0;border:0;padding-top:0">
  The last two columns are the estimators this page builds toward &mdash;
  <strong>v2</strong> centres credit on the parent's score, <strong>v3</strong> adds a scale
  remembered across steps. Both are defined further down; they appear here so the same rollouts
  can be compared under all four rules at once.</p></div>
<div class="legend">
  <span><i class="sw" style="background:color-mix(in srgb,var(--teal) 45%,transparent)"></i>improvement</span>
  <span><i class="sw" style="background:color-mix(in srgb,var(--amber) 45%,transparent)"></i>no-op (same score as parent)</span>
  <span><i class="sw" style="background:color-mix(in srgb,var(--crimson) 40%,transparent)"></i>regression</span>
  <span><i class="sw" style="background:var(--muted)"></i>failure</span>
</div>
__EX_QW__
<div class="col">
  <div class="callout bad"><span class="lab">What GRPO gets wrong</span>
  <p>Look down the GRPO column: <strong>+0.246 for every single valid rollout.</strong> The
  rollout that improved by 4×10⁻⁶ and the rollout that copied the parent character-for-character
  receive <em>identical</em> credit. GRPO passes reward differences through at 1:1, so a 10⁻⁸
  improvement produces a 10⁻⁸ difference in gradient — invisible beside the ±2.6 failure signal.
  Half of GRPO's total gradient magnitude goes to the three failures.</p></div>
  <p>GRPO isn't malfunctioning. It optimizes expected reward, and under expected reward
  <em>copying really is the best strategy</em>: a guaranteed 2.6256 beats a gamble that pays
  10⁻⁸ and risks 2.6256. For the gamble to be worth taking, the mutation would need to add less
  than <strong>3 chances in 100 million</strong> of failing. Qwen's 60% copy rate is a rational
  response to the objective we gave it.</p>
</div>
</section>

<section><div class="col">
  <div class="eyebrow">Estimator two</div>
  <h2>TTT-Discover: tilt, then normalize</h2>
  <div class="eq"><span class="nm">TTT-Discover</span>
    weight = e<sup>β·reward</sup> ÷ (average weight of the other 31)<br>
    advantage = weight − 1<br>
    <span style="color:var(--muted)">β chosen per group so the weights carry exactly one bit of selectivity</span></div>
  <p>Because everything is a <em>ratio</em>, the reward scale cancels: adding a constant to all
  32 rewards changes nothing. The floor comes free — weights can't go negative, so no advantage
  is below −1. On the same qwen group, the TTT-D column above shows what that buys: the best
  rollout gets <strong>+13.4</strong> where GRPO gave +0.246, and the group is finally ordered.</p>
  <div class="callout"><span class="lab">What it solves</span>
  <p>Discovery signal exists at all. The 10⁻⁶ improvement is now separated from the copy by
  <strong>13.8 units of advantage</strong> instead of 0.000. This is the thing GRPO fundamentally
  cannot do, and it is why gemma — whose rewards cluster tightly — gains so much from this
  objective.</p></div>
  <h3>But look again at the same column</h3>
  <p>Nine of the twelve <em>improvements</em> in that group received <strong>negative</strong>
  advantage (−0.372, −0.394…). Their crime was improving by 5×10⁻⁹ instead of 4×10⁻⁶. Because
  the softmax measures everything against the group's best, a genuine improvement that isn't
  near the top is pushed <em>down</em>, almost as hard as an exact copy (−0.396).</p>
  <p>And on gemma the failure is starker. Same step, a group with 8 failures, 4 improvements
  and <strong>20 regressions</strong>:</p>
</div>
__EX_GM__
<div class="col">
  <div class="callout bad"><span class="lab">Both estimators pay for damage</span>
  <p>Every regression in this group — rollouts that made the construction measurably
  <em>worse</em> — receives <strong>positive</strong> advantage under both GRPO (+0.656) and
  TTT-Discover (+0.7 to +1.9). A rollout that degraded the answer by 1.6×10⁻⁶ is rewarded more
  than one that improved it by 1.4×10⁻⁸. Neither objective can tell "worse" from "better",
  because neither knows what the parent scored — they only know the group's own statistics, and
  in a group where a quarter of rollouts scored zero, being mediocre still looks above average.</p></div>
</div>
</section>

<section><div class="col">
  <div class="eyebrow">The knob</div>
  <h2>β, and when it stops working</h2>
  <p>β is an inverse temperature — a volume knob. At β = 0 the weights are uniform and no rollout
  is preferred; as β grows the distribution <strong>sharpens</strong> toward the best sample.
  The solver turns β up until the weights carry exactly <strong>one bit</strong> of concentration
  — "as selective as keeping half and dropping half."</p>
  <p>The catch: that one bit gets spent on the <em>coarsest distinction available</em>, and
  telling failures from successes is much coarser than telling good constructions apart.</p>
</div>
<div class="tw"><table>
  <tr><th>group composition</th><th class="num">failures</th>
      <th class="num">β solved on all 32</th><th class="num">β solved on valid only</th></tr>
  __BETAS__
</table></div>
<div class="col">
  <p>Read the first three rows: with 3 failures β is 216,000; with 16 failures it is
  <strong>9.1</strong>; with 22 it is <strong>1.0</strong>. A <strong>20,000× collapse</strong>,
  caused by nothing but the failure count. When half a group fails, separating valid from failed
  <em>already costs the whole bit</em>, and the discovery signal never gets amplified at all.</p>
  <p>That is not hypothetical. Here is a real gemma group where 15 of 32 failed:</p>
</div>
__EX_GH__
<div class="col">
  <p>β collapsed to <strong>4.19</strong>. Look at the consequences: the no-ops get +1.308 and a
  rollout that regressed by 3×10⁻³ gets +1.272 — <em>indistinguishable</em>. In a group like
  this, TTT-Discover degenerates into "did you avoid failing," which is exactly GRPO.</p>
  <div class="callout warn"><span class="lab">And the failure mode that matters most</span>
  <p>Look back at the β columns in the training table. For converged qwen, β sits at the
  solver's ceiling in <strong>88% of groups at step 12 and 94% at step 14.</strong> Once the
  valid rollouts are near-identical, one bit of selectivity is <em>unattainable at any finite
  β</em>, so the solver runs to its arbitrary maximum. At convergence — precisely when discovery
  matters most — the "adaptive" scale has stopped adapting and the credit magnitudes are set by
  a constant somebody typed into the source.</p></div>
</div>
</section>

<section><div class="col">
  <div class="eyebrow">Estimator three</div>
  <h2>v2: measure against the parent</h2>
  <p>Both failures above share a cause: the zero point is a <em>group statistic</em>, so it moves
  with the company a rollout happens to keep. But there's a fixed, meaningful zero available —
  <strong>the score of the parent the group was asked to improve.</strong></p>
  <div class="eq"><span class="nm">v2 — parent baseline</span>
    advantage = clamp( β × (reward − <strong>parent's reward</strong>), −1, +10 )<br>
    <span style="color:var(--muted)">β solved on the valid rollouts only, so failures can't eat the budget</span></div>
  <p>Every sign now means what it says: copy → 0, improve → positive, regress → negative,
  fail → floor. In the qwen table the v2 column gives the top improver +2.982, the tiny ones
  +0.004, and the copies exactly 0.000. In the gemma table it is the only column besides v3 where
  the twenty regressions are <strong>negative</strong>.</p>
  <div class="callout"><span class="lab">What v2 fixes</span>
  <p>Copying earns nothing instead of +0.246. Damage is penalized instead of rewarded. Failures
  can no longer consume β's calibration budget — in the half-failed gemma group, v2's β is
  700,000 where TTT-D's collapsed to 4.19.</p></div>
  <div class="callout bad"><span class="lab">What v2 does not fix</span>
  <p>β is still solved per group, so it inherits the ceiling problem. In that same half-failed
  group v2's β is <em>at the cap</em> — and at convergence it will be capped in nearly every
  group, just like TTT-D. Two rollouts making the same improvement in different groups still get
  different credit, and when the cap binds, the credit is arbitrary.</p></div>
</div>
</section>

<section><div class="col">
  <div class="eyebrow">Estimator four</div>
  <h2>v3: remember the scale instead of re-deriving it</h2>
  <p>The remaining problem is that each group re-measures the volume knob from its own 32 samples
  — like re-measuring room temperature from 32 air molecules every second. Usually fine;
  occasionally the 32 you grabbed are all identical and the thermometer pegs at its maximum.</p>
  <p>So stop asking the group. Keep a slow running estimate of <strong>how large a typical
  improvement has been lately</strong>, and divide by that.</p>
  <div class="eq"><span class="nm">v3 — remembered scale</span>
    scale ← 0.9 × scale + 0.1 × (typical improvement seen this step)<br>
    advantage = clamp( (reward − parent's reward) ÷ scale , −1, +3 )<br>
    <span style="color:var(--muted)">copies (within a tie threshold) → exactly 0 · failures → −1</span></div>
  <p>Same meaning as β — "how big is this improvement compared to what's normal" — with three
  properties per-group β cannot give:</p>
  <div class="two">
    <div class="card"><h4>Stable</h4><p>One degenerate group can't peg or crater the knob. A
    group of pure copies simply contributes nothing, instead of triggering a cap.</p></div>
    <div class="card"><h4>Fair</h4><p>The same child earns the same credit regardless of which
    siblings it happened to be sampled with.</p></div>
    <div class="card"><h4>Still adaptive</h4><p>The remembered scale tracks the 10⁻³ → 10⁻⁸
    collapse over a few steps, so late micro-improvements still get full-size gradients.</p></div>
  </div>
  <p>And with the scale external, the cap becomes <em>meaningful</em> rather than arbitrary:
  capping at 3 says "no rollout pushes harder than a three-sigma-unusual improvement." In the
  tables above, v3 is the only column where copies are 0, every improvement is positive and
  ordered, and every regression and failure is negative.</p>
  <div class="callout warn"><span class="lab">Where v3 is weakest — visible in the data</span>
  <p>Early in training, improvements are far larger than the remembered scale, so <em>many</em>
  rollouts saturate at +3 and their ordering is lost. In a real step-3 qwen group, eight separate
  improvements — spanning 1.1×10⁻⁴ to 4.6×10⁻⁴ — all clamp to exactly +3.000, while TTT-D still
  ranks them 4.7, 3.4, 3.3, 2.9… A smooth compressor (<code>tanh</code>, or dividing by
  <code>1 + |Δ|/scale</code>) would keep the bound and the ordering. That is the first change I
  would make.</p></div>
</div>
</section>

<section><div class="col">
  <div class="eyebrow">Summary</div>
  <h2>The four columns, side by side</h2>
</div>
<div class="tw"><table>
  <tr><th></th><th>a copy</th><th>a tiny improvement</th><th>a regression</th><th>a failure</th><th>scale set by</th></tr>
  <tr><td class="tag">GRPO</td><td>+0.246 <span style="color:var(--bad)">paid</span></td>
      <td>+0.246 — same as a copy</td><td>+0.656 <span style="color:var(--bad)">paid</span></td>
      <td>−2.38, half of all gradient</td><td>nothing (raw scale)</td></tr>
  <tr><td class="tag">TTT-Discover</td><td>−0.396</td>
      <td><span style="color:var(--bad)">−0.394 — punished</span></td>
      <td>+1.9 <span style="color:var(--bad)">paid</span></td><td>−1.00</td>
      <td>per group; collapses on failures, caps at convergence</td></tr>
  <tr><td class="tag">v2 parent-β</td><td>0.000</td><td>+0.004</td><td>−0.15 to −1.00</td>
      <td>−1.00</td><td>per group; caps at convergence</td></tr>
  <tr><td class="tag">v3 EMA</td><td>0.000</td><td>+0.274</td><td>−1.00</td><td>−1.00</td>
      <td>remembered history</td></tr>
</table></div>
<div class="col">
  <div class="callout"><span class="lab">The through-line</span>
  <p>Every estimator here is answering "how unusual is this rollout?" They differ in
  <strong>unusual compared to what</strong>. GRPO says <em>compared to this group's average</em>,
  which is dragged around by failures. TTT-Discover says <em>compared to this group's best</em>,
  which punishes modest improvements and can't see the parent. v2 and v3 say <em>compared to the
  state we were asked to improve</em> — which is the only reference that makes "improvement" mean
  improvement. v3 then adds: <em>and measure it against how large improvements have typically
  been</em>, rather than re-deriving that from 32 samples that are often all the same.</p></div>
  <div class="note">
  <p><strong>What isn't settled.</strong> These are worked examples from real groups, not a
  controlled comparison — v2 and v3 have not been trained. Three open questions we can test
  offline with data already archived: are "no-ops" literal duplicate programs or merely
  score-ties (we store the parsed code)? Do 10⁻⁸ improvements actually produce better
  descendants, or is that tail evaluator dust? And should regressions be pushed negative at all,
  or merely given zero — penalizing every regression assumes discovery is uphill-only, which for
  a model that regresses half the time may suppress the very exploration that occasionally
  wins.</p></div>
</div>
</section>

</div>"""

import base64
_img = base64.b64encode(open("/n/fs/vision-mix/sk7524/SkyRLTpu-league/results/story/fig_trace_grid.png","rb").read()).decode()
HTML = HTML.replace("__TRACEIMG__", _img)
HTML = (HTML.replace("__DISTSVG__", open(f"{S}/dist.svg").read())
            .replace("__TRACE__", trace_rows)
            .replace("__BETAS__", beta_rows)
            .replace("__EX_QW__", extable(qw, {"improve":6,"noop":1,"fail":1}))
            .replace("__EX_GM__", extable(gm, {"improve":3,"regress":4,"fail":1}))
            .replace("__EX_GH__", extable(gh, {"improve":2,"noop":2,"regress":2,"fail":1})))

out="/n/fs/vision-mix/sk7524/SkyRLTpu-league/results/story/advantage_walkthrough.html"
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out,"w").write(HTML)
print("wrote", out, os.path.getsize(out)//1024, "KB")
