import base64, re
H = open("advantage_walkthrough.html").read()
b64 = lambda f: base64.b64encode(open(f, "rb").read()).decode()
img = lambda f, alt: f'<div class="figbox2"><img alt="{alt}" src="data:image/png;base64,{b64(f)}"></div>'

CSS = """
.tabs{display:flex;gap:8px;margin:18px 0 6px}
.tabbtn{font:inherit;font-size:.88rem;padding:7px 16px;border:1px solid var(--grid);
  border-radius:8px 8px 0 0;background:transparent;color:var(--muted);cursor:pointer}
.tabbtn.active{background:var(--panel,#fff);color:var(--ink);border-bottom-color:transparent;font-weight:600}
.figbox2{background:#fff;border:1px solid var(--grid);border-radius:10px;padding:8px;overflow-x:auto;margin:12px 0}
.figbox2 img{max-width:100%;display:block;min-width:640px}
.cav{border:1px solid var(--grid);border-left:4px solid #c0392b;border-radius:8px;padding:10px 14px;font-size:.86rem}
"""
H = H.replace("</style>", CSS + "</style>", 1)

PANE = f"""
<nav class="tabs">
  <button class="tabbtn active" data-t="walk">The estimators</button>
  <button class="tabbtn" data-t="xmodel">Cross-model diffs</button>
</nav>
<div id="pane-walk">
"""
H = H.replace("</header>\n", "</header>\n" + PANE, 1)

X = f"""</div>
<div id="pane-xmodel" hidden>
<section style="border-top:none;padding-top:0"><div class="col">
  <div class="eyebrow">Flagship runs, one per model</div>
  <h2>The same training loop, four models</h2>
  <p>The best run per base model: <strong>gpt-oss-120b</strong> (entropic, 64×8, 4e-5 — its era predates our GRPO
  arms), <strong>qwen3.5-27B</strong> (GRPO 1.5e-4, <code>lr-n</code>), <strong>gemma-4-31B</strong> (GRPO 4e-5,
  strict-grader rerun), <strong>Muse-Glimmer-30B</strong> (GRPO 4e-5, 7/15 steps at time of writing). Same problem,
  same 512-rollout steps, same reward <code>r = 1/(1e-8+C₅)</code>.</p>
  {img("figx1.png","Six-panel metric grid across the four flagship runs")}
  <p>Three regularities. <strong>Correctness rises under GRPO and only under GRPO</strong> — qwen 0.25→0.95,
  muse 0.78→0.97, gemma 0.67→0.75, while the entropic 120b drifts sideways at ~0.5 and its format rate
  <em>decays</em> 1.00→0.93. GRPO is the only estimator here that charges for failures every group.
  <strong>Mean reward tells the same story</strong>: the GRPO models climb toward the 2.63 ceiling
  (reward mass = usable-rollout mass); the entropic run peaks at step 6 and slides back — it has no
  incentive to lift the mean, only the max. <strong>The advantage panels replay the β autopsy at fleet
  scale</strong>: entropic's max/spread sit pinned near 10–20 all run (the β ceiling), while GRPO's
  signal shrinks as groups homogenize — qwen's max advantage ends at 0.05, which is why estimator E
  re-normalizes the improver mean to 1 regardless of scale.</p>
</div></section>

<section><div class="col">
  <div class="eyebrow">What the rollouts were doing</div>
  <h2>Failure / regression / no-op / improvement, per step</h2>
  {img("figx2.png","Stacked category fractions per archived step for qwen, gemma, muse; committed tree states for 120b")}
  {img("figx3.png","Per-step heatmaps of signed reward-delta magnitudes for the three league runs")}
  <p>The stacks are per-rollout ground truth from the trajectory archives (the steps each run archived);
  the 120b panel is different in kind — only <em>committed</em> tree states survive there, and the commit
  gate admits almost nothing but improvements, so it shows the tree's diet, not the model's behavior.</p>
  <p>The heatmaps are the signature plot: each column is a step, each row a delta magnitude.
  <strong>Qwen</strong> is a fine-grained no-op machine — a dominant exact-no-op band plus improvements
  concentrated at 10⁻⁷–10⁻⁸, and near-zero regression mass. <strong>Gemma</strong> is a coarse explorer:
  its no-op band collapses (0.40→0.02 across steps 5–14) into a broad 10⁻⁵–10⁻³ <em>regression</em> band,
  paid for by improvements a hundred times larger than qwen's (10⁻⁵–10⁻⁶). <strong>Muse</strong> starts
  on a fresh pool with 10⁻¹-scale improvements, then converges within six steps into the same
  no-op-plus-regression regime. One estimator has to price all three of these economies — that is the
  design problem the other tab walks through.</p>
</div></section>

<section><div class="col">
  <div class="eyebrow">GRPO vs TTD, within model</div>
  <h2>Four head-to-head pairs</h2>
  {img("figx4.png","Best-C5 curves, GRPO vs TTD/entropic, one panel per model")}
  <p>qwen: GRPO by 6.3e-6 (stage-A pair, both 4e-5). gemma: <strong>TTD by 4.8e-5</strong> — the one
  clear TTD win. muse: GRPO by 6.6e-5, but the TTD arm ran at 1.5e-4 (a rate at which muse GRPO also
  failed), so this pair is LR-confounded. gpt-oss-20b (16×32): GRPO by 1.6e-4, the largest margin of
  the four — both arms ended early of their planned 30 steps (prod OOV regression), curves shown to
  the last logged step.</p>
</div></section>

<section><div class="col">
  <div class="eyebrow">The interaction question</div>
  <h2>Does starting validity pick the winning algorithm?</h2>
  {img("figx5.png","Initial correctness vs GRPO edge; correctness trajectories under both estimators")}
  <p><strong>Not monotonically.</strong> The scatter refuses the clean story: lowest-validity qwen gives
  GRPO only a hair, mid-validity gpt-oss gives it the biggest win, high-validity gemma flips to TTD —
  and high-validity muse flips back (confounded). What the data does support is sharper:</p>
  <p><strong>1. In all four pairs, correctness ends higher under GRPO than under its TTD twin</strong>
  (0.73 vs 0.44, 0.75 vs 0.68, 0.97 vs 0.81, 0.53 vs 0.25). GRPO repairs validity; entropic lets it
  drift — and in the gpt-oss pair actively decay, because failures cost nothing under a softmax.</p>
  <p><strong>2. GRPO's margin tracks the correctness it actually harvested, not the starting level.</strong>
  gpt-oss-20b harvested +0.21 correctness and won by 1.6e-4. The stage-A qwen pair at 4e-5 harvested
  little headroom difference and the margin was 6e-6 — qwen's big GRPO edge only appeared at 1.5e-4
  (<code>lr-n</code>), where correctness reached 0.95. Gemma had no headroom to harvest (0.67 start,
  +0.08 gained), so GRPO's failure-pricing channel bought nothing, and entropic's sharp exploitation
  of gemma's unusually <em>coarse</em> improvement tail (10⁻⁵-scale, see the heatmaps) won instead.</p>
  <p>Working hypothesis for the E arms: <em>GRPO-family estimators win where correctness headroom
  exists and the LR is high enough to harvest it; exponential-tilt estimators win where validity is
  already high and progress lives in a genuinely graded improvement tail.</em> E is built to do both —
  failures priced at −K, improver credit tilted and mean-normalized.</p>
</div></section>

<section><div class="col">
  <div class="eyebrow">One more axis</div>
  <h2>Batch shape, entropic without CE distillation</h2>
  {img("figx6.png","Best-C5 curves for 16x32, 64x8, 8x64 entropic no-CE runs at 20b and 120b")}
  <p>Within 20b, 16×32 finishes <em>worst</em> (0.381084 vs 64×8's 0.381001) — the shape our GRPO league
  cells use is not what the entropic runs liked. Within 120b, 8×64 sprints (below 0.38095 by ~step 5)
  but 64×8 keeps descending late and overtakes (0.380888 vs 0.380939). Fewer, wider groups pay off
  late for entropic; note the 8×64 run carries elite=2 vs 64×8's elite=0, so the late-game comparison
  is not perfectly controlled.</p>
</div></section>

<section><div class="col">
  <div class="eyebrow">Provenance</div>
  <h2>Caveats</h2>
  <div class="cav"><p><strong>Grader era.</strong> All gpt-oss numbers (120b flagship, the 16×32 pair,
  both shape contrasts) were logged under the pre-fix lenient grader; the 120b flagship's 0.380887659
  is independently re-verified, the others are not. League runs: gemma is the post-fix strict rerun;
  qwen/muse trees were checked clean at the top.</p>
  <p><strong>Coverage.</strong> Category stacks and heatmaps exist only for archived steps (qwen 11–14,
  gemma 5–14, muse 0–6 — per-rollout archiving shipped mid-campaign; it is on by default for every run
  since). Muse and the E arms are still running; the 16×32 pair ended early (OOV). The muse
  GRPO-vs-TTD pair is LR-confounded. The 120b category panel counts committed states only.</p>
  <p>Data: <code>xmodel_data.json</code> (built from run <code>metrics.jsonl</code>, trajectory archives,
  and the 120b final tree snapshot) via <code>make_xmodel.py</code>, both in
  <code>results/story/</code>.</p></div>
</div></section>
</div>
<script>
document.querySelectorAll(".tabbtn").forEach(b=>b.addEventListener("click",()=>{{
  document.querySelectorAll(".tabbtn").forEach(x=>x.classList.toggle("active",x===b));
  document.getElementById("pane-walk").hidden = b.dataset.t!=="walk";
  document.getElementById("pane-xmodel").hidden = b.dataset.t!=="xmodel";
  window.scrollTo(0,0);
}}));
try{{const t=localStorage.getItem("advtab");if(t==="xmodel")document.querySelector('[data-t="xmodel"]').click();}}catch(e){{}}
document.querySelectorAll(".tabbtn").forEach(b=>b.addEventListener("click",()=>{{try{{localStorage.setItem("advtab",b.dataset.t)}}catch(e){{}}}}));
</script>
"""
# close pane-walk before final </div> of .wrap and insert the new pane
i = H.rstrip().rfind("</div>")
H = H[:i] + X + H[i:]
open("advantage_walkthrough.html", "w").write(H)
print("bytes:", len(H))
