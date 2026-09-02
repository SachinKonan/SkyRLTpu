"""One-shot patch: refresh companion figures (tlr-n 14 steps), add the live-fleet
best-so-far section (figx9) and the 120b plateau autopsy section (figx8)."""
import base64, re
H = open("advantage_walkthrough.html").read()
b64 = lambda f: base64.b64encode(open(f, "rb").read()).decode()

def swap(H, alt, f):
    m = re.search(r'<img alt="' + re.escape(alt) + r'"\s+src="data:image/png;base64,[^"]+"', H)
    assert m, alt
    return H[:m.start()] + f'<img alt="{alt}" src="data:image/png;base64,{b64(f)}"' + H[m.end():]

H = swap(H, "Category stacks for the TTD/entropic companion runs", "figx2b.png")
H = swap(H, "Delta-magnitude heatmaps for the TTD/entropic companion runs", "figx3b.png")

# 1) live-fleet section right after the flagship section (before "What the rollouts were doing")
anchor = '''<section><div class="col">
  <div class="eyebrow">What the rollouts were doing</div>'''
fleet = f'''<section><div class="col">
  <div class="eyebrow">Refreshed 2026-09-01</div>
  <h2>Best C₅ so far — the live fleet</h2>
  <div class="figbox2"><img alt="Best-C5-so-far curves for every live arm, refreshed" src="data:image/png;base64,{b64("figx9.png")}"></div>
  <p>Since the previous build only <code>tlr-n</code> banked new steps (12 → 14; the companion
  stacks and heatmaps above are refreshed to match) — every other arm spent the interval in
  post-reboot bring-up or waiting for capacity. The E arms have 1 (qwen), 3 (muse) and 0 (gemma)
  steps; too early to read.</p>
  <p>The black line is the run holding the <strong>best verified construction, 0.38085904871926474</strong>
  (<code>m-meta</code>, muse GRPO from the meta experiment; sum(h) = n/2 exact, recomputed correlate
  bit-identical). Read its provenance carefully: it is flat from step 1 because it <em>started</em> from
  a pre-seeded pool already at 0.3808591578 and gained only 1.1e-7 in seven steps. The best
  <em>from-scratch</em> cell is still <code>lr-n</code> at 0.380859354887961. Both facts belong on
  any chart that quotes the number.</p>
</div></section>

'''
assert anchor in H; H = H.replace(anchor, fleet + anchor, 1)

# 2) 120b plateau autopsy — after the batch-shape section, before Provenance
anchor2 = '''<section><div class="col">
  <div class="eyebrow">Provenance</div>'''
plateau = f'''<section><div class="col">
  <div class="eyebrow">Autopsy</div>
  <h2>Why the 120b curve sat flat for nine steps, then dropped</h2>
  <div class="figbox2"><img alt="120b 64x8: pool best vs best parent sampled vs best child per step, with incumbent rollout counts" src="data:image/png;base64,{b64("figx8.png")}"></div>
  <p>The flagship gpt-oss-120b run (entropic, 64×8, <strong>elite = 0</strong> — it predates the elite
  feature) holds 0.380957655 from step 3 to step 11, then drops 4.8e-5 at step 14/15 and 2.0e-5 at
  step 17/18. Its per-rollout tables show the plateau was not the model failing to improve. It was
  <strong>the search never asking it to</strong>: the incumbent state got exactly one batch of 8
  rollouts the step after it was created (step 3: 4 valid, 0 improved) and was then sampled
  <strong>zero times for ten consecutive steps</strong> (shaded). PUCT spread the 512 rollouts across
  50–60 distinct parents per step with only 5–17% on the best three; the best parent it actually
  expanded sat 1e-5 to 3e-5 <em>worse</em> than the incumbent the whole time. Meanwhile 30–45% of
  rollouts were improving on their own (mid-ranked) parents — progress the running-min curve cannot
  show.</p>
  <p>The step-14 drop is <strong>one rollout</strong>: a child of a step-13 state at 0.380999832 — a
  parent 4.4e-5 <em>worse</em> than the incumbent — that gained 9.2e-5 in a single move, the largest
  single-rollout gain after step 3. Its parent was hot (5 of 6 valid siblings also improved on it). The
  code diff against the parent (74% line similarity) shows no new technique family: the same smooth-max
  gradient descent + annealing + FFT-correlate + projection program, but with the α-continuation
  ladder extended ~10× (to α ≈ 2.6·10⁶), an L-BFGS polish (<code>maxiter 8000, ftol 1e-13</code>) and a
  new deterministic <code>_pairwise_local_search</code> stage. The earlier 1.6e-6 step (step 11) had
  added annealing to a gradient-only lineage; the step-17 gain came from the incumbent itself, adding
  Adam and pairwise local search.</p>
  <p>Two consequences. First, this is the cleanest evidence we have that <strong>elite slots exist for a
  reason</strong>: a single unlucky batch (0/4) buried the frontier for ten steps in a run with no
  elite reseeding, and every league cell since runs elite = 2. Second, "flat best-so-far" is an
  ambiguous signal — here it meant a healthy population improving under an incumbent nobody was
  expanding, which is why the per-rollout category stacks and the best-parent-sampled trace are the
  diagnostics to trust, not the headline curve.</p>
</div></section>

'''
assert anchor2 in H; H = H.replace(anchor2, plateau + anchor2, 1)
open("advantage_walkthrough.html", "w").write(H)
print("bytes:", len(H))
