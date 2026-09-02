import base64
H=open("advantage_walkthrough.html").read()
b64=lambda f: base64.b64encode(open(f,"rb").read()).decode()
anchor='''<section><div class="col">
  <div class="eyebrow">One more axis</div>'''
sec=f'''<section><div class="col">
  <div class="eyebrow">The gradient budget</div>
  <h2>How many rollouts actually push the policy, and where the mass goes</h2>
  <p>Two questions behind the model × validity × algorithm interaction: how many rollouts per step are
  <em>usable</em> (valid, hence able to carry an improvement signal), and how the advantage mass is
  actually distributed over them. The first comes from every run's metrics; the second from the
  <em>applied</em> advantages stored in the gpt-oss wandb tables and, for league runs, advantages
  recomputed from the trajectory archives with each arm's own estimator (GRPO: reward minus group mean;
  entropic: the leave-one-out tilt with the one-bit β solve). The 20b 16×32 pair ran <strong>elite 2</strong>
  (not 4); its two arms differ in <code>adv_estimator</code> only.</p>
  <div class="figbox2"><img alt="Valid rollouts per step, GRPO vs TTD, six panels" src="data:image/png;base64,{b64("figx10.png")}"></div>
  <p><strong>Usable rollouts.</strong> In every pair the GRPO arm ends with more: qwen 374 vs 224 (4e-5),
  485 vs 406 (1.5e-4), gemma 383 vs 348, muse 497 vs 417, gpt-oss-20b 272 vs 129. The size of the gap
  tracks starting validity: the two low-validity starters (qwen ≈ 0.2, 20b ≈ 0.33) open gaps of
  150–250 rollouts per step, and they open <em>late</em> — after step 8–10 for qwen, after step 12 for
  20b, where the entropic arm turns downward while GRPO keeps climbing. Gemma, starting at 0.67, shows
  almost no gap (both arms live at 350–400 all run). Muse is the interesting one: it starts high (0.78)
  and GRPO still climbs to 497, while TTD dips to 330 at step 4 before recovering — entropic
  destabilizes validity even where there was little to gain. The gpt-oss entropic runs (right panel)
  show the same shape at every batch size: rise to ~400 by step 5–8, then decay once the search
  converges, because nothing in the objective holds validity up.</p>
  <div class="figbox2"><img alt="Effective sample size, top-8 share of positive mass, and positive-advantage counts per step" src="data:image/png;base64,{b64("figx11.png")}"></div>
  <p><strong>Where the mass goes.</strong> The right panel is the cleanest statement of the mechanism:
  GRPO gives a <em>positive</em> push to 350–470 rollouts per step, entropic to 100–270. The middle
  panel says why — under entropic, the top 8 rollouts carry 13–30% of all positive mass (peaking at
  47–48% for muse TTD and qwen entropic), under GRPO 4–7%. Entropic's positive signal is a narrow
  beam on the winners; GRPO's is a floodlight on everything above a failure-dragged mean, including
  the no-ops and regressions the other tab documents. That floodlight is exactly what repairs validity:
  a policy rewarded for <em>every</em> valid sample keeps producing valid samples, and the usable-rollout
  curve compounds. The left panel adds a caution about reading "gradient" too simply: converged GRPO
  on qwen and muse has an effective sample size of only 70–150 — its |advantage| mass is concentrated
  on the few failures (each at −mean ≈ −2.5) while the hundreds of valid samples sit within 0.1 of the
  mean. Late GRPO is mostly a "don't fail" gradient with a faint "improve" gradient on top; entropic on
  8×64 groups (one bit over 64 samples) keeps ESS near 465 but spends it on the tail.</p>
  <p><strong>So is there a model × validity × algorithm law?</strong> The data support this version:
  the algorithm decides whether validity is <em>reinforced</em> (GRPO, broad positive mass) or merely
  <em>tolerated</em> (entropic, failures at a capped −1 that never scales with severity); the model's
  starting validity decides how much that reinforcement is worth. Low starters (qwen, gpt-oss) win with
  GRPO because the reinforced validity turns into 150–250 extra usable rollouts per step late in the
  run — "more gradient later" is literally visible in the curves. High starters split: gemma has nothing
  to reinforce and entropic's precision wins; muse has a little (0.78 → 0.97) and GRPO still wins,
  though that pair is LR-confounded. The one thing no run here does is <em>both</em> — reinforce
  validity broadly <em>and</em> aim the positive mass at the improvement tail. That combination is
  what estimator E is: −K on every failure (GRPO's floodlight on the negative side), tilted
  mean-normalized credit on improvers only (entropic's beam on the positive side). The E arms are the
  test of this paragraph.</p>
</div></section>

'''
assert anchor in H; H=H.replace(anchor, sec+anchor, 1)
open("advantage_walkthrough.html","w").write(H); print("bytes:",len(H))
