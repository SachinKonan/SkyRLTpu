# Establishing legacy environment equality and TPU validation

This is the remaining validation procedure for `agent/ray-runtime-parity`.
It is a plan, not evidence that remote inventories or TPU validation have run.
The local CPU test results are in `RAY_RUNTIME_PARITY_REVIEW.md`.

## 1. Select the actual legacy reference

For each intended model (Qwen, Gemma, Muse), identify a legacy run that completed
an optimizer step, saved weights and optimizer state, and reloaded its adapter
into inference. Record its job ID, TPU family/topology, run/checkpoint URI,
source bundle SHA256, source revisions, model/tokenizer snapshot revisions,
launch argv/environment and effective engine/trainer configuration.

Use the deployed source and installed role interpreters. The current shell
installer, a moving branch, or an old partial `pip freeze` is insufficient.
Trainer and serving are distinct environments and may intentionally use different
JAX versions. Compare each role to the corresponding role, including the client.
A Qwen reference alone does not certify Muse or Gemma.

If a suitable live legacy environment no longer exists, reconstruct one from its
immutable source/artifacts on a dedicated validation slice and first prove that
it completes the legacy cycle. Label that baseline reconstructed; do not claim
it was captured from the historical worker.

## 2. Capture and compare all three role environments

`runtime_inventory.py` is standalone and imports no JAX/TPU packages. Place the
same collector version on the reference host and run it using each role's actual
Python. Example with paths replaced by verified deployed paths:

```bash
/path/to/legacy/trainer/bin/python /path/to/runtime_inventory.py \
  --source /path/to/legacy/deployed-source --output legacy-qwen-trainer.json
/path/to/legacy/serving/bin/python /path/to/runtime_inventory.py \
  --source /path/to/legacy/deployed-source --output legacy-qwen-serving.json
/path/to/legacy/client/bin/python /path/to/runtime_inventory.py \
  --source /path/to/legacy/deployed-source --output legacy-qwen-client.json
```

Keep these captures immutable with their provenance. In addition to the collector,
record the installed binary/wheel provenance for JAX, libtpu, torch/torchax,
vLLM/TPU-inference, tokenizers and other compiled dependencies. The collector
hashes selected Python sources; it does not prove binary equality or record every
model/tokenizer asset. Record those assets and MaxText patches separately.

Build fresh Ray role environments from the candidate bundle, capture them with
the same collector, and compare with `--expect`. Do not use a cached Ray venv as
proof that its installation recipe reproduces the legacy reference.

Classify each difference explicitly:

- Compute stack: Python, JAX/JAXlib, libtpu, MaxText, torch/torchax, vLLM,
  TPU-inference, Transformers, tokenizers, Tinker and transitive dependencies.
  Resolve these to exact versions/revisions; do not mask a mismatch by copying
  the candidate inventory over the legacy baseline.
- Intentional runtime/source differences: Ray supervision and the native
  completion implementation. Keep a reviewed difference record and test their
  functional behavior. Whole source hashes will differ here by design.
- Workload choices: context/buckets, meshes, batching, memory utilization,
  thread caps, routing/timeouts and KL diagnostics. For a controlled legacy/Ray
  comparison, hold these fixed on both sides or identify the exact intervention.

Muse's serving recipe and the client's Transformers/Python differences require
resolution here. Do not install `transformers@main` as the final reproducible
answer: identify the successful commit/version and pin it. Produce complete
role/model locks, including transitive dependencies, and have both installers
consume them. These shared locks are still to be implemented.

`runtime_baselines/{model}-{role}.json` is an acceptance gate for the final
reviewed Ray environment. Its provenance must link to the legacy capture and the
reviewed differences. The current comparator is strict, not an allow-list engine;
a raw legacy source inventory with intentional native source differences will
correctly fail. Preserve the original captures separately. Every future Ray setup
and reuse must match the approved candidate inventory exactly.

## 3. Use a dedicated validation slice and fixed inputs

Choose one idle slice matching the target profile's TPU family and host layout.
Confirm ownership, free devices/ports, sufficient disk/RAM and no unrelated
processes before launch. Preserve shared model/compile caches. Stage an immutable
bundle containing the parent commit, committed Discover revision and explicit
profile. No validation should silently change the current fleet's running jobs.

Use the existing Ray v2 runtime, production context/buckets, and intended meshes.
For the distributed warmup test, use a profile with four trainer hosts; a
single-host v5p smoke cannot exercise the nested-RPC defect. Additional inference
hosts must fit the chosen slice. Keep warmup off in production until this passes.

For correctness comparisons, use identical recorded prompts/trajectories,
behavior logprobs, base weights, initial adapter weights and optimizer state.
Do not compare two independently sampled training batches and call it numerical
parity. Record both cold compilation and warm-cache timings separately.

## 4. Validation sequence and pass criteria

| Stage | What to execute | Required evidence |
| --- | --- | --- |
| Environment/startup | All intended role environments and inference engines | Inventories match approved baselines; effective vLLM config confirms chunked-prefill behavior and token limits; every engine becomes healthy |
| Four-host warmup | Adapter creation with warmup enabled and both 18,432/22,528 buckets, where configured | All ranks complete the same local backward sequence; no nested RPC, collective timeout or TPU error; weights/optimizer and accumulation counters are preserved |
| Native completion | Qwen/Gemma/Muse with real pinned tokenizers, n=1 and n=32 | Natural closure, forced transitions, early finishes, truncation and low-headroom fallback agree with two-phase token/mask/logprob semantics on controlled fixtures; cap-boundary EOS must STOP without another request, an intentional difference from legacy |
| One real update | Production 16x32 workload, packing and accumulation settings | Grading completes; expected groups are consumed; all backward passes and exactly one optimizer step complete; adapter exports and reloads into every engine |
| Numerical replay | Same recorded training batch through legacy and Ray | Exact preprocessing/masks/advantages and counters; compare losses, gradient norms and gradient/optimizer/adapter arrays with numerical tolerances established from reference repeatability before evaluating Ray |
| Durable restart | Save a completed step, stop the test cleanly, restart from it | Weights, optimizer state, client/search state and completed-step identity agree; resumed update succeeds without replaying partial gradients |
| Lifecycle | Controlled cancellation/relaunch of only the test workload | Owned processes/devices/ports released, artifacts preserved, repeat launch succeeds; no cross-workload cleanup |

Native stops when the server stops, including a stop/EOS exactly on the thinking
cap. No second request is sent and no post-EOS tokens are injected or trained.
This is a reviewed, intentional correction of legacy's length-only continuation
bug. Test both EOS during reasoning and EOS after a complete answer, for n=1 and
n=32. The legacy `_two_phase` algorithm itself remains unchanged; the native
insufficient-headroom fallback is also retained. Real tokenizer coverage must
not be replaced by fixture-only success.

Validate gpt-oss paired-engine worker hooks separately if deploying that topology:
prove the function imports on the second host, Ray's assigned TPU visibility is
preserved at the correct lifecycle point, and host-local cache/adapter paths are
valid on both hosts. Single-host Qwen/Gemma/Muse engines do not prove that path.

## 5. Decision and rollout

A successful smoke is necessary, not an environment-equality certificate. Publish
one result record per tested model/topology with immutable inputs, environment
inventories, effective settings, timings, peak HBM, optimizer/checkpoint IDs,
adapter reload evidence and lifecycle results. Keep failed stages explicit.

Deploy incrementally only after the applicable stages pass. The full durable
completion certificate, bounded recovery policy and active storage reclamation
remain separate implementation work; manually verifying a smoke does not mean
those mechanisms have been implemented.

## Commit and distribution order

Commit Discover first; then commit the parent gitlink, source contract fingerprints,
executor changes and tests together. A parent gitlink alone does not publish the
referenced submodule object. Before sharing via a remote clone, publish the
Discover commit to its configured upstream, then publish the parent branch and
verify a fresh recursive checkout plus bundle construction. Local commits alone
do not establish remote availability. This turn does not publish either repo.
