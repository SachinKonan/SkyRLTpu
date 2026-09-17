# Ray runtime parity: baseline, implementation, and review gates

## Scope and provenance

This worktree is `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ray-runtime-parity`, branch
`agent/ray-runtime-parity`. It starts at science-placement commit
`86f476a7a0e5bd379e237f2ff767ae75ea37ab88` (2026-09-17). Its predecessor
`9b17c675` checkpoints the science bootstrap, Xplace starts and CPU placement
grading. The parent worktree was clean at branching and is not modified here.

The requested contract is:

1. Match the working legacy **trainer** environment with the Ray trainer, and
   the working legacy **serving** environment with Ray serving. Trainer and
   serving need not use the same JAX version as each other.
2. Native thinking preserves the supported two-phase completion semantics.
3. The same effective model/training settings reach every appropriate process.
4. Ray replaces SSH/tmux coordination with owned, repeatable startup/shutdown.

Seeding is explicitly outside this review. Existing experiment choices (22k
context, two trainer buckets, TP/FSDP, adaptive PWC, problem prompts, CPU grading,
bootstrap) are not to be reverted just to resemble legacy defaults. No TPU
jobs have been launched, cancelled or modified by this worktree.

## The legacy reference we actually have

The legacy comparison checkout is
`/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost`, commit `b48b5b61`, branch
`agent/tunix-multihost`. The reference launch chain is:

```
tpu/swarm/run_v5p32_cell.sh
  -> tpu/jobman/ensure_orbax_ckpt.sh
  -> tpu/jobman/cell_worker.sh
  -> tpu/start_colocated_vllm_tinker.sh
  -> tpu/start_vllm_tpu.sh
  -> tpu/launch_cell.sh
  -> tpu/jobman/cell_monitor.sh
```

The 25 previously submitted native recovery jobs, 855–879, used this immutable
base archive (not the entire present science checkout):

- URI: `gs://sk7524-tinker-tpu-us-east5/code-bundles/tpuswarm-skyrl-v5p32-cells-v22.tar.gz`
- SHA256: `515ed9a5f2c3021b31ec78bd5eb9baaab983e2e20aa7c84fede9188f02d93a12`
- Local evidence: `/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/benchmarks/smoke9-v5p/build/base-v22.tar.gz`
- Submission manifest: `/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/benchmarks/checkpoint-redeploy-20260914/submissions.json`

The preceding audit verified all 25 archive hashes, 365 source overlay entries,
and 1,058 packaged code/lock comparisons. Comparing 1,153 relevant base-archive
files with the legacy checkout found eight differences. Therefore “v22”,
“current legacy checkout” and “current science checkout” are distinct baselines.

`ray_train/host.py` records that its package pins were obtained from a working
legacy Qwen job 340 on 2026-09-08. This is useful provenance, **not a complete
installed-environment inventory for all three models**. The current audit has
not reconnected to that worker or independently reconstructed its entire venv.

| Role | Ray's recorded key pins | Baseline limitation |
| --- | --- | --- |
| Trainer | JAX/JAXlib 0.11.1, libtpu 0.0.46, Transformers 5.8.0, Tokamax 0.0.13 | Frozen TPU lock followed by a second package installation; transitive drift remains possible |
| Serving | vllm-tpu 0.23.0, JAX/JAXlib 0.10.1, libtpu 0.0.41, Torch 2.10.0, torchax 0.0.11, tokenizers 0.22.2 | Legacy Muse instead requests tokenizers >=0.23.1,<0.24.0 and Transformers from main |
| Client | Python 3.12, frozen lock, Tinker 0.22.7 plus native SDK patch | Legacy client provisioning uses Python 3.11; tokenization/SDK behavior needs comparison |

Do not manufacture an “exact legacy lock” from these partial pins or blindly
downgrade Muse. Capture the working environments first, then choose and validate
the common environment for each model/role. Include MaxText/fork revisions,
installed patches and model/tokenizer artifact identities, not only pip versions.

## Current science path

`build.py`/`science/package_training.py` -> immutable bundle -> `bootstrap.py`
on every host -> private Ray runtime -> controller/host actors -> ordinary
Tinker trainer and vLLM subprocesses -> Discover client. Training requests still
use Tinker/SQLite; Ray owns placement and lifecycle, not a replacement loss.

Prepared CPU-placement profiles use four trainer hosts and four TP4 inference
hosts: Qwen/Muse TP8 x FSDP2; Gemma TP4 x FSDP4. They use 16 x 32 rollouts,
18,432/22,528 trainer buckets, 22,528 total context and a 16,384-token
prompt-plus-thinking ceiling. The 18,432 trainer bucket is not the thinking cap.
These are intentional settings, not runtime parity failures.

Science selects a newer Tunix backend for distributed inputs and repeated/tied
KV heads. A same-workload legacy comparison must use that same selected backend;
the earlier finding that the single-host native sweep retained untouched v22
trainer code does not apply to these science profiles.

## Changes implemented in this worktree

| Area | Change | Scope / limitation |
| --- | --- | --- |
| Mesh validation | Inherit parent's rejection of conflicting MaxText TP/FSDP/CP | Already implemented by `86f476a7`; not claimed as new here |
| Environment isolation | Strip inherited model/compiler/debug variables before applying the role's explicit settings | Keeps authentication/system environment; intentional diagnostics must be in the profile |
| Override validation | Reject conflicting row shard, executor-owned topology/storage overrides, duplicate managed CLI flags and contradictory engine flags | Does not silently repair a conflicting profile |
| Inference controls | Explicit custom token buckets and model/sampling serialization fields | Defaults match the corresponding legacy exports |
| Runtime inventory | Capture trainer, serving and client Python/packages/origins/source hashes, including installed client and tokenizer Python sources; compare on reuse; optionally require a reviewed model/role baseline | No baseline is fabricated or enabled by default; a populated `runtime_baselines` profile fails setup on any mismatch |
| Launch records | Persist trainer, inference and client argv, role settings, explicit profile overrides and relevant system environment with credentials excluded | Uploaded with run logs; records operational paths as well as model settings |
| Native contract | Verify selected-model markers/cues, inheritance and reviewed method ASTs after base extraction and source overlay installation, also on reuse | Does not confuse checkout source with deployed client; inference-only sources do not require Discover; live tokenizer/TPU tests remain necessary |
| Backward warmup | Optional backend warmup uses production buckets/row counts with scratch accumulators; synchronizes execution and propagates failure | No optimizer call; restores previous accumulator/count even on failure; remains OFF pending TPU validation |
| Packaging | Include `$code/tpu` in generated PYTHONPATH | Carries the known pallas_arena import fix into this branch |
| Shutdown status | Required shutdown/final run-writeback errors make an otherwise successful job fail | Compile-cache upload failure alone remains nonfatal; a full checkpoint completion certificate is still needed |
| Lifecycle tests | Update the stale FLCE test imports to the current helper's contract | Lets the existing lifecycle suite collect again |

Implementation entrypoints are `environment.py`, `config.py`, `commands.py`,
`runtime_inventory.py`, `launch_contract.py`, `host.py`, `serving.py`,
`thinking_budget/contract.py`, `controller.py`, and
`skyrl/backends/backward_warmup.py`. This branch now changes `third_party/discover/ttt_discover/tinker_utils/completers.py`
in its isolated submodule checkout. That file is included in the native training
source overlay. Discover is now committed at `2444247284d2ef766a13b8a892b28d90f1bc4c33`;
the parent snapshot records that gitlink alongside the matching contract hashes. TPU-inference submodule contents are unchanged.

### Warmup specifics

Enable explicitly with `trainer.backward_warmup=true`; keep `TTD_WARMUP_FB=0`.
The new helper executes after adapter creation, before any rollout request.
With uniform length disabled, it validates that the explicit buckets cover the
maximum training length and warms the largest legal row count in each bucket,
including the packer's FSDP-padded singleton exception when the budget is smaller
than one sharded row. It calls `_model_pass` locally on each rank; it never
re-enters the distributed `forward_backward` RPC from inside `create_model`.
It is currently limited to single-adapter importance-sampling training without
carried/fresh adapter mixing. It does not invoke Adam, save a fake checkpoint,
or include dummy examples in the real step's normalization.

The old client warmup added a zero-gradient example to the accumulation count.
It also warmed only the first bucket in the two-bucket configuration. Re-enabling
that flag is not an acceptable substitute for the new warmup.

Review must still verify on TPU: multi-rank collective ordering, preserved
weights/optimizer/counters, both exact shapes, compile cache hits on real FB,
and peak HBM with trainer/inference running together. A zero-weight synthetic
pass is a compilation/allocation check, not a realistic gradient correctness test.

## Remaining implementation and validation gates

These are **not represented as completed fixes**. Another agent should review
the patch and this list before any deployment.

1. **Full package equality:** capture working legacy trainer/serving/client
   inventories per model; compare with freshly built Ray environments; resolve
   Muse's package discrepancy; produce complete immutable locks used by BOTH
   installers. Current version-list recipes do not establish this guarantee.
2. **One shared launch specification:** the new emitted records make comparisons
   concrete, but the legacy shell and Ray builders still independently encode
   launch settings. Refactor both to consume one resolved specification, with
   allowed differences limited to addresses, paths, topology and supervision.
3. **Native hardware/tokenizer validation:** exact-cap EOS/stop continuation and
   insufficient-headroom behavior now use legacy continuation/fallback on CPU
   fixtures. Validate these on real pinned tokenizers and TPU, including
   stop/forced-transition conflicts, natural/forced/partial markers, early answers,
   n=1/n=32, truncated output and all training masks. Do not claim identical random draws from differently
   scheduled requests. Preserve sampled behavior logprobs.
4. **Numerical parity:** feed identical recorded samples through both paths;
   compare datums, masks, advantages, gradient normalization, gradients, optimizer
   state and exported/reloaded adapters at each intended mesh/bucket.
5. **Durable completion certificate:** verify the latest real completed optimizer
   step, matching client/search state, and checkpoint weights + optimizer state.
   A final checkpoint row or exit zero is insufficient; the client can write a
   final row using the configured iteration limit. Publish the matching objects
   behind a commit manifest rather than inferring atomicity from separate copies.
6. **Bounded recovery:** decide explicit engine/application retry policies and
   test restarting from the last complete checkpoint. Existing profiles retain
   zero engine/application-error retries; silently enabling retries before
   recovery validation could repeat a correctness bug.
7. **Active storage reclamation:** implement verified checkpoint retirement and
   reference-aware payload GC. Preserve pending requests, readable results and
   snapshots during races; never reintroduce age-only request deletion. Current
   inactive-run cleanup does not solve active-run growth.
8. **Idempotence under failure:** exercise cancel during provisioning/download,
   compiler failure, actor/driver death, node loss, repeated start/stop, shutdown
   during upload and partial bootstrap role transition. Verify device/port/process
   release without touching another workload. Parent-death cleanup and locks
   exist, but safe duplicate-start refusal alone is not complete reconciliation.

## Review and reproducible validation

Review changes against `86f476a7`, not against the moving science branch:

```bash
git diff 86f476a7 --stat
git diff 86f476a7 -- tpu/swarm/ray_train skyrl/backends tests/tpu_swarm
git -C third_party/discover diff 1d662eb..2444247 -- ttt_discover/tinker_utils/completers.py
```

New files are included in the parent commit; inspect its complete diff against
`86f476a7`, including the recorded Discover gitlink.
Tests must expose this worktree's Discover import path and avoid creating caches:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/third_party/discover" \
/scratch/gpfs/ZHUANGL/sk7524/.cache/multi-lora-tests/bin/python -m pytest \
  -p no:cacheprovider tests/tpu_swarm/test_runtime_parity.py \
  tests/tpu_swarm/test_ray_train_mesh_conflicts.py \
  tests/tpu_swarm/test_ray_train_commands.py \
  tests/tpu_swarm/test_ray_train_api_leader.py \
  tests/tpu_swarm/test_ray_train_lifecycle.py \
  tests/tpu_swarm/test_ray_train_writeback.py \
  tests/tpu_swarm/test_native_training.py \
  tests/tpu_swarm/test_native_thinking_budget.py \
  tests/tpu_swarm/test_native_external_transport.py \
  tests/tpu_swarm/test_native_pipeline_failure.py \
  tests/tpu_swarm/test_science_training.py -q
```

Use the CPU Slurm partition for this check on the shared HPC machine. The live
Qwen-tokenizer test needs `THINKING_TEST_TOKENIZER`; skipping it is not a pass.
No CPU test establishes successful TPU compilation, numerical parity or fleet
recovery.

### Initial validation results, superseded by remediation below (2026-09-17)

- All 59 existing science profiles validate without changing their settings.
- Broad CPU suite: 228 passed, one skipped, and one failure caused by the new
  runtime-baseline test fixture omitting `Host.source`. The fixture now includes
  that required attribute; rerunning the complete runtime-parity module passed
  all 19 tests. Across the broad run and focused rerun, all 229 non-skipped test
  cases have passed. The entire suite was not repeated after this fixture-only fix.
- The skipped test requires the real Qwen tokenizer fixture. No tokenizer or
  TPU parity is inferred from this skip.
- `git diff --check` passes. Native marker/cue validation also runs successfully
  with Python site packages disabled (no JAX requirement in the build checker).
- No TPU smoke, role installation, live legacy inventory capture or deployment
  was performed. Backward warmup remains disabled in every existing profile.

This is a reviewable implementation of the changes listed above, not completion
of the remaining gates. Exact legacy parity and deployment readiness are not yet
established.


## Remediation of CRITICISM_MD.md (2026-09-17)

The criticism file remains unchanged as the original review record. This section
tracks the response; it does not erase the initial warmup defect or relabel the
initial 11-file suite as a full-directory pass.

| Finding | Implemented response | Evidence / limit |
| --- | --- | --- |
| Nested warmup RPC desynchronizes ranks | Call the local `_model_pass` on every rank, preserve accumulators and propagate error responses | CPU regression instantiates actual DistributedTunixBackend leader and TunixBackend workers; any nested RPC fails the test. TPU collectives still need a smoke. |
| Undersized warmup budget rejected | Match padded singleton admission in the real packer | Shape regression for a budget smaller than one FSDP row |
| CLI alias bypass / false flags | Normalize `-tp/-pp/-dp`, underscore spellings and `=value`; cover emitted LoRA/path/executor flags; preserve legacy omit-when-false behavior | Managed-flag and all-profile emission regressions; the erroneous negative-flag change was reverted |
| Owned cache/plugin overrides | Reject engine environment overrides of owned cache, topology and routing; expose `inference.plugins` as a typed setting | Existing Muse plugin behavior retained through the typed field; conflicts rejected |
| Missing client inventory | Capture and validate client inventory on install and reuse; include selected client baseline in environment identity | Fresh/reuse and failed-install tests; no real legacy baseline fabricated |
| Missing launch inputs / corrupted SQLite URL | Record all three roles, common runtime/client keys and explicit profile overrides; preserve SQLite absolute-path slashes | Credential redaction, client settings and SQLite regression. Unknown inherited environment keys are deliberately not dumped. |
| Weak native contract check | Resolve annotated/tuple constants, verify inheritance, reject post-class mutation, fingerprint the reviewed methods, check open/end markers and only selected model | Mutation tests; clear ValueError contract errors; Python 3.10/3.12 AST compatibility |
| Wrong source at build time | Remove the misleading checkout-client build check; verify extracted base plus overlays before installation and on source reuse | Host setup is the authoritative gate. Build alone does not certify deployed client parity. |
| Inference-only dependency | Do not require a Discover client when no training client is used | Slim inference-only source test |
| Second engine host differs | Carry bucket/serialization flags and sanitize inherited model/compiler settings through Ray's worker setup hook | Pure production-hook regression and explicit paired-engine source-overlay test. Real Ray/vLLM pair smoke remains pending. |
| Gemma TP8 override uses wrong key | Use `global_num_kv_heads` for Gemma, `base_num_kv_heads` otherwise | Pure configuration regression |
| Stale client installation tests | Supply real required fixture attributes and assert inventory sequencing | Client dependency tests now participate in validation |
| Native stop exactly on cap | Continue the same sampled prefix with legacy phase-2 cue, budget, mask and logprobs | Qwen/Gemma/Muse, n=1/n=32, with/without an already-open answer channel |
| Native insufficient answer headroom | Invoke the original two-phase algorithm for this unsupported native shape; invalid thinking caps raise fatal NativeCompletionError | Wall-ending behavior equality test; AST comparison proves extracted legacy algorithm body unchanged |

### Native completion scope after this fix

Ordinary native requests still use one serving request. Two rare cases now use
legacy continuation: a completed response exactly at the cap, and a request with
insufficient headroom for the native forced transition plus one answer token.
Continuation uses the same sampling client and original sampled prefix; routing
of a second request remains the legacy client's routing behavior. This does not
promise engine affinity or identical random draws across different schedulers.
No global native flag is toggled during fallback, so concurrent requests are not
silently switched into another mode.

The original two-phase algorithm was extracted into `_two_phase`, with a test
comparing its AST body to Discover commit `1d662eb4`. Updated contract hashes
include this local change. Do not interpret those hashes as proof of mathematical
parity: they prevent later unreviewed drift; comparison tests supply the evidence.

### Retained differences and decisions still needed

- Preserve existing workload profiles: max-num-seqs 16, profile-specific batched
  tokens (often 1024/4096), memory utilization (often 0.65–0.8), meshes, context,
  ingress routing and longer timeouts. These differ from legacy defaults and
  are deliberate current experiment/resource settings. Matching compiled shapes
  may reuse cached programs; altered shapes require different compilations.
- Preserve OMP/OpenBLAS caps of 1 and offline trainer/serving loading for now.
  Those are explicit executor choices, not legacy equality. In grader sandboxes
  a CPU allocation of 2/4 does not grant 2/4 BLAS threads under these caps.
- Preserve current KL-measurement default 0; legacy defaults to 1 for non-Erdos.
  This remains a known diagnostic/workload difference requiring a decision for a
  strict numerical/performance baseline. Enabling it silently would change 167
  current non-Erdos profiles and can change first-step memory pressure.
- TPU_VISIBLE_CHIPS, zero-threshold persistent caching, explicit single-host
  mesh kwargs, different client Python/locks, and pinned-versus-moving MaxText
  revisions remain visible differences. Package inventories and a common launch
  specification are still needed to certify equality.
- The Qwen science ragged-convolution setting and existing server adapter
  reload semantics are retained, not newly certified as matching legacy.
- Warmup remains opt-in, runs at model creation including resume, and does not
  retain JIT programs across the backend's `jax.clear_caches()` calls. It can
  populate disk compilation caches and detect startup failures; no measured
  steady-state speedup is claimed.
- Superseded source directories are not deleted by this patch. Active-run
  checkpoint/payload reclamation and completion certification remain open.

### Remediation validation

- Focused implementation/native/client/command/pair suite: **250 passed,
  2 skipped** (real tokenizer fixture and opt-in client installation).
- All **222 profiles** validate without editing their experiment settings.
- Full-directory results and any final follow-up checks are recorded below.
- No TPU jobs, live fleet environments or deployment bundles were changed.

Final local validation:

- Executor/native/science-training/RPC selection (`test_runtime_parity.py`,
  `test_ray_train_*.py`, `test_native_*.py`, `test_science_training.py`,
  `tests/backends/test_rpc.py`): **450 passed, 2 skipped**, 45.47 seconds.
- After extending launch-record credential handling, the runtime-parity module
  passed **51 tests**, 17.16 seconds.
- The full `tests/tpu_swarm/` run (with collection errors retained): **757 passed,
  11 failed, 3 skipped, 1 collection error**, 119.65 seconds. Failures are the
  previously reported Orbax recovery fixture (1), stale gradient-probe fixtures
  missing `training_failed` (3), missing TPUSwarm checkout files (6), and missing
  SkyPilot module (1). The collection error is missing `tpuswarm`. This is not
  reported as an all-green repository suite; none is a native/executor regression
  observed in this run. These unrelated fixtures/dependencies were left unchanged.
- A final code review also found and fixed a baseline-key mismatch: host placement
  calls the serving role `inference`, so it now canonicalizes that role before
  looking up the `serving` inventory. Otherwise an approved serving baseline could
  be silently skipped. The final targeted regression passed **3 tests** covering
  trainer, inference/serving and client paths.
- `git diff --check` passes in the main checkout and the Discover submodule.

The changes are recorded as local commits in Discover and its parent. Review
the new files and Discover commit `2444247` as well as the parent diff. No
changes were made to `CRITICISM_MD.md`, the parent science-placement worktree,
or running jobs.


## Second-review fixes and committed snapshot

- Restored omit-when-false for both prefix caching and chunked prefill. Across
  all 222 profiles, **219 omit the chunked-prefill flag and 3 explicitly enable
  it; zero emit a negative chunked-prefill flag**. No profile settings changed.
  The regression test checks all profile-generated commands, not just a single
  hand-built command. False continues to mean "use vLLM's default", not off.
- Python 3.13 method hashing now explicitly includes empty AST fields, preserving
  fingerprints across Python minor versions. The no-site-packages contract check
  passes on Python 3.13.7 as well as the test interpreter (3.12) and system Python.
  The regeneration command is documented next to `METHODS`; it prints hashes
  without editing or silently approving them.
- Discover commit: `2444247284d2ef766a13b8a892b28d90f1bc4c33`. Its parent gitlink,
  matching contract code, regression tests and this review are recorded together
  in the parent snapshot. No upstream push was performed; publish the child
  commit before the parent branch for fresh remote-clone reproducibility.
- Final executor/native/science-training/RPC suite: **455 passed, 2 skipped**,
  38.69 seconds. This supersedes the earlier selected-suite counts. The known
  unrelated full-directory failures above have not been represented as fixed.
- The remaining environment and hardware validation procedure is documented in
  `LEGACY_PARITY_TPU_VALIDATION.md`. No TPU launch or legacy-host capture was
  performed in this turn; environment equality is not yet certified.
