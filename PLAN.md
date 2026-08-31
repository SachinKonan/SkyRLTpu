# Tunix Multi-Host v6e Plan

Worktree: `/n/fs/vision-mix/sk7524/SkyRLTpu-multihost`
Branch: `agent/tunix-multihost`
Planning date: 2026-08-26

## Goal

Run the production Tunix/MaxText LoRA trainer for Qwen3.5-27B,
Gemma-4-31B, and Muse-Glimmer-30B on v6e without changing the existing RL
algorithm or requiring a large training batch.

The chosen training topology is physical-host-local tensor parallelism:

| Target | Physical hosts | TPU VMs/processes | Chips | MaxText mesh | Global trainer batch | Rows per FSDP rank |
|---|---:|---:|---:|---|---:|---:|
| First target | 2 | 4 | 16 | TP=8, FSDP=2 | 4 | 2 |
| Scale target | 4 | 8 | 32 | TP=8, FSDP=4 | 4 | 1 |

Cloud TPU multi-host v6e slices expose half-host VMs with four chips each:
`v6e-16` is two physical hosts but four TPU VMs, and `v6e-32` is four physical
hosts but eight TPU VMs. The `v6e-8` inference-optimized shape is the exception
that exposes a full eight-chip host as one VM. Thus each TP8 group should span
the two half-host VMs on one physical host, while FSDP spans 2 or 4 physical
hosts. This device-to-mesh packing is a hardware gate, not an assumption.

The trainer continues to receive small batches streamed from the inference
side. One training payload contains four rows. Multiple four-row payloads may
bank gradients before `optim_step`, so an effective update batch of `4 * K`
does not require holding `4 * K` rows in an individual forward/backward pass.

## Fixed decisions

1. Start with `ici_tensor_parallelism=8` and
   `ici_fsdp_parallelism=2` on v6e-16.
2. Keep the global forward/backward batch at four rows. Do not increase it just
   to fill the device mesh.
3. Set `TUNIX_ROW_SHARD` equal to the FSDP degree: 2 on v6e-16 and 4 on the
   training-only v6e-32.
4. Use rank-32 LoRA for capacity and acceptance runs. Smaller-rank smoke tests
   do not establish production HBM headroom.
5. Long-context acceptance inputs must contain 22,528 valid tokens per row;
   short examples merely padded to that shape are useful compilation probes
   but do not establish dense-attention behavior.
6. With uniform sequence length `L`, set `TUNIX_TRAIN_TOKEN_BUDGET=4*L` when
   the desired compiled forward/backward shape is four rows. Token-budget mode
   overrides `TRAIN_MICRO_BATCH_SIZE`, so leaving the historical value at 1
   must not accidentally split the batch.
7. Preserve the existing memory work: FLCE, `num_vocab_tiling`, full remat where
   already selected, token-budget packing, input sharding, donated in-JIT
   gradient accumulation, base-state release, host offload, and cache eviction.
8. Do not modify or deploy to the running v5p fleet until all v6e correctness
   and resume gates pass.

## Resource layouts

### Layout A: v6e-16 trainer

```text
workers 0,1: physical host 0, FSDP rank 0, TP ranks 0..7
workers 2,3: physical host 1, FSDP rank 1, TP ranks 0..7
worker 0: Tinker API/RPC coordinator; workers 1..3: collective workers

global batch [4, L]
  FSDP rank 0 addressable rows: [0, 1]
  FSDP rank 1 addressable rows: [2, 3]
```

This layout consumes all four VMs on both physical hosts for training. An end-to-end RL run therefore
needs already-running external inference endpoints, or it can first be tested
with synthetic/direct trainer calls.

### Layout B: mixed v6e-32 cell

```text
workers 0..3: TP8 x FSDP2 trainer (16 chips, two physical hosts)
workers 4..7: vLLM serving (16 chips, two physical hosts)
```

This is the first self-contained end-to-end layout. It proves multi-host Tunix,
LoRA publication, sampling, Jobman recovery, and GCS resume without requiring a
second TPU slice. The trainer is still FSDP2 because only two physical hosts
train. The initial serving shape is one TP4 engine per four-chip VM; a TP8
engine would require a separate multi-VM serving path.

### Layout C: training-only v6e-32

```text
workers 0,1: physical host 0, FSDP rank 0, TP ranks 0..7
workers 2,3: physical host 1, FSDP rank 1, TP ranks 0..7
workers 4,5: physical host 2, FSDP rank 2, TP ranks 0..7
workers 6,7: physical host 3, FSDP rank 3, TP ranks 0..7
worker 0: Tinker API/RPC coordinator; workers 1..7: collective workers

global batch [4, L]
  one addressable row per host
```

This consumes every chip in the v6e-32. Inference must run in a separate TPU
slice/fleet and be supplied to Tinker as comma-separated vLLM URLs. A v6e-32
cannot simultaneously be TP8/FSDP4 and reserve two of its hosts for serving.

Serving topology remains model-specific. Start Qwen/Gemma conservatively with
one TP4 engine per four-chip serving VM. Do not carry Muse's
v5p TP2 layout directly to 32 GB v6e chips: a 30B bf16 base is already roughly
30 GB/chip at TP2 before runtime and KV memory. Muse must prove TP4 or TP8 fit
and throughput on v6e even though that pads its two KV heads.

## Current system being extended

The current v5p-32 cell uses worker 0 for Tunix/Tinker and workers 1-3 for
independent vLLM engines. `tpu/jobman/cell_worker.sh` selects per-model MaxText,
FLCE, sequence, cache, and serving settings, then invokes
`tpu/start_colocated_vllm_tinker.sh`.

The repo already has the pieces that must remain intact:

- Tunix/MaxText LoRA training and external vLLM adapter publication.
- Fused linear cross-entropy with tiled custom VJP and target-logit/logsumexp
  loss, avoiding a full `[B*T,V]` logits allocation.
- Uniform/token-budget packing and `TUNIX_ROW_SHARD` support.
- Durable model, optimizer, tree, trajectory, and recurrent sampling state.
- GCS-backed code, Hugging Face, Orbax, vLLM XLA, and trainer JAX caches.
- Jobman leases, retry loops, completion probes, state re-registration, and
  recurrent spot recovery.

The JAX backend already implements RPC-over-JAX-broadcast, but Tunix is guarded
to a single training host in `tpu/start_colocated_vllm_tinker.sh`. The Tunix
backend also assumes that arrays are fully addressable by one process in its
input, result, checkpoint, and PEFT export paths.

## Implementation workstreams

### 1. Generic distributed backend control plane

Extract the backend-neutral pieces of `skyrl/backends/jax.py` into a small
shared module:

- `RpcPayload` and command serialization.
- Coordinator `broadcast_command`.
- A generic collective worker loop with an INIT payload containing the backend
  name and backend config.
- Ordered method dispatch and clean shutdown.
- Error propagation that returns a failure to the coordinator instead of
  leaving another process blocked in a collective.

Keep the existing JAX backend behavior compatible. Add a generic worker module
instead of making Tunix workers pretend to be `skyrl.backends.jax`.

Acceptance:

- A two-process CPU stub backend receives the same ordered command stream.
- Positional/keyword payloads and exceptions round-trip correctly.
- Single-process JAX behavior is unchanged.

### 2. Distributed Tunix facade and initialization

Add a `DistributedTunixBackend` facade that mirrors the existing distributed
JAX facade. Collective methods include model creation/deletion, forward,
forward/backward, optimizer step, checkpoint save/load, and sampler checkpoint
save. Sampling remains coordinator-side because vLLM is external.

Add `coordinator_address`, `num_processes`, and logical process metadata to the
Tunix configuration. Call `jax.distributed.initialize` before model loading or
any other JAX operation on every process. Keep MaxText's
`skip_jax_distributed_system=True`, because SkyRL owns initialization.

The launcher must map possibly non-contiguous TPU worker IDs to contiguous JAX
process IDs. Process 0 remains the Tinker API and RPC coordinator.

Acceptance:

- v6e-16 reports `jax.process_count()==4`, four local devices per process, and
  sixteen global devices.
- The MaxText mesh reports tensor degree 8 and FSDP degree 2.
- Each TP group is confined to the two VMs of one physical host; the
  device-to-mesh mapping is logged and asserted.

### 3. Correct global batches and outputs

The RPC command is broadcast, so every process initially sees an identical
host copy of the four-row input. Do not pass that full copy directly to
`jax.make_array_from_process_local_data`, which could reinterpret it as each
process's local contribution and duplicate the global batch.

Implement one explicit global-array construction path:

1. Determine the global batch shape and target `NamedSharding` from the MaxText
   mesh.
2. Select or construct only the process-addressable slice for each host.
3. Build the global JAX array using addressable indices/callbacks or verified
   process-local data semantics.
4. Assert global shape `[4,L]`, local row count `4/FSDP`, and a row-ID checksum
   proving that every input row appears exactly once globally.

For forward/backward outputs, all processes must reach the result collective.
Gather per-token loss and target-logprob shards before process 0 reconstructs
the API response. Non-coordinator processes discard the assembled host result
only after the collective completes.

Acceptance:

- FSDP2 produces two addressable rows per host; FSDP4 produces one.
- No row is duplicated or dropped.
- Output ordering matches the original four-row request.
- Loss, target logprobs, gradients, and adapter deltas match a single-process
  reference within the established numeric tolerance.

### 4. Checkpoint and LoRA export correctness

Create one leaf-wise global gather helper for sharded Tunix state, then refactor
all persistence/export callers to consume the gathered flat state.

This must cover more than `_flat_numpy`: `_peft_tensors_native`,
`_peft_tensors_gptoss`, and `_peft_tensors_maxtext` currently call
`jax.device_get` directly and therefore also need conversion.

Rules:

- Every process participates in state gathers.
- Only process 0 writes `.npz`, tarballs, GCS objects, or PEFT files and sends
  adapter HTTP requests.
- No collective is placed inside a process-0-only branch.
- Process-0 I/O success/failure is broadcast before the next command so a
  write error cannot strand workers at a barrier.
- Each process may initially download the same small LoRA/optimizer archive on
  restore, but each leaf must be placed with its target global sharding rather
  than restored as an unsharded `jnp.asarray`.
- In-memory sampler snapshots remain valid on every process; only external
  publication is coordinator-owned.

Acceptance:

- A gathered per-leaf hash is identical before save and after restore.
- Checkpoint round-trip preserves global shape, dtype, sharding, optimizer
  count, and accumulated-gradient reset semantics.
- PEFT output from distributed training is byte- or tensor-equivalent to the
  single-process export.
- Exactly one archive write and one adapter publication occur per call.

### 5. Launcher and trainer-host provisioning

Update `tpu/start_colocated_vllm_tinker.sh`:

- Remove the Tunix single-host guard.
- Add distributed coordinator fields to the Tunix backend JSON.
- Launch the generic backend worker on every nonzero trainer process.
- Validate `TP * FSDP == global trainer device count` and
  `global_batch % FSDP == 0`.
- Make v6e device geometry explicit and accelerator-aware. For multi-host v6e,
  each process owns a four-chip `2,2,1` tile; `v6e-16` uses process bounds
  `2,2,1`, yielding the global 4x4 slice. Keep the full-host `v6e-8` exception
  separate instead of applying its eight-chip geometry to multi-host slices.
- Accept an explicit comma-separated external-vLLM URL override when
  `START_VLLM=0`, enabling the training-only v6e-32.

Provision every trainer host identically before starting JAX:

- Same repo bundle, virtualenv, Tunix, MaxText commit, aqtp/tokamax dependencies,
  and FLCE source patch.
- Same model/MAXTEXT configuration and tokenizer/config availability.
- Local Orbax base-checkpoint cache on every trainer host.
- Local `~/jax_cache` restore on every trainer host.
- Same `TUNIX_UNIFORM_SEQ_LEN`, token budget, row shard, sequence buckets,
  dtype/remat/offload knobs, and coordinator metadata.

Use a shared GCS JAX-cache seed for reads. Initially allow only the coordinator
to publish cache entries; if hardware testing shows process-specific entries,
publish process-indexed prefixes and merge/promote them deliberately rather
than racing multiple `rsync` writers against one prefix.

Acceptance:

- Rendered coordinator and worker scripts have matching package versions,
  patches, model knobs, and cache locations.
- A cold worker cannot reach JAX initialization with a missing tokenizer,
  MaxText install, Orbax base checkpoint, or FLCE patch.
- Existing v5p single-trainer defaults remain unchanged.

### 6. Jobman, recurrent runner, and GCS recovery

Make topology a durable job property, not an ad-hoc coordinator export:

- Job configs set `TRAIN_WORKERS`, `VLLM_WORKERS`, TP, FSDP,
  `TUNIX_ROW_SHARD`, global trainer batch, token budget, and accelerator type.
- `tpu/jobman/cell_worker.sh` honors job-provided trainer/serving worker lists
  instead of hardcoding worker 0 as the trainer.
- `tpu/jobman/ensure_orbax_ckpt.sh` runs on every selected trainer host.
- HF/tokenizer and JAX-cache restore also run on every trainer host.
- Trainer health treats all distributed processes as one atomic unit. If one
  trainer process dies or becomes unhealthy, stop and relaunch the complete
  trainer group; do not perform the current worker-0-only surgical restart.
- Process 0 remains the owner of durable run-dir sync, checkpoint registration,
  sampling-cache publication, and completion probes.
- Preserve strict checkpoint-set agreement and monotonic global-step resume.

Add three explicit job/runner shapes:

1. v6e-16 training smoke: workers 0-3 train; synthetic or external inference.
2. v6e-32 mixed cell: workers 0-3 train, workers 4-7 serve.
3. v6e-32 training-only: workers 0-7 train, external vLLM URLs supplied from a
   separately managed serving slice.

Acceptance:

- A recurrent spot replacement reconstructs the whole trainer group and
  resumes from the last mutually complete checkpoint.
- Existing tree/trajectory/sampling caches prevent completed rollout work from
  being repeated.
- A partially restored Orbax or run cache is detected and repaired instead of
  being mistaken for a valid cache.

### 7. Tests before TPU deployment

Add CPU tests using multiple forced host devices and two local JAX processes:

- Generic RPC ordering, INIT dispatch, exception propagation, and shutdown.
- Single-process `DistributedTunixBackend` equivalence.
- Global input construction: shape, row ownership, and no duplication.
- Result gathering and original-row ordering.
- Gathered LoRA and optimizer state.
- Checkpoint save/load preserving target sharding.
- Native, GPT-OSS, and MaxText PEFT export paths use gathered state.
- Only process 0 performs files, GCS mocks, and adapter HTTP calls.
- Rendered worker provisioning is equivalent to coordinator provisioning.
- No collective appears behind a process-only gate.

Run the existing Tunix, JAX, checkpoint re-registration, and preemption suites
in addition to the new tests.

## Hardware rollout gates

### H0: local/CPU gate

- All targeted Python files compile.
- New multi-process tests pass repeatedly without hangs.
- Existing Tunix/JAX/resume tests remain green.
- Shell configs render the intended meshes and four-row global batch.

### H1: v6e-8 environment smoke

Use one host briefly as TP8/FSDP1. This is not the production topology; it only
proves the v6e image, eight-chip device geometry, MaxText build, Orbax restore,
FLCE patch, tokenizer, and compile cache before debugging distributed failures.

### H2: v6e-16 TP8/FSDP2 correctness

Use two physical trainer hosts/four TPU VMs and a four-row payload:

- Confirm global/local device counts and logged TP/FSDP groups.
- Create one LoRA model and record per-process HBM after load and after template
  construction.
- Run one forward, one banked forward/backward, and one optimizer step.
- Verify local row count 2, gathered adapter hashes, loss/logprob parity, and
  optimizer-step count.
- Save training and sampler checkpoints, restart both trainer processes, load,
  and repeat the hashes and next-step loss.
- If external inference is available, publish the adapter and sample through
  every endpoint.

Do not advance on a collective hang, duplicate rows, divergent adapter state,
or checkpoint mismatch.

### H3: mixed v6e-32 end-to-end cell

Run workers 0-3 as TP8/FSDP2 and workers 4-7 as inference. Stream four-row
training payloads from the real RL client. Validate:

- Sampling-to-training overlap remains intact.
- Adapter publication reaches every serving engine exactly once.
- No new compile shape appears merely because the request came from the live
  client.
- Throughput, trainer duty cycle, LoRA publication latency, HBM headroom, and
  GCS cache behavior are recorded per model.

### H4: training-only v6e-32 TP8/FSDP4

Run all four physical hosts/all eight TPU VMs as the trainer and point it at a separate serving fleet.
Keep the global batch at four, giving one row to each FSDP rank.

Compare against H2 at the same model, lengths, four rows, and seed:

- Peak and steady HBM per chip.
- Forward/backward wall time and trained tokens/sec.
- Cross-host collective time.
- Compile time and cache-hit behavior.
- Loss, gradient norm, and adapter-delta parity.

FSDP4 is accepted only if it is stable and its capacity/recovery benefit is
worth the additional cross-host communication. It is not assumed to be twice
as fast as FSDP2.

### H5: spot and resume qualification

For both H3 and H4, force termination at these points:

1. During initial compile.
2. After sampling cache publication but before optimizer step.
3. After optimizer step but before checkpoint upload completes.
4. After checkpoint upload but before local registration completes.
5. While one nonzero trainer process is running a collective.

The replacement must either resume the last complete step or deliberately
replay one incomplete step. It must never combine mismatched model/optimizer
states, skip a committed step, publish a partial adapter, or resample a
completed rollout batch.

## Measurements and decision record

For every hardware run, save the following beside the Jobman attempt record:

- Git SHA, code-bundle object/version, MaxText SHA, TPU image, model, LoRA rank,
  sequence shape, token budget, FLCE tile, and `num_vocab_tiling`.
- Physical device list and logical mesh mapping.
- Per-process HBM at model load, LoRA creation, compile peak, and steady state.
- Compile duration and local/GCS cache hit status.
- Forward/backward seconds per four rows and trained tokens/sec.
- FSDP/TP collective timing when a profile is captured.
- Adapter export/push duration and inference throughput.
- Checkpoint upload, restore, and total spot-recovery duration.

Keep TP8 as the selected topology through H2. Only open a TP4 comparison if
TP8 fails for a diagnosed model-layout or communication reason; do not change
TP, FSDP, batch size, sequence length, and memory knobs simultaneously.

## Completion criteria

This project is complete when:

1. TP8/FSDP2 runs a real four-row RL stream on v6e without correctness drift.
2. Checkpoint, PEFT publication, Jobman recurrence, sampling cache, and GCS
   restore survive forced spot loss.
3. The mixed v6e-32 layout is usable as a self-contained fallback.
4. TP8/FSDP4 runs on a training-only v6e-32 against external inference and has
   a recorded throughput/recovery comparison with FSDP2.
5. Existing v5p jobs retain their current defaults and behavior.

## Operational safety

This worktree splays from the live league code, but nothing here should be
packed into the live fleet's code-bundle object until H2 passes. Use a separate,
versioned v6e code-bundle path and separate Jobman config names during testing.
Never reuse topology-incompatible v5p/vLLM/JAX compile-cache prefixes; cache
keys and GCS paths must include model, accelerator, TP, FSDP, sequence shape,
and relevant engine count.

## Status log

- 2026-08-26: initial multi-host audit and RPC plan written; implementation not
  started.
- 2026-08-26: topology fixed to TP8/FSDP2 for v6e-16 and TP8/FSDP4 for the
  training-only v6e-32, with a global streamed trainer batch of four. Plan
  corrected to include all PEFT export paths, per-trainer-host provisioning,
  explicit external inference for FSDP4, and atomic distributed recovery.
- 2026-08-26: corrected Cloud TPU v6e VM geometry after validation: multi-host
  slices use four-chip half-host VMs, so v6e-16 has four JAX processes and
  v6e-32 has eight. Logical TP8/FSDP2 and TP8/FSDP4 are unchanged.
- 2026-08-26: implemented the generic distributed RPC/Tunix facade, global
  input/result/state handling, PEFT/checkpoint gathers, four-process trainer
  provisioning, recurrent topology controls, and the v6e-16 Qwen smoke. New
  RPC/sharding tests pass (7); existing JAX/engine/preemption/resume regressions
  pass (69) on Slurm lowprio CPU. Published the isolated v6e code bundle and
  started resumable Jobman job 000681. Hardware allocation is waiting for an
  east5-b queued-resource slot (project zone limit 40/40); east5-c has zero v6e
  spot quota and east5-a rejects v6e-16 submissions for this project.
- 2026-08-27: launched the Qwen H2 request concurrently in us-east5-b (Jobman
  000681), us-east1-d (000684), and asia-northeast1-b (000685). Each region has
  a local code-bundle object and an exact 41,892,404,608-byte Qwen Orbax mirror;
  no checkpoint is read cross-region during bring-up. The first hardware pass
  reached model creation but exposed MaxText's physical 4x4/logical TP8 mesh
  split guard; `allow_split_physical_axes=true` is now pinned. A later pass
  proved the four-host TPU-only runtime but exposed an API child-process launch
  assumption; direct-Python Tunix API/engine launch is now covered by targeted
  CPU tests.
- 2026-08-27: fixed Jobman's existing-queued-resource path so a preserved
  WAITING/PROVISIONING request is polled instead of resubmitted under the same
  name every minute. The live editable checkout and this worktree's Jobman
  submodule carry the same two passing CPU regression tests. Google service
  preemptions and failed provisioning cycles now requeue without disturbing the
  other regional queue positions.
- 2026-08-27: published code bundle v7 independently into all three regional
  buckets (SHA-256 `5725f7b50a2373f238c0855e584abc74f4db39dd8cb7a022a14e69bfa7b93cad`).
  Its H2 probe fingerprints a baseline forward, verifies an optimizer-induced
  change, restores model plus optimizer state, and deterministically replays
  the update. It also carries model-specific Gemma and Muse profiles; Muse is
  pinned to its parity-proven MaxText revision because the Qwen/Gemma revision
  does not contain Muse. Qwen H2 hardware success remains pending capacity.
- 2026-08-28: raised the production acceptance profile to rank-32 LoRA and four
  dense 22,528-token rows. The four-row request is accumulated as four
  one-sequence trainer tiles (`TUNIX_TRAIN_TOKEN_BUDGET=22528`) on
  TP8/FSDP2. Published isolated bundle v17 in east5, east1, and Asia
  (SHA-256 `95da38d6234946f63984b302934b50a03b2fb866b1677bca78715f28ff5ea48b`);
  its failure path preserves diagnostic JSON and the compile cache without
  weakening strict replay assertions.
- 2026-08-28: Qwen3.5-27B passed the full rank-32 dense-22,528 transaction on
  v6e-16: forward/backward, Adam update, exact checkpoint restore, and exact
  replay (`grad_norm=0.0072021484375`, restore/replay delta `0`). Durable result:
  `gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/qwen35-tp8-fsdp2-r32-s22528-b22528-tokamax-auto-dense-v1.json`.
- 2026-08-28: Muse-Glimmer-30B also passed the full rank-32 dense-22,528
  transaction (`grad_norm=0.005889892578125`, restore/replay delta `0`). Durable
  result:
  `gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/muse-glimmer-30b-tp8-fsdp2-r32-s22528-b22528-tokamax-auto-dense-v1.json`.
- 2026-08-28: Gemma-4-31B passed the rank-32 dense-22,528 capacity path without
  OOM (four backward tiles and Adam update, first `grad_norm=0.048583984375`)
  and restored the pre-update checkpoint, but failed strict Tokamax replay:
  replay `grad_norm=0.059326171875` and updated-logprob max delta
  `38.09947204589844`. Treat Gemma as capacity-proven but correctness-pending;
  do not loosen the replay gate without diagnosing the fused backward/reduction
  behavior.
- 2026-08-28: verified that Gemma's saved `before` and `restored` checkpoint
  payloads are byte-identical (LoRA weights, optimizer state, and metadata),
  while dropout, Tokamax's experimental scheduler, and three-step dQ reduction
  are all disabled. Bundle v18 wires the declared trainer topology through the
  smoke worker and adds a Gemma-native TP4/FSDP4 control: it keeps 16-way total
  sharding but uses the checkpoint's native four global KV heads instead of the
  TP8 path's 4-to-8 padding. The rank-32 dense-22,528 control is queued in
  east5-b under Jobman 000681 after a service preemption recycled the retained
  spot slice. Bundle SHA-256:
  `fd7033af2678f398d76241053360dca50e346b04ce99e2410bd71d5521b13f7d`.
- 2026-08-28: east5 Cloud Audit Logs showed each replacement-node attempt
  failing with provider code 8, `There is no more capacity in us-east5-b`,
  while the queued resource continued retrying. Kept that queue position and
  launched the same TP4/FSDP4 control in asia-northeast1-b as Jobman 000685.
  The Asia bucket now has a one-time Gemma Orbax mirror byte-identical in
  aggregate to east5 (`47,837,127,780` bytes) and a directly uploaded regional
  v18 bundle (`21,041,186` bytes); neither runtime reads cross-region.
- 2026-08-29: the TP4/FSDP4 full-22,528 control proved unusable as a
  discriminator because its 56.46 GiB temporary exceeded v6e's 31.24 GiB HBM;
  Gemma TP8/FSDP2 at 4,096 tokens remained exactly replayable.  Replaced the
  control with concurrent TP8/FSDP2 rank-32 probes: one real 22,528-token row
  in east5 and the production four-row streamed accumulation in Asia, each
  replayed three times.  Bundle v22 records backward-returned logprob hashes,
  global and per-leaf gradient hashes/norms, and pre-update model/gradient
  sharding metadata.  It also disables the production-only minimal backward
  response for the small smoke batch so the backward fingerprints are real.
  Scoped Slurm lowprio CPU tests pass (10).  Regional objects are byte-identical
  (`22,513,474` bytes, SHA-256
  `915fc632a63648b360257a987d23ccf9020ba1c870b68dab114ac2164a10dc36`);
  east5 is retrying provider code-8 capacity failures and Asia is
  `WAITING_FOR_RESOURCES` without any cross-region checkpoint reads.
- 2026-08-30: both v22 TP8/FSDP2 diagnostics acquired independent v6e-16
  slices and reproduced the same slice-wide failure immediately after their
  first completed gradient accumulation (one tile in east5, four tiles in
  Asia).  The originating worker reported an unexpected multihost launch peer
  with HLO module `jit_integer_pow`; the remaining workers then aborted with
  `SLICE_FAILURE_SW_INJECT_ERROR`.  This isolates a diagnostic/gradient-norm
  dispatch failure from both physical-slice health and Gemma's original replay
  mismatch.  Bundle v23 computes the diagnostic norm from the global gradients
  already gathered for per-leaf hashes and uses one jitted `optax.global_norm`
  program when diagnostics are disabled, avoiding hundreds of eager per-leaf
  launches.  Focused Slurm CPU tests pass (12/12).  The regional v23 objects
  are byte-identical (`25,896,140` bytes, CRC32C `K9HiCw==`, MD5
  `UaTOstzkVnZrQTiZjn5MgQ==`, SHA-256
  `ddca11b0d2db4e45afedd0edf0ed9313bd24193be5041204a365b2ec9574827c`),
  and Jobman 000681/000685 were restarted against their existing queued
  resources so the next healthy allocation restores only its regional v23
  bundle.
- 2026-08-31: bundle v24 fixed the diagnostic environment propagation on the
  three nonzero JAX workers and completed the four-row Asia Gemma probe.  Raising
  `TUNIX_TRAIN_TOKEN_BUDGET` to `45056` made the same `[2,22528]` compiled shape
  carry two real rows per tile, reducing the four-row transaction to two forward
  and two backward tiles while retaining the existing JAX cache; peak HBM was
  20.0 GiB.  Three restored replays were mutually identical but differed from
  the baseline (`0.04817155592406464` versus `0.058228973133218405`).  Per-leaf
  diagnostics isolated every mismatch to the 41 LoRA-B leaves: the pre-update
  state was replicated (`P()`), but the value-only checkpoint restored those
  leaves into the post-update target's canonical TP/FSDP placements.
- 2026-08-31: checkpoint format v2 now saves topology-relative `PartitionSpec`
  and committed placement metadata for every LoRA and optimizer leaf, and uses
  the saved layout during training and sampler restore.  Legacy v1 checkpoints
  retain the target-layout fallback.  Bundle v25 (25,893,682 bytes, SHA-256
  `c5adb38b876718da32d0ea8a048cec25ad6abcc2e5c58798cdecc52ba00a1bc5`) was
  uploaded independently to the Asia and east5 regional buckets.  The Asia
  v6e-16 TP8/FSDP2 hardware run passed rank-32, four dense 22,528-token rows and
  three checkpoint replays: all four gradient norms were exactly
  `0.04800766350685909`, and restore, backward, and post-update replay deltas
  were all exactly zero (`acceptance_pass=true`, peak HBM 20.0 GiB).  Durable
  result:
  `gs://sk7524-tinker-tpu-asia-northeast1/v6e-smoke-results/gemma4-tp8-fsdp2-r32-s22528-rows4-replays3-layout-v2.json`.
- 2026-08-31: probed a single real `[4,22528]` Gemma tile by raising the token
  budget to `90112`, with an isolated `b90112` compile-cache prefix.  The
  forward compiled and ran in 22.06 seconds and still showed the previous 20.0
  GiB runtime allocator peak, but backward compilation failed its HBM plan:
  `jit_forward_backward_fn` required 67.70 GiB of temporaries versus 31.24 GiB
  available.  Therefore two real rows (`45056` tokens) remain the maximum
  proven tile; four streamed rows continue to run as two sequential tiles.  The
  recurrent retry was stopped, the live Jobman config was restored to the
  accepted `45056` profile, and the failed-shape log is preserved at
  `gs://sk7524-tinker-tpu-asia-northeast1/v6e-smoke-logs/gemma4-tp8-fsdp2-r32-s22528-rows4-one-tile-b90112-oom-v1.log`.
