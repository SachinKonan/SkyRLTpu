# Multihost handoff: inherited League work and GPT-OSS acceptance

This document is the source of truth for handing `agent/tunix-multihost` to a
worker whose checkout predates the 2026-09-03 rebase.

## Synchronizing a pre-rebase checkout

The former remote tip was `2165c10b`. The first force-pushed, post-rebase tip
was `0379e0e5` and has this shape:

```text
29601970
  `-- 29 League commits through 45cada2d
        `-- replayed 40 multihost commits through 2ad1eb21
              `-- post-rebase integration fix 0379e0e5
```

Do not merge or rebase the old and new multihost histories. If the old
checkout is clean, preserve its current head, fetch, and adopt the remote:

```bash
git branch backup/multihost-before-rebased-sync
git fetch origin
git reset --hard origin/agent/tunix-multihost
git submodule update --init --recursive
```

Preserve any dirty changes before resetting.

## League history newly visible to the remote branch

The old multihost remote did not contain the 29-commit range
`29601970..45cada2d`. Inspect it directly:

```bash
git log --reverse --oneline 29601970..45cada2d
git diff --stat 29601970..45cada2d
```

The executable changes include:

- `discover@a2b529e`: `parent_baseline_adaptive_beta`, centered on the parent
  state instead of a group statistic.
- `discover@636b7de`: bounded `parent_relative` credit. Mode C is MAD-scaled;
  mode D is piecewise.
- `discover@aa06361`: mode E, incumbent-baselined entropic expected
  improvement, and float64 deltas for all parent-relative modes.
- Superproject `80121b8b`: six C/D cells across Qwen, Gemma, and Muse.
- Superproject `5f8d18cf`: three estimator-E cells at each model's working
  GRPO learning rate.
- Superproject `3bb20af5`: corrected stale Jobman identities in all 15 cloned
  cell configs.
- Superproject `afdb1cc1`: `META_SEED_ONLY=1`, which prevents a failed cold
  tree from replacing the intended seed before a generation banks metrics.

The inherited evidence and analysis include:

- [`results/meta-tree/STATE-2026-09-03.md`](../results/meta-tree/STATE-2026-09-03.md):
  independently reverified 53-run fleet audit and Stage A-E state.
- [`results/story/advantage_walkthrough.html`](../results/story/advantage_walkthrough.html):
  estimator derivation and cross-model evidence.
- [`results/story/single_agent_ledger.html`](../results/story/single_agent_ledger.html):
  experiment chronology.
- [`results/story/chasing_c5.html`](../results/story/chasing_c5.html): research
  narrative and controlled conclusions.
- [`results/erdos-records/record_mmeta_0380859049.json`](../results/erdos-records/record_mmeta_0380859049.json):
  exact overall verified C5 construction, `0.38085904871926474`. This is a
  seeded Muse result; Qwen `lr-n` at `0.380859354887961` remains the best
  verified from-scratch cell.

The audit says the mutable `stagea-league.tar.gz` bundle was built at
`afdb1cc1`, while estimator E and `discover@aa06361` were committed later.
Verify a deployed bundle's manifest or republish an immutable bundle before
relaunching League cells.

## GPT-OSS status

The implementation and acceptance design are documented in
[`gpt-oss-mxfp4-lora.md`](gpt-oss-mxfp4-lora.md). The runtime/queue contract is
documented in [`tpu/swarm/README.md`](../tpu/swarm/README.md) and
[`third_party/TPUSwarm/README.md`](../third_party/TPUSwarm/README.md).

- GPT-OSS 20B sparse-LoRA training passed live on v6e-16, TP8/FSDP2. Its
  durable trainer result reports `acceptance_pass: true` at
  `gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/gptoss20b-sparse-lora-tp8-fsdp2-r32-s256-0e5e43a3-d388c5478.json`.
- The live vLLM MXFP4/LoRA gate is being run on v6e-8. Do not call inference
  validated until the result below exists and reports `acceptance_pass: true`:
  `gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/gptoss20b-vllm-mxfp4-lora-v6e8-v1.json`.
- GPT-OSS 120B checkpoint conversion is complete. The marker exists at
  `gs://sk7524-tinker-tpu-asia-northeast1/skyrl-maxtext-ckpts-gptoss120b-bf16-d388/gpt-oss-120b/CHECKPOINT_COMPLETE`.
- GPT-OSS 120B training is not yet accepted. Success requires
  `gs://sk7524-tinker-tpu-asia-northeast1/v6e-smoke-results/gptoss120b-sparse-lora-tp8-fsdp4-r32-s1024-v1.json`
  with `acceptance_pass: true`.

The v6e-8 inference gate proves native MXFP4 load, zero-adapter parity,
nonzero expert-factor execution, A-to-B-to-A replacement without drift,
ordinary router LoRA, expert clear, and final immutable-base parity. The
client is [`tpu/gptoss20b_vllm_lora_smoke.py`](../tpu/gptoss20b_vllm_lora_smoke.py)
and the remote runner is
[`tpu/run_gptoss20b_vllm_lora_smoke.sh`](../tpu/run_gptoss20b_vllm_lora_smoke.sh).

## Remaining operational warning

The local TPUSwarm and SkyPilot API services were stopped after a consolidation
mode launch expanded to 64 controller processes and exhausted host threads via
NumPy/OpenBLAS. Cap or repair that controller runtime before restarting it.
Do not claim GPT-OSS 120B complete until the full v6e-32 result is durable.
