# Gemma and Muse science training configuration audit

Prepared 2026-09-15. These are configurations to validate, not claimed TPU successes.
The running Qwen bundles are immutable and do not receive these code changes.

| Model | Training chips | TP | FSDP | Padded rows per microbatch | Token budget | KV layout |
|---|---:|---:|---:|---:|---:|---|
| Gemma 4 31B | 16 across four hosts | 4 | 4 | 4 | 90,112 | Native local 16 x 256; global 4 x 512 |
| Muse Glimmer 30B | 16 across four hosts | 8 | 2 | 2 | 45,056 | Native 2 x 128, repeat-interleaved to 8 tied logical heads |

Both retain LoRA rank 32, seed 1, full rematerialization, learning rate 4e-5,
18,432/22,528 sequence buckets, context 22,528 and phase-one cap 16,384.
Full discovery remains 16 groups x 32 answers, one request per group,
importance_sampling, mean_baseline, prefix caching and 16 active sequences per
inference engine. The replay driver uses recorded data instead of those 512
new rollouts. It does not change tokens, advantages, masks or behavior logprobs.

## Why these choices

Muse job 894 failed with 75.38 GiB temporary HBM for the short shape and 94.10
GiB for the long shape, against 30.75 GiB available. Its TP2/FSDP8 mesh padded
microbatches to eight rows even when the token budget suggested fewer. Lowering
the token budget below that minimum does not eliminate the padding. TP8/FSDP2
reduces the minimum to two rows and increases tensor partitioning. The actual
memory reduction must be measured; dividing the old HLO estimate by four is
not a valid guarantee.

The pinned Muse MaxText loader already repeats checkpoint KV heads when the
logical kv_heads axis is expanded. The backend now also ties KV LoRA gradients,
Adam moments and native-shape export for Muse. This preserves the model's
native two-head policy rather than allowing the copies to learn independently.
Local eight-device gradient tests, multi-step Adam equivalence, and the actual
Muse export mapping pass. Full-model TPU backward and export/inference remain
required before declaring this configuration usable.

Gemma is not a uniform-head model: the pinned configuration has sixteen local
KV heads and four global KV heads, with different head dimensions and shared
KV projections in global layers. A generic TP8 override of base_num_kv_heads
would both miss the global constraint and potentially reduce native local
heads. Keep TP4/FSDP4 first. Preserve the 256-token attention backward tiles and
dq_reduction_steps=3, use a 512-token fused-cross-entropy tile, and deploy the
current distributed-input-sharding backend in the replay bundle. Gemma's old
science job was cancelled before training; it provides no 22k success evidence.

## Validation artifacts

- profiles/science-gemma-v4-tp4-fsdp4-replay-001.json
- profiles/science-muse-v4-tp8-fsdp2-replay-001.json

Profiles live under tpu/swarm/ray_train. Each replay runs its recorded requests
twice, requires finite nonzero optimizer gradients, exports an updated adapter,
and generates 64 tokens with aligned finite logprobs. The Gemma fixture comes
from native-v5p-gemma-ac2-grpo-lr15e4-s1-002 and is a compiler regression input,
not evidence of science-task quality. The Muse fixture preserves the complete
eight-example science-placement-v4-muse-grpo-002 requests 4 and 5 from job 894.

Pinned model sources inspected:
- Gemma MaxText 0fd409939977ac0ab79a4e64d21730936f253567
- Muse MaxText 4f65ba509
- Muse HF config a4e59da52a7bc87ae7251dd5545c0dd437c44b68

Source snapshots and the read-only Muse failed-run database are retained under
.science/model-config-audit. CPU tests establish transformation correctness;
they do not establish TPU memory fit or throughput.
