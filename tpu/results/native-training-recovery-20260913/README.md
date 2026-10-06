# RG-LRU and native training checkpoint

This branch consolidates the recurrentgemma integration and the deployed native
training sweep into Git. Its parent is `agent/recurrentgemma-env` at
`9665d86aa217e7618ccb732945eb813a26461de0`, originally created from
`agent/gptoss-multi-lora`. The original dirty worktrees were left intact.

## Source history

- `SkyRLTpu-recurrentgemma`: RG-LRU environment/grader, Ray v2 integration,
  model sampling, and native thinking support. Its tracked and untracked
  `tpu/` and `tests/` changes were copied before applying the deployment snapshot.
- `frozen12-v4-64`: frozen inference workloads, incorporating recurrentgemma
  runtime/native code and AC2 grader dependency fixes.
- `smoke9-v5p` and `smoke9-v5p-tp4-ac2`: training/inference smokes, sequence
  buckets, and Gemma TP4/FSDP1/DQ-reduction memory configuration.
- `native-training-tp2`: native training transport/SDK/masking, Muse TP2,
  adaptive PWC, answer-only extraction, and sweep profiles.
- `native-gemma-grid`: deployed runtime plus Gemma profiles, excluding the
  unfinished dedicated RG grader-actor prototype.
- `native-external-fix`: corrected external inference forwarding and regression
  tests. This is the source of the replacement jobs recorded here.

The latter directories were source snapshots, not Git worktrees. The separate
`third_party/discover` commit records prompts, adaptive PWC, native completer,
rollout handling, and answer extraction. Its gitlink is part of this checkpoint.
The unfinished dedicated RG grader-actor prototype remains in the original
`native-training-tp2` snapshot; it is not a deployed or validated feature here.

## Confirmed defect and correction

Training used `ExternalInferenceClient`, while the initial native integration
patched `VllmSamplingClient`. The external proxy omitted the outgoing
`thinking_token_budget` and discarded returned `loss_mask`/`thinking_budget`.
Saved EXTERNAL futures and code on failed worker 454 confirmed the omissions.
Native clients rejected these results before training.

Both external forwarding paths now propagate the budget, validate enforcement
evidence, preserve masks/audits through output serialization, and zero behavior
logprobs only for forced tokens. Native client parity checks remain enabled.

## Validation

From this Git checkout: 119 native control, API, forwarding, serialization,
and actual-completer regression tests passed; one optional pinned-tokenizer
test was skipped. The discover advantage and answer-extraction tests also
passed (40 tests). Forwarding tests execute the actual forwarding method bodies
with mocked HTTP model responses and real output schemas. They are not hardware
end-to-end tests. The user authorized direct sweep replacement without waiting
for separate per-model training smokes.

Every source-overlay file and profile in all 24 replacement archives was checked
byte-for-byte against this checkout. `deployment-record.json` records immutable
archive URIs, source hashes, profiles, and SkyPilot IDs. The runtime uses its
profile-pinned base source bundle; an archive is not an export of the entire
Git checkout.

## Deployment

The 24 replacements cover Qwen, Muse, and Gemma; AC2 and circle packing n=32;
GRPO and adaptive PWC; learning rates 4e-5 and 1.5e-4. They retain seed 1,
15 iterations, 16 parent groups x 32 completions, 22528 context, 16384
prompt-plus-thinking first phase, and trainer buckets 18432/22528.
RG-LRU sweep profiles remain deferred.

All old sweep attempts were terminal before replacement. The 11 still-active
attempts were cancelled; the other attempts had already failed or been cancelled.
Replacement IDs are 778 through 801 in `tpuswarm-v5p32-east5a-erdos`.

The launch task prepends `clean_host_audit.py` on every host before starting the
executor. This checks 30 GiB free home storage, 10 GiB free temporary storage,
free inodes, TPU ownership, and recognizable leftover workload processes.
The executor additionally locks its root, stops its own stale Ray processes,
checks ports, and performs a host preflight barrier. Preserve this launch-task
preamble when rebuilding these jobs; it is separate from the code archive.
No unrelated workloads or caches are automatically deleted by the audit.

Submission is not proof of a successful optimizer step or updated-adapter
inference. Live monitoring records remain under
`tpuswarm-state/benchmarks/native-external-fix/sweep/`.
