# Native thinking caps for the RG-LRU sampler

Ported from `SkyRLTpu-thinking-budget`, parent branch `agent/thinking-budget-optin`
(commit `450718ba`) and TPU inference core commits `ec325cfdc` and `012a2cc5d`.
The source checkout and its uncommitted benchmark work are left unchanged.

This package applies only the sampler/runner diff to an isolated copy of the
SHA-pinned serving bundle. It does not replace the complete TPU inference fork.
`patch --dry-run --fuzz=0` must pass first. All patch/module bytes enter the
source identity, so an old serving installation cannot satisfy the new marker.
Ordinary profiles retain their original entrypoint and sampler.

The Ray v2 sampler sends one token-ID completion request with
`thinking_token_budget`. Control is fused into the sampler JIT, retains the
request and KV cache, and emits the forced transition during ordinary decode.
There is no second completion or repeated prompt prefill. Stable decode adds no
control-state host/device transfer or separate JIT dispatch. Exact throughput
of the extended marker logic remains to be measured on TPU.

The API now matches the default (`min_think_tokens=0`) Gemma/Muse/Qwen
two-phase completers' forced continuation and marker recognition. The earlier
job 703 used bare transitions and the original reasoning-only counter; its
measurements do not validate this subsequent compatibility change.

The forced prefix is the model's `THINK_CLOSE` followed by the exact inherited
`ANSWER_CUE`: `Here is the final complete program:\n\n` and an opening Python
code fence plus newline. Constants are checked against the actual completer
classes in the regression suite.

| Model | Original reasoning marker | Completer's close substring | Forced transition before ANSWER_CUE |
| --- | --- | --- | --- |
| Qwen | `<think>` | `</think>` | newline, close marker, blank line |
| Gemma | `<\|channel>thought` plus newline | `<channel\|>` | `<channel\|>` |
| Muse | ` to=self<\|message\|>` | `to=user` | `<\|eom\|><\|start\|>assistant to=user<\|message\|>` |

For API requests, the budget counts every generated token before the answer
marker, including model-emitted channel headers. It does not wait for an opening
thinking marker. Markers inside the input prompt are ignored, matching the old
client's check of generated output only. A literal closing marker inside the
reasoning also releases the cap: this deliberately preserves the existing
completer's substring semantics. At a cap inside an incomplete marker, the
complete forced prefix is appended, matching the old client rather than repairing
the partial header. The original channel-aware mode remains available only to
the low-level control/oracle tests, not through the default API entrypoint.

Decoded substring matching is compiled once per tokenizer into a shared token
class table and small finite-state table. This recognizes markers across token
boundaries, including alternate tokenizations, without decoding output strings
on the host each step. The tables are immutable device inputs to the fused
sampler, shared across requests. They add roughly one MiB per chip (the replicated token
class table), plus a small transition table, and no per-token state transfers.
The supported ASCII markers contain no whitespace; tokenizer decoding retains
special tokens and disables whitespace cleanup. Fixture cases retain decoded
text from the actual pinned tokenizers for comparison with client methods.

`thinking_token_budget` is the generated-token phase-one allowance, i.e.
`phase1_max_tokens - actual_prompt_tokens`. The audit labels its basis as
`phase1_generated_tokens` and exposes `counted_phase1_tokens`; the legacy
`thinking_tokens` field is retained as an alias and now includes emitted headers.
The caller supplies the total generation allowance. Matching the existing
completer's continuation context margin requires subtracting its 50-token
buffer; this API does not silently change the caller's `max_tokens`.
Minimum-thinking continuation forcing is not implemented.

Inference keeps each sequence on its original engine through the answer.
Our non-streaming `n>1` response still waits for all choices, although finished
sequences stop decoding individually. The arena uses independent `n=1` requests
and submits each result for grading as soon as it arrives. The existing training
`sample_group` path awaits phase one and all phase-two continuations before
starting rollout environment steps. Native control does not itself remove that
response/grading barrier or add streaming. Prefix caching can already save
repeated prefill in the old path, so an end-to-end speedup needs a matched
comparison; grouped sampling and native continuation are separate choices.

Responses retain token IDs, forced positions, raw model logprobs when requested,
`loss_mask` (zero for injected tokens), and an enforcement audit. The sampler
saves these and rejects a missing/over-budget audit. Training enablement is
rejected until a training client explicitly preserves the forced-token mask.

## Rerun profile

`profiles/rglru_three_models_v4_32_thinking.json`, run ID
`rglru-three-models-v4-32-006`:

- 32 independent samples/model; concurrency 8/model; one TP4 engine/model.
- 32,768 generated tokens total, with a maximum of 20,000 thinking tokens.
- Forced transition tokens count toward the total; early closure leaves more
  room for the answer. The input prompt is separate.
- 40,960 engine context to accommodate the largest prompt plus full output.
- Same task, seed, graders, temperature, top-p, chunk sizes and 0.8 memory fraction.
- Same HF caches; prior run's compile caches seed distinct new writeback prefixes.
- Short warmups use a cap of 8 and 64 total tokens and are saved separately.

The original 8k profile remains unchanged. This is a source integration, not a
claim that the new three-model TPU configuration has passed. The previous v4-32
pool has no ready workers and the target-zone v4-32 VM inventory is empty at the
integration check. No pool targets, other workloads or reservations were changed.
The earlier post-completion Ray status-actor shutdown error is separate and is
not fixed by this port.

## Verification

`tests/tpu_swarm/test_native_thinking_budget.py` ports the original control
regressions, executes the actual patched sampler on CPU, and extends coverage to
20,000-token caps, real-tokenizer markers, multi-token natural closure, partial
transitions, masking and one-request API handling. The fixture archive comes
from the exact run005 base bundle; its provenance is beside it.

`test_ray_train_arena_sampling.py` checks model configuration, source packaging,
context headroom, preserved cache seeds and baseline settings. Hardware canary
and throughput measurements remain pending.

## v4-64 submission profile

`profiles/rglru_three_models_v4_64_thinking.json` runs the same 96-candidate
comparison on one eight-host slice. Rank 0 grades on its four chips; ranks 1-2
serve Qwen, 3-4 Gemma, and 5-6 Muse as independent TP4 engines. Rank 7 joins the
executor but advertises no TPU resources and runs no model. Ray Serve balances
requests across each model's two replicas. There is one sampling cohort per
model: 32 samples and concurrency 8 total per model, not per replica.
Generation throughput therefore measures two engines/model and must not be
compared directly to the prior one-engine/model numbers. Kernel grading remains
single-chip with the original six cases and forward/backward contract.
