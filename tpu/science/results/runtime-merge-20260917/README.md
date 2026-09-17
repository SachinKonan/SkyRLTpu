# Shared v4-64 science runtime deployment — 2026-09-17

Merged runtime-parity into science-placement at `7ebc1e9e`; Discover merge
`cf6db33f0ac96931415722f8b9006f086c27159d` preserves search-ahead recovery and
traceback logging alongside the audited native-completion contract.
The parity worktree and its uncommitted fourth-review notes were not modified.

User requested cancellation of existing runs and a common runtime for all six
Qwen/Gemma/Muse × circuit/qubit jobs on v4-64. Cancelled IDs 974–976;
Gemma 975 ended FAILED_CONTROLLER during cancellation, others CANCELLED.
New submissions: circuit Muse 977, Qwen 978, Gemma 979; qubit Qwen 980,
Gemma 981, Muse 982. Submission is not evidence of an optimizer step.

## Preserved settings and recovery

Four trainer hosts and four TP4 inference engines per slice. Qwen/Muse trainer
TP8/FSDP2; Gemma TP4/FSDP4. 16 groups × 32 generations; 16,384 prompt-plus-thinking
ceiling; 22,528 total context. Inference max sequences 16/engine, 1,024-token
chunks; memory utilization .80 Qwen/Gemma, .75 Muse. Backward warmup stays off.
CPU grading: 16 slots/host, 4 CPUs and 8 GiB/task, for both science tasks.
Circuit reuses Xplace inputs and CPU-verified seeds: Qwen 18/32, Gemma 3/32,
Muse 9/32. Seeds staged with no-clobber upload and exact readback verification.
Qubit preserves original run namespaces: Qwen checkpoint 3, Gemma checkpoint 1
plus newer search state, Muse six verified seeds and no completed checkpoint.
Fresh local roots force restore from durable state. Bounded shutdown, atomic
client restore, pending-request retirement, and capped retries are preserved.
Circuit IDs use the requested terminology; internal task identifiers stay
unchanged for compatibility.

## Validation

Broad merge check: 282 passed, 2 skipped, 5 subtests passed; one pre-existing
recovery fixture assumed CPU affinity outside the Slurm allocation. Explicitly
mocked that hardware precondition and reran all 11 recovery tests: all passed.
The shutdown regression now also checks recorded timeout errors.
All six packages pass native contract checks and contain their installed prompt
and recovery module. All packages record the same parent/Discover commits;
circuit/qubit installed source overlay manifests match per model.
Authentication, ADC refresh, TPU/storage reads and SkyPilot GCP checks passed.
Read-only host audit passed on all eight hosts of workers 385, 398, 406.
Workers 392/397 lost SSH configs and 402 timed out as pool capacity changed;
these are not claimed clean. Each new launch also performs its per-host audit.

This deployment does not claim numerical equality with the legacy v5p stack.
Dependency recipes remain pinned as before; no unverified Transformers upgrade,
legacy-environment baseline, or optional backward warmup was enabled.
