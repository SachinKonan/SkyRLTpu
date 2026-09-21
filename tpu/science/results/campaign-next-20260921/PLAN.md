# Continuing the ten-step science campaign

Authorized goal: monitor v6e central/east, v4-64, and dedicated v5p-64;
keep successors queued in priority order; follow completed AC2 with models
starting from the best prior solution. Every new run here is capped at 10 steps.

- v6e east: preserve current Qwen/Gemma/Muse AC2 and Gemma RG-LRU recovery.
  Qwen RG-LRU job 1342 is SUCCEEDED with saved checkpoint/metrics step 10.
- v6e central: use two idle workers for new Gemma and Qwen qubit runs.
- v6e east successor: Muse qubit, lower priority (100) than AC2 (120) and the
  existing RG-LRU recovery (110). No existing run is cancelled or overwritten.
- v4-64: preserve Qwen/Muse qubit and queued Gemma qubit. These deployments use
  their current local inference; changing running jobs to borrow is not part
  of this submission.
- Dedicated v5p-64: preserve the managed Gemma -> Muse -> Qwen circuit queue.
  Gemma attempt 5 has passed cache preparation and reached client execution.
- v5p-32: separate existing experiments and inference-farm jobs are preserved.

The three v6e qubit jobs use fresh adapters/optimizers and each model's original
verified bootstrap pool, matching the seed source of its prior qubit campaign.
They have new run IDs, independent checkpoints and compilation-cache write
prefixes. Existing compatible caches are read as seeds. Same 16 groups x 32
rollouts, native thinking budget, 22,528 context, GRPO/importance_sampling,
full 72-case grading, and model-specific learning rates remain.

New jobs opt into the existing run-scoped inference-farm admission/discovery
service. They wait at most 300 seconds for initial borrowing, then use local
inference while discovery can add compatible free farm capacity. Existing AC2
leases remain higher priority. Deployment validates bundle bytes, seed bytes,
identity, provider read access, and duplicate run IDs before submission;
worker startup runs the existing clean-host audit.

## AC2 second round

Do not take a moving in-flight winner as a final result. After the original
three AC2 runs finish their 10 steps, inspect terminal status, checkpoint 10,
and final candidate pools. The proposed follow-up gives all three models the
same highest-scoring valid construction from those completed runs, resetting
PUCT visit counts and optimizer/adapter state, for 10 additional steps in new
run IDs. Preserve source run, state ID, score, source generation and checksum.
Recompute the construction's AC2 score before promotion. This does not count
as launched until submission receipts and live queue records exist.

The shared-winner interpretation is the current default, communicated to the
user; they may still steer it before the first round completes.
`check_completion.py` checks live terminal status, durable checkpoint 10, and
the final step-10 pool for all three original AC2 runs.

## Submission receipts

- 1359: Gemma qubit, central, priority 100, 96 original bootstrap seeds.
- 1360: Qwen qubit, central, priority 100, 46 original bootstrap seeds.
- 1361: Muse qubit, east, priority 100, 48 original bootstrap seeds.

All three immutable bundles and seed objects were uploaded and read back with
matching SHA256 checksums. Archive inspection verified the ten-step limit,
seed pin, optional farm admission, and compilation-copy retry fix. Focused
seed/farm validation passed 62 tests plus three subtests (Slurm 14217483).
No existing jobs were cancelled.

## Executable AC2 handoff

`prepare_ac2_handoff.py fetch` checks live completion and downloads all three
final pools at pinned GCS generations. It currently reports WAIT and creates no
handoff while the first round is running. Once ready, run its `build` action on
a CPU allocation. `tpu.science.ac2_handoff` chooses the maximum valid saved score,
recomputes it with the trusted AC2 evaluator without executing candidate code,
and resets PUCT history. All three recipients get byte-identical seed pools and
new run IDs with fresh adapters/optimizers. Source run, state, generation,
verifier hash, and candidate hash are recorded. Configurations retain their
model-specific optimizer recipe and use independent cache-write prefixes.

Inspect the resulting `ac2-shared-best-20260921/prepared.json` and provenance,
then use `submit.py --ac2-handoff` for the same upload verification, duplicate
checks, durable submission intent and receipts as the qubit campaign. Its AC2
priority is 120. This handoff is not yet built or submitted.

Validation: six focused tests passed (Slurm 14217850), including unfinished
round rejection, saved-score mismatch rejection, construction-only evaluation,
fresh root history, recipient isolation, and preserving the source recipe.
