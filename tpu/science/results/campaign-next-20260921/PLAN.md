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

## Supervised AC2 follow-up

`watch_ac2.py` wraps the same fetch/build/submit commands. Arm it after review
with `--arm` to record HEAD and a checksum of the tracked working diff (excluding
the changing completion observation). It polls the live completion gate every
60 seconds. Once all three originals succeed with checkpoint 10 and final pools,
it freezes the sources, builds on a CPU Slurm allocation, independently checks
the winner, profiles, identical seeds, bundle hashes and ten-step settings,
then submits and confirms the receipts against the live queue.

`skyrl-ac2-followup.service` runs this watcher. Observation-command failures are
retried; source changes, preparation errors, and uncertain submission intent
stop it in `needs_review`. It has no automatic restart on failure, never cancels
jobs, and never retries an uncertain launch. Fully confirmed partial receipts
can be resumed without resubmitting those runs. The service exits after all
three follow-ups have confirmed queue entries; experiment monitoring continues.

Runtime evidence is under `.science/ac2-shared-best-20260921/`. A future code
change requires revalidation and explicit replacement of the source guard
before restarting this service. Do not bypass a `needs_review` state without
checking its cause and the existing submission receipts.

Validation: 16 focused handoff/watcher tests passed on CPU Slurm job 14219921,
including unfinished-gate refusal, source-change refusal, uncertain-intent
refusal, and receipt-to-live-queue identity checks.

## East maintenance recovery

The 06:22 UTC east maintenance interruption displaced all original east jobs.
Gemma and Muse AC2 were still waiting for replacement capacity while central
had idle workers. Their original run IDs and immutable bundles were resumed
in central as jobs 1363 and 1364, from saved steps 2 and 4, after cancellation
of the old jobs 1335 and 1336 was verified. Only task placement changed; see
`../ac2-central-recovery-20260921/README.md` and its receipts. Qwen retains its
existing east recovery. The completion gate follows the newest job per run ID.

Gemma RG-LRU's displaced job 1343 was likewise still waiting in east. It was
resubmitted as 1365 on central worker 176 with its exact repaired bundle and
verified 29-candidate bootstrap, after cancellation was confirmed. It has no
optimizer checkpoint yet and retains ten total steps; see
`../rglru-central-recovery-20260921/README.md`.

## Central capacity update

A third central worker became READY while Muse successor 1361 was unassigned
in east. That entry was cancelled before replacement job **1362** was submitted
to central with the same seeds and training recipe, still capped at 10 steps.
See `../campaign-muse-central-20260921/` for the exact profile difference, seed
checksums, cancellation evidence, and new submission receipt. Jobs 1359 and
1360 are now RUNNING and preparing caches/reference checks.
