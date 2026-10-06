# Muse qubit recovery and idle v4-64 workers — September 23, 2026

Muse job 1600 completed optimizer step 15 and saved its final checkpoint, but
the executor rejected successful completion during final run-state writeback.
The subsequent recovery failed the disk admission check on worker 726. Its
replacement is job 1625, continuing the same run to 25 total updates with strict
checkpoint resume from at least step 15 and the existing farm-enabled recipe.

## Evidence and failure sequence

- Worker 747's client logged `done — loop returned without raising`; the
  controller recorded `client_finished` with exit code 0.
- The only recorded completion failure was `final_run_writeback` on rank 0:
  `previous run-state writeback is still running`. Periodic database snapshot
  creation had not returned before the final flush's 330-second lock wait.
  The exact cause of the slow snapshot is not established by this repair.
- GCS contains `model_30eabd6a/000015.tar.gz` (2,331,471,244 bytes),
  `model_30eabd6a/final.tar.gz` (2,331,471,243 bytes), the step-15 PUCT snapshot,
  metrics/checkpoint indices, and the portable database backup.
- Best saved reward is 0.5255169946709173. This recovery does not reset the
  optimizer or search pool and does not change the learning/grading recipe.
- The restart on worker 726 failed before training: host `t1v-n-48f3c6fb-w-3`
  (SSH alias worker4) had only 12.39 GiB free.

## Worker inventory and disk repair

The pool showed five READY workers but only three running jobs: circuit Gemma
1591 on 744, circuit Qwen 1602 on 761, and qubit Gemma 1603 on 765.
Muse qubit 1600 and AC2 Muse 1624 retained assignments to workers 726 and 747
after failed local attempts. All sixteen hosts of those two workers were
reachable, with no TPU device owners or running workload systemd services.
Worker 747 had sufficient disk space; this was not evidence of two broken TPUs.

On worker 726's affected host, old Gemma checkpoint archives occupied about
40 GiB. Removed 18 obsolete local archives only after matching each archive's
size and MD5 against GCS metadata. Kept local `final` and `000010` archives,
all databases, logs, search state, and all GCS objects. Reclaimed 32.845 GiB;
free disk increased to 45.237 GiB. No active workload was stopped for cleanup.

Job 1600 was cancelled after durable-state verification. Once its reservation
was released, AC2 1624 was reassigned to repaired worker 726. We did not cancel
AC2, alter circuit jobs, restart the shared Sky API, or delete TPU resources.

## Continuation handling

`qubit_continue25.source_ready` now supports an explicit audited shutdown-failure
handoff. It requires a terminal old job, the exact source job/run/step, successful
client exit, a final-writeback failure, and a hash-pinned local evidence file.
The existing independent checkpoint, metrics, PUCT, archive, immutable bundle,
and single-active-writer checks still run before submission. It does not change
SkyPilot's failed/cancelled job history into a success.

Validation: 13 continuation tests passed on Slurm CPU job 14315196, including
rejecting active old writers, absent evidence, changed evidence, and mismatched
source identities.

The replacement uses the already prepared immutable bundle
`e79448782d48d6ebd4739609a81221c72b9ac2a18a9e98321f50f4132c058324`.
This repair does not claim a general fix to slow runtime writeback or the pool
scheduler's stale-reservation behavior. Runtime code in that bundle is unchanged.

Private operational evidence is under
`.science/routing-relaunch-20260921/muse-repair-20260923/`; the replacement launch
receipt is `.science/routing-relaunch-20260921/extend25/1600/submitted.json`.

## Verified after launch

Both assigned workers then had existing `sky.exec` requests stuck PENDING.
Validated their job IDs, run names, worker assignments, and task bundle hashes,
and dispatched those exact requests through SkyPilot's normal request executor.
Both requests completed successfully; no duplicate managed jobs were created.

- Worker 747: replacement Muse qubit 1625, local job 5 RUNNING. All eight hosts
  passed clean-host admission and were compiling routing dependencies.
- Worker 726: existing AC2 Muse 1624, local job 6 RUNNING. All eight hosts passed
  clean-host admission and were installing the CPU grading environments.
- The final pool query showed all five assigned jobs RUNNING: 1591, 1602, 1603,
  1624, and 1625. The two recovered jobs were in startup, not yet proven to have
  resumed sampling or completed an optimizer update.

This did not require directly editing pool reservation records. Cancelling the
completed Muse attempt and normal reassignment released the blocked capacity.

## Subsequent checkpoint-registry failure

The first attempt of job 1625 reached client startup but failed strict resume:
the API returned HTTP 404 for `model_30eabd6a/000015`. The client checkpoint index
and GCS archives were at step 15, while the portable database snapshot contained
only registrations through step 14. A snapshot uploaded after step 15 had begun
earlier, so its upload timestamp did not establish registry freshness. The
original stopped job1600 database still contained all completed step-15/final
registrations and the exact original model metadata.

The previous continuation preflight checked existence of the portable database
but did not check the selected checkpoint's registration inside it. Added
`require_checkpoint_registry` and a generation-pinned database download before
new continuation submission. Missing models, missing checkpoints, and pending
or failed checkpoint registrations now reject submission. The expanded suite
passed all 18 tests on Slurm CPU job 14317229.

The targeted repair script under `muse-repair-20260923/registry-fix/repair.py`
holds the pool scheduling lock, verifies all eight hosts are idle and there are
no in-flight launch requests, preserves the original GCS database under a unique
recovery prefix, verifies the four saved archives against GCS MD5/size, copies
only their completed registry rows from the original database, and publishes
the repaired portable database with a generation precondition. It also fixes
the stopped worker's local registry. Optimizer tensors and search data are not
modified. The original model LoRA metadata is retained, including its seed.

After this repair the script conditionally releases only job1625's stale worker
assignment, leaving the same managed job queued for recovery. Unlike the first
repair above, this repeated failure requires the narrow reservation repair.

The repair completed with four restored entries and published GCS database
generation `1790179950003591`. The original version is preserved in
`recovery/registry-before-1790179017851425.db` under the run's GCS prefix.
Worker747's local job6 then started for the same managed job1625. All eight hosts
passed admission; compilation caches were warm (roughly 1,050 inference entries
reused per host). Checkpoint loading still required runtime verification at this
snapshot.

The existing execution-request dispatcher now accepts validated RECOVERING jobs
as well as STARTING jobs; it still requires an exact job/run/bundle/worker match
and a PENDING request, and uses SkyPilot's normal locked request executor. The
expanded 21-test continuation suite passed on Slurm CPU job 14317577.

## Confirmed resume at 12:25 p.m. EDT

The new attempt passed all 72 reference cases on all eight hosts. The client
then logged `member muse resumed from tinker://model_30eabd6a/weights/000015`,
`Resume optimizer batch=15 search snapshot=15`, and
`step 15: sampling muse:16g [pipelined dataflow]`. This confirms checkpoint and
optimizer continuation, not merely a RUNNING controller record. No additional
optimizer update is claimed yet.

All four local inference replicas registered and committed the resumed adapter.
Serve reported 16 outstanding requests: four dispatched locally and twelve
queued, with no fatal error. No farm was leased at this snapshot.

The fresh controller query listed five RUNNING jobs and no PENDING/RECOVERING
jobs in this v4-64 pool: circuit Gemma1591/worker744, circuit Qwen1626/worker761,
qubit Gemma1603/worker765, AC2 Muse1624/worker726, and qubit Muse1625/worker747.
The other four jobs were not independently audited for optimizer progress in
this final check. Evidence: `registry-fix/resume-proof.json`.
