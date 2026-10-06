# Gemma circuit checkpoint recovery, 2026-09-21

Attempt 5 completed its first optimizer update at 07:47:13 UTC. It saved a
2,938,939,279-byte training archive locally, but the gcloud upload exited
unsuccessfully. The sampler save ran next and succeeded, installing adapter
model_a0eb148b_000001. The client retained its old sampling handle because the
combined save failed. Subsequent requests received HTTP 409, then the batch
failed with zero rollout groups. The controller stopped and the queue held
Gemma rather than advancing to Muse. There is no evidence of a trainer HBM
failure or host OOM. The original uploader's terminating cause is unproven;
the old helper omitted the exit code from its error.

Recovery preserves the first update. A serial gcloud retry uploaded the existing
training archive successfully. Both cloud archives were verified against local
MD5 checksums. Training and sampler LoRA NPZ payload hashes agree. The optimizer
step and Adam count are exactly 1; the database records one completed optimizer
request (125), failed training save 126, and successful sampler save 127. The
step-1 PUCT snapshot and global step also exist. No later forward/backward request
was submitted. The offline repair backs up the database, marks only that verified
checkpoint materialization completed, and writes the missing standard client
index. Failed request 126 remains intact as historical evidence. The backup and
client state are published before restart. See registration-repair.json when
publication completes.

The runtime change bounds checkpoint-upload concurrency to one process/thread
and retries nonzero exits three times with 1/2-second backoff. Exhaustion reports
the final exit code. Timeout and size-verification failures still propagate.
This is resilience for interrupted uploads, not a claim that the original signal
source has been identified. New circuit bundles change only overlay support,
the overlay manifest, and the checkpoint mirror helper. Recipe, 10-step budget,
model, sources other than that helper, topology, and caches remain unchanged.
The installer was exercised on all three packaged overlays. The existing source
digest changes, so source_ready installs a new pinned source tree.

An independent AC2 watcher issue was exposed when circuit-queue-state.json
changed from launched to blocked: its source guard treated this observation as
a code change. The guard now excludes that runtime state file and the three circuit launch
receipts alongside the existing completion poll output. A real Git test checks both observations are
ignored while recipe edits still invalidate the guard. The focused checkpoint
and watcher suite passed 17 tests on CPU allocation 14221310; bundle installation
and unchanged-file checks passed on allocation 14221328.

Attempt 6 was dispatched on all eight hosts after successful clean-host audits.
Its immutable Gemma bundle is `808ec4ff390c67bfe7dcf401f143a085e58cbec2b17c6da9e081d8edd88a66b3`.
Actual optimizer restore and resumed sampling remain to be observed.

At 08:20 UTC attempt 6 completed CREATE_MODEL 144, LOAD_WEIGHTS 145 from
model_a0eb148b/000001, and sampler export 146. The client logged checkpoint-1
resume and sampling step 1 with 16 groups of 32. All seven engines registered,
16 requests were active, and ingress reported no fatal or exhausted engines.
See resume-proof.json. Step 2 is not yet completed. A bounded signal trace
observed 45 later uploads without capturing SIGTERM; the original uploader's
terminating cause remains unproven.
