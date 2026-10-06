# Routing recovery, 2026-09-17

Recover the three models on v4-64 with four trainer hosts and four TP4
inference hosts. CPU grading uses all eight hosts, sixteen slots per host,
four logical CPUs and eight GiB per slot. This is 128 simultaneous full-suite
candidate evaluations, not 128 individual routing cases. Each candidate must
pass all 72 cases across Q20, Willow, and Heron; reward is unchanged.

- Qwen: restore model_94fc0c91 checkpoint 000003 with optimizer, and pool step 3.
- Gemma: restore model_0bfeb293 checkpoint 000001 with optimizer, retaining
  the newer step-2 graded search pool. Checkpoint 000002 failed and is absent
  from durable storage. The training loop resumes at batch 1; it does not claim
  the unsaved optimizer update. Prior search work is retained via the explicit
  TTD_RESUME_SEARCH_AHEAD option; the default behavior is unchanged.
- Muse: all three seed shards succeeded (965/966/967). Their validated merge
  has 110 valid submissions among 1,024 drafts/repairs and six retained distinct
  seeds. No regeneration is required. The seed hash is pinned in the profile.

Downloaded Qwen/Gemma training archives are readable and contain LoRA weights,
optimizer state, and sharding metadata. Both database snapshots pass quick_check;
a full local restore, including embedded request/result blobs, succeeded.
Recovery copies of the latest pools, best code and database snapshots are saved
under each existing run's recovery-20260917 GCS prefix. Original checkpoint
objects and completed results are retained.

Qwen's stopped workload left approximately 72 GiB of grading artifacts on its
head. After confirming no model/grader processes or grading units were active,
only per-candidate evaluation/target and evaluation/rust directories were
removed on its eight hosts, restoring approximately 71-80 GiB free per host.
Sources, routed outputs, logs, verdicts, client state and checkpoints remain.

Recovery protections:

- Private Rust builds are cleaned in finally, including worker exceptions.
- Serve shutdown has a bounded wait, followed by owned process shutdown and
  final writeback; a dead request cannot block that shutdown phase forever.
- Recovery profiles enable three application-error retries plus spot failover.
- Required minimum checkpoints prevent silent fresh starts for Qwen/Gemma.
- Client downloads stage before publication, so interrupted restores can retry.
- In checkpoint-resume mode only, pending dead-client requests are retired
  before API startup. Completed results and checkpoint registrations remain.
- Storage operations in seed handoff retry; job submission is not blindly retried.
- Model/optimizer restoration remains strict; a missing checkpoint is an error.

Validation: 106 focused tests and seven subtests passed; subsequent atomic client
restore changes passed 17 targeted tests. Real snapshot restoration also passed.
The broader database test module could not collect in this local test environment
because SQLAlchemy is absent; this is separate from the successful real restores.

Launch packages are under .science/routing-recovery-20260917/package-{model}.
Gemma's best program independently reproduced exactly: all 72 cases passed,
reward 0.5386220506405557, weighted cost improvement 14.341058%, 93,172 SWAPs.
Total replay time was 1,288.23 seconds with four CPUs and eight GiB. This is
reproduction on the same discovery suite, not a held-out evaluation.

Latest saved search pools contain 67 Qwen and 65 Gemma programs. Muse starts
with six retained seeds. Runtime code is pinned to commit 5981aa41, with the
Discover search-recovery change at submodule commit 8d55c16.

The observed Gemma policy uses squared graph distances, layer-decayed lookahead,
a 30-pair lookahead set with weight 0.5, mild SWAP-decay penalties, and a dense
connected-subset initial layout. These are source observations, not ablations
establishing which change caused the improvement.
