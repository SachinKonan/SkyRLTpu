# Parallel v4/v6e Q20 GRPO continuations

Replaces the fresh v6e GRPO comparison arms 1068 (Gemma), 1070 (Muse), and
1072 (Qwen), which were cancelled at the user's request. The v4 GRPO jobs
1059/1061/1063 continue unchanged; PWC and circuit jobs are untouched.

The purpose is independent progress on both hardware pools, so either pool can
survive the loss of the other. These are independent branches from the copied
state, not continuously synchronized replicas or bit-identical trajectories.

Gemma and Qwen start at their latest durable GRPO checkpoint, step 2, including
model/optimizer checkpoint, database registration, client progress and matching
PUCT search pool. Muse has no saved optimizer checkpoint and starts from its
same step-zero bootstrap pool. Run names, client log directories, checkpoint
mirrors, root directories and writable compilation caches are distinct from
v4 and every other v6e run. No estimator changes: all three use mean_baseline
GRPO and the original importance_sampling loss.

Profile parity tests allow only hardware/region changes and output namespaces;
training schedules, sampling, learning rates and CPU grading limits match v4.
Gemma/Qwen use NUM_EPOCHS=17 from step 2; Muse uses 15 from step 0.

The cancelled jobs were confirmed CANCELLED, followed by a clean-host audit on
all 24 released hosts (TPU ownership, private Ray, grading units, ports, disk
and memory). Credentials, GCP/storage access and TPU inventory were refreshed.
All 16 focused profile/prompt tests passed.

Snapshot copies use generation-qualified source objects and verify size and
CRC32C at the destination. A replaced source generation is accepted only if
its contents match the captured metadata. Live database backups had acquired
new in-flight requests, so the immutable database used to initialize v4 at the
same step-2 checkpoint was used instead, after matching its size/CRC32C to the
initially captured database. Qwen's repeatedly replaced client objects were
also copied from their immutable input replica after verifying exact content
hashes against the observed v4 state. No in-flight optimizer work is claimed
as saved.
The client checkpoint is checked
against the matching PUCT pool and archive; restored databases pass SQLite
quick_check. Bundle hashes and selected profiles are verified before launch.
Artifacts record exact source generations and deployed bundle hashes.

The profiles were packaged before the requested post-submission commit.
Controller status does not imply completed optimizer steps.

## Submitted replacements

| Model | Source v4 job | New v6e job | Copied step | Valid programs |
|---|---|---|---|---|
| gemma | 1059 | 1075 | 2 | 70 |
| muse | 1061 | 1076 | 0 | 7 |
| qwen | 1063 | 1077 | 2 | 50 |
