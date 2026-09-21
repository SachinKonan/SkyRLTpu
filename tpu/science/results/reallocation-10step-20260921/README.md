# Ten-step science reassignment

User-approved target: one active copy per model/problem. Every training profile
sets `NUM_EPOCHS=10`, a total cap including restored steps, not ten additional
steps. The existing ensemble loop uses `range(start_batch, cfg.num_epochs)`.

| Role | Model/task | Submission |
|---|---|---|
| v4-32 farm | Qwen | 1330 |
| v4-32 farm | Gemma AC2 | 1331 |
| v4-32 farm | Gemma RG-LRU | 1341 (1332 withdrawn to prioritize Muse) |
| v4-32 farm | Muse | 1333 |
| v6e-32 | Qwen AC2, restore checkpoint 1 | 1334 |
| v6e-32 | Gemma AC2, restore checkpoint 1 | 1335 |
| v6e-32 | Muse AC2, restore checkpoint 2 | 1336 |
| v6e-32 | Gemma RG-LRU, existing bootstrap | 1337 |
| v4-64 | Qwen full qubit routing, latest checkpoint >=2 (3 verified at cutover) | 1338 |
| v4-64 | Gemma full qubit routing, checkpoint 1 verified at cutover | 1339 |
| v4-64 | Muse full qubit routing, restore checkpoint 3 | 1340 |
| dedicated v5p-64 | Circuit placement: Gemma, then Muse, then Qwen | systemd queue |

Both v6e pools (`us-central1-b`, `us-east5-b`) are approved destinations. At
submission central had zero ready workers and east had five; all four initial
v6e jobs were consequently placed in east. This does not install automatic
cross-pool migration of a live experiment. Future reassignment must cancel and
verify the former owner before submitting the same canonical run elsewhere.
No duplicate central copies were submitted.

## Latest checked runtime (2026-09-21 02:18 UTC)

All three AC2 jobs have active application controllers. Muse has all four local
inference engines registered and is waiting for its external farm. Qwen and
Gemma are still preparing/compiling engines. The Qwen/Gemma v4-32 farms are in
engine startup; Muse farm 1333 has been assigned worker 177 and is starting.
The spare Gemma farm 1341 remains pending. Gemma RG-LRU 1337 is recovering.

Qwen qubit 1338 has completed eight-host environment preparation and is setting
up its science reference; Gemma and Muse qubit are pending capacity. Circuit
Gemma attempt 4 has validated topology and reached cache preparation on all
eight hosts. No new completed training update is verified in this snapshot.
`runtime-summary.json` and `live-jobs.json` contain the observations and times.

## Recipe and state

The jobs preserve 16 groups x 32 rollouts, importance-sampling loss, mean-baseline
advantages, seed 1, native thinking, 22,528 context, and 16,384 phase-one budget.
Each farm is four TP4 engines, one adapter version, exclusive ownership. AC2 uses
four local inference engines plus its farm. RG-LRU uses three local inference
engines, one dedicated TPU grader and its farm. AC2 reserves matching farms first;
a spare Gemma farm can then serve RG-LRU. Contract hashes exclude incompatible
or legacy farms before assignment.

Canonical run IDs and GCS roots are preserved for resumes, even where the name
contains the old hardware (for example `fresh-v4-qwen-ac2...` now runs on v6e).
No optimizer states or parent trees are merged between trials. Saved database
integrity checks passed, and all six selected training checkpoint registrations
were `COMPLETED`; Gemma AC2 still lacks its old metrics record. Runtime loading
and useful new updates remain live acceptance checks.

`prepared.json` records profiles, destination pools, total step caps, immutable
bundle hashes and superseded job IDs. `uploads.json` records verified GCS bundle
generations. `preflight.json` records source pools, checkpoint references,
database generations and mirrored archives. `recovery-snapshots.json` records
preserved recovery metadata. `launches.json` records new managed-job IDs.
The operator database inspection is at
`.science/reallocation-10step/database-audit.json` (not committed).

Cancelled old submissions: 1287-1294, 1301-1303, 1318, 1321-1324. The four idle
farms 1250, 1252, 1253, 1254 were claimed with maintenance leases before targeted
cancellation/replacement. Obsolete Muse farm 1255 subsequently recovered; it
was verified idle, maintenance-leased and retired as well. With three ready
farm workers, Qwen/Gemma AC2/Muse have priority 120; the spare Gemma RG farm
has priority 110 and waits for capacity. Unrelated v5p-32 Erdos submissions were left alone.

## Standalone circuit queue

Only `vision-mix/us-central1-a/shu-v5p-64-on-demand-machine1` is used. The other
v5p-64 machine is not modified. Each model runs one trainer and seven TP4
inference engines, with CPU circuit grading using the existing
`ibm17-proxy-v1` helper and original per-model v5p trainer settings.
Gemma reuses its saved parent pool; Muse and Qwen complete their bounded
bootstraps before training.

`skyrl-circuit10-queue.service` is a user-systemd service on the controller.
It launches the same pinned Ray v2 task through per-host systemd units, tracks
all eight hosts, coordinates at most three attempts, and advances only after
step-10 metrics and a step-10 checkpoint exist and every previous host is stopped.
A failed model is blocked after the retry budget; it is not silently skipped.
`circuit-queue-state.json` and per-attempt launch/audit receipts are live records.

The first two Gemma attempts failed during environment preparation because the
machine's uv 0.8.22 did not know the pinned Python 3.12.12 release. Repair uses
isolated `uv==0.12.17` to install that exact interpreter, preserving the original
uv executable and the training/serving package locks.

Attempt 3 reached cache preparation but the compilation-cache download process
exited without an explicit storage error. No kernel OOM was recorded for that
interval; its exit cause remains undetermined. A separate replay of the same
validated cache restore succeeded (177 reused + 62 restored = 239 seed objects).
`compile-preflight.txt` records that check. One explicit operator retry (attempt
4) was granted after this repair, retaining the exhausted automatic counter;
another failure blocks again. `circuit-repair-retry.json` records the decision.

Before launch the head had only 28 GiB free. Eleven local CP26 checkpoint
archives were matched byte-for-byte to GCS using MD5 and size before removal.
Their obsolete adapter-upload copies and stopped runtime environment/source
caches were also removed. No cloud checkpoints or local result logs were deleted;
free space increased to 54 GiB. `standalone-cache-cleanup.txt` is the receipt.

## Validation and operation

- 50 integration tests passed, with 12 subtests, after repairing a test request
  fixture missing `headers`.
- Seven focused reallocation/queue tests passed, including bounded retries,
  completion gating and incompatible-farm exclusion.
- All 14 bundles passed source/config validation and upload checksum verification.
- The managed discovery supervisor was updated to a frozen copy of the new
  admission code and includes both approved v6e pools.

No completed farm-assisted optimizer cycle or measured speedup is claimed merely
from submission. Monitor remote-completed groups, adapter hashes, successful
training metrics, checkpoint publication and disk headroom. First two new cycles
count toward the ten-step cap. Require checkpoint resume and farm-loss fallback
checks before calling the migration fully validated.

`prepare.py` builds without cloud mutation. `operations.py` imports the established
credential environment helper; it never prints credentials. `deploy.py` uses
exact IDs, checksums, a lock and write-ahead launch receipts. Do not rerun an
uncertain submission: reconcile its receipt with the live queue first.
