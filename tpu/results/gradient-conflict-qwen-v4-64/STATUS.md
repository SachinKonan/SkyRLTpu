# v4-64 gradient-conflict pilot

## Completed: 2026-09-07 01:56:34 UTC

SkyPilot job **326 SUCCEEDED** on v4-64. All **six groups / 192 samples**
completed, with no selected samples excluded. The durable `result.json`
reports `complete: true` and `groups_completed: 6`.

All six signed valid/invalid gradient cosines are negative: median **-0.340155**,
range **-0.481575 to -0.280777**. Invalid/valid sum-norm ratios range from
**0.571621 to 1.434055**. The combined gradient has a positive dot product
with both subset gradients in every group, so opposition alone does not imply
an adverse combined SGD direction for either subset.

The independent reconstruction check passed at **0.545% relative error**.
All 25 live exports share one parameter fingerprint; all 24 cached partitions
match their live export metadata and expected sample counts. Result geometry,
per-parameter totals, input identity, and SkyPilot terminal state were verified.

- [Full results and interpretation](RESULTS.md)
- [Plot](gradient-conflict.png) and [SVG](gradient-conflict.svg)
- [Raw result](result.json) and [completion audit](verification.json)
- [Export metadata](gradient-export-metadata.json) and [deployment manifest](manifest-v3.json)

The dated updates below record earlier stages of this now-completed run.

## Update: 2026-09-07 01:39 UTC

**Five of six groups are complete** on job 326. Group 5
(`step000002-seed01`, 14 valid / 18 invalid) has signed-gradient cosine
**-0.280777**, valid sum norm **1.070917**, and invalid sum norm **1.163578**.
All five completed groups have negative cosine. The final group
(`step000001-seed04`) is scoring. SkyPilot still reports **RUNNING**;
the final success artifact and job completion are not yet verified.

## Update: 2026-09-07 01:23 UTC

**Four of six groups are complete** on job 326. Group 4
(`step000001-seed08`, 12 valid / 20 invalid) has signed-gradient cosine
**-0.431907**, valid sum norm **1.264992**, and invalid sum norm **1.088227**.
All four completed groups have negative cosine. Group 5
(`step000002-seed01`) has begun scoring. Full completion is still pending.

## Update: 2026-09-07 01:06 UTC

**Three of six groups are complete** on job 326. Group 3
(`step000002-seed02`, 10 valid / 22 invalid) has signed-gradient cosine
**-0.296060**, valid sum norm **1.047128**, and invalid sum norm **0.924793**.
All three completed groups have negative cosine. Group 4
(`step000001-seed08`) is forward scoring, with the probe process and API
confirmed live. The three comparisons are saved in GCS `progress.json`.
The full experiment is not yet complete.

## Update: 2026-09-07 00:50 UTC

**Two of six groups are complete** on job 326. Group 2
(`step000000-seed00`, 7 valid / 25 invalid) has signed-gradient cosine
**-0.481575**, valid sum norm **2.064770**, and invalid sum norm **1.455319**.
Group 3 (`step000002-seed02`) has begun forward scoring. The live monitor
continues; full completion is still pending.

## Update: 2026-09-07 00:39 UTC

Job **326 is RUNNING**. **One of six groups is complete**, and the second
group has finished scoring all 32 samples and begun its gradient partitions.
The first group (`step000000-seed02`, 3 valid / 29 invalid) has signed-gradient
cosine **-0.366077**, valid sum norm **1.191742**, and invalid sum norm
**0.681225**. Its matched random split has cosine **0.027437**.

The independent full-group reconstruction check **passed** with relative error
**0.00544514** (0.545%, below the 2% tolerance). Parameter fingerprints remain
unchanged. These preliminary measurements are saved in GCS `progress.json`;
the full experiment has not yet completed.

## Update: 2026-09-06 23:48 UTC

Job 320 **FAILED** at the checkpoint space guard (16 GB free; 51 GB required).
The affected worker retained 54 GB of canary inference weights under
`ray-serve-canary-v4-64-v3/model`, outside the usual HF cache. Its total canary
directory occupied 66 GB; the TPU itself was healthy.

Revision 3 adds targeted inference-weight cleanup on assigned training hosts,
preserving canary results/logs, environments, source, metadata, and XLA caches.
**67 targeted tests pass.** Replacement v4-64 **job 326** is confirmed **RUNNING**
on the same slice. Topology validation passed, selecting trainer ranks 0,2,4,5.
The cleanup ran on the affected host: available space rose from 16 GB to 67 GB,
and checkpoint download began successfully (free space then decreases as it
downloads). This verifies that the disk-space guard has been passed.
`manifest-v3.json` records its overlay. No gradient comparison has completed yet.

Older status updates follow.

## Update: 2026-09-06 23:10 UTC

Job 317 **FAILED** after downloading the base checkpoint and creating the model:
the probe's checkpoint-registration subprocess hit `sqlite3.OperationalError:
database is locked`. No gradient measurements were produced.

The fix registers the checkpoint before creating the SDK session/model and
increases registration's SQLite busy timeout to 30 seconds. Registration was
verified successfully on the live slice. **65 targeted tests passed**, including
a real SQLite writer lock held longer than the old five-second timeout.

Replacement v4-64 **job 320**, `qwen-v4-64-gradient-conflict-v1-r2`, is confirmed
**RUNNING**, with no recoveries, and has started topology setup on a different slice.
The overlay is recorded in `manifest-v2.json` at the same GCS experiment prefix.
The following section describes job 317's initial startup, not current success.

## Original startup

SkyPilot job **317**, `qwen-v4-64-gradient-conflict-v1`, was confirmed **RUNNING**
on 2026-09-06 at approximately 22:43 UTC in
`tpuswarm-v4-64-central2-qwen35-erdos`. The previous pending v4-32 job 314 is
confirmed **CANCELLED**.

The assigned eight-host slice began startup successfully. Its topology selector
chose trainer ranks **0,1,5,6**; validation of that physical row passed, and
checkpoint staging began. Training retains the original 16-chip TP8/FSDP2 mesh. The other
four hosts do not run inference for this diagnostic.

The same 192 archived samples and frozen rank-32 LoRA checkpoint are used.
Scoring chunks and completed gradient partitions are saved durably for retries.
This status records successful submission and startup, not completed measurements.

Artifacts:
`gs://sk7524-tinker-tpu-us-central2/gradient-conflict/20260906-qwen-v4-64-v1/`

- `manifest-v1.json`: exact overlay and task hashes.
- `progress.json`: partial comparison results and reconstruction status.
- `result.json`: successful completed experiment.
- `cache-v5/`: resumable scoring and gradient artifacts.
- `logs/`: startup, API, and measurement logs.

Configuration: `tpu/gradient_conflict/qwen_v4_64.yaml`.
Validation: **64 targeted tests passed**, plus shell/Python syntax checks and
`git diff --check`. The selected trainer ranks are passed through the common
worker wrapper; its default four-host behavior remains available to v4-32 jobs.
