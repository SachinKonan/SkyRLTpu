# AC2 grading capacity rollout — September 23, 2026

User-authorized change: advertise 64 Ray CPUs per host and raise client grader admission from 128 to 256 for the existing Qwen, Gemma and Muse adaptive-PWC AC2 continuations. Candidate allocation remains two CPU threads with the existing 1100-second evaluator timeout and 1000-second prompted search budget. All runs retain the 15-total-step target, optimizer, LoRA, search pool, learning configuration and inference-farm borrowing configuration.

## Implementation

`Config.ray_cpu_capacity` explicitly overrides the Ray host capacity. Its default of zero retains every existing workload-specific default. Validation rejects negative and noninteger values. `bootstrap.py` already consumes `ray_cpus_per_host` when starting each Ray node.

Deployment bundles are derived from each run's exact existing archive. Only the archived config implementation and selected profile changed. Muse's older source baseline was preserved; it was not replaced by the current checkout. Deployment profile adjustments besides capacity/admission are a fresh local cache root and minimum resume checkpoint. Durable GCS run namespaces and compilation-cache locations are unchanged.

The running configuration reserves approximately eight Ray CPUs per host for model actors, with one extra CPU on the head. The expected scheduler ceiling is approximately 223 two-CPU graders over eight hosts, below the new 256 client admission limit. This is scheduling capacity, not measured achieved throughput.

## Cutover

| Model | Original job | Pool | Required saved step before cutover |
|---|---:|---|---:|
| Qwen | 1606 | tpuswarm-v6e32-central1b | 5 |
| Gemma | 1607 | tpuswarm-v6e32-central1b | 5 |
| Muse | 1597 | tpuswarm-v4-64-central2-qwen35-erdos | 10, plus original job completion |

The watcher checks consistent durable checkpoint index, global step, metrics and search-pool snapshot; checkpoint archive existence; and a healthy SQLite backup with the matching completed training checkpoint. It cancels only the two specified Qwen/Gemma controllers after those checks, then submits continuations on the same pools. Muse finishes its original ten-step job first. In-flight work beyond the saved checkpoint may be replayed. No pool is resized or released.

The old AC2 continue-to-15 watcher is marked superseded for Muse to prevent duplicate submissions. Its Qwen/Gemma entries were already superseded by the central migration. The replacement user service is `skyrl-ac2-grading64.service`.

Live state and receipts:

- `.science/ac2-grading64-20260923/status.json`
- `.science/ac2-grading64-20260923/watch.log`
- `.science/ac2-grading64-20260923/{qwen,gemma,muse}/submitted.json`
- `.science/ac2-grading64-20260923/{qwen,gemma,muse}/manifest.json`

An ambiguous submission blocks automatic retries; reconcile its recorded intent with the SkyPilot database before retrying. The watcher never cancels replacement jobs or unrelated workloads. A restart of the watcher after a successful submission reads the saved receipt rather than submitting again.

## Validation and limits

49 local tests passed: explicit-capacity validation, preservation of candidate environment, and executor command tests. Each exact archived deployment was separately loaded and checked for 64 Ray CPUs, 256 grading threads, two CPUs per task, 1100-second timeout, 16 groups of 32 generations, 15 total steps and enabled inference borrowing.

A live inventory found 558–627 GiB available per v6e host and 282–318 GiB per Muse v4 host. These snapshots support increasing admission; they do not establish peak utilization or throughput. Per-task memory remains a 1-GiB Ray scheduling request, not a hard cap. Candidate processes currently share the systemd runtime cgroup; this change does not add per-candidate cgroups.

After startup, verify 64 CPU resources on all eight Ray nodes and the effective client environment's 256-worker admission. Measure achieved simultaneous graders, queue time, per-candidate runtime, host RAM, and inference/backward performance before claiming a speedup.
