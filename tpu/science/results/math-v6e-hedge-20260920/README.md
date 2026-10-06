# Parallel v6e math and qubit runs

The v4 runs remain scheduled with their existing progress. These eight v6e runs
are independent branches of the same fresh-bootstrap GRPO experiment; nothing
in this campaign cancels or modifies a v4 job.

| Experiment | v6e pool | New v6e job | Existing v4 job |
|---|---|---:|---:|
| Gemma AC2 | tpuswarm-v6e32-central1b | 1275 | 1228 |
| Muse AC2 | tpuswarm-v6e32-central1b | 1273 | 1229 |
| Gemma circle packing n=26 | tpuswarm-v6e32-central1b | 1276 | 1231 |
| Muse circle packing n=26 | tpuswarm-v6e32-east5b-qwen35 | 1274 | 1232 |
| Qwen circle packing n=26 | tpuswarm-v6e32-east5b-qwen35 | 1277 | 1230 |
| Qwen qubit | tpuswarm-v6e32-east5b-qwen35 | 1278 | 1233 |
| Gemma qubit | tpuswarm-v6e32-east5b-qwen35 | 1279 | 1234 |
| Muse qubit | tpuswarm-v6e32-east5b-qwen35 | 1280 | 1235 |

`jobs.json` records profiles, immutable package hashes, and submission paths.
`submissions.json` records confirmed job receipts after dispatch.

## Recipe

- Bootstrap: all eight hosts run TP4 inference, 32 requests of 16 completions;
  generate at most 1,024 drafts, stopping at the configured 512 distinct valid
  programs target. No separate repair round.
- Training: four physically adjacent trainer hosts and four TP4 inference
  engines; 15 GRPO steps, 16 groups of 32, learning rate 1.5e-4 for Qwen and 4e-5 for Gemma/Muse.
- Gemma trainer TP4/FSDP4; Qwen and Muse TP8/FSDP2. Native thinking budget and 22,528-token
  context are inherited from the validated v6e model configurations.
- Task prompts, grading, and rewards are inherited from the matching v4
  profile. Math grading retains two CPUs per task. Qubit uses all three
  topologies (Q20, Willow, Heron), 16 grading slots per host, four CPUs and
  8 GiB per candidate; its reward and per-case feedback match the v4 run.
- Each run has an independent output namespace and writable compilation-cache
  destinations. Recovery restores only that run's own state.

## Cache and worker preparation

`replicate.py` copies the two models' required HF weights, model-specific Orbax
trees, and v6e compilation seeds into us-central1. Source generations are pinned;
existing destinations must match size and CRC32C. The complete verified inventory
is saved in `cache-copies.json`: 1,424 objects, 201.56 GiB verified, with 1,347
objects newly copied. Compilation seeds are from v6e runs, not v4.

`clean-hosts.json` records successful checks on all 24 hosts of central1b workers
4, 6, and 8. The previously idle east5b worker 5227 became unreachable during the
audit; it is not certified healthy. Submission targets the pool, not that worker.
The subsequent pool inventory showed only central1b workers 4 and 8 ready and
both ready east5b slices occupied; jobs may queue for replacement capacity.
Every task also runs the clean-host gate before starting its runtime. Existing
RG-LRU and circuit jobs are outside this campaign and remain untouched.

## Submit

From this worktree, using the established SkyPilot API environment and approved
compute service-account identity for both gcloud and ADC:

```bash
/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python \
  tpu/science/results/math-v6e-hedge-20260920/submit.py --hardware v6e --stage all
```

Complete each model's required cache replication before submitting that model.
Use repeated `--run-id` arguments to select ready runs while another model's
cache is still copying. The submitter checks package hashes, verifies
uploaded archives, and persists an attempt receipt before dispatch. Re-running
skips confirmed submissions and refuses uncertain attempts until reconciled.
No extra validation test suite was run for this deployment, as requested.

## Future v6e to v4 handoff

Cross-pool takeover is an explicit operation, not automatic failover. Keep both
branches separate while they are active. If the v6e branch fails and its progress
is selected for continuation on v4:

1. Select the latest complete durable training step and preserve both branches.
2. Transfer its model/optimizer checkpoint, API database and referenced state,
   client checkpoint journal, and matching PUCT/search snapshot together.
3. Create a separate v4 resume namespace; reconcile run IDs and checkpoint paths
   and require that committed step through the strict resume guard. Do not
   overwrite an active v4 namespace or combine divergent optimizer histories.
4. Verify checkpoint restoration under the v4 trainer mesh before advancing.

Bootstrap completion is bound to configuration/implementation fingerprints, so
copying a v6e bootstrap directory into an existing v4 run is insufficient. Equal
chip counts alone do not prove checkpoint portability. A future transfer must
pass real restoration; weights-only recovery would lose optimizer/search state.

## Additional east5b submissions

`prepare_qwen_cp26.py` adds Qwen circle packing; `prepare_qubit.py` adds all
three qubit models through the science packager, including Rust runtime setup.
The original submitted artifacts are not rebuilt. New runs start their own
bounded bootstrap and GRPO training; they do not import partial v4 progress.
Qwen circle packing has priority 100 and qubit priority 90, retaining math
priority. All four use independent east5 output and compilation-cache prefixes,
with existing v6e caches as read-only seeds.

Both idle east5b slices 5241 and 5242 passed all eight host checks before
submission; `extra-host-checks.json` records the 16 successful checks. The
startup gate also checks the actual assigned hosts. At preparation there were
only two idle ready slices, so some additions may wait for pool capacity.
