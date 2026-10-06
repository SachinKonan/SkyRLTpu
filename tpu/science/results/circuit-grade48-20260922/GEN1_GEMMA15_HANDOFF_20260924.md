# Circuit Gen-1: Qwen and Muse from Gemma step 15

Authorized September 24: wait for Gemma Gen-0 to reach 15 saved training steps, then submit two independent Gen-1 branches, each with **10 additional training steps**. This is a program-pool handoff: each recipient starts its own base model, fresh LoRA and optimizer. No Gemma model weights are imported.

Donor: `circuit300-v464-gemma-pwc05-three-starts-10step-20260922` in `gs://sk7524-tinker-tpu-us-central2/ray-training/`.

Recipients:

- `circuit300-gen1-qwen-on-gemma15-pwc05-10step-20260924`
- `circuit300-gen1-muse-on-gemma15-pwc05-10step-20260924`

Both submit to `tpuswarm-v4-64-central2-qwen35-erdos`, retaining their model-specific Ray v2 executor, local inference configuration and dynamic inference-farm borrowing. They queue for available workers; the monitor does not cancel any existing jobs or release workers.

## Transfer and recipe

Import the complete saved step-15 PUCT program pool, including code, constructions, scores, feedback, IDs and ancestry. Validate every code-bearing state against the full IBM17 suite. Reset pool step, per-state timestep and PUCT visit counters for the recipient's new timeline. Preserve the unmodified donor snapshot separately. Both recipients receive the same content-hashed pool. No new bootstrap or regrading is requested.

Retain adaptive piecewise-centered advantages with rho 0.5, importance_sampling loss, 16 parent groups × 32 rollouts, three XPlace starts, the 300-second placement budget and 48 grading slots per host. Preserve recipient-specific seeds, context/token budgets, trainer layout, and inference settings. The only profile changes are new run/storage/cache identity, imported-pool mode, fresh resume boundary and 10 epochs.

## Durable gate and duplicate protection

The monitor requires a checkpoint record at batch 15, nonempty corresponding optimizer and sampler archives, successful final-step metrics (life_step 14), and the step-15 pool object. It does not require a successful SkyPilot terminal state. Source GCS generation and content hashes are recorded before staging the recipients.

Templates were packaged on Slurm CPU job `14353753`; no compute-intensive packaging runs on the login host. The template's imported-pool placeholder is bound to the final pool hash without changing executable bytes. Archive and task hashes are checked before submission. GCS destination writes are create-only and verified against existing content if retried.

Each branch writes an exclusive submission receipt before calling SkyPilot. A restart reconciles uncertain submissions against the unique run ID; an ambiguous or missing controller record blocks automatic resubmission rather than risking duplicates. The existing packaged clean-host audit runs before TPU startup.

## Monitor and evidence

Persistent user service: `circuit-gen1-gemma15-20260924.service` (enabled; Restart=on-failure). Poll interval: 60 seconds.

Worktree-relative artifacts: `.science/launch/circuit-gen1/`:

- `plan.json`, `qwen-template/`, `muse-template/`: launch specification and frozen bundles.
- `watch.py`, `watch.log`, `status.json`: monitor and current state.
- Upon completion: `donor.json`, `source-step15.json`, `seed-pool.json`, `*-bound/`, and per-model submission receipts.

Verified after installation: service active, waiting for donor checkpoint **14 → 15**; neither Gen-1 job submitted yet. Real step-14 fixture validation retained all **537** programs and their scores, verified IBM17 coverage, and rejected that fixture as a step-15 donor. Both recipient profiles validate. This verifies preparation and the waiting state, not Gen-1 training execution.

Inspect with `systemctl --user status circuit-gen1-gemma15-20260924.service` and the status/receipt files above. The service exits after both jobs are submitted; it is a handoff monitor, not a supervisor of their subsequent training.

## Regional storage audit (September 24)

Checked live Muse job 1646's profile from its executing code directory and its VM metadata (`us-central2-b`). Checked queued Gemma job 1663's submitted SkyPilot YAML and downloaded its immutable, SHA256-verified code bundle. Also inspected both frozen Gen-1 template profiles. All profile GCS references resolve to `sk7524-tinker-tpu-us-central2`; queried GCS bucket metadata confirms location `US-CENTRAL2`. Six storage references each for Muse/Gemma and eight each for the Gen-1 profiles, including compile-cache seeds. No cross-region profile storage references found. Evidence: `.science/launch/circuit-gen1/region-audit/report.json`.

The Gen-1 monitor now checks actual bucket locations for every GCS reference in the packaged profile and for the code bundle before staging or submitting each job. Unknown/inaccessible bucket metadata or a mismatched location blocks submission. Matching/mismatching location tests passed; monitor restarted and confirmed active, waiting at Gemma checkpoint 14. Running TPU jobs were not restarted. This audit covers circuit trainer storage, not the independently managed farm workers' startup caches or historical billing.

## Handoff recovery September 24

Gemma completed step 15 with best reward 0.5176607143635323; donor snapshot contains 567 programs. The monitor passed donor and regional checks but could not find `gcloud` in its systemd PATH, before creating any submission receipt. Replaced the bare command with `/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud`, added traceback logging, and restarted the monitor. Subsequent submission state is recorded in the per-model receipts; this note alone does not establish submission success.
