# Circuit grading concurrency relaunch — September 22, 2026

Requested change: increase grading capacity on the existing v4-64 hosts.

Each run has eight hosts, now configured for 48 concurrent cases per host (384 aggregate; previously 256). Each case retains four CPU slots, 4 GiB RAM, 300 seconds placement search, 310 seconds candidate process allowance, and 180 seconds verification; the worker envelope remains 510 seconds. All 17 IBM cases, seed 42, three XPlace starts, adaptive PWC rho 0.5, and the ten-step target remain unchanged.

Read-only inspection of all 24 existing hosts found 240 available logical CPUs each. The minimum available RAM by run was Gemma 287.0 GiB, Qwen 268.1 GiB, Muse 259.7 GiB. The new partition dedicates 192 CPUs to grading and leaves 48 for services. The cache reserve increases from 240 to 256 GiB: 192 GiB grading plus 64 GiB service allowance. 64 concurrent cases would require more CPUs than these hosts have.

Implementation updates Ray admission capacity, host-wide locks, CPU affinity for graders and service processes, runtime resource validation, and explicit bootstrap reuse. The existing 32-slot default remains. Different active CPU maps are rejected. Completed bootstrap provenance is pinned by its original contract and pool hashes; it is not rewritten. Reuse allows placement concurrency and reserve changes, retaining strict comparison of scientific settings.

Relaunch preserves the original run IDs and durable GCS namespaces. Qwen and Gemma profiles require checkpoint step >=1. Muse resumes its completed bootstrap and any subsequently available saved checkpoint. In-flight uncheckpointed training work may repeat. Before cancellation, bundles were uploaded and SHA256-verified, and durable bootstrap, database backup, and Qwen/Gemma checkpoint objects were checked. Old jobs: Gemma 1548, Qwen 1550, Muse 1564. Pool workers were not deleted.

Farm evidence before relaunch: Qwen completed borrowed generations via 10.130.1.1:24800 and Muse via 10.130.0.88:24800; both map to v4-32 TPU VMs in us-central2-b. Gemma's latest completions were local. v5p Qwen 1577 and Muse 1579 were controller-RUNNING, Gemma 1578 RECOVERING; that does not establish circuit use of those v5p farms. Farm services were not modified.

Validation: 32 tests passed across test_placement_300.py, test_bootstrap_reuse.py, and test_abuplace_starts.py; git diff --check passed. Deployment bundles built by Slurm CPU job 14277452.

Operational receipts, hardware inventory, bundle manifests, cancellation output, and submission IDs are in `.science/launch/v4-grade48/`. Profiles end in `-grade48.json` but retain the original internal run IDs.

Submission snapshot: Gemma 1584 STARTING on worker 744; Qwen 1585 STARTING on worker 748; Muse 1586 STARTING on worker 749. Workers may be reassigned by the pool scheduler. This snapshot establishes submission, not resumed training or measured throughput improvement.

## Gemma cache-only retry

After job 1584 failed, the user approved reducing only the RAM-cache cap and explicitly declined changing startup ordering. The `-grade48-cache64.json` profile differs from `-grade48.json` only in `cache.trainer_gib` and `cache.inference_gib`, both 128 -> 64. The reserve remains 256 GiB, giving a 320 GiB fresh-cache admission threshold. Bootstrap reuse now permits these operational cache capacity changes for the modern placement runtime; seed-pool hashes and scientific settings remain checked. The 48-slot limit and all per-case time budgets are unchanged. Existing startup reference checks remain enabled and may still fail independently of cache admission.

Build: Slurm CPU job 14286783. Validation: 23 bootstrap-reuse tests passed; exact profile comparison confirmed only the two cache-capacity fields differ; bundle SHA256 and durable bootstrap/database/checkpoint state verified before submission. Submission receipts are in `.science/launch/gemma-cache64/`. No worker cleanup or changes to Qwen/Muse were performed in this retry.

## Last verified operational snapshot (2026-09-22 23:57 UTC)

Qwen 1585 was RUNNING on worker 750, with step 3 saved. Gemma cache64 replacement 1591 was RUNNING on worker 744, with step 1 saved and adapter loaded on farm endpoint 10.130.0.184:24800. Muse cache64 replacement 1592 was STARTING on worker 727, with step 2 preserved but no new training process confirmed. These are historical observations, not a claim of current fleet health. Both cache64 replacement receipts are included beside this document.
