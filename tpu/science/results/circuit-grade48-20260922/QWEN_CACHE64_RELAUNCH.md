# Qwen checkpoint-preserving cache64 relaunch, 2026-09-23

Job 1585 exhausted three error retries after recovery. Worker 761's driver
traceback ended in Host.prepare -> admit_cache -> mount_cache with
`not enough available memory for RAM cache plus runtime reserve`.
The old profile required 128+256=384 GiB free; head preflight had 379.45 GiB.

Replacement profile: circuit300-v464-qwen-pwc05-three-starts-10step-20260922-grade48-cache64.json.
Only changes from grade48: trainer/inference cache limits 128 -> 64 GiB;
resume_min_checkpoint_step 1 -> 4. Retains original run ID, durable state,
bootstrap reuse contract, farm assistance, 48 grading slots/host, adaptive
PWC rho 0.5, and ten-step cap. No loss, sampling or grading changes.

GCS sizes checked: Qwen trainer weights 39.015 GiB; HF cache 51.769 GiB;
trainer/inference compile caches approximately 0.273/0.271 GiB. Both fit 64 GiB.
Revalidated bootstrap contract, database backup and checkpoint objects, plus
checkpoint manifest at step 4. Built on Slurm CPU job 14297441, verified archive
hash and packed config, uploaded and downloaded bundle for checksum validation.

Live inventory: workers 761 and 764 READY and unoccupied. All sixteen hosts
passed clean_host_audit; no TPU owners or workload processes. Disk free 76–80
GiB; RAM available 343–380 GiB. Fourteen leftover 128 GiB tmpfs cache mounts
were idle (fuser checked) and remounted at 64 GiB, preserving their contents.
Head hosts had no cache mount. No worker was released or unrelated job stopped.

Operational evidence: .science/launch/qwen-cache64 (per-host audits, resize
before/after records, checkpoint inventory, manifest, submission receipt).

Submitted job **1602**, assigned audited worker **761**. Initial controller
status STARTING; this is not yet proof of restored training.
