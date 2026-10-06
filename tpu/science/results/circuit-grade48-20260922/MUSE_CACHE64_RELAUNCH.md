# Muse cache-only recovery

User authorized cancellation of Muse 1586, storage/RAM/process checks, and relaunch on the available worker without releasing it. Startup ordering remains unchanged.

Worker 727 was inspected on all eight hosts before submission. All had 240 logical CPUs; RAM available was 371.99 GiB on the head and 389.31–389.69 GiB on workers. No executor/model processes, TPU device owners, or stale model-cache tmpfs mounts were found. Disk free was 22.98 GiB on the head and 59.31–67.94 GiB on workers. The head's uv package cache was cleared with `uv cache clean`; it freed only about 44 MiB because installed environments retain shared files. Durable run state and unrelated workloads were preserved. The head still has about 23 GiB free; this was not a broad disk purge.

Cache sizing from durable GCS objects: Muse Orbax weights 40.585 GiB, full HF cache prefix 55.490 GiB, trainer compile cache 0.128 GiB, inference compile cache 0.151 GiB. Each role's measured assets fit in 64 GiB, including the downloader's 2 GiB headroom. Future cache growth can still consume the remaining margin.

The new profile is `circuit300-v464-muse-pwc05-three-starts-10step-20260922-r4-grade48-cache64.json`. Relative to the grade48 profile, only trainer/inference cache caps (128 -> 64 GiB) and the required minimum restored checkpoint (0 -> 2) change. It retains 48 grading slots per host, 256 GiB reserve, the original run ID/GCS namespace, all placement/scoring budgets, farm assistance, adaptive PWC rho 0.5, and the ten-step cap.

Job 1586 was verified CANCELLED. Bootstrap hashes, database-backup presence, saved step >=2, checkpoint objects, and uploaded bundle SHA256 were checked before submission. Slurm CPU build job: 14287275. Receipts and eight-host inventory: `.science/launch/muse-cache64/`.
