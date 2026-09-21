# Circuit v5p-64 cache recovery

The dedicated `shu-v5p-64-on-demand-machine1` remained READY while the Gemma
attempt-4 runtime had failed on all eight hosts. The queue daemon was alive but
its durable state was `blocked`, so no workload was using the slice and Muse and
Qwen could not advance.

The controller's first recorded failure was inference rank 3's compilation-cache
restore. Its copy process exited within the first 0.5-second poll interval, without an
error message in the copy log. The old wrapper discarded the process exit code.
The checked head and rank-3 hosts had ample RAM and disk space and no kernel OOM
records in the failure window. The exact original process-exit cause is unknown;
successful later copies do not establish it.

## Recovery performed

1. Verified all eight hosts had no TPU owners, competing runtime processes, or
   active grading units.
2. Resumed the failed inference cache restore independently. Its remaining 99
   objects copied successfully and passed the pinned-generation checksums.
3. Restored and verified both seed and current compilation-cache prefixes on
   **all eight** hosts: 239 trainer files and 355 inference files per engine.
   Unlike the earlier trainer-only preflight, this covered the failed role and
   every inference host. Existing model caches and durable training state remain.
4. Commit `7b76cf74` records failed transfer exit codes and adds up to three
   compilation-copy attempts with bounded backoff. Retried copies include only
   files still missing or invalid against the original pinned generations.
   Cancellation exits are not retried. Final checksum validation remains strict.
   The cache suite passed **41 tests** (Slurm job 14216563).
5. Repackaged the three existing circuit bundles, changing **only**
   `tpu/swarm/ray_train/cache.py`; all other archived file contents match their
   prior bundles. This deliberately excludes concurrent inference-farm worktree
   changes. Uploaded bytes were downloaded and SHA256-verified before activation.
6. Paused only the blocked queue daemon, atomically updated its bundle receipts,
   and resumed Gemma as **attempt 5**. The exhausted automatic retry counter was
   retained; another failure will block rather than cause unlimited restarts.

The order is still **Gemma → Muse → Qwen**, with **10 total steps** per run.
The systemd launch receipts confirm dispatch on all eight hosts. The new
controller has joined the complete topology (one trainer, seven inference hosts).
The actual restarted runtime passed both compilation-cache prefixes on all eight
hosts, as recorded in `circuit-attempt5-cache-check.json`. At 04:06 UTC on
September 21, the cache barrier was complete, all seven inference hosts were
prepared, and the trainer process had started. No runtime failure was recorded.
See `circuit-repair-live.json` for that snapshot; a completed optimizer step has
not yet been established.

Evidence: `circuit-copy-failure-audit.json`, `circuit-all-host-prewarm.json`,
`circuit-cache-repair.json`, and the attempt-5 audit and launch receipts.
