# Gemma RG-LRU continuation on central capacity

The east maintenance event at approximately 06:22 UTC displaced job 1343.
It remained PENDING/RECOVERING on its old worker 5597 while central had idle
capacity. After confirming that state again, only job 1343 was cancelled.
Cancellation and the absence of another active writer were verified before
submitting replacement **1365**, assigned **central1b-176**, priority 110.

The canonical run is still
`fresh-v6e-gemma-rglru-grpo-lr4e5-s1-20260919-fix1`. Only the SkyPilot task's
`resources.zone` changes. The repaired bundle from job 1343 is reused exactly:
`5d9638d7ebe5fa15b43546f5d3cbc0545bbc881e0fa4fbba595a4090857ec382`.
This is the bootstrap-reuse repair, not the older bundle that rejected the
completed bootstrap. The original task and archive SHA256 values were verified,
and the archived profile was compared with the repaired profile on disk.

The completed bootstrap contract, completion record, and step-zero pool were
read at pinned GCS generations and matched byte-for-byte against the prior
verified preflight. All 29 retained candidates are preserved; there was no
optimizer checkpoint at cutover. Training remains capped at ten total steps,
with identical model, optimizer, sampling, grading, caches and farm behavior.
No bootstrap regeneration or change to its explicit reuse pins was introduced.

`prepared.json` records the source artifacts and current cloud generations.
`submission.json`, `cancelled.json`, and `queue-after.json` record the cutover.
At submission verification the replacement was STARTING, not yet executing a
training step. Future workload checks must establish bootstrap reuse and client
startup on the new worker. The AC2 follow-up watcher was rearmed after this
placement-only commit; its winner-selection and completion rules are unchanged.
