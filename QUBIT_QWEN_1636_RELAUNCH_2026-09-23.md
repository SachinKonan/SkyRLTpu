# Qwen qubit 1636 recovery — September 23, 2026

User authorized host storage/process checks and relaunch after the step-18 trainer
failure. Inspected the existing v4-64 pool and all eight hosts of worker747.
Worker747 had no TPU owners or active skyrl services, but its head had only
15.65 GiB disk free. Cleanup was prepared but its ownership guard stopped it
before mutation: SkyPilot had reassigned Qwen to worker773. No files were
removed and no unrelated jobs or farms were cancelled by this recovery action.

Audited all eight worker773 hosts before dispatch:

- Disk free: 82.15–82.52 GiB per host.
- RAM available: approximately 380–395 GiB per host.
- No TPU-device owners, active skyrl runtime/grading units, or old workload
  actors. SkyPilot Ray services and idle Ray workers remained intact.
- Previous worker-local job2 was already CANCELLED.

Downloaded Qwen's durable database and verified checkpoint registrations for
model_fe1af1c7/000018. Final SQLite quick_check returned `ok`. An earlier local
integrity read overlapped two downloads to the same destination and is invalid;
only the completed-download check is evidence of database integrity.
Verified nonempty GCS training/sampler archives and step-18 PUCT snapshot.
The metrics and checkpoint index record 18 completed updates, best reward
0.5505525090247733. This is checkpoint availability, not a completed restore test.

Existing SkyPilot recovery execution request
`a9a1c962-5f5a-47db-b955-15607f72e742` remained PENDING. After matching managed
job1636, run, worker773, and code bundle against the continuation manifest,
dispatched that same request through the existing guarded SkyPilot executor.
Request SUCCEEDED; no duplicate managed job was submitted.

Managed job1636 and worker-local job3 are RUNNING; local job start epoch
1790203935.4259577. All eight hosts downloaded and checksum-verified the original
code and CPU runtime bundles. Runtime/model restoration is still in progress;
optimizer restoration and step19 have not yet been observed. Same target of25
steps, learning recipe, central2 artifacts, v4 caches, and farm-borrowing setup.
No code change was made to the SQLite contention issue; the lock holder/root
cause of the previous failure remains unproven.

Evidence: `.science/routing-relaunch-20260921/qwen1636-relaunch-773/`, including
host-0.json through host-7.json, requests.json, dispatch.log,
registry-verified.json, and startup-latest.txt. Previous-worker checks and
checkpoint object metadata are in the adjacent `qwen1636-relaunch/` folder.
