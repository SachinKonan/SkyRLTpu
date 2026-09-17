# Live systemd lifecycle probe

Run only on an exclusively reserved spare four-host TPU slice with systemd,
passwordless sudo, Python with Ray 2.58 and psutil, and unused ports 24899,
24900, 25479-25486, 25490-25492 and 25500-25531. No model weights are loaded.
The probe creates real Ray clusters and runs a remote task from every host.
It deliberately leaves detached processes and separate resource-limited grader
services for the runtime to clean up.

Stage these three files together in a small SkyPilot workdir:

- this `probe.py`
- `tpu/swarm/ray_train/runtime_service.py`
- `tpu/science/cgroup_limits.py`

Run `python -u probe.py matrix` as the SkyPilot task on every host, using the
Ray environment's Python. SkyPilot supplies the task ID, host list and rank.
The probe checks normal completion, a launcher SIGKILL, a supervisor SIGKILL,
a preflight failure before any Ray node starts, and reuse of the same ports.
It verifies the exact detached process identities have disappeared and that
pre-existing Ray process identities survive. Grader services use the same
owner-dependency helper as the production CPU graders.

The final phase intentionally waits for **real SkyPilot cancellation**. Once
all four hosts print `ready for real SkyPilot cancellation`, cancel only this
probe's managed job within 210 seconds. Independently verify its units are
inactive/failed with empty cgroups, its daemon/grader PIDs are dead, and its
ports can be rebound. Do not infer cleanup from the CANCELLED status alone.

The checked-in probe uses `~/skyrl-systemd-probe-v5` for evidence. For a new run,
change this path to a fresh directory before packaging; do not reuse old
markers. Keep failed-unit logs as evidence. No broad `ray stop` or process-name
kill is used, and no pre-existing private cluster is adopted or cleaned up.

Verified 2026-09-17 on pool `tpuswarm-v4-32-central2-smoke`, worker 130,
managed job 990 (worker-local job 9). All six cases passed on all four hosts.
`evidence/host-*.json` contains five in-job checks plus the independent
post-cancellation unit/PID/port audit. `manifest.json` pins the tested runtime
and probe hashes. The test used Ray 2.58.0, systemd 249 and cgroup v2.
This validates lifecycle containment, not model training or TPU computations.

Earlier probes exposed issues in the test harness: a deliberately uncooperative
child makes systemd report a stop timeout; systemd service main processes are
already session leaders; Ray renames some agent processes, so graceful cleanup
must include descendants rather than only matching command lines. The final
probe includes those corrections and retains strict success checks for normal
exit. The runtime preserves failure exit codes for forced shutdowns.
