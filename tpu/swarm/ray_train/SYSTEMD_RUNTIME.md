# Systemd-owned runtime (opt-in)

Set `systemd_runtime: true` in a Ray v2 profile. The default remains false.
This runs the per-host bootstrap inside a transient systemd service under the
same Unix user, filesystem, IP addresses and workload ports. It is not a
container and creates no network namespace. Passwordless `sudo systemd-run`
and `sudo systemctl` are required on each host.

The SkyPilot task process remains a launcher. The independent service watches
that process's PID and `/proc` start time, and exchanges slice heartbeats on
`systemd_port` (default 24900). A dead launcher, failed bootstrap, lost peer,
or expired setup deadline aborts the service. Before any Ray node starts,
every bootstrap must finish its existing local cleanup and port checks.

The service owns all descendants through `KillMode=control-group`. Normal
shutdown gives bootstrap `systemd_shutdown_grace` seconds (default 90) to
checkpoint and stop its services. Systemd then removes remaining descendants,
including Ray daemons and processes that detach from their parent's session.
A killed supervisor also causes systemd to clean its remaining cgroup.
Science graders use separate resource-limited units, linked to their local
runtime with BindsTo/After/PartOf. These links stop graders when their runtime
stops or dies; their existing four-CPU/eight-GiB limits remain unchanged.
A descendant that outlives the stop timeout is force-killed and systemd marks
the unit failed rather than reporting a clean exit.
Peer heartbeat timeout defaults to 30 seconds; tune only after measuring
host stalls. A whole-host failure requires no local process cleanup, while
surviving hosts stop after their peer timeout.

Each attempt writes `systemd-<uuid>/unit.json`, `supervisor.json`, `service.log`
and `result.json` under the run directory. `service.log` contains the bootstrap
output; `result.json` records supervisor completion, not proof that the final
cgroup cleanup has finished. The launcher waits for systemd's final unit state.
The environment payload is mode 0600 in a mode 0700 directory and is removed
as soon as the supervisor reads it. Never copy that payload into diagnostics.

This does not clean processes created before the service existed, change
checkpoint/resume semantics, or make a SkyPilot CANCELLED record a cleanup
certificate. A subsequent attempt still performs the ordinary port checks;
a conflict makes it fail closed instead of joining an unrelated Ray head.
Production enablement should follow the live cancellation/failure tests.

The wrapper copies the launcher's process resource limits (including NOFILE
and MEMLOCK) and removes systemd's default task-count ceiling. It does not
change candidate CPU/memory limits. The live probe and reproduction procedure
are in `tests/tpu_swarm/live_runtime_service/`.
