# Qubit Gen-1 priority and startup verification — September 24, 2026

User clarified that qubit has priority over circuit Gemma → Muse, and requested stable operation.

## Changes

Persisted managed-job and DAG priorities:
- Qubit Gemma20 → Qwen **1705**: **180** (previously 140).
- Qubit Gemma20 → Muse **1706**: **170** (previously 130).
- Circuit Gemma15 → Qwen **1701**: unchanged **160**.
- Circuit Gemma15 → Muse **1702**: **125** (previously 150).
- Deferred AC2 Gemma **1711**: unchanged **90**.

Before cancelling circuit Muse, the live controller assigned a newly available v4-64 worker **790** to qubit Qwen. Consequently no circuit cancellation was necessary to start qubit Qwen. Both circuit jobs remained untouched apart from Muse's priority change. These are configured priorities, not a claim that the pool retry scheduler guarantees strict ordering among already active recovery loops.

## Actual startup proof

- Worker 790: all eight hosts inspected, approximately 84 GiB free disk per host, zero TPU owners, no leftover workloads or active grading units.
- Existing queued `sky.exec` request `d51d8154-49b9-4baf-8c2a-605699ce2d78` was verified against managed job1705, worker790, and the clean-host audits before dispatch through the native request executor. Its result was SUCCEEDED.
- Head-host SkyPilot local job **2** matches the exact qubit Qwen Gen-1 name and is RUNNING.
- Executable bundle SHA256: `7dda3356cddace9d84f0e71c3e79596f10fa376152698c4928c7500ee75066a4`.
- Actual startup log shows clean-host preflight passed, code and CPU-runtime checksums passed, and active runtime dependency compilation.
- Selected model, Orbax, compilation seed/output and code-bundle configuration uses `gs://sk7524-tinker-tpu-us-central2`.
- At first probe: no generated/grading/training completion established; zero durable recipient optimizer steps. Do not call startup success sustained training stability.

## Persistent observation

User systemd unit on this login host:
`qubit-qwen1705-stability-20260924.service`

It checks job1705 every two minutes for up to 24 hours, resolves its current worker on every check, reads head-host job/log/service status, disk space, and Central2 metrics/checkpoint indices. It records low-disk/error/stall observations. It is read-only and does not automatically cancel jobs, submit duplicates or restart shared services. SkyPilot remains responsible for job recovery.

Evidence and scripts: `.science/qubit-priority-stability-20260924/`.
- `priority-changes.json`
- `host0-audit.json` through `host7-audit.json`
- `qubit-dispatch-intent.json`, `qubit-dispatch-result.json`
- `latest-probe.json`
- `stability-latest.json`, `stability-history.jsonl`

Inspect the monitor with `systemctl --user status qubit-qwen1705-stability-20260924.service` and `journalctl --user -u qubit-qwen1705-stability-20260924.service`.
