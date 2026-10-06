# Qwen circuit borrowing trial with discovery and strict local failure handling

Replaces independent experiment 1270. Fresh Qwen model/optimizer, circuit fast
proxy on all 17 IBM cases, GRPO, 16 groups x 32 completions. Four v6e trainer hosts
(TP8/FSDP2), four local TP4 inference engines. Up to four additional TP4 engines
borrowed from one exclusive Qwen v4 farm per sampling phase. CPU grading remains
16 tasks per host, each reserving four CPUs and 8 GiB.

Pool: `tpuswarm-v6e32-central1b`. Read-only metadata on idle worker 82 confirmed
`us-central1-b` and `v6e-32`; this zone was added to the runner's validated list.
Output and compilation-cache writes have their own run namespace in central1.
The earlier trial's compile-cache prefixes are read-only warm-start sources;
there is no reuse of its model or optimizer state.

The profile has no hardcoded farm IPs. A controller-local supervisor discovers
healthy farms by exact model and pushes their current URLs to this job only.
The borrower still verifies its lease and adapter on every engine before use.

Local service failure is fatal: strict engine restart budget, pinned ingress
identity, periodic status checks, fatal local-generation errors, and zero
SkyPilot application-error retries. Failed remote requests fall back to healthy
local inference. Remote transient health failures have a 90-second grace period.

Validation: 77 focused borrowing/supervisor/lifecycle/diagnostic tests passed,
then 53 tests including one additional local-error propagation check, launcher
commands, and exported science package checks passed (the test sets overlap).
`artifact-check.json` records checks on the exact submitted archive, isolated
imports, all 17 input hashes, helper compilation and strict/dynamic configuration.
`submission.json` is the durable launch receipt. Live generation, grading and an
optimizer step must be observed before claiming end-to-end success.

Submitted as job **1324**, assigned worker **82**. Job **1270** was verified
CANCELLED. The supervisor PID is recorded in `supervisor.json`; it initially
found both Qwen farms (jobs 1250 and 1251) and waits for the new ingress before
updating it. No other training jobs were cancelled or reconfigured.
