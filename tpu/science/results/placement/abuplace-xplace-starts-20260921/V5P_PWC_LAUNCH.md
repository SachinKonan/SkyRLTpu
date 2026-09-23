# Full circuit training submissions — 2026-09-21

Gemma 1477, Muse 1478, Qwen 1479 submitted to tpuswarm-v5p32-east5a-erdos.
Each: v5p-32, Ray v2 executor, run-scoped farm assistance with dynamic endpoint
discovery, adaptive centered PWC rho=0.5 / invalid reward=0, importance_sampling
loss, 10 training steps, 16x32 groups, fresh bounded bootstrap up to 1024 drafts
targeting 512 valid seeds, CPU300 grading with 32 four-CPU/four-GiB slots per host.
No smoke training run. Gemma STARTING, Muse/Qwen PENDING at submission verification;
this is not proof of completed bootstrap, optimizer steps or active farm leases.

All 51 layouts passed independent scoring. ibm17/rudy_hv required CPU-only
legalization retry (264.35s) and scoring (151.09s); original raw output reused.
Portfolio published under tpu/science/results/placement/abuplace-xplace-three-starts-v1.
Bundles uploaded and SHA256 verified before submission. ADC and gcloud checked
against the required service account; live provider/controller reads succeeded.
Discovery service is active and includes the v5p-32 trainer pool. Other jobs were
not cancelled. Superseded Slurm preprocessing remains held; its Central-targeted
dependent bundle job must not be released for these runs.

## Rho assessment

Exact current advantage code replayed in float64 over 1024 historical Qwen
bootstrap grades from science-placement-v4-qwen-xplace-bootstrap-l2-001, preserving
32-sample groups and separating draft and repair layers. rho=0.5 ranking bonus
RMS was 33.40% / 29.00% of GRPO advantage RMS. Maximum advantage rose from
0.42516 to 0.60777 / 0.65581. Failure advantages unchanged; zero valid-negative
advantages. Valid counts were only 38/512 and 48/512. This is a plausibility
check on older data, not evidence of optimal rho or a measurement of the fresh
three-start/300s bootstrap; it does not measure parameter-gradient norms.

Full artifacts: .science/launch/v5p-pwc (bundles, manifests, uploads, submissions,
queue snapshots) and .science/launch/rho-audit (generation-pinned source list,
reward records, exact-code analysis, advantage-audit.json).

## Farm borrowing recovery

Confirmed late-discovery race: Muse began bootstrap at 00:20:43 UTC before its
first farm-list update at 00:21:20; Gemma began at 00:29:03 before its update at
00:29:05. Both startup admissions returned reserved=false. Discovery updated
URLs but old RunBorrower never retried inside the active phase.

Recovered jobs 1477/1478 through the existing reservation acquire endpoint,
checking base-model version=None first and matching run/instance. No restart,
no direct farm takeover, no bootstrap deletion. Gemma acquired farm
10.130.0.178:24800; Muse acquired 10.130.0.184:24800. Both attest four engines;
Muse subsequently showed four active borrowed requests. Gemma was ready with
zero active borrowed requests at the sampled instant. awaiting_adapter is the
expected farm state for a leased base-model bootstrap, not a failure.

Source fix in run_borrowing.py retries preparation on late URL updates and on
run-watchdog ticks, preserving the current adapter archive and respecting
expected_n, existing healthy leases, active preparation, closed/ended phases,
and normal acquire reconciliation. Regression suites: 74 passed, 17 dependency
deprecation warnings. Existing remote processes were recovered through their API;
this source patch has NOT been loaded into those already-running processes or
retroactively inserted into their immutable uploaded bundles.
