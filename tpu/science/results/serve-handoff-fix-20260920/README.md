# Ray Serve bootstrap handoff fix — 2026-09-20 UTC

The bootstrap-to-training graph update unintentionally restarted surviving inference engines. Ray 2.58 generates random deployment versions when omitted, so stable deployment names alone did not preserve the engines. Qwen AC2 job 1227 completed bootstrap but failed the transition with `inference recovery budget exhausted`.

The fix derives each engine deployment version from its own initialization inputs and serving source. Retiring neighboring engines therefore preserves surviving replicas. The ingress still updates to the reduced engine list; the restart guard is unchanged. Ray 2.58 rejects the public `options(version=...)` API but still consumes `Deployment._version`; the isolated compatibility helper uses that pinned contract.

## Verification

A CPU-only real Ray 2.58 Serve probe on worker 678 reproduced the failure: reducing an unversioned application replaced the surviving engine PID (255122 to 255400). The fixed application retained the exact survivor identity and PID (255603). See `probe.py`, `probe.json`, and `probe.log`. This is a lifecycle regression, not a TPU training smoke. A production handoff and optimizer step after relaunch are not yet verified.

## State preservation and rollout

Each replacement patches only `tpu/swarm/ray_train/serving.py` inside its original immutable archive. All other archive files and the experiment configuration are preserved byte-for-byte. Run IDs, storage buckets, cache destinations, bootstrap journals, and output namespaces are unchanged. Only managed job IDs, task display names, and bundle URI/checksum change.

Uploaded bundles were downloaded and checksum-verified before stopping workloads. Assigned runtimes received cooperative stop requests where available, then the 23 explicit old job IDs were cancelled. The independent borrowing job 1270 and its farms were untouched. Read-only cleanup checks passed on 80 reachable hosts across 10 slices. Stale v4 worker 662 was unreachable and is not certified healthy; replacement startup retains the clean-host gate.

Saved bootstrap config and implementation contracts match replacement bundles. Completed Qwen AC2 (291 retained seeds) and Qwen circle packing (172 retained seeds) pool hashes match their completion markers. Partial bootstrap journals remain in their original namespaces. No completed optimizer/checkpoint progress was found in the affected namespaces at shutdown.

The bulk stop followed by sequential submissions caused temporary idle workers on v6e. All 23 replacement submissions are now confirmed. STARTING is controller status, not proof of service readiness or training progress.

## Replacement jobs

Snapshot: 2026-09-20T01:47:33.160976+00:00

| Experiment | Pool | Old job | New job | Controller status | Assigned worker |
|---|---|---:|---:|---|---|
| qwen ac2 | tpuswarm-v4-64-central2-qwen35-erdos | 1227 | 1281 | STARTING | tpuswarm-v4-64-central2-qwen35-erdos-678 |
| gemma ac2 | tpuswarm-v4-64-central2-qwen35-erdos | 1228 | 1282 | STARTING | tpuswarm-v4-64-central2-qwen35-erdos-681 |
| muse ac2 | tpuswarm-v4-64-central2-qwen35-erdos | 1229 | 1283 | PENDING | None |
| qwen cp26 | tpuswarm-v4-64-central2-qwen35-erdos | 1230 | 1284 | STARTING | None |
| gemma cp26 | tpuswarm-v4-64-central2-qwen35-erdos | 1231 | 1285 | STARTING | None |
| muse cp26 | tpuswarm-v4-64-central2-qwen35-erdos | 1232 | 1286 | STARTING | None |
| qwen qubit | tpuswarm-v4-64-central2-qwen35-erdos | 1233 | 1287 | STARTING | None |
| gemma qubit | tpuswarm-v4-64-central2-qwen35-erdos | 1234 | 1288 | PENDING | None |
| muse qubit | tpuswarm-v4-64-central2-qwen35-erdos | 1235 | 1289 | STARTING | None |
| qwen circuit | tpuswarm-v4-64-central2-qwen35-erdos | 1236 | 1290 | STARTING | None |
| gemma circuit | tpuswarm-v4-64-central2-qwen35-erdos | 1237 | 1291 | STARTING | None |
| muse circuit | tpuswarm-v4-64-central2-qwen35-erdos | 1238 | 1292 | STARTING | None |
| qwen rglru | tpuswarm-v6e32-east5b-qwen35 | 1259 | 1293 | STARTING | tpuswarm-v6e32-east5b-qwen35-5217 |
| gemma rglru | tpuswarm-v6e32-east5b-qwen35 | 1260 | 1294 | STARTING | tpuswarm-v6e32-east5b-qwen35-5229 |
| muse rglru | tpuswarm-v6e32-east5b-qwen35 | 1261 | 1295 | STARTING | tpuswarm-v6e32-east5b-qwen35-5236 |
| muse ac2 | tpuswarm-v6e32-central1b | 1273 | 1296 | STARTING | tpuswarm-v6e32-central1b-4 |
| muse cp26 | tpuswarm-v6e32-east5b-qwen35 | 1274 | 1297 | STARTING | tpuswarm-v6e32-east5b-qwen35-5239 |
| gemma ac2 | tpuswarm-v6e32-central1b | 1275 | 1298 | STARTING | tpuswarm-v6e32-central1b-7 |
| gemma cp26 | tpuswarm-v6e32-central1b | 1276 | 1299 | STARTING | tpuswarm-v6e32-central1b-8 |
| qwen cp26 | tpuswarm-v6e32-east5b-qwen35 | 1277 | 1300 | STARTING | tpuswarm-v6e32-east5b-qwen35-5242 |
| qwen qubit | tpuswarm-v6e32-east5b-qwen35 | 1278 | 1301 | STARTING | tpuswarm-v6e32-east5b-qwen35-5249 |
| gemma qubit | tpuswarm-v6e32-east5b-qwen35 | 1279 | 1302 | STARTING | None |
| muse qubit | tpuswarm-v6e32-east5b-qwen35 | 1280 | 1303 | STARTING | None |

`jobs.json` records exact archive identities and original package paths; `uploaded.json`, `preserved.json`, `graceful-stop.json`, and `clean-hosts.json` contain rollout evidence. `submissions.json` preserves confirmed submission receipts and this status snapshot. Package-local receipts prevent blind retry after uncertain dispatch. The scripts are campaign-specific operational records, not a general fleet launcher.
