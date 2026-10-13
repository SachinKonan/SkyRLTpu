# Qwen3.8 regional storage and farm connectivity — 2026-10-07

Live audit for the Qwen3.8 comparison campaign in `SkyRLTpu-qwen-comparison`.

| Region | Work | Inference weights | Converted trainer checkpoint |
|---|---|---|---|
| us-central2 | Six planned v4-64 hybrid trainers; circuit/qubit bootstrap producers | Verified, 18 shards / 55,563,006,776 bytes | Verified, 41,882,259,914 bytes |
| us-east5 | Two v6e-8 trainers; queued v5p-32 inference farms | Verified, identical shard checksums | Verified, identical revision |
| asia-northeast1 | Three live v6e-8 inference farms | Verified, identical shard checksums | Absent and unnecessary for these inference-only farms |

GCP bucket metadata confirms the actual regional locations. Model revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` in all three regions. HF prefixes are
`gs://sk7524-tinker-tpu-<region>/hf-cache-qwen38-1d4bf0f2`.
Trainer prefixes in Central2 and East5 are
`gs://sk7524-tinker-tpu-<region>/skyrl-maxtext-qwen38-1d4bf0f2/qwen3.5-27b`.
The final directory is the MaxText architecture alias; completion markers identify Qwen3.8.

Inspected the deployed profiles on trainer jobs 2125/2145 and farm jobs
2119/2120/2121. Also extracted profiles from the immutable, SHA-verified bundles
recorded in SkyPilot's queued jobs 2129/2131/2133/2135/2137/2139/2141/2143/2144.
All storage paths, including runtime bundles, model caches and compilation seed
paths, point into each job's regional bucket. All ten planned campaign profiles
(two retained trainers, six shared-pool consumers, two bootstrap producers) pass
regional checks. This is a campaign audit, not an all-fleet assertion.

HTTP connectivity succeeded from both East5 trainers and Central2 worker 935 to
all three Asia farms: `10.146.0.124:24800`, `10.146.15.196:24800`, and
`10.146.15.198:24800`. These are observed addresses, not static configuration.
East5 trainer 2125 held two Asia farms; 2145 held one. Central2 health checks prove
reachability, not a completed Central2 training update. Queued East5 farms have no
assigned endpoint yet, so their future endpoints cannot be tested now.

The campaign discovery supervisor updates run-scoped candidates across regions.
Model/runtime compatibility and adapter readiness remain required. LoRA archives
are uploaded as HTTP bytes; farms do not fetch a trainer's checkpoint bucket.
Cross-region HTTP requests, answers and adapters remain cross-region network
traffic. Region-local bucket access does not mean that this traffic is free.

The launch watcher now checks the *bundled* profile immediately before submission:
actual bucket location, every GCS URI in the profile and task, completion markers,
HF shard counts/bytes, converted checkpoint bytes for trainers, and local runtime
bundle existence. A failed check prevents launch; it never falls back to another
region's bucket. `single_bucket=true` remains enabled at runtime.

One shared bootstrap per problem supplies both seeds. The controller may stage
the small immutable Erdős/AC1 step-zero pool once from East5 into Central2 before
launch. The TPU jobs then read the Central2 copy. No optimizer, trained adapter,
trainer DB or later search checkpoint is copied for these fresh-seed comparisons.
Circuit/qubit pools originate in Central2 already.

Evidence, including timestamped JSON, is in
`.science/qwen-comparison-20261006/regional-preflight-report.json`,
`regional-model-inventory.json`, `regional-runtime-audit.json`,
`queued-farm-artifacts.json`, and `regional-farm-reachability.json`.
The regional and shared-bootstrap targeted suite passed 100 tests.

After the checks, submitted circuit bootstrap job **2180** and qubit bootstrap
job **2181** to the Central2 v4-64 pool. These generate shared pools only; they
perform no optimizer updates. Six dependent v4-64 training runs remain behind
the verified-pool gate. The admission/discovery watcher was restarted with the
regional launch check enabled. East5 jobs 2125 and 2145 were retained.

## Retention correction, October 7

The initial implementation incorrectly required 512 valid programs. The user
clarified that 512 is a maximum: grade exactly 1,024 drafts, then retain the best
up to 512 unique valid programs. Circuit retained 428; qubit retained 170. Their
saved grade journals were promoted without generation or regrading, with the
original generation contracts preserved and explicit capped-retention evidence.
Both training seeds use byte-identical pools per problem. Four hybrid v4-64
training jobs were submitted: circuit 2218/2220, qubit 2219/2221. They await pool
capacity; no optimizer step is claimed.

Erdős/AC1 replacements **2225/2226** resume 544 graded Erdős drafts and
272 generated / 263 graded AC1 drafts. Superseded pending jobs 2125/2145 were
cancelled after checking for optimizer history. Root and draft journals were
server-side copied within East5; only run identity and the completion-policy
flag changed in their generation contracts. Sampling, prompts, implementation
hash and rewards remain fixed. Neither replacement has completed its bootstrap
at submission. Regional/capped-pool validation passed 75 targeted tests.


## Circuit RAM admission correction and worker cleanup check

Circuit jobs 2218/2220 restored the shared 428-program pool but failed before an optimizer update: the trainer requested a 128-GiB RAM cache plus a 256-GiB runtime reserve, above approximately 380 GiB available on the head. Both were cancelled and replaced by jobs 2227/2228 with a 96-GiB trainer cache and unchanged 256-GiB runtime reserve. Shared pool identity and science settings are unchanged. Replacement bundles include safe shrinking of a reused tmpfs cache when its used bytes fit the smaller limit. The targeted cache/admission/bootstrap/campaign suite passed 75 tests.

The cleanup inspection covered eight hosts each on former workers 974 and 988. No mounts were removed: an initial safety audit could not import psutil under system Python, so it preserved every mount. Before retrying, worker ownership had changed to unrelated running jobs 2224 and 2223 respectively. Manual cleanup was therefore deferred; the helper now checks active ownership and uses the run controller environment. Checkpoints and bootstrap pools were not deleted, and worker 954 was untouched.

New qubit worker 1016 head had 78.1 GiB free disk and 328.73 GiB available RAM, with only its current run RAM mount. Job 2221 encountered a dpkg frontend-lock conflict on worker 1023 rank 3. A subsequent live check found no apt/dpkg process or lock holder and 83 GiB free disk there; no package process was killed or lock file deleted. Job 2221 was recovering at this snapshot.

The campaign watcher was restarted (PID 3634996), and successful monitoring ticks confirmed the corrected circuit jobs queued. Startup, completed generation, and optimizer updates for the replacements are not yet verified. Evidence is in `.science/qwen-comparison-20261006/{idle-circuit-cleanup-evidence.json,post-relaunch-host-check.json,circuit-ram96-plan.json,admission-watch-capped-pools.log}`.


## Verified stale-claim recovery and campaign guard

Circuit 2227's failed worker attempt 4 left `current_cluster_name` set to worker 1023. The live recovery controller repeatedly reported "No idle replicas" because the resource allocator counted that nonterminal managed-job claim against its own idle worker. After verifying the worker's latest attempt was FAILED and repairing its interrupted dpkg state, the claim was cleared under the native pool lock using SkyPilot's state setter. The same controller dispatched worker attempt 5 and both controller and worker reported RUNNING; no duplicate managed job was submitted.

The campaign watcher now uses `tpu/swarm/ray_train/pool_recovery.py` to perform that narrow recovery automatically. It requires an explicitly scoped run, PENDING/RECOVERING with recovering_from_failure, sole claim ownership, an exact matching terminal worker attempt older than 120 seconds, no active worker jobs, a dead driver, and unchanged controller state after inspection. SSH errors and observation timeouts preserve ownership. The 15 targeted tests pass. Watcher PID 4192440 completed a live tick after deployment.

Latest verified progress: qubit 2219 and circuit 2228 have committed their initial adapters and each reports 16 active local generation requests. Jobs 2221 and 2227 remain in startup. AC1 2226 has 351 bootstrap grades and one held farm. Erdos 2225 is waiting for capacity after worker 2853 became unreachable and was removed; only one East5 v6e-8 worker was ready. No optimizer update is verified yet. This is intermediate progress, not an end-to-end training success claim.


### Follow-up monitoring

Live provider inventory confirmed multiple East5 v6e-8 replacements in CREATING, with only worker 2864 ready; Erdos recovery is waiting on real capacity rather than an idle healthy worker claim. AC1 bootstrap advanced to 388 graded candidates. Qubit 2219 and circuit 2228 each have four registered inference engines and 16 admitted generation requests; qubit 2221 logs show ongoing XLA compilation and circuit 2227 has finished all eight host preparations. No optimizer step is verified.

The first-step monitor now accepts local `hybrid_generated` completion events as well as `borrow_generated` after an updated adapter commit. Previously its remote-only completion condition could never prove a working hybrid run using local inference. This changes monitoring only; training and evaluation are unchanged. Syntax checking passed and watcher PID 106766 completed live ticks.

### October 7, approximately 13:37 EDT: training signal and capacity audit

All four hybrid jobs have committed an initial adapter and started generation. Qubit 2219 has eight completed grouped generation requests and eleven completed forward/backward requests; five completed requests contain nonzero advantages, including both signs across the batch. The earlier zero-advantage requests came from all-invalid groups, corroborated by the client PWC logs. Grader artifacts include valid routing scores and generated Rust compilation errors; zero advantages alone were not evidence of a broken reward pipeline. Circuit 2228 has four completed grouped generation requests and one completed forward/backward request with nonzero advantages (14 valid candidates in its first graded group). No optimizer update or post-update adapter reload is verified yet. Evidence: `.science/qwen-comparison-20261006/training-signal-evidence.json`.

AC1 2226 advanced to 411 bootstrap grades and holds two attested four-engine farms, at `10.130.0.58:24800` and `10.130.0.95:24800`. Erdos 2225 remains in recovery after its worker stopped accepting SSH; the controller is live and waiting for replacement capacity. East5-b has 40 queued resources, matching the quota rejection in provisioning logs. Of the pool's 15 reservations in the provider snapshot, 13 were PROVISIONING, one WAITING_FOR_RESOURCES, and one ACTIVE. Suspended reservations in that snapshot belonged to other users and were left untouched. No training-code failure was established for Erdos's lost worker.

All 32 hosts of our four v4-64 workers responded to resource checks. Minimum free disk / available RAM in GiB: worker 1016, 68.3 / 264.2; worker 974, 31.5 / 262.9; worker 988, 43.0 / 262.0; worker 1023, 68.4 / 267.5. Worker 974's lowest-disk host merits continued monitoring; it is not presently full. No active caches or processes were removed. Evidence: `.science/qwen-comparison-20261006/current-worker-resources.json`.

At 13:37 EDT AC1 worker 2864 was also lost. The provider explicitly reports `UNHEALTHY_MAINTENANCE`, with a maintenance event at `2026-10-07T17:37:22.572927251Z`; the pool controller retired the worker and job 2226 entered recovery. Thus the preceding AC1 running/lease snapshot is no longer current. Its East5 bucket contains 426 grade files and 27 saved generation groups; Erdos has 544 grade files and 34 groups. These were verified directly in regional storage after the loss, so recovery can reuse saved work. Evidence: `.science/qwen-comparison-20261006/math-bootstrap-recovery-durability.json`. No duplicate jobs were submitted and no other users' reservations were removed.

At 13:46 EDT the two expired AC1 farm leases had been reclaimed through normal discovery/acquisition: Qubit 2219 uses `10.130.0.95:24800` and Circuit 2228 uses `10.130.0.58:24800`. Both borrowers report ready, four engines, four active remote requests, and their own verified adapter archive hashes. This proves lease expiry, rediscovery, adapter publication, and remote dispatch after the owner loss; a completed remote answer is still pending at this snapshot.

Training request completions advanced to 22 for Qubit 2219 and 12 for Circuit 2228. Live trainer logs show continued `[2,22528]` forward/backward tiles at approximately 14 seconds per warm tile; a first tile took 73 seconds. These are tile timings, not complete optimizer-step timings. The monitor now records read-only request counts by type/status, latest training completion, and oldest pending training request. Syntax validation and live reads on both jobs passed; watcher PID 643011 completed a fresh tick after its local restart. No training process was restarted. Optimizer updates remain unverified.

Follow-up: provider audit history for Erdos worker 2853 records `tpu.nodes.terminate` at `2026-10-07T16:42:30.686452991Z`, with `terminateReason: UNKNOWN`. This does not prove maintenance or a workload crash; retain that uncertainty. The complete retrieved events are in `.science/qwen-comparison-20261006/erdos2853-provider-events.json`. A later live East5 inventory contained nine owned v6e-8 nodes in CREATING and none READY.

On worker 974 rank 1, `uv cache prune --cache-dir ~/.cache/uv --offline --no-config` completed with "No unused entries found". It freed no space (31.44 GiB remained), and no model caches, run artifacts, active environments, or processes were removed. Subsequent repeated live checks showed no failed training requests; forward/backward completions reached 36 for Qubit 2219, 10 for Qubit 2221, and 24 for Circuit 2228. Circuit 2227 completed its first four generation groups; grading was still pending. These remain intermediate observations rather than optimizer-step proof.

### October 7, 14:15 EDT: hybrid watchdog fix and East5 recovery

Circuit 2228's API watchdog cancelled and redispatched eight generation requests after 3,611 seconds. These groups were legitimately queued behind long completions. The ingress shields work from disconnected HTTP callers, so redispatch created duplicate engine work while abandoning original replies. `trainer_environment()` now defaults the three age-based external watchdog settings to zero for ingress hybrid borrowers with the external scheduler. Orphan recovery and bounded inference HTTP timeouts remain enabled; explicit hybrid overrides and legacy direct mode are preserved. The command/remote-only suites passed 122 tests. The reconciled campaign suite passed 19 tests.

Jobs 2227, 2219, 2228, and 2221 were cancelled for corrected immutable replacements. No durable optimizer checkpoint existed; the partial first training batch is lost, while the completed circuit/qubit shared bootstrap pools and regional compilation caches are preserved. All 24 hosts on surviving workers 974, 988, and 1023 passed retired-run ownership/process/mount checks after shutdown drained. No manual process kill or cache deletion was necessary. Worker 1016 had independently suffered provider maintenance.

The replacement plan, bundle hashes, cancellation evidence, and all-host audits are under `.science/qwen-comparison-20261006/watchdog-*`. New run IDs append `wd`; bundle profiles contain the corrected startup environment. Model, evaluator, seeds, shared pool hashes, token budgets, loss, and training-step limits are unchanged. Runtime GCS remains Central2 for v4 and East5 for v6e.

Erdos 2225 recovered to healthy East5 worker 2885 and restored 544 bootstrap grades / 34 saved generation groups. Its discovery list initially was empty after recovery; a scoped discovery update supplied the available compatible farm. Live status then confirmed a ready, attested lease on `http://10.130.0.95:24800`, four engines, and four active requests. The head had 78.46 GiB free disk. This verifies resumed dispatch, not yet a newly completed answer or optimizer step. AC1 2226 still awaited capacity after worker 2864's documented maintenance event; its 426 grades / 27 groups had already been verified in East5 storage. No duplicate East5 job was submitted.

Corrected managed submissions are Circuit seed 0 **2238**, Qubit seed 0 **2239**, Circuit seed 1 **2240**, and Qubit seed 1 **2241**. The watcher restarted as PID 1030720 and completed a fresh discovery/monitoring tick. Circuit 2238 dispatched to worker 988; the other three were pending capacity. Workers 974 and 1023 were subsequently assigned unrelated synthgrad jobs 2232 and 2231, respectively, and were left untouched. Thus a READY pool worker is not necessarily available to this campaign. Erdos subsequently acquired both compatible Qwen3.8 farms, `.95` and `.58`, advertising eight engines total, with four requests active in the latest snapshot.

At approximately 14:23 EDT, two live scrapes of Erdos's active farm `.95` measured 94,912 additional generated tokens over 61.12 seconds (1,553 tokens/s aggregate). Each of its four engines had 16 running sequences and zero waiting sequences. These counters prove ongoing generation, not completed bootstrap groups or a training update. Raw samples: `.science/qwen-comparison-20261006/farm-token-samples.jsonl`. Circuit 2238 restored the exact 428-program shared pool and advanced from topology checks through cache setup into trainer environment setup. No new failure was observed; remaining campaign consumers were still capacity-bound.

### 14:32 EDT: terminal provider reservation cleanup

Provider revalidation found two FAILED v4-64 queued resources left behind by pool workers 1028 and 1030. Their launch logs explicitly deferred deletion while the reservations were still PROVISIONING; they subsequently failed with provider error 8, "no more capacity in the zone". Before deletion, both were re-described as FAILED, verified to belong to sk7524 and v4-64, checked against live TPU inventory (no corresponding VM), and checked for active managed-job claims (none). Both were deleted without force or terminating any live worker. Exact resource names and evidence: `.science/qwen-comparison-20261006/failed-v4-reservations-cleanup.json`. This removes terminal reservations; it does not create physical TPU capacity. East5 contained only one ready owned v6e-8, occupied by Erdos, while other owned reservations remained provisioning/waiting.

### 14:50 EDT: AC1 recovery and farm handoff

AC1 2226 recovered on East5 worker 2881, restored 426 bootstrap grades / 27 saved generation groups, and passed startup with 81.8 GiB free head disk. Its ready ingress had no farm candidates because Erdos 2225 held both available Qwen3.8 farms. Discovery intentionally preserves leases and does not rebalance incumbents.

Using the existing identity-fenced service-list and reservation APIs, Erdos's idle second farm (`10.130.0.58:24800`, farm job 2217) was released and acquired by AC1. Erdos retained `.95`. Preconditions verified donor ownership, zero generation activity, zero grading queue depth, recipient readiness, and an additional retained donor farm. The discovery watcher was suspended briefly around the handoff and resumed in `finally`; no farm/trainer restart or global admission-policy change occurred. Evidence and replayable guarded helper: `.science/qwen-comparison-20261006/{erdos-ac1-farm-handoff.json,handoff_idle_erdos_farm.py}`. Existing lease health snapshots may retain the released farm until the next heartbeat observes the changed owner.

Circuit 2238 finished engine initialization: all four engine health endpoints returned HTTP 200, the training API accepted a session, CREATE_MODEL completed, and the initial SAVE_WEIGHTS_FOR_SAMPLER request was pending at the first verified post-startup snapshot. This is still not optimizer-step proof.
