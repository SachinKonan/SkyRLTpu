# Q20 on v6e: original bootstrap comparison launch

**Update:** GRPO jobs 1068, 1070 and 1072 were cancelled at the user's request.
They are replaced by continuations of the v4 GRPO state; see
[parallel continuation records](../q20-v6e-hedge-20260918/README.md).
The three PWC jobs remain unchanged. The text below records the original launch.

Six additional runs, authorized after the three fresh circuit PWC runs. These
augment the six v6e circuit runs and leave the three v4 Q20 GRPO continuations
unchanged. Each model's two new runs use its original Q20 step-zero bootstrap
pool: Gemma 7, Muse 7, Qwen 6 retained valid programs. Both arms start from fresh
model/adapter/optimizer state. No trained GRPO checkpoint is imported into PWC.

GRPO uses `mean_baseline`. PWC uses
`piecewise_valid_entropic_centered_adaptive`, rho 0.5, invalid reward 0.
Other per-model settings match exactly after normalizing run destinations.
Separate run, checkpoint and writable compile-cache destinations isolate every
experiment. `checkpoint_resume=true` supports recovery within each new run;
it does not select a prior experiment's trained state.

Each slice has four trainer hosts and four inference hosts (four TP4 engines,
16 sequences per engine). Qwen/Muse train TP8/FSDP2; Gemma TP4/FSDP4. There are
16 CPU grading slots per host, each 4 CPUs/8 GiB. Training runs 15 iterations,
16 groups of 32 samples, 16,384 prompt/thinking tokens plus 6,144 answer tokens.
The Q20 reward and independent validation are unchanged.

## Verification and deployment

- Gcloud and refreshed ADC identity matched the compute service account;
  SkyPilot GCP checks and TPU/storage read probes passed.
- All 88 hosts on the 11 idle READY v6e slices passed device-owner, private-Ray,
  grading-unit, memory/disk and workload-port checks before submission.
  Occupied circuit slices were excluded from this idle-host audit.
- Each destination initially contained exactly two objects: the hash-verified
  original `puct_sampler_step_000000.json` and `seed-import.json`. No database,
  checkpoint, trained metrics or optimizer state was copied.
- 13 focused tests passed, including per-model estimator pair equivalence,
  original bootstrap hash, zero checkpoint floor and independent namespaces.
- The exact selected profile was read back from every package archive before
  submission. Bundle upload hashes were verified. Systemd runtime is enabled.
- `provenance.json`, `health.json`, `submissions.json`, `manifests.json` and
  `post-submit-status.json` preserve evidence. Bundles were built before the
  requested post-submission commit; bundle SHA identifies the deployed bytes.

Controller STARTING/RUNNING labels alone do not establish engine readiness or
completed optimizer updates. The status snapshot records submission state only.

## New Q20 jobs

| Model | GRPO | Adaptive PWC rho=0.5 | Original seeds |
|---|---|---|---|
| gemma | 1068 | 1069 | 7 |
| muse | 1070 | 1071 | 7 |
| qwen | 1072 | 1073 | 6 |
