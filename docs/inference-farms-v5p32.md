# v5p-32 inference farms

Three inference-only profiles in `tpu/swarm/ray_train/profiles/`:

| Model | Profile / managed job name |
| --- | --- |
| Qwen3.5-27B | `inference-farm-v5p32-qwen-20260921.json` |
| Gemma4-31B | `inference-farm-v5p32-gemma-20260921.json` |
| Muse-Glimmer-30B | `inference-farm-v5p32-muse-20260921.json` |

Each requests one `tpu-v5p-32` in `us-east5-a`: four hosts, four TP4 engines,
zero trainer hosts, 16 active sequences per engine (64 per farm), 80% inference
memory utilization, 1,024-token prefill chunks, 22,528-token context, and native
thinking-budget enforcement. The consumer supplies its thinking budget; the
campaign uses 16,384 prompt-plus-thinking and 6,144 answer tokens. Prefix caching
is disabled, matching the current shared farms. Only one adapter/lease owner is
served at a time. The systemd runtime owns the processes and cleanup.

These are configuration choices, not measured v5p farm throughput. Compilation
and full-length serving on these three new profiles still need live validation.
No trainer, optimizer, or grading task runs on a farm.

## Discovery

The controller's `borrowing_supervisor` discovers RUNNING managed jobs whose
names contain `inference-farm`, case-insensitively, across pools visible to its
configured SkyPilot API server. It also accepts the existing `--farm-pool`
selector for legacy v4 jobs. Name matching only identifies candidates; the
four-engine health, lease protocol, model/runtime fingerprint, and committed
adapter checks still control actual use. Arbitrary public OpenAI endpoints and
jobs on another SkyPilot API server are not automatically discovered.

Training targets remain explicitly scoped by job ID or trainer pool and must
opt into discovery updates. Existing reservations are retained. Each training
run can use one farm at a time; additional same-model farms support other runs
or replacement after a farm becomes unavailable. No running model is moved just
because v5p capacity appears. Private network connectivity between consumer and
farm is required and checked by the borrower.

The deployed `skyrl-hybrid-farm-discovery.service` includes these training pools
(updated 2026-09-21):

- `tpuswarm-v6e32-east5b-qwen35`
- `tpuswarm-v6e32-central1b`
- `tpuswarm-v4-64-central2-qwen35-erdos`

Preserve all three `--trainer-pool` arguments when regenerating the service.
Farm discovery still uses `--farm-name-contains inference-farm` plus the legacy
`--farm-pool tpuswarm-v4-32-central2-smoke` selector. Adding a training pool
allows updates only for opted-in, run-scoped borrowers; it does not launch or
restart training jobs.

## Cache isolation

Each profile writes under its own `gs://sk7524-tinker-tpu-us-east5/` run and
compile-cache prefixes. No v4 executable cache is used as a v5p seed. Qwen and
Gemma reference existing v5p seed prefixes; compatibility misses compile anew.
Muse has no configured compile seed and may require a cold compile.

Model weights use existing model-cache sources. Gemma's established manifest
remains in `us-central2`; model weights are portable across TPU generations.
The inference RAM cache cap is 128 GiB, with 128 GiB reserved for runtime.
For additional concurrent farms, copy the profile and change `run_id`, `root`,
`cache.trainer_compile`, and `cache.inference_compile` to unique destinations.

## Build and submit

From this checkout, using the configured operations environment (GCP project
`vision-mix` and the existing compute service account):

```bash
BUILD_PY=/scratch/gpfs/ZHUANGL/sk7524/tinker-cookbook/.venv/bin/python
SKY=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky
for model in qwen gemma muse; do
  run="inference-farm-v5p32-${model}-20260921"
  "$BUILD_PY" -m tpu.swarm.ray_train.build \
    "tpu/swarm/ray_train/profiles/${run}.json" \
    --output ".science/v5p-inference-farms/${model}" --upload
  "$SKY" jobs launch ".science/v5p-inference-farms/${model}/${run}.yaml" \
    --pool tpuswarm-v5p32-east5a-erdos --yes --detach-run
done
```

The builder packages and checksums the runtime. Omit `--upload` and the `sky`
command to prepare artifacts locally only. Before submitting, inventory the
pool and verify credentials/cache access using the usual launch procedure.
Do not submit another active job with the same run name/cache destinations.

## Six-farm deployment (September 21)

The authorized deployment uses two independent runs per model. Their profiles
append `-1` or `-2` before the date, for example
`inference-farm-v5p32-muse-1-20260921.json`. Each has its own run root and both
compile-cache destinations. The three unnumbered profiles above remain templates.

Submission IDs and immutable artifact receipts are recorded in
[`launches.json`](../tpu/science/results/v5p-inference-farms-six-20260921/launches.json).
All six target `tpuswarm-v5p32-east5a-erdos` in `us-east5-a`. Once healthy, they
are automatically eligible for the already deployed name-based discovery.
Together they provide 24 engines and a configured maximum of 384 simultaneous
sequences; achieved throughput and full-length capacity require live observation.
