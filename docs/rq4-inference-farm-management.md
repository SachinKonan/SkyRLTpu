# RQ4 inference-farm management

This checkout owns farm launch, live acceptance, discovery, leasing, and retirement for the RQ4 frozen-policy campaign. `third_party/discover` is owned by the frozen-policy work and must not be modified here until its commit is handed over.

## Discovery groups

Use these exact values in trainer profiles under `inference.external_pool_discovery_group`:

| Model | Discovery group and farm-name selector | Zone | Storage boundary |
|---|---|---|---|
| Qwen3.5-27B | `rq4-qwen-asia` | `asia-northeast1-b` | `gs://sk7524-tinker-tpu-asia-northeast1` only |
| Gemma4-31B | `rq4-gemma-east5` | `us-east5-b` | `gs://sk7524-tinker-tpu-us-east5` only |
| Muse-Glimmer-30B | `rq4-muse-east5` | `us-east5-b` | `gs://sk7524-tinker-tpu-us-east5` only |

A separate supervisor serves each group. The group is presented on every trainer discovery read and write, while the same string in each farm job name limits the farm inventory. This prevents an unrelated global supervisor or a supervisor from another region from changing the trainer's candidate list. Model identity and the serving compatibility hash remain independent admission checks.

The initial profiles are:

- `inference-farm-rq4-qwen-asia-v6e8-001.json`: one v6e-8 host, two TP4 engines, 32 sequences per engine.
- `inference-farm-rq4-gemma-east5-v6e8-001.json`: one v6e-8 host, one TP8 engine, 128 sequences.
- `inference-farm-rq4-muse-east5-v6e8-001.json`: one v6e-8 host, one TP8 engine, 128 sequences, default RPA, 70% HBM allocation.

All three use a 22,528-token model limit, a 16,384-token phase-one budget, one LoRA slot, lease fencing, runtime attestation, and regional HF/compile caches. Qwen uses 85% HBM and ragged conv1d. Gemma uses 85% HBM. Prefix caching starts disabled so acceptance matches the previously exercised farm path; enabling it is a separate identical-prompt test.

Do not copy a profile to add replicas without changing `run_id`, `root`, `cache.trainer_compile`, and `cache.inference_compile`. Farm quantity is a launch-time gate and has not been authorized yet.

## Build and launch

Build from this checkout so the runtime and profiles come from `origin/main` plus the reviewed farm-management changes:

```bash
PY=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python
SKY=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky
run=inference-farm-rq4-qwen-asia-v6e8-001
"$PY" -m tpu.swarm.ray_train.build \
  "tpu/swarm/ray_train/profiles/${run}.json" \
  --output ".rq4/farms/${run}" --upload
"$SKY" jobs launch ".rq4/farms/${run}/${run}.yaml" \
  --pool tpuswarm-v6e8-asia-ne1b --yes --detach-run
```

Use `tpuswarm-v6e8-east5b` for the Gemma and Muse profiles. Before upload, verify that every `gs://` value in the rendered profile belongs to the profile's region. No farm may fall back to a model, base-bundle, Orbax, or compile-cache path in another region.

## Live acceptance gate

A farm is eligible for discovery only after all of these pass:

1. `/health`, `/status`, and `/v1/models` are reachable through the worker SSH config.
2. `/status.expected_engines`, `/status.replicas`, and the physical engine topology agree with the profile.
3. The advertised base model and `capabilities.compatibility_sha256` match the intended trainer.
4. Lease acquire, renew, release, and reacquire succeed; a request with a stale lease is rejected.
5. One native-thinking request returns a complete audited response.
6. The bootstrap shape succeeds without missing or duplicated completions: first one `n=16` request, then four concurrent `n=16` requests. Increase only if the trainer profile requires a larger shape.
7. A 30-minute mixed-length soak has no engine restart, SparseCore halt, nonfinite logprob, truncated JSON response, or lease leak.
8. Release drains admitted work and the next owner can acquire the farm.

Record profile hash, packaged-code hash, model revision, compatibility hash, engine count, KV capacity, HBM use, tokens/s, request latency, and failure count. A healthy control plane without successful generation is not an accepted farm.

## Discovery services

Render one systemd unit per group with distinct output and lock files. The trainer pools are supplied after the no-training jobs are submitted. Example for Qwen:

```bash
python -m tpu.swarm.ray_train.borrowing_service \
  --checkout "$PWD" \
  --python /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python \
  --environment /ABSOLUTE/PATH/discovery.env \
  --ssh-dir /scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh \
  --farm-name-contains rq4-qwen-asia \
  --discovery-group rq4-qwen-asia \
  --trainer-pool TRAINER_POOL \
  --lock-file /ABSOLUTE/PATH/rq4-qwen-asia.lock \
  --output /ABSOLUTE/PATH/skyrl-rq4-qwen-asia.service
```

Repeat with `rq4-gemma-east5` and `rq4-muse-east5`. First run each supervisor once with `--dry-run`; require the intended trainer identities, only the intended regional farms, and no incompatible assignment. Then install and start the units. The older `skyrl-hybrid-farm-discovery.service` has no group credential, so grouped trainers reject its reads and writes. Stop it only after all active legacy consumers are confirmed absent and the three RQ4 services are healthy.
