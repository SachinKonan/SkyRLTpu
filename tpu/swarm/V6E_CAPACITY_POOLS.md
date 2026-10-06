# v6e Capacity-Only Pools

As of 2026-09-08, v6e pool bootstrap follows the lightweight v5p pool pattern.
It installs basic tools, stages a validated code bundle, checks the eight-host
topology, and publishes readiness. It does not acquire model weights, wait for
an HF cache marker, initialize TPU devices, or start trainer/vLLM processes.
Submitted jobs own those steps. The legacy `prepare_qwen35_v6e32.sh` remains
available but is no longer called by pool setup.

Both `examples/v6e32-qwen35-grpo-pool.yaml` and
`examples/v6e32-qwen35-grpo-single-pool.yaml` retain the setup retry loop:
failed bootstrap attempts keep the allocated worker, remove their incomplete
bundle staging directory, and retry. Completed bundles are reused by GCS
generation. No model/checkpoint caches are deleted by pool bootstrap.

## Deployment

| Pool | Zone | Fixed target | Regional bucket |
| --- | --- | ---: | --- |
| tpuswarm-v6e32-east5b-qwen35 | us-east5-b | 32 | sk7524-tinker-tpu-us-east5 |
| tpuswarm-v6e32-asia-qwen35 | asia-northeast1-b | 40 | sk7524-tinker-tpu-asia-northeast1 |
| tpuswarm-v6e32-europe-w4a-qwen35 | europe-west4-a | 2 | sk7524-tinker-tpu-europe-west4 |

Each bucket contains:
`code-bundles/tpuswarm-skyrl-v6e-capacity-20260908-v1.tar.gz`.
The archive was built using the existing bundle builder with its explicit
dirty-tree flag, retaining the manifest's source provenance. Its SHA256 is
`5dcfb0dab8f1269fe18061cd3fff8865b924f910f6e7d7a9e2406799b86b62b0`.
The working tree is shared; this publication was not a commit or a push.

Updates use the ordinary `sky.jobs.pool_apply` SDK or its `/jobs/pool_apply`
API endpoint with rolling mode; server validation and credential checks remain
enabled. The direct API was used for Europe after shared-filesystem latency
delayed SDK imports and the sequential client's wait timed out.
Before applying a regional copy of the single-pool YAML, set all three
`pool.workers`, `pool.min_workers`, and `pool.max_workers` fields to the target;
set resources region/zone and the `ZONE` environment variable; replace the
bundle and all GCS cache prefixes with the corresponding regional bucket.
Set `PROJECT=vision-mix` and
`SKYRL_REPO_DIR=/home/gcpuser/SkyRLTpu-tpuswarm`.

Do not combine a YAML and `--workers` on this fork's CLI: despite its help
text, that combination is rejected. For subsequent size-only updates,
`sky jobs pool apply -p POOL --workers N` updates both bounds without changing
the worker setup.

The rollout preserves the SkyPilot API, TPUSwarm service, and v4/v5p pools.
A new pool version excludes old-version setup failures from the autoscaler's
unrecoverable-startup guard. No direct database deletion or pool teardown is
needed. Rolling updates retain old nonterminal capacity until sufficient
replacement workers are ready; old and new requests can coexist temporarily.

The submitted rollout request IDs are:

- East5b: `109803bf-0477-4772-9adf-ec696822f596` (version 6 applied).
- Asia: `c173cecc-c109-4234-955a-70aad1afdcaf`.
- Europe: `1e2800a3-35f1-4813-99be-11fa572da39d`.

Check these existing requests before retrying an apply. A client wait timeout
does not cancel the server-side update. Asia and Europe were still pending
completion during the 2026-09-08 filesystem maintenance window; do not assume
that submission alone means their new version is active.

## Verification and Limits

`tests/tpu_swarm/test_v6e_capacity_bootstrap.py` executes both setup scripts
with fake cloud commands and deliberately failing model entrypoints. It checks
that an interrupted bundle transfer is retried, incomplete staging is removed,
the completed bundle is reused, and both head/peer ranks become ready without
running the model entrypoints.

Pool readiness is not a successful training or inference test. Existing queued
jobs retain their job parameters. In particular, this migration does not prove
that the previous v6e Qwen HBM-exhaustion configuration will run successfully,
nor does it migrate those jobs to the Ray v2 executor.
