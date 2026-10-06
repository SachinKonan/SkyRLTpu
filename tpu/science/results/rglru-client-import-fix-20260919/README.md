# RG-LRU v6e east5b: frozen-client grader import fix

The original jobs 1256/1257/1258 all passed the executor-side six-case TPU
self-test, then failed the frozen training-client transport check because
`pallas_arena.judge.ray_pool` was missing from that source tree. Those jobs
are cancelled. Neither bootstrap generation nor optimizer steps had begun.

The source overlay now includes ray_pool, worker, grader, child_runner,
cache, gates, aot_gate and the problem registry/base modules. This gives
Ray tasks deserialized from the client source their subprocess dependencies.
The standalone executor package already contained these files.

The user requested immediate deployment without waiting for validation tests.
The local run was interrupted: 23 tests passed; the new real-child test could
not complete because the local test Python lacks JAX. Its dependency gate
has been added but it was not rerun. This is not a hardware validation claim.
Existing production startup gates remain unchanged: executor-side valid and
invalid TPU tests, then the frozen-client invalid-program transport check.
No additional full-suite check has been added to startup.

Three `-fix1` profiles preserve the original GRPO/model/mesh/bootstrap/reward
settings and use new run roots, checkpoint paths and writable compile caches.
This is a fresh retry because the originals had no generated seeds or steps.
The normal read-only clean-host gate runs before any workload starts.

```bash
# Using the same Python/PYTHONPATH as the fresh campaign:
python tpu/science/results/rglru-client-import-fix-20260919/prepare.py
# Using the approved gcloud/ADC identity and existing TPU-only Sky API:
python tpu/science/results/rglru-client-import-fix-20260919/submit.py --hardware v6e --stage rglru
```

[jobs.json](jobs.json) records package hashes and configuration paths.
[submissions.json](submissions.json) records the replacement job IDs.
The submit helper verifies identity and archive checksum, and journals each
attempt before dispatch so uncertain submissions cannot silently duplicate.
