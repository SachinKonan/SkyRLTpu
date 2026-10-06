# Reliability follow-up to the September 20 review

Implemented locally on top of `a732b9f7` in `agent/hybrid-inference-migration`.
No TPU jobs, deployed bundles, profiles, or systemd services were changed in this
follow-up. Live fleet status was not polled. The original review file is unchanged.

## Changes

1. **Lost acquire replies:** new farms advertise `acquire_protocol=1`. Borrowers
   send an acquire UUID and farm ingress instance. If the reply is lost, the
   borrower records the unresolved attempt and retries `/cancel_acquire` at the
   next phase/reservation attempt. The farm remembers cancellation even if it
   arrives before the acquire, prevents a queued acquire from granting, and
   drains an existing grant before acknowledging. Only a matching cancellation
   acknowledgement allows the borrower to claim another farm. Cancellation
   tombstones last for the ingress lifetime; a timer alone cannot safely resolve
   an acquire queued behind server work.
2. **Bounded initial admission:** `external_pool_initial_wait_seconds=300` is the
   default. Busy or unreachable farms no longer leave trainers waiting forever.
   At the deadline the controller logs `farm_admission_local_fallback` and proceeds
   locally, including when `external_pool_require_initial=true`. Later phases can
   recover borrowing. Readiness also pins the run heartbeat identity if initial
   service discovery was unavailable.
3. **Request errors:** engine generation responses 400/404/413/422/429 propagate
   through Ray as request errors and retain their HTTP status. They do not latch
   the farm catalog's fatal flag. Engine transport errors and 5xx responses retain
   the strict failure behavior and bounded diagnostics.
4. **Status polling:** a single transport error or 5xx response no longer kills a
   strict run. Three consecutive unsuccessful polls are fatal; a valid response
   resets the count. Replaced ingress identities, malformed status, exhausted
   engine budgets, and explicit fatal state still fail immediately.

Lease-token security remains deferred as requested. This change does not establish
legacy numerical parity or expand v4-64 borrowing.

## Validation

The following CPU-only suite passed: **111 tests**, with 17 dependency deprecation
warnings. Slurm job `14215631`, runtime 12.43 seconds:

```bash
srun -p cpu --cpus-per-task=2 --mem=4G --time=00:05:00 \
  .venv/bin/python -m pytest -q --disable-warnings \
  tests/tpu_swarm/test_farm_leases.py \
  tests/tpu_swarm/test_inference_borrowing.py \
  tests/tpu_swarm/test_hybrid_inference.py \
  tests/tpu_swarm/test_local_inference_fatal.py \
  tests/tpu_swarm/test_ray_train_engine_diagnostics.py
```

Tests exercise the actual HTTP ingress with mocked engines: cancellation before
arrival, while queued behind upload/drain, and after grant; draining admitted work;
lost/mismatched cancellation replies; recovery in both phase and run scopes;
legacy conservative fallback; bounded startup; request-error propagation through
serialized Ray exceptions with both scheduling paths; recovery after a rejected
request; transient versus persistent status failures; retained engine diagnostics.
This is not a new real-TPU deployment or end-to-end training validation.

## Deployment boundary

Roll out upgraded farms before relying on lost-acquire recovery in new training
bundles. New borrowers can use legacy farms, but an ambiguous acquire against a
legacy farm remains blocked for that borrower process. An unreachable farm or a
changed ingress identity also remains blocked until ownership is resolved; local
training can proceed. Existing running bundles do not acquire these fixes from a
local commit. Repackage and validate a farm/trainer canary before wider rollout.
