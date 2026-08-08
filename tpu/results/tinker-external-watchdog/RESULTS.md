# Test results

All runs on a neuronic compute node via sbatch (`-c 16`, `--mem=64G`,
`JAX_PLATFORMS=cpu`). Nothing was run on the login node beyond the fast targeted
suite, and nothing was run on a TPU slice.

## Green gate — job 3655863

```
uv run --extra dev --extra jax --extra tinker pytest tests/tinker/ -q \
  -k "not tunix" --ignore=tests/tinker/test_tunix_backend.py \
  --ignore=tests/tinker/test_tunix_backend_maxtext.py
77 passed, 15 skipped, 9 deselected in 713.98s          EXIT 0

uv run --extra dev --extra jax --extra tinker pytest tests/utils/ -q
2 passed in 0.24s                                        EXIT 0
```

The 77 include the full end-to-end `test_api.py[jax]` set — a real API server
subprocess with its engine child — so `asample`'s dispatcher wiring, the lifespan
construction/teardown of the watchdog, and the `create_session` heartbeat seed are
all exercised against the real server, not just in unit tests.

## New tests — 25 total

`tests/tinker/test_external_dispatch.py`, 22 passed in 4.66s (CPU-only: temp-file
SQLite plus a fake inference client; no engine, no backend, no model):

```
test_dropped_request_blocks_client_forever_without_watchdog   <- the reproduction
test_watchdog_recovers_a_dropped_request
test_orphan_from_a_previous_api_process_is_redispatched
test_hung_worker_is_cancelled_and_redispatched
test_redispatched_attempt_gets_a_fresh_inflight_clock
test_slow_but_alive_request_is_not_redispatched               <- the no-false-positive guard
test_complete_external_future_is_exactly_once                 <- idempotency
test_redispatched_request_cannot_be_completed_twice           <- idempotency, end to end
test_watchdog_does_not_redispatch_an_already_completed_row
test_request_is_failed_after_max_redispatch
test_abandoned_rows_are_failed_not_regenerated
test_dispatcher_holds_a_strong_reference_to_in_flight_tasks
test_aclose_leaves_inflight_rows_pending_for_the_next_process
test_watchdog_loop_survives_a_failing_sweep
test_create_session_starts_the_staleness_clock
test_format_exception_keeps_the_type_for_blank_messages
test_failed_sample_records_an_attributable_error
test_replay_request_matches_the_client_contract
test_replay_request_prefers_string_stops_when_no_stop_tokens
test_redispatch_replays_the_stored_payload
test_watchdog_reads_env_configuration
test_non_external_pending_rows_are_ignored
```

`tests/tinker/test_engine.py`, 3 new:

```
test_cleanup_stale_sessions_reclaims_slots_with_no_heartbeat[never_heartbeat]
test_cleanup_stale_sessions_reclaims_slots_with_no_heartbeat[stale_heartbeat]
test_cleanup_stale_sessions_keeps_a_live_session
```

## Pre-existing failures, not caused by this change

Recorded for honesty; both reproduce independently of this work.

**1. tunix backend — 21 failures (job 3655682).** Every `[tunix]` parametrization of
`test_api.py` and all of `test_tunix_backend.py`:

```
TypeError: Qwen3.__call__() got an unexpected keyword argument 'skip_lm_head'
  .venv/.../qwix/_src/interception.py:177
```

The installed `google-tunix` predates this branch's FLCE vocab-tiling work in
`skyrl/backends/tunix_backend.py`. This change touches neither file. Under the
documented `--extra dev --extra jax` invocation (no `--extra tunix`) the same tests
fail earlier with `ModuleNotFoundError: No module named 'tunix'`.

**2. `tests/tx/` — failures then a segfault at ~30% (job 3655682).**

```
Fatal Python error: Segmentation fault
  transformers/core_model_loading.py:946 in _materialize_copy
  torch/storage.py:471 in __getitem__
```

A crash in `transformers`' multi-threaded HF weight materialization, which takes the
whole pytest process down and makes the documented single-command invocation
(`pytest tests/tx/ tests/tinker/ tests/utils/`) unusable as a gate in this
environment — the crash masks everything after it. `tests/tx/` imports exactly one
symbol from the tinker package, `skyrl.tinker.types.LoraConfig`, and `types.py` is
not modified by this change (`git diff --stat` covers `api.py`, `engine.py`,
`dispatch.py`, `extra/*`, tests, and one shell script).

## Lint

```
ruff 0.11.9   — clean on every file this change touches
black 24.10.0 — clean on every file this change touches
```

`skyrl/tinker/api.py` still has two pre-existing black violations (the
`create_async_engine(...)` call and the `route_external = (...)` assignment), both
present at the base commit and both outside this diff; they were left alone rather
than mixing an unrelated reformat into the fix. `skyrl/tinker/engine.py` also has a
pre-existing ruff `E702` in the `_edbg` debug helper around line 689 — leftover
`/tmp/enginedbg.log` tracing that is still actively writing on the live slices and
is probably worth removing separately.
