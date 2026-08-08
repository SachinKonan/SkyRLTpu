# Stranded EXTERNAL sampling requests in the SkyRL tinker server

Investigation + fix, 2026-08-08. Branch `agent/ttd-discover-erdos`, commits
`c100f2ac` (watchdog) and `79a3c02f` (slot leak).

## TL;DR

The framing I was handed — "requests are accepted, never dispatched, and dropped
before reaching vLLM" — **is not what the evidence shows**, and the difference
matters for the fix.

EXTERNAL requests *are* dispatched and *do* reach vLLM. What is actually broken:

1. **Nothing bounds how long a request may sit PENDING.** The client's progress
   timeout is 900s; the server's read timeout is **7200s**. A wedged or merely
   overloaded vLLM holds the row for up to two hours *after* the client has given
   up and restarted. This is the dominant cause of the 104/248-row PENDING piles.
2. **A genuine drop window exists and fired 24 times in 12 hours**: the terminal
   DB write sat outside the `try/except`, so a failed write threw away a result
   the model had already generated, leaving the row PENDING with nothing watching.
3. **Nothing ever re-scans PENDING EXTERNAL rows**, so (1) and (2) are both
   terminal. That part of the framing is exactly right and is what the fix adds.
4. **The slot leak is real and is a config bug**: `--session-timeout-sec 86400`.
   47 leaked adapter slots in one 12-hour run.

## Evidence

Read-only peek at the live slices (`sk7524-tunix-qwen35-v5p32-dbtest-{d,e}`,
project `vision-mix`, zone `us-east5-a`). Nothing on the hosts was modified,
restarted, or written. DB path on the host, from the live engine cmdline:
`/home/sk7524_princeton_edu/SkyRLTpu/skyrl/tinker/tinker.db`; API log
`~/skyrl-logs/tinker-api.<MMDD-HHMMSS>.log`. `sqlite3` is not installed on the
hosts, so queries went through `python3 -c` with a `file:...?mode=ro` URI.

### dbtest-e, rotated `tinker.db.1786193598.bak` (12h run, 00:36 → 12:48 UTC)

```
('EXTERNAL', 'FAILED', 709)      ('EXTERNAL', 'PENDING', 248)
('EXTERNAL', 'COMPLETED', 547)   ('CREATE_MODEL', 'COMPLETED', 47)
```

* Pending EXTERNAL `created_at` spans `10:48:21` → `12:37:43`. The oldest is
  **exactly 2h00m** before the snapshot — i.e. `external_inference_timeout_sec =
  7200.0`. These rows are not "dropped"; they are parked on an httpx read that
  has not yet timed out.
* Pending rows group into the last 8 sweep iterations, 32 apiece
  (`24 @10:48:21`, then `32 @11:03:57`, `32 @11:19:33`, … `32 @12:37:42`).
* Log exception histogram over the whole run: **733 `httpx.ReadTimeout`**
  (all from `_forward_to_engine`'s `POST /completions`), 3 `ValueError`.
  Zero `QueuePool limit`, zero `Task was destroyed but it is pending`, zero
  `Task exception was never retrieved`.
* **733 error log lines but only 709 FAILED rows.** The 24-row delta matches the
  oldest stranded batch exactly. Those are results that were computed, logged as
  failed, and then lost when the terminal write did not land.
* Every FAILED row reads `{"error": "", "status": "failed"}`, because
  `str(httpx.ReadTimeout())` is the empty string.

### Live state at 15:14 UTC

| | dbtest-e bak (12:48) | dbtest-e live | dbtest-d live |
|---|---|---|---|
| EXTERNAL COMPLETED / FAILED / PENDING | 547 / 709 / 248 | 200 / 0 / 88 | 0 / 0 / 32 |
| oldest PENDING age | 2h00m | 45m | 11m |
| CREATE_MODEL futures | 47 | 9 | 1 |
| models `created` / `unloaded` | 47 / 0 | 9 / 0 | 1 / 0 |
| sessions `active` / NULL heartbeat | 47 / 0 | 9 / 0 | 1 / 0 |
| `External engine error` in log | 733 | 0 | 0 |

Measured *legitimate* end-to-end EXTERNAL latency on dbtest-e: **18 to 55
minutes** (32 concurrent group requests fanned across 3 colocated vLLM workers).
That number sets the watchdog's in-flight ceiling — anything under ~1h would
cancel healthy work.

Note dbtest-d: 32 rows PENDING for 11 minutes, zero errors in the log, client
heartbeating normally. That is slow vLLM, not a lost request. **The throughput
problem on these slices is upstream of this fix** — a sampling round taking
18-55 minutes will blow a 900s client progress timeout no matter how good the
recovery path is. The watchdog bounds the damage; it does not make vLLM faster.

## Loss mechanism, with file:line

Confirmed against the code actually deployed on the hosts (an rsync'd tree with
no `.git`, matching this branch before the fix).

**`skyrl/tinker/engine.py:468`** — `find_single_requests` excludes EXTERNAL:

```python
.where(FutureDB.request_type != types.RequestType.EXTERNAL)
```

So the engine loop will never touch these rows. Correct by design, but it means
the API process is the *only* thing that can ever resolve one.

**`skyrl/tinker/api.py:1234` (pre-fix)** — fire-and-forget, return value dropped:

```python
asyncio.create_task(
    req.app.state.external_inference_client.call_and_store_result(...)
)
```

The event loop holds only a weak reference to a task; CPython documents that an
untracked task can be garbage collected mid-execution. Its exception is also
never retrieved, so a task that dies is completely silent. (No
`Task was destroyed` lines appeared in the live logs, so this was not the active
mechanism here — but it is why failures were invisible.)

**`skyrl/tinker/extra/external_inference.py:109-114` (pre-fix)** — the actual
confirmed drop, 24 rows:

```python
        except Exception as e:
            logger.exception("External engine error")
            result_data = {"error": str(e), "status": "failed"}
            status = RequestStatus.FAILED

        async with AsyncSession(self.db_engine) as session:   # <-- outside try
            future = await session.get(FutureDB, request_id)
            future.result_data = result_data                   # <-- AttributeError if None
            future.status = status
            future.completed_at = datetime.now(timezone.utc)
            await session.commit()
```

Anything raised here — pool timeout, `database is locked`, `future is None` —
kills the task with the row PENDING and the result gone.

**`skyrl/tinker/config.py:47`** — `external_inference_timeout_sec = 7200.0`,
8× the client's 900s progress timeout. This is what turns a slow worker into a
two-hour strand.

## What the watchdog does

New module `skyrl/tinker/dispatch.py`.

* `ExternalDispatcher.dispatch()` replaces the bare `create_task` and keeps a
  **strong reference** in `_inflight`, with a done-callback that unregisters the
  task and logs any exception.
* `complete_external_future()` performs the terminal write as a single Core-level
  `UPDATE futures SET ... WHERE request_id = ? AND status = 'pending'`, retried 3×
  with backoff. Conditional-on-PENDING is what makes recovery safe: a resurrected
  original attempt matches zero rows and its result is discarded. **A request can
  never be completed twice or produce duplicate results.** If the write genuinely
  cannot land, the row is left PENDING *on purpose* — the watchdog can recover
  that; a FAILED row would strand a client that could have been saved.
* `sweep_once()` scans PENDING EXTERNAL rows every poll interval and, per row:
  * **has a live task, under the ceiling** → left completely alone. This is the
    guard that protects a legitimately slow generation, and it is exact: liveness
    is a fact about the task, not a guess about the clock.
  * **has a live task, past `INFLIGHT_SEC`** → the worker is wedged. Cancel and
    re-dispatch; the client's round-robin sends the retry to a *different* vLLM.
  * **no live task, older than `STALE_SEC`** → orphan (task died, was cancelled at
    shutdown, or belonged to a previous API process). Re-dispatch by replaying the
    stored `SampleInput` through duck-typed shims.
  * **past `MAX_REDISPATCH` attempts** → fail explicitly with a descriptive error
    so the client gets a 400 and can retry, instead of hanging.
* Every re-dispatch logs at ERROR with request id, age, and attempt number, plus a
  per-sweep summary line, so the rate is visible in the API log.
* `aclose()` (wired into the lifespan) leaves in-flight rows PENDING rather than
  failing them — the next API process's watchdog picks them up.

`format_exception()` keeps the exception type when `str(exc)` is empty, so FAILED
rows stop reading `{"error": ""}`.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `SKYRL_EXTERNAL_WATCHDOG_ENABLED` | `1` | Master switch. |
| `SKYRL_EXTERNAL_WATCHDOG_POLL_SEC` | `30` | Sweep interval. |
| `SKYRL_EXTERNAL_WATCHDOG_STALE_SEC` | `300` | Age at which a row with **no live task** is re-dispatched. |
| `SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC` | `3600` | Age at which a row **with** a live task is cancelled + retried. `<= 0` disables. |
| `SKYRL_EXTERNAL_WATCHDOG_MAX_REDISPATCH` | `2` | Attempts before the row is failed. |

`INFLIGHT_SEC=3600` sits above the measured 18-55min legitimate latency and below
the 7200s read timeout. An operator who would rather burn a duplicate generation
than let a row outlive the client's 900s progress timeout should set it to ~900.

## Slot leak

**Finding: it is a config bug, not the NULL-heartbeat hole I first suspected.**

`cleanup_stale_sessions` is the only path that frees a model's adapter slot after
its client dies. On the live DBs **no session ever had a NULL `last_heartbeat_at`**,
so my initial hypothesis was wrong. What actually happens: each sweep client
heartbeats ~93 times (~15.5 min), exits, and its session then sits `status='active'`
with a frozen heartbeat — while `--session-timeout-sec 86400` means "stale" is
23h45m away. Result: 47 CREATE_MODEL futures, 47 models `created`, **zero ever
`unloaded`**. One leaked slot per sweep iteration, exactly the behaviour the sweep
driver works around by restarting the engine per variant.

Fixed:

* `tpu/start_colocated_vllm_tinker.sh`: `SESSION_TIMEOUT_SEC` default
  `86400 → 1800`. The tinker SDK heartbeats every 10s from a background thread for
  the life of the client process, so 1800s is ~180 missed beats of slack; a
  live-but-idle client is never at risk. This is the whole fix for the observed leak.
* `engine.py`: `last_heartbeat_at < cutoff` → `coalesce(last_heartbeat_at,
  created_at) < cutoff`. `NULL < cutoff` is NULL in SQL, so a session that died
  before its first heartbeat was invisible to the sweep forever. Not the observed
  mechanism, but a real hole on exactly the path a crash-looping client takes.
* `api.py create_session`: seed `last_heartbeat_at` at creation.

I did **not** add a force-unload-by-model API or an idle-model reaper. Both are
more invasive than the problem warrants now that the timeout is sane: a reaper
keyed on model idleness (rather than session liveness) would unload a model that a
live client is simply not using at that moment, and there is no server-side signal
that distinguishes the two. If a stronger reclaim is ever needed, the safe version
requires (a) a client-visible lease/refresh on the *model* rather than the session,
(b) the engine refusing requests against a reclaimed model with a distinguishable
error, and (c) the SDK transparently re-creating the model on that error. That is a
protocol change, not a patch.

## Test evidence

`tests/tinker/test_external_dispatch.py` — 19 tests, CPU-only (temp-file SQLite +
a fake inference client; no engine, no backend, no model). Highlights:

* `test_dropped_request_blocks_client_forever_without_watchdog` — **the
  reproduction.** The completion write raises, the task dies; a poll loop standing
  in for the SDK's `retrieve_future` returns nothing, the row is PENDING with
  `result_data is None`, and `_inflight` is empty — the exact production signature
  of a stuck row with an idle machine.
* `test_watchdog_recovers_a_dropped_request` — same failure; one sweep
  re-dispatches and the row reaches COMPLETED with the real result.
* `test_orphan_from_a_previous_api_process_is_redispatched` — a row with no task at
  all recovers on age; a 5s-old row is left alone.
* `test_hung_worker_is_cancelled_and_redispatched` — a worker that accepts and never
  answers is untouched below the ceiling and cancelled + retried above it.
* `test_slow_but_alive_request_is_not_redispatched` — `stale_after=0` and the row
  backdated **one hour**; five consecutive sweeps make zero re-dispatches because
  the task is alive. Exactly one client call is ever made.
* `test_complete_external_future_is_exactly_once` /
  `test_redispatched_request_cannot_be_completed_twice` — idempotency: the second
  and third writers return `False`, the first result stands unchanged, and a
  resurrected original cannot overwrite a re-dispatched success.
* `test_request_is_failed_after_max_redispatch` — budget exhausted → FAILED with
  `"could not be recovered"`, total calls = original + 2.
* `test_dispatcher_holds_a_strong_reference_to_in_flight_tasks` — survives a forced
  `gc.collect()` while suspended; `_inflight` drains on completion.
* `test_create_session_starts_the_staleness_clock`,
  `test_failed_sample_records_an_attributable_error` (no more `{"error": ""}`).

`tests/tinker/test_engine.py` — 3 new tests: NULL and stale heartbeats both reclaim
the slot and expire the session; a session inside the timeout keeps its model.

Suite status: see `RESULTS.md` in this directory.

## Deployment recipe

The watchdog lives entirely in the **API process**. The engine needs a restart only
for the `coalesce` change; the launcher change needs a fresh launch.

Files to copy to each host (`~/SkyRLTpu/` on the train-coordinator worker):

```
skyrl/tinker/dispatch.py                              (new)
skyrl/tinker/api.py
skyrl/tinker/engine.py
skyrl/tinker/extra/external_inference.py
skyrl/tinker/extra/skyrl_train_inference_forwarding.py
```

`tpu/sync_skyrl_to_tpu.sh` already does this; a targeted rsync of `skyrl/tinker/`
is sufficient.

**Restart required: yes, both processes.** The API imports `dispatch` at module
load and starts the watchdog in `lifespan`; the engine's stale-session query is
compiled at call time but the process must be restarted to pick up the new file.
Since `api.py` spawns the engine as a subprocess, restarting the API restarts both.
No DB migration — no schema change.

Recommended env for the sweep slices (client progress timeout is 900s):

```bash
export SKYRL_EXTERNAL_WATCHDOG_STALE_SEC=300
export SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC=1200   # aggressive; default 3600 is safer
export SESSION_TIMEOUT_SEC=1800                    # now the launcher default
```

Also worth considering independently of this fix: lower
`--external-inference-timeout-sec` from 7200 to something under the client's
progress timeout. A 2-hour read timeout cannot help any client that gives up after
15 minutes; it only converts a slow worker into a long strand.

### Verifying the fix from the futures table

On the host, read-only:

```bash
python3 - <<'PY'
import sqlite3
db = sqlite3.connect("file:/home/sk7524_princeton_edu/SkyRLTpu/skyrl/tinker/tinker.db?mode=ro", uri=True)
for row in db.execute("select request_type, status, count(*) from futures group by 1,2"):
    print(row)
# NB: SQLAlchemy stores enum NAMES, so these are uppercase in the DB.
print("oldest pending EXTERNAL:", db.execute(
    "select min(created_at), max(created_at), count(*) from futures "
    "where request_type='EXTERNAL' and status='PENDING'").fetchone())
print("models by status:", db.execute("select status, count(*) from models group by 1").fetchall())
PY
```

Healthy after deployment:

1. **No PENDING EXTERNAL row is older than `INFLIGHT_SEC + POLL_SEC`.** Pre-fix the
   oldest reached 2h00m; post-fix the age of the oldest pending row is the headline
   metric.
2. `grep -c WATCHDOG ~/skyrl-logs/tinker-api.log` — each line carries request id,
   age and attempt. Zero is fine (nothing was lost). A steady nonzero rate means
   requests *are* being dropped or wedged and you now have the count; a rate that
   climbs with load points back at vLLM, not at the dispatcher.
3. **No FAILED row has a blank error**:
   `select count(*) from futures where status='FAILED' and json_extract(result_data,'$.error')='';`
   should be 0 for rows created after the deploy.
4. **Slot leak closed**: `select status, count(*) from models group by 1` should show
   `unloaded` growing as sweep variants finish, instead of `created` only. Expect the
   first unloads ~30 min after a client exits.
5. `select count(*) from futures where request_type='create_model';` should stop
   tracking 1:1 with live adapter slots — leaked models now return to the pool, so the
   engine no longer needs a per-variant restart.

### Rollback

`SKYRL_EXTERNAL_WATCHDOG_ENABLED=0` disables all re-dispatch and cancellation
behaviour while keeping the (strictly safer) retried, idempotent terminal write and
the strong task references. Full revert is `git revert c100f2ac 79a3c02f`.
