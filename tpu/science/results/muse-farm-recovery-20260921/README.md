# Muse inference farm recovery, 2026-09-21

The Muse farm failed at 02:32 UTC during generation. Its vLLM TPU runner raised
`RESOURCE_EXHAUSTED: E0101: RuntimeProgramAllocationFailure` while loading
`jit__substitute_placeholder_token`. The exact traceback is in original-error.txt.
All four host controllers completed cleanup, the remote Sky job was FAILED,
and the serving endpoint refused connections. Managed job 1333 nevertheless
remained PENDING with its old worker association. The scheduler counted that
association as occupied, preventing reuse of the otherwise idle v4-32 worker 177.

Cancellation and identical resubmission first released that association (1333
became 1366). Lower-priority second Gemma farm 1341 acquired the slot in between;
it was requeued as 1367 to give Muse AC2 precedence over Gemma RG-LRU. Inspecting
the full failure traceback then identified the program-allocation failure, so
1366 was replaced before application startup with the memory mitigation below.
The pending second Gemma request was briefly withdrawn during this replacement
to preserve the AC2 ordering.

| Workload | Current replacement job | Priority | Placement at submission |
|---|---:|---:|---|
| Muse farm | 1368 | 120 | v4-32 worker 177 |
| Second Gemma farm | 1369 | 110 | Pending |

Only Muse `inference.memory_utilization` changes, from 0.80 to 0.70. This leaves
more HBM headroom for runtime programs. It is a mitigation requiring generation
validation, not proof that every E0101 failure is solved. Model, context length
22528, max sequences 16, TP4, four engines, native thinking, prefix caching off,
one leased adapter, package/source versions, run ID, cache paths, and all
training settings stay identical. Active Qwen/Gemma farms 1330/1331 and the
Muse AC2 training process were not restarted.

The new immutable archive is
`b5c2aab680c39031fd5d3cd36ed39cbcd31c7d7b680086674c9d7fef16cffac6`.
It was derived from the original immutable farm archive
`8c45ac6a23fccfd178c142bb273abba09ec5ae7998a32b3c0c4ec079c322ba86`.
CPU allocation 14220597 verified every one of the 434 archive entries: only the
one profile field changes; other file bytes and entry metadata match. Task YAML
changes only code URI and archive hash. GCS upload size and MD5 were verified.
`prepared.json` and `upload.json` record hashes and object generation.

`prepare.py` and `replace.py` are the exact operational scripts, run from the
repository root. Their durable intents live under `.science/muse-farm-memory-fix-20260921`;
they deliberately refuse blind resubmission. Receipts and the post-submit live
queue are included here. Old jobs 1366/1367 were confirmed CANCELLED before
replacement. Check actual application readiness, four engines, lease admission,
and Muse checkpoint-4 resume before declaring recovery complete.

During the same check, Gemma RG-LRU passed its 300-second optional farm wait and
started sampling locally; it need not wait for the second Gemma farm to place.
The AC2 follow-up service is rearmed after this reviewed profile/receipt commit.
