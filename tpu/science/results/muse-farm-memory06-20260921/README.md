# Muse farm follow-up memory mitigation, 2026-09-21

The .70 memory-utilization farm passed its short canary, then failed a real
borrowed generation at 08:14:31 UTC on engine 10.130.0.29. The serving process
reported E0101 RuntimeProgramAllocationFailure loading
jit__substitute_placeholder_token through the asynchronous token substitution
path. This is a new real-request failure, separate from the earlier invalid
greedy n=16 canary. The complete captured engine tail is in failure.json.

At startup each TP4 chip held 13.36 GiB model weights and 21.52 GiB after KV
allocation, out of 30.75 GiB. The .70 budget still did not make the realistic
request stable. The next mitigation changes only inference.memory_utilization
from .70 to .60. It leaves approximately 3.07 GiB more HBM per chip for runtime
allocations, at the cost of KV capacity and potentially throughput. It remains
unvalidated for long requests until a real training batch completes.

The immutable archive is ffb757e66c364ab5d67fc53fbee8c066b64445fafa4cc70e6ff841362c11ba08.
CPU allocation 14221630 checked all 434 archive entries. Only the profile field
changes; other bytes and entry metadata remain identical to the .70 bundle.
The task changes only code URI and SHA256. Context 22528, max sequences 16,
TP4, four engines, native thinking, prefix caching off, lease policy, and
training recipes are unchanged. The upload size and MD5 were verified.

Before replacing the stopped farm, remote Sky jobs were all terminal and the
API request database contained no live exec for worker 177. The exact managed
job cancellation is confirmed before submission, and intent is persisted so
an ambiguous API result cannot be blindly retried. Existing Qwen/Gemma jobs
and queued second Gemma are left in place; the pool decides allocation.

Managed job 1368 was confirmed CANCELLED; replacement 1373 is submitted at
priority 120. Queue receipt records STARTING with no assigned worker yet.
A successful launch alone does not validate long-request stability.
