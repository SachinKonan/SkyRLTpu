# Science seed bootstrap

`bootstrap_layers` defaults to `0`; existing profiles keep their training path.
`1` generates drafts; `2` adds one repair layer for invalid programs only.
`bootstrap_all_hosts` defaults to `false` and requires bootstrap to be enabled.
This version supports single-model, native-thinking, TP4 v4-64 science runs.

The six `science-{routing,placement}-v4-{qwen,muse,gemma}-bootstrap-l2-001`
profiles use 16 groups of 32 drafts. Up to 16 distinct invalid programs that
fit the context are selected for 32 repairs each, spread across error signatures.
Valid drafts are never selected for bootstrap repair. If fewer invalid programs
are eligible, fewer repair groups run; if none are eligible, that layer is empty.
Each repair sees the full task/API instructions, its failed generated program,
and actual grader feedback. The original starter is omitted from repair prompts.

Bootstrap uses the frozen base policy through the native inference endpoint.
It does not construct a training client, trainer, or optimizer, and none of its
trajectories enter a gradient update. Its generation parameters retain the grid's
16,384-token prompt-plus-thinking cap and 22,528-token total context (including
the existing 50-token context reserve), temperature 1, and group size 32.

Results use the existing science graders and reward transforms. Only valid
results enter the promoted PUCT snapshot, retaining normal per-parent top-k,
deduplication, and visit accounting. An invalid repair parent is recorded in the
bootstrap journal, never as a persistent PUCT state. Its successful repairs are
attached to the original task root; their actual generating parent is recorded
in `repair_parent_id`. Negative state timestamps distinguish bootstrap from
optimizer steps. An empty valid seed pool blocks training.

Routing starts with eight inference hosts. Placement reserves its TPU grading
host and starts with seven inference hosts. After bootstrap the controller:

1. Persists the seed pool and bootstrap journal.
2. Closes ingress admission and drains all outstanding requests.
3. Removes the four future trainer hosts from routing, adapter fanout, and the
   catalog; retired engines cannot register again or trigger intentional restart.
4. Stops their owned inference processes and deploys the reduced Serve graph.
   Stable engine names preserve the remaining inference replicas.
5. Verifies endpoint membership/health and that retired TPU devices have no owner.
6. Flushes inference compilation caches, prepares the four physically contiguous
   trainer hosts, reserves their TPU resources, and starts the distributed trainer.
7. Requires trainer/API readiness and the reduced endpoint's health before the
   normal training client starts at optimizer step zero.

`client/bootstrap/` stores the immutable contract, parent plans, native generation
responses, per-candidate grades, and layer summaries. Completed generations and
grades are reused after restart. Pool promotion can be replayed without duplicate
visits. A completed bootstrap skips expansion on recovery and resumes the normal
training checkpoint. Changing the configuration or bootstrap contract requires a
new run ID. The normal optimizer counter/checkpoint format remains unchanged.

The new profiles request 15 optimizer steps, with one-step smoke mode disabled.
The one-day smoke wrapper does not apply. Success must be established separately
for each of the six runs: valid retained seeds, retired-host exclusion and device
release, trainer readiness, actual optimizer progress, and adapter/checkpoint
writeback. Local tests alone do not establish TPU execution success or speedup.

Bootstrap runs from the executor bundle directory, not the frozen model source
directory. Python's `-m` invocation puts the working directory ahead of
`PYTHONPATH`; the frozen source can otherwise shadow the current `Config` and
sampling helpers. Before deploying inference, the controller runs the same
bootstrap command with `--check-only` in the actual client environment. This
validates the complete configuration and imports, and records both module paths
in `bootstrap-import.log`. The packaged subprocess regression test deliberately
installs an incompatible frozen-source configuration to exercise this boundary.

Placement observations include bounded per-netlist W/D/G/cost/timing fields and
terminal error details. Subprocess command strings and initialization logs do not
replace the actual candidate/grader exception. The 3000-character observation is
complete JSON. An offline feedback-only migration must preserve original grade
records, leave code/scores/metrics/generations unchanged, and finish before a repair
plan exists; see the dated recovery record under `results/bootstrap-l2-20260915/`.

## Separate v4-32 routing seed shards

`bootstrap_only=true` is opt-in and requires routing, v4-32, four inference
hosts, zero trainer hosts and native TP4 inference. It runs the same bootstrap
and Ray CPU graders, uploads the journal and pool, then shuts down. It does not
create a trainer or optimizer. Existing training profiles retain their defaults.

The Muse `science-routing-v432-muse-seeds-l2-s{0,1,2}-001` profiles allocate
6, 5 and 5 groups respectively: 512 drafts total, followed by at most 512 repairs.
Each shard selects distinct invalid repair parents from its own drafts, with the
same error-stratified selection. This differs from selecting 16 repair parents
from a single global draft journal; valid programs are never repair parents.
All three slices keep the native 16k prompt-plus-thinking and 6k answer allowance,
TP4, 16 concurrent sequences per engine, and memory utilization 0.75.

`seed_pool.merge` checks the completed journals, prompt/model/sampling contracts,
invalid repair ancestry, counts and pool hashes. It replays all candidate records
through normal PUCT admission to apply global code deduplication, parent top-k
and visit accounting. It requires all 16 task roots and positive valid seeds.

`seed_handoff` waits for the three named jobs to succeed, downloads their durable
results, verifies and merges them, uploads a step-zero pool, and binds its checksum
into an immutable training template. Only then does it submit v4-64 training.
The training controller verifies that pool on restore before starting the trainer.
An incomplete shard, changed contract or failed job blocks automatic submission.
The handoff has a persisted status file and a 36-hour deadline.
Temporary SkyPilot queue failures or timeouts leave the state unknown and are
retried within that deadline; they never count as successful seed completion.

The Muse routing `-002` shard and training profiles replace the failed `-001`
attempts with `inference.batched_rpa_kernel=false`, selecting standard RPA.
They retain TP4, sequence concurrency 16, memory utilization 0.75, chunk size
1024, token budgets, and the 6+5+5 sampling allocation. The shared Muse preset
and older profiles are unchanged. The old full-serving failure reported
`jit_compute_logits_func` allocation failure after an earlier TPU device fatal
interrupt; the final exception alone does not establish an HBM shortage.
An isolated synthetic attention test did not reproduce that full-model failure.
The replacement full routing run passed the failing prefill transition and
generated tokens on all four engines at concurrency 16. This validates the
serving workaround, not completion of bootstrap, grading, or optimizer steps.
