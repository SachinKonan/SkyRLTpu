# Circuit IBM17 on v5p-32 — prepared, not submitted

Each model gets one independent v5p-32 slice: four hosts, four TPU chips per
host. Host 0 trains; hosts 1–3 serve inference. All four hosts also grade on CPU.

| Model | Trainer TP / FSDP | Trainer token budget | Inference layout | Active sequences |
|---|---|---:|---|---:|
| Qwen | 1 / 4 | 90,112 | 3 hosts × 1 TP4 engine | 16/engine, 48 total |
| Muse | 1 / 4 | 90,112 | 3 hosts × 2 TP2 engines | 16/engine, 96 total |
| Gemma | 4 / 1 | 22,528 | 3 hosts × 1 TP4 engine | 16/engine, 48 total |

All three use GRPO (`mean_baseline`, `importance_sampling`), learning rate
4e-5, LoRA rank 32, seed 1, 16 groups × 32 completions, 15 epochs, native thinking
budget enforcement, and 80% inference memory utilization. Context is 22,528:
16,384 for prompt plus thinking and 6,144 for the answer. No separate bootstrap,
no imported seed pool, and no imported trained weights/optimizer. Fresh run IDs
allow subsequent strict recovery of their own progress.

CPU execution: 16 case tasks per host, 4 logical cores and 8 GiB per process tree;
64 case tasks across the slice. This reserves at most 64 CPU cores and 128 GiB
for graders on each host, separate from model/runtime and the 128 GiB tmpfs
cache cap. Host preparation checks available CPU affinity and memory. Candidate
work remains 180 s/case (170 s supplied to place()); trusted grading gets 180 s,
and the worker envelope is 390 s. Queue-inclusive client timeout is 172,800 s.
At the full worker envelope, 512×17/64 case waves require about 14.7 hours; this
is a capacity bound, not a measured step runtime. Complete-case grading is much
more expensive than the former four-case objective.

Every submission must pass all 17 cases: ibm01–ibm04 and ibm06–ibm18. Scientific
cost is the arithmetic mean of W + 0.5 D + 0.5 G. Any invalid/missing case or
hard-macro overlap invalidates the entire submission. Reward is
max(1e-6, 1/(1+mean_proxy_cost)). No Xplace or per-case normalization is applied.
Local results remain unverified by competition judges.

The immutable `xplace-start-ibm17-v2` starts were independently rescored with the
existing trusted evaluator: all 17 legal, mean proxy 1.37480890750885. Old
four-case prompts/results are retained, and old subset-scored pools are rejected
at restore and prompt construction. The new prompt injects the C-backed
Evaluator and preserves all 17 per-case diagnostics.

## Legacy comparison

Reviewed `tpu/jobman/cell_worker.sh` and `tpu/launch_cell.sh`. They confirm one
trainer / three inference hosts and Muse's two TP2 engines per inference host.
Their older defaults have shorter client contexts, different packed token
budgets, and more active sequences. Those defaults are superseded here by the
native-thinking reference profiles:

- native-v5p-qwen-ac2-grpo-lr4e5-s1-retryfix-004
- native-v5p-muse-ac2-grpo-lr4e5-s1-retryfix-004
- native-v5p-gemma-ac2-grpo-lr4e5-s1-bwd256-retryfix-004

Tests require exact trainer/inference dataclass equality to these profiles.
In particular, Gemma retains the backward block-size 256 settings; neither its
mesh nor Muse's engine layout is inferred from the older generic defaults.

All profiles target us-east5-a and the east5 bucket. Every new run has its own
run directory and trainer/inference compile write destination. Existing native
compile caches are read as seeds; trained checkpoints and optimizer states are
not compile caches and are not imported. Reference cache availability on GCS and
new slice startup have not been tested in this preparation task.

Profiles: `tpu/swarm/ray_train/profiles/science-circuit-v5p-{qwen,muse,gemma}-ibm17-helper-grpo-20260919.json`.
Validation: 152 distinct tests and 12 subtests passed across science, profiles,
cache, commands, bootstrap, mesh and lifecycle checks. The helper matched the
trusted scorer on all 17 starts within 1.83e-6 maximum component error. Local
bundle inspection checks all 17 NPZs, native files, current prompt and restore
guard.

Local review bundles: `.science/packages/circuit-ibm17-v5p-20260919/{qwen,muse,gemma}/`.
No upload, pool mutation, or job submission was performed.
