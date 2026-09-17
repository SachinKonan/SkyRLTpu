# CPU science benchmarks on the native Ray v2 workload

This worktree is based on native-training commit
`3f57333f7bf7e1e0ddf29d202ce355eec6e34823`. Qwen is the first model pilot.
Portfolio and qubit routing have real isolated graders. Chip placement has a
schema/parser and native-client adapter, but is blocked on `plc_wrapper_main`;
no native placement score or timing has been validated.

## Resource and reward contracts

| Task | CPU / aggregate memory | Candidate time | Trusted grading | Reward |
|---|---|---|---|---|
| Routing | 4 / 8 GiB | 900 s build + route, 72 cases, 20/20 trials | 900 s; 1800 s overall | weighted B/(B+C) |
| Portfolio | 4 / 8 GiB | 180 s fit, 60 s inference/replay | 300 s overall | sigmoid(sample Sharpe) |
| Placement, provisional | 4 / 16 GiB | 60 s/netlist, four netlists | 600 s overall | 1/(1+normalized proxy cost) |

Invalid results have reward and correctness zero. Every valid result receives
at least `1e-6`. `raw_score` equals normalized reward for higher-is-better state
ranking; scientific metrics remain separate. Routing invalidates the whole
submission if any case fails. Its baseline denominators are the upstream
suite's stored reference counts: the current seed can outperform them and
therefore need not score exactly 0.5.

Fixed package installation and Rust dependency warming happen before candidate
timing. Changed policy compilation counts. Queue admission is separate from
the candidate budget. The resource profiles are pilot choices; placement's
limits remain provisional until its real native evaluator is available.

Portfolio allows NumPy, SciPy, pandas, scikit-learn (`sklearn`), statsmodels,
SymPy, matplotlib, CPU XGBoost and CVXPY. The full import and solver list is
generated into its prompt from `resources.py`. JAX, Flax, Optax and PyTorch are
excluded. Library threads share one four-core allocation; native BLAS defaults
to one thread per library worker to avoid nested oversubscription. Memory is
shared across all libraries and descendants, not granted separately per library.

## Execution and isolation

`ray_cpu.grade` runs on the existing private workload Ray cluster. Every one of
the eight prepared hosts is eligible, including inference hosts. The initial
pilot defaults to two four-core CPU slots per host. Routing profiles can opt into
`science_routing_slots_per_host=16`: sixteen concurrent tasks, 64 logical CPUs
and at most 128 GiB of combined task memory per host. Host-wide slot locks prevent
new payload directories from duplicating those slots. Each Ray task reserves four
logical CPUs and 8 GiB, then launches an individually named systemd service with
`CPUQuota=400%`, a four-CPU cpuset, `MemoryMax=8G`, no swap, `TasksMax=128`, and a
hard overall deadline. The namespace sandbox runs candidate code without host
credentials, networking or accelerator devices. Import restrictions provide
reproducibility; the sandbox and cgroup enforce isolation and resource bounds.

The expanded routing profile advertises 72 Ray CPUs per host (64 for grading and
eight for actors), and bootstrap admission scales to the host count. Candidate
budgets, rewards, and placement/portfolio concurrency are unchanged. Candidate
`/tmp` is tmpfs charged inside its 8 GiB limit; the separate model/compilation
RAM cache is capped at 128 GiB for expanded profiles. A tmpfs capacity is not a
physical memory reservation. Retired cache mounts and model/runtime allocations
still count against host RAM. Existing running jobs keep their packaged settings;
apply this change through a new package and controlled checkpoint resume, not a
live edit of imported Ray functions. Old and new executors must not overlap on a
host because the old pilot used payload-local slot locks.
Completed routing evaluations discard their private `evaluation/target` and
`evaluation/rust` build trees after the worker stops. Candidate source, build/run
logs, routed outputs and verdicts remain available. Keeping a roughly 380 MiB
private Cargo target per candidate indefinitely can fill the host disk.

Cooperative cancellation stops the service. If Ray kills the worker process,
an owner-PID/start-time watchdog exits the service, and systemd kills all its
children. The hard service deadline also applies. Candidate pickle objects are
only loaded within the candidate sandbox, never in the controller or scorer.

Routing compiles the pinned native Rust scaffold, runs the suite, and validates
every output with Qiskit. It independently counts validated output SWAPs rather
than trusting the reported count. Portfolio fitting receives inner train/valid
only. A fresh inference sandbox receives the frozen model and one causal
observation at a time. Each action forks the same frozen state so candidate
mutations cannot carry information across observations. The trusted parent
applies fees, returns, drift, and scoring.

## Data and provenance

`sources.json` pins SimpleTES, TradeMaster and Circuit Training revisions.
`requirements-cpu.lock` pins CPU libraries. `portfolio_data.py` builds six
causal features from TradeMaster DJ30 price fields for 29 assets. Decisions use
features through close[t], execute at open[t+1], and earn the subsequent
adjusted-open return through open[t+2]. Split-crossing labels are purged.

The split sizes are 1719 inner-train days, 250 inner-valid days, 251 discovery
days, and 250 final-test days. Raw inner-training arrays occupy about 1.55 MiB;
the budget allows substantially larger fitted models and intermediate arrays.
The final-test NPZ is deliberately excluded from deployed worker inputs.
Discovery feedback is validation, not an untouched test. Public historical
prices cannot be claimed uncontaminated by model pretraining.

The candidate/observation schema, costs, dates, assets, feature definitions and
hashes are included in the rendered prompt. Fees are 0.001 times absolute
changes in asset weights, deducted before returns; the cash leg is excluded
from turnover and cash return is zero. Sharpe uses sample standard deviation
(`ddof=1`) with the explicit 1e-8 floor.

## New price/headline dataframe

The separate [portfolio data v2 guide](PORTFOLIO_V2.md) documents the 2020–2023
train, 2024 inner validation, 2025 discovery and 2026 final split, with 2019
warm-up. It joins a fixed historical 30-equity universe with archived news,
causal price/volume features and lagged Treasury yields. Its model-facing card
includes equity/year coverage and the CPU library contract.

This is **data version v2**, distinct from **prompt version v2** in the model
pilots below. Those pilots use the original TradeMaster-derived v1 data. The new
dataframe is not yet connected to Ray scoring; corporate-action and unavailable
execution-price handling must be pinned before that integration.

## Evidence collected so far

**Portfolio v1 timing issue:** the legacy replay exposed next-open holdings to
an earlier close-time decision. Archived portfolio pilot scores are diagnostic
only and do not establish a causally valid ranking. Several candidates use
`current_weights`, so they require a corrected replay and rerun. Routing results
are unaffected. The v2 replay separates close-time observations from next-open
execution and has a future-price mutation regression test.


| Reference | Execution | Result |
|---|---|---|
| Native routing seed | isolated Slurm CPU, 72 cases | 898.6 s; 101,618 SWAPs; reward 0.519364 |
| Native routing seed | Ray on v4 CPU, 72 cases | 1109.9 s; identical SWAP count and reward to Slurm |
| Equal-weight portfolio | v4-host CPU | reward 0.631134; Sharpe 0.537086 |
| All-library portfolio learner | Ray, all eight v4 hosts concurrently | 8/8 valid; identical reward 0.674651 and Sharpe 0.729299; 16.2–17.1 s; 231–262 MiB peak |
| Invalid actions/imports/Rust wrapper | Ray | all rejected with zero reward |
| Cooperative and forced cancellation | Ray + systemd | service and cgroup removed; forced cleanup about 1 s |
| Discover environment bridge | actual locked Discover client + Ray | reward, scientific diagnostics and higher-is-better next-state value verified |

Evidence is in `results/`. The older standalone profile is labeled separately
from the later Ray measurements. These reference scores are not Qwen scores.
Full-suite Ray routing and the first Qwen inference pilot are complete. The
matched comparison uses routing prompt v1 and portfolio prompt v2. Portfolio
v2 clarifies the ordinary dictionary/NumPy array interface and includes a
working equal-weight submission; its scorer and resource limits are unchanged.
The initial portfolio prompt results remain archived separately.

| Qwen, four completions per task | Valid | Best scientific result | Best reward | Mean reward, including failures |
|---|---|---|---|---|
| Routing v1 | 2/4 | 101,447 SWAPs; 171 fewer than the native seed | 0.519664 | 0.259757 |
| Portfolio v2 | 1/4 | Sharpe 0.433826 | 0.606787 | 0.151697 |

These are unmodified generated programs, not repaired submissions. The valid
portfolio program trades but scores below equal weight. Both valid routing
programs passed all 72 cases. This is a small inference pilot, not an RL sweep
or a statistically established model ranking. Per-candidate programs, verdicts,
generation audits and summaries are in `results/qwen-matched/`.

The matched Gemma and Muse pilots are also complete. All 24 generated programs
were archived and checked for matching prompts, sampling settings, native
thinking-budget enforcement and source/verdict consistency. The comparison is
in `results/model-comparison.json`.

| Model, four samples/task | Portfolio valid | Best Sharpe | Routing valid | Best SWAP count |
|---|---|---|---|---|
| Qwen3.5-27B | 1/4 | 0.433826 | 2/4 | 101,447 |
| Gemma-4-31B | 4/4 | 0.646266 | 2/4 | 100,934 |
| Muse-Glimmer-30B | 3/4 | 0.548275 | 1/4 | 101,046 |

Gemma had the best observed result on both tasks in this four-sample pilot;
this does not establish a robust model ranking. Portfolio results use the
original v1 historical data, not the new joined price/headline dataframe.
Gemma and Muse's CPU portfolio reference checks also passed on all eight hosts
with identical scores. All three serving pilots were stopped after archiving.
Device-owner checks confirmed all eight hosts of each pilot released their TPU
devices (`results/pilot-shutdown.json`). Intentional controller exit 143 can
appear as FAILED in SkyPilot; it does not invalidate completed benchmark results.

## Running on a prepared workload

Prepare the same immutable source/data/library bundle on every worker with
`prepare_cpu_host.sh`. The directory must contain `.science/routing-task`,
`.science/cargo`, `.science/rustup`, and `.science/data/portfolio` (train, valid,
discovery and manifest only). Fixed dependency installation requires host
administration permissions already available in the TPU workload image.

Run the driver with the workload controller's Ray/Python environment:

```bash
python -m tpu.science.ray_pilot \
  --address HEAD_IP:19679 --namespace RUN_ID --root /ABS/CPU_RUNTIME \
  --task portfolio --sources tpu/science/seed_portfolio_libraries.py \
  --output .science/new-pilot --all-nodes
```

For discovery integration, set `SCIENCE_WORKER_ROOT` and initialize the existing
workload Ray cluster before constructing `PortfolioAllocationEnv` or
`QubitRoutingEnv` from `environment.py`. Use `problem_type=portfolio_v1` with
`eval_timeout=300`, or `problem_type=qubit_routing_v1` with `eval_timeout=1800`;
both require `num_cpus_per_task=4` and `eval_backend=ray`. Start with small groups
relative to worker capacity. Do not launch the original 512-completion training
batch until its admission/throughput profile has been measured.

`sample_suite.py` uses the launched grid's native token-budget implementation:
22528 context, 16384 **prompt plus first-phase thinking** tokens, then roughly
6144 answer tokens (with the original 50-token reserve). It checks the native
budget audit, token IDs and forced-token loss masks before accepting samples.
The first pilot uses four completions per task and temperature/top-p 1. It
preserves the grid's engine-default sampling RNG (the TPU ingress does not
accept per-request seeds); candidate portfolio fitting uses seed 1.

Local contract tests: `.science/venv/bin/python -m pytest tpu/science/tests -q`.
The opt-in `test_ray_failures.py` checks actual worker rejection/cancellation on
an already prepared Ray cluster. No model-comparison or placement-success claim
should be made until corresponding real evaluation records exist.
