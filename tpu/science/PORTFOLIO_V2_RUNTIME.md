# Portfolio v2 CPU evaluation

The runtime derivative refines the original joined dataframe's return accounting
and connects its pandas interface to the same isolated Ray CPU executor used for
the original science tasks. It does not overwrite the verified base dataframe.

The model implements `fit(train, valid, rng, *, time_budget_s)` with pandas
DataFrames and `act(model, obs)`. The observation has a 30-row `market` DataFrame
and `current_weights[31]`, cash first. It contains the current decision session,
rolling price indicators, and the seven-day headline window. The full contract,
equity/year coverage, allowed libraries, solver list, and library thread controls
are in [the model prompt](prompts/rendered/portfolio_v2.txt).

## Accounting and timing

Each decision uses the preceding NYSE close and is made 15 minutes later. Old
holdings first experience the next overnight price move and distributions. New
target-weight orders then execute at that open, paying 0.001 times asset-weight
turnover. Daily wealth is marked at the ensuing close. The holdings shown to the
next decision are marked at that close, with no following-open information.
The first 2025 discovery decision uses December 31, 2024 information; every one
of the 250 completed 2025 sessions contributes a daily return. Cash earns zero.

The supervised fit label is next-open to following-open total return. This
asset-level learning target differs from daily portfolio PnL, which also reflects
old positions held overnight before a new order fills. Split-crossing fit labels
are purged. The reward is the bounded sigmoid of annualized sample Sharpe; failure
is zero. Return, drawdown, volatility, turnover, phase times, peak cgroup memory
and rejected orders are returned as feedback alongside the scientific Sharpe.

The [reviewed action manifest](manifests/portfolio-v2-actions.json) distinguishes
real AAPL/WMT share splits from five distribution adjustments in Yahoo prices:
UTX→CARR/OTIS, PFE→VTRS, MRK→OGN, IBM→KD, and MMM→SOLV. Undoing vendor adjustment
factors reconstructs contemporaneous share units, subject to vendor price and
dividend rounding. Each spin-off keeps the parent share count unchanged and
liquidates fractional child shares at their first regular-way open, with the
same proportional transaction fee. Source links and distribution ratios are
recorded in the manifest.

Walgreens settles for $11.45 cash per share on August 28, 2025, with its
nontransferable DAP right recorded but valued at zero. This is an explicit
conservative valuation convention, not a claim that the right had no economic
value. The grader does not fabricate a traded price after termination. Requested
new exposure to a terminated security remains cash; an unexplained missing quote
anywhere else aborts preparation. The only prelisting gap is DOW before the
vendor's first March 20, 2019 quote, during warm-up.

These are simplified replay conventions: fractional shares, ideal open/close
fills, ex-date dividend cash credit, no taxes or market impact, and immediate
modeled settlement. Public archive revisions and unavailable original headline
versions remain limitations. This is not a broker-exact securities ledger.

## CPU envelope

Keep 4 CPU cores and 8 GiB aggregate memory per candidate. Fit has 180 seconds;
inference/replay has 60 seconds; the overall task has 300 seconds. These include
candidate imports, data loading inside its sandbox, and library initialization.
The fitted-model limit is 128 MiB. BLAS defaults to one thread per library worker;
joblib, XGBoost and solver parallelism share the four-core cgroup allowance.

The Slurm sandbox pilot completed equal-weight in 5.1 seconds, TF-IDF/Ridge in
18.6 seconds, and TF-IDF/SVD + XGBoost/CVXPY in 21.4 seconds. Peak cgroup memory
was 409, 655 and 449 MiB respectively. These establish feasible reference
workloads, not a guarantee that every learning algorithm will fit. Ray host
measurements are required before treating these as TPU-host timings.

## Reproduce

Use the separate data preparation environment to build the derivative:

```bash
srun -p cpu -c 4 --mem=8G --time=00:10:00 \
  .science/data-venv/bin/python -m tpu.science.prepare_portfolio_v2_runtime \
  --data .science/data/portfolio-v2 --raw .science/data/portfolio-v2-raw \
  --output .science/data/portfolio-v2-runtime-new
```

The discovery worker receives only `train.json.gz`, `valid.json.gz`,
`discovery_observations.json.gz`, `discovery_market.npz`, and `manifest.json`.
It never receives final-test files. Only train/valid are mounted during fitting;
inference receives one causal observation at a time. Grading never unpickles
candidate models. The original historical research parquet and corrected audit
parquet stay outside the deployed runtime.

Deploy that runtime at `.science/data/portfolio-v2-runtime` under a prepared
worker root. `ray_cpu.grade` accepts task `portfolio_v2`; the Discover bridge
accepts problem type `portfolio_v2`, four CPUs and `eval_timeout=300`. Both use
the existing systemd/cgroup wrapper and namespace sandbox.

## Legacy result correction

The old v1 replay exposed next-open holdings to an earlier close-time decision.
Several generated programs used those weights. Its archived portfolio scores
therefore cannot establish an unbiased model comparison and require a corrected
rerun. Routing scores are unaffected. The new replay includes a regression that
mutates a future opening price and verifies that earlier observations and
holdings do not change.
