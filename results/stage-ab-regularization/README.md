# Stage A (regularization) + Stage B (base model)

Two experiments on one harness. Every cell: one v5p-32, 15 steps, and a
preemption-proof loop in which **a step exists iff its gradient landed** — a
failed train step crashes the client and replays rather than banking a row.
Runs live at `gs://sk7524-tinker-tpu-us-east5/skyrl-runs/stageA2-*` and
`.../stageB-*`.

**Stage A** — {GRPO, TTD-Discover} x {N, K, R} x {Erdos, JSSP}, qwen3.5-27B.
N = no regularizer, K = KL penalty 0.1 toward base, R = fresh-weights restart
when a step's gain falls below 1% of the life's peak.

**Stage B / Phase 2** — Stage A's winning arm (N) on gemma-4-31B, three problems.

## Results

### Erdos min-overlap (C5, lower is better)

| cell | model x objective x arm | C5 |
|---|---|---|
| grpo-n | qwen GRPO N | **0.380861631** <- record, verified |
| g-ttd-n | gemma TTD N | 0.380863196 |
| ttd-n | qwen TTD N | 0.380867900 |
| *SimpleTES* | *gpt-oss-120b, 50 steps, inference* | *0.380868561* |
| grpo-r | qwen GRPO R | 0.380870435 (1 restart) |
| *TTT-Discover* | *gpt-oss-120b, 50 steps, training* | *0.380875* |
| ttd-k | qwen TTD K | 0.380891700 |
| ttd-r | qwen TTD R | 0.380907260 (3 restarts) |
| g-grpo-n | gemma GRPO N | 0.380917550 <- corrected, see below |
| *AlphaEvolve* | *published* | *0.380924* |
| grpo-k | qwen GRPO K | **did not complete** (see below) |

### JSSP / frontier_algo 46 (higher is better)

| cell | | score |
|---|---|---|
| g-grpo-n-j | gemma GRPO N | **0.227394** |
| g-ttd-n-j | gemma TTD N | 0.222985 |
| grpo-r-j | qwen GRPO R | 0.181880 (2 restarts) |
| ttd-n-j | qwen TTD N | 0.163495 |
| grpo-k-j | qwen GRPO K | 0.162678 |
| ttd-r-j | qwen TTD R | 0.153788 (1 restart) |
| grpo-n-j | qwen GRPO N | 0.150085 |
| ttd-k-j | qwen TTD K | 0.135748 |

### ac_inequalities (higher is better) — IN PROGRESS at time of writing

g-grpo-n-a -1.505522 (12/15) | ttd-n-a -1.506568 (13/15) |
g-ttd-n-a -1.507314 (15/15) | grpo-n-a -1.513835 (11/15)

## Conclusions

1. **Plain N is the right default.** It holds the record and wins both TTD
   trios. TTD-Discover's own published KL penalty costs it 17% on JSSP.
   Restarts help only GRPO-on-JSSP; on TTD they always hurt.
2. **Why KL hurts: it suppresses length growth.** Every unregularized qwen arm
   roughly doubles generation length over a run (~6k -> 13k tokens). ttd-k
   plateaus at ~8k with sigma=767 vs ~2250 for its siblings, and finishes worst
   of its trio. On JSSP all six arms sit at 12.9-13.2k regardless of treatment
   — no length growth to suppress, and the arms differ only in score.
3. **The base model outweighs the regularizer.** On JSSP gemma beats every qwen
   cell by >=25%, against a total N/K/R spread of 0.136-0.182. On Erdos the two
   models finish 1.5e-6 apart.
4. **Length growth is qwen-specific, not a property of RL discovery.** Six
   gemma cells across three problems shed their cold-start step and then hold a
   narrow band for 13+ consecutive steps while improving (JSSP 5448+/-546,
   ac1 5035+/-317, Erdos 6026+/-353). Qwen climbs to its 13824-token thinking
   cap and saturates it.

## Correction: g-grpo-n, 2026-08-20

g-grpo-n was published at **0.380909993** and is actually **0.380917550**.

The Erdos grader returned the program's self-reported `c5_bound` rather than
the value it had just recomputed, guarded only by `np.isclose(atol=1e-4)` --
wider than this problem's entire competitive span. A program could therefore
build a density with slightly too much mass, evaluate the un-normalized
correlate, and report that; the grader renormalized to `sum(h) == n/2` before
measuring but returned the claim anyway, so the excess mass was scored and
never paid for. Fixed in `third_party/discover` ae0a9bb (returns the
recomputed score, archives the projected `h`).

Every completed Erdos cell was then re-scored by **best true value over all
states**, not by re-scoring the top claim -- those are different questions, and
the distinction matters here. g-grpo-n's best-claimed state is contaminated and
re-scores to 0.380999297, but a *different, legitimate* state in the same tree
scores 0.380917550, and that is what the cell actually earned.

Six of seven cells were unaffected: their headline was already the honest best.
Contaminated states existed in all of them (1-7 per tree) but never at the top.
The table ordering does not change -- g-grpo-n still sits between ttd-r and
AlphaEvolve. `verify_erdos.py` now gates on `sum(h)` and reports a signed delta
so this class is caught automatically.

One tree was unusable rather than merely wrong: `tsw-n` (a Stage C tree-swap
arm) had 94 of 596 states overclaiming ~5.7e-5. That cell was killed rather
than corrected.

## Caveats

- Single seed per cell; contrasts inside ~2x are screening results.
- Each model runs at its own validated context length (qwen 18432, gemma
  10240), so this is "each model as canonically configured", not matched-context.
- **Stage A is 11/12.** grpo-k (GRPO x KL x Erdos) never completed a step: HBM
  *fragmentation*, not capacity — OOM summary showed InUse 38.7G of 95.74G with
  `LargestFreeBlock: 0`. It is the only cell combining the base-model KL pass,
  18k-token sequences and 512 rollouts/step; ttd-k halves the volume, grpo-k-j
  has metronomic lengths (17.5k +/- 563 vs Erdos 14.7k +/- 2611), grpo-n has no
  KL pass. Both dataflows were tried: pipelined -> no fragmentation but the
  trainer wedges at the fb drain; non-pipelined -> fragmentation crash loop.

## Files

- `make_bars.py` — the three-panel results figure (`nkr_bars.png`).
- `make_plots3.py` — league ctrl-vs-ours analysis (`league_ctrl_vs_main.png`):
  best-over-time, restarts, per-program improvement +/- 1 std, format rate.
- `pull_wandb.py` — recovers per-life history from wandb (each guardian
  relaunch created a new run, so the union of runs is the full record).
