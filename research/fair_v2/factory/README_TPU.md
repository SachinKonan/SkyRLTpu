# TPU grader for the FAIR feature factory

`factory/tpu_grader/` is a JAX port of `factory/grade.py`. It runs one grade per v5p chip and is used by the MOMO
`fair_features` problem. The package is self-contained: it needs numpy, jax and the standard library, with no
torch, scipy or sklearn and no imports from `research/fair_v2`. Copy the directory verbatim; the MOMO bundle
vendors it at `<bundle>/tpu/fair/tpu_grader/`.

## Requirements

Validated with the env that `venv.tar.gz` ships:

- Python 3.12.15 (uv-managed, at `/dev/shm/fair/py`)
- jax 0.10.1, jaxlib 0.10.1, libtpu 0.0.41
- numpy 2.3.5

The venv also holds flax, optax and orbax, which the grader does not use. The grader uses only `jax.jit`,
`lax.scan`, `jnp.fft`, `jax.random`, `jax.tree.map`, `jnp.searchsorted`, `jax.clear_caches` and the persistent
compilation cache, so it should also run on jax 0.10.2 and numpy 2.4.x (the SkyRL `uv.lock` versions). That
combination has not been tested.

## Running it

MOMO calls it from `tpu/fair/ray_task.py`, one fresh process per grade:

```
PYTHONPATH=<bundle>/tpu/fair FAIR_FACTORY_ROOT=/dev/shm/fair/factory \
TPU_VISIBLE_CHIPS=k TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_PORT=$((8476+k)) ... \
/dev/shm/fair/venv/bin/python -m tpu_grader.grade cand.py --out DIR --mode rl --stack v2
```

This writes `DIR/result.json` (everything, including the `hidden` block) and `DIR/proposer_view.json` (no
hidden-shift numbers, plus a top-level `hidden_ok`).

Note for `ray_task.grade`: it reads `result.json` and drops `hidden`. As a result, `hidden_ok` does not reach
the trainer. If you want the boolean there, read `proposer_view.json` instead.

There is also an in-process API, which the self-test uses:
`from tpu_grader import grade as G; G.init(chip, 'v2'); G.grade_source(src, 'rl')`. It is safe to call
repeatedly:

- Each candidate gets a fresh module namespace.
- Candidate errors, bad outputs and timeouts return `ok=False` with the reason.
- The matmul precision is reset after every candidate.
- Memory stayed flat over 24 grades.

Run `python -m tpu_grader.selftest --chip k` to check it.

## Candidate contract

`features(maps, mask)`:

- `maps` is a `jnp.float32` array of shape (64, 1424, 176) on the chip, zero outside the footprint.
- `mask` is a concrete `jnp.bool_` array of shape (1424, 176).
- It returns (64, d) with d in [1, 4096].
- The grader jits it at one batch shape. Partial batches are zero-padded.

The grader enforces these hard checks:

- Finite output.
- Per-map behaviour: replacing other maps in the batch, zero padding, or shuffling the order must not change a
  map's features (more than 1% of entries off by over 1e-3 relative fails).
- Determinism.
- Runtime budget in TPU seconds per 10k maps: 15 for rl, 60 for full and screen. For reference, dirpower takes
  1.2 and dog about 3.6.

## Modes

| mode | fit rows | shift rows | likelihood | configs | use |
|---|---|---|---|---|---|
| `rl` | all, 2 views | all held-out | full (3000 / 3000 / 150×3) | 11 | MOMO reward |
| `full` | same as rl | same | same | 11 | same reward, looser budget |
| `screen` | 1/5, 1 view | 1/3 | none (ridge vote only) | 1 | not a usable ranker |

`rl` is the full computation with a tighter runtime budget.

- On TPU the likelihood refits are cheap: a cached 3000-step fit takes about 0.13 s per 1000 steps.
- I tested a one-fit-view rl variant. It saved about 3.5 s per grade but ranked dirpower8 (+0.0031) only just
  above random features (+0.0025), against +0.0085 and +0.0013 in full. It was dropped.
- Screen mode ranks dirpower (−0.0003) below random (+0.0023), as on della.

The likelihood member is `jlik.JLikelihood` ported to JAX: an MLP for θ → (μ, log σ²); torch OneCycleLR with
momentum cycling; AdamW; OAS correlation; a Gaussian copula with an eigenvalue floor of 0.02; a regressor
initialiser; and an Adam profile with 3 shared-offset restarts and the Δz and hull priors.

Programs are compiled once per shape, and later grades reuse them:

- Feature dimensions are padded to a bucket (16 … 4096) and masked.
- Rows are padded to a fixed size for each mode.
- The persistent JAX cache in `$FAIR_FACTORY_ROOT/jax_cache` keeps these across processes. It holds only the
  grader's own programs; candidate programs are never written (about 39 MB after 3 buckets).

## Stacks

- **v1** = v3-12 without Oh: `Sd .5, C 1, Dr 2.5, xSL 1, consP8 .5, consRplus 3, N2 .5, PsBh .5, SS 1`, with the
  12 consRplus views. All-19 LCO is 0.4350.
- **v2** = v1 plus dirpower at its best full configuration: misfit member `Xdp` at weight 2, and dirpower's ridge θ
  vote added as a 13th consRplus view. All-19 LCO is **0.4405** (visible 0.4824, hidden 0.3496).

  `factory/build_stack_v2.py` builds it from the della full grade (`runs/gh_dirpower_full`). The default for MOMO
  is v2, so Qwen cannot score by rediscovering dirpower variants: their v2 rewards are about 0.

## Data: bucket and staging

Prefix `gs://sk7524-tinker-tpu-us-east5/fair-factory/v1/`, 20.8 GB, the same region as the east5 v5p hosts:

- `manifest.json`: `{version, gcs, files: [{path, sha256}] × 34, untar: [{archive: venv.tar.gz, into: .}]}`.
  Paths are relative to `/dev/shm/fair`. The current version is `fair-factory-v1-20261010T201241`.
- `factory/noisy/{fit_v0,fit_v1,calib_v0,calib_v1,audit_v0}.npy`: the frozen noisy dumps (float16, valid pixels),
  bit-identical to the della dumps.
- `factory/noisy/shift_<s>_v0.npy`: 19 shift dumps restricted to the 1476 held-out audit rows.
- `factory/meta.npz`: labels, per-fold masks, held-out rows, the footprint mask and the shift order.
- `factory/cache_{v1,v2}/fold{0,1,2}.npz` + `stack.json`: stack members (`comp/<m>/<set>`) and consRplus views
  (`TH/<view>/<set>`) for each fold. These are plain npz, identical to the della pickles.
- `venv.tar.gz` (426 MB): `venv/` plus the `py/` interpreter it links to. It only works when unpacked at
  `/dev/shm/fair`.

How to rebuild it:

1. On della, run `factory/tpu_prep.py --upload` with the fair-env Python. This builds `meta.npz`, the npz caches
   and the shift subsets, hashes everything, writes `manifest.json` and uploads anything missing.
2. Build the venv tarball on a TPU host and pass it to `tpu_prep.py` as `--extra venv.tar.gz=<sha256>`.

`ray_task.stage()` downloads every file, checks its sha256 and untars the venv once per host. `factory/tpu_test/`
holds a Ray-free harness (`drive.py`, `run4.sh`) that runs the real `stage()` and `grade()` from `ray_task.py`
with a stub `ray` module.

## Validation on job 2365 (worker 3314)

Full mode, stack v1, TPU vs della. The della numbers come from `runs/*_full` and the replay loop:

| candidate | della reward (hidden_ok) | TPU reward (hidden_ok) | TPU all-19 |
|---|---|---|---|
| dirpower | +0.0086 (T) | **+0.0083 (T)** | 0.4405 (della 0.4405) |
| dog | −0.0043 A100 / −0.0018 GH200 (F) | −0.0025 (F) | 0.4340 |
| psband | −0.0019 (F) | −0.0017 (F) | 0.4352 |
| random_feats | +0.0023 (F) | +0.0013 (F) | 0.4365 |
| dirpower8 (replay r000c00) | +0.0062 (T) | +0.0085 (T) | 0.4411 |
| dirpower_wide (replay r001c01) | +0.0081 (T) | +0.0084 (T) | 0.4405 |
| batch_violator | fails per-map | fails per-map | – |

- **Feature parity:** on the same 256 audit maps, JAX and torch dirpower features differ by a median of
  5e-7. The worst column is off by 0.009. The cause is 15 frequency pixels that sit exactly on |kx| = |ky|,
  which float32 `atan2` puts in different wedges in torch and numpy. That is a constant column offset, which
  the normal model absorbs. The pipeline reproduces all-19 0.4405 exactly.
- Grades are deterministic on TPU: the same candidate gives the same reward in a warm worker, in a fresh process
  and on any chip.

Rewards on stack v2 (rl, which equals full):

| candidate | reward |
|---|---|
| dirpower | −0.0003 |
| dog | −0.0035 |
| psband | −0.0014 |
| random | +0.0014 (hidden_ok T by chance) |
| dirpower8 | +0.0006 |
| dirpower_wide | +0.0013 |

Accept a candidate only if:
- reward > 0.006,
- every fold is positive,
- `hidden_ok` is true.

`hidden_ok` alone behaves like a coin flip for noise.

## Timings

All on one v5p host, rl mode.

Cold process per grade, run through the real `ray_task.grade()` on 4 chips at the same time, after `stage()`:

| | wall (ray_task) | in-grade | extract | likelihood (3 folds) |
|---|---|---|---|---|
| first grade in a new dimension bucket | 56–62 s | 47–51 s | 10–29 s | 28 s (compiling) |
| dirpower-class, cache warm | **31–37 s** | 24–25 s | 10.4 s | 5.3–6.3 s |
| random_feats, cache warm | 21.5–22 s | 15 s | 7.6 s | 5.0 s |
| dog (heavy extraction), cache warm | 58–62 s | 50–51 s | 28.6 s | 6 s |

- **Per-process overhead:** wall minus in-grade is about 7–11 s. That covers the guard, Python and JAX import,
  TPU init, and loading the memmaps and stack caches; `init()` alone takes 3–5 s.
- **Candidate compile:** about 6 s for dirpower, inside "in-grade".
- **Warm worker** (`selftest`, the in-process API): about 22 s per dirpower-class grade, 47 s for dog.
- **Concurrency:** 4 chips at once ran no slower than one.
- **Staging** (`stage()` on a clean `/dev/shm/fair`): 147 s for 34 files and 20.8 GB, sha256-checked, venv
  unpacked.

## Per-slot resources

| resource | per slot |
|---|---|
| TPU | 1 chip. HBM use is ≤ 0.2 GB, but without `XLA_PYTHON_CLIENT_PREALLOCATE=false` JAX preallocates 75%. |
| host cores | `OMP_NUM_THREADS=8` in ray_task is enough for the numpy part (ridge, consistency, metrics) when d < 500. Give a few more for d near 4096. |
| private RAM | about 6–7 GB while grading (RssAnon) |
| shared RAM | 20.8 GB of `/dev/shm/fair/factory`, shared by all slots on the host; counted as RssFile |

## Caveats

- The timeout uses SIGALRM, and only in the main thread. A candidate that catches BaseException in a Python loop,
  or one stuck inside a compiled XLA call, is stopped only by the caller's process timeout (ray_task:
  `CHILD_TIMEOUT_S` = 900).
- Candidates can read files and import modules. Sandboxing is the caller's static check plus the process guard.
- The stacks' cached members come from the torch pipeline on della. Only the candidate's own family is refit on
  TPU.
