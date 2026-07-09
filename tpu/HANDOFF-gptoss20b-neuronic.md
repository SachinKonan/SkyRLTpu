# Handoff — gpt-oss-20b ttt-discover run (→ neuronic cluster)

Written for a **fresh Claude session on the neuronic cluster**, which has better CPU
availability than della. Read this top-to-bottom before doing anything.

## 1. What this is / the goal
Recreate the ttt-discover **"Erdos min-overlap"** RL method with **`openai/gpt-oss-20b`**
against the **real Thinking Machines Tinker API** (prod), on branch `agent/ttd-discover-erdos`.

- The model runs on **hosted Tinker** — we do NOT use the SkyRL TPU backend here (that is a
  separate Qwen-on-TPU effort). The only local compute we need is **CPU, for grading** the
  generated Erdos programs.
- Everything runs from the `discover` client (`third_party/discover`), driven by the wrappers
  in `tpu/`.

## 2. Why we moved to neuronic
**Program grading is the entire bottleneck.** On della the `cpu` partition forces `QOS=test`
(2 concurrent jobs) for small jobs, big jobs wait ~5.5h, and the GPU partitions require holding
a GPU. So grading ran only 2-wide (~30h per step). **Sampling and training themselves are
seconds-fast** (Tinker handles them) — see timing below. neuronic should let grading run wide.

## 3. Status — validated & committed
Pipeline validated end-to-end via a smoke test (`group_size=1`): sample from Tinker → grade →
one train step, completed cleanly. Committed work:
- **Runner**: `tpu/run_ttd_gptoss20b.sh` → calls the fully env-parameterized entrypoint
  `tpu/run_ttd_smoke_gptoss20b.py` (despite the name, it is the general runner; `[smoke]` log
  prefix is cosmetic).
- **discover-repo fixes** (committed in `third_party/discover`, see its `git log`):
  - OpenSSL first-init race — pre-init `ssl.create_default_context()` on the main thread.
  - KL-penalty logprob alignment — real Tinker returns `len(prompt)` logprobs w/ leading None;
    slice `[1:]` only when `len(base)==len(sampled)+1` (works for both real Tinker and SkyRL).
  - wandb-logger `IndexError` guard (`loggers[2]` needs `len>=3`, not `>=2`).
  - Final checkpoint gated on `save_every>0` → `SAVE_EVERY=0` persists **nothing**.
- **Config** = authors' canonical gpt-oss defaults (`third_party/discover/docs/api.md`):
  renderer `gpt_oss_high_reasoning`, `group_size=8`, `groups_per_batch=64` (=512 rollouts/step),
  `lr=4e-5`, `kl_penalty_coef=0.1`, `phase1_max_tokens=26000`, `context_window=32768`,
  `lora_rank=32`. **`SAVE_EVERY=0`** (no LoRA adapters saved — user requirement). wandb online.

## 4. Timing reality (from the completed smoke, group_size=1)
| phase | time |
|---|---|
| sampling (Tinker generation + 1 grade) | ~45 s |
| **train** (forward_backward + optim) | **~5 s** |
So at full scale a step is basically `seconds (gen) + seconds (train) + grade(512 programs)`.
**Grading concurrency is the only lever that matters.**

## 5. What to do on neuronic (step by step)
1. **Secrets** — create `third_party/discover/.env` (gitignored) with:
   ```
   TINKER_API_KEY=<real Tinker prod key>
   WANDB_API_KEY=<wandb key>
   ```
   (The della `.env` is NOT in git and won't come with the clone.)
2. **Get the discover code WITH our fixes** — ⚠️ the submodule points at upstream
   `github.com/test-time-training/discover` and our 5 fix commits are NOT pushed there, so a
   plain `git submodule update` will fail to check out the pinned commit. Use the bundled patch
   instead:
   ```bash
   cd third_party
   rm -rf discover && git clone https://github.com/test-time-training/discover.git
   cd discover
   git checkout 6c40e82dab9d5de7416ac873ad5cd3106084aaed   # base our fixes are on top of
   git apply --3way ../../tpu/discover-fixes.patch          # our fixes (SSL/KL/wandb/save-gate/eval-backends/qwen-completer)
   ```
   (`tpu/discover-fixes.patch` is committed in this repo; it is `diff origin/main..our-HEAD`.)
   Then the venv: `uv sync --extra math --python 3.11` → creates `.venv-ttd-discover`.
3. **Pick the grading backend** (the whole point of neuronic — figure out what runs widest):
   - **Preferred if neuronic *compute nodes reach the internet*** (test: from a compute node,
     `curl -sS -m8 https://<any external host>`): run the WHOLE client on one big compute node
     (`sbatch`, e.g. 60–96 cores) with **`TTD_EVAL_BACKEND=local`** → grading runs in-process
     across all the node's cores (~N-wide), and sampling hits Tinker directly. Simplest — no Ray,
     no dispatch.
   - **If compute nodes have no internet** (client must stay on a login node for Tinker):
     use **`TTD_EVAL_BACKEND=submitit`** if neuronic's cpu QOS allows many concurrent jobs
     (verify with a burst test — submit ~50 quick jobs and count how many RUN at once), OR use
     the `tpu/ray_graders_slurm.sh` pattern (grab a node, start a Ray head, client connects via
     `RAY_ADDRESS`, `TTD_EVAL_BACKEND=ray`).
   - Set `NUM_CPUS_PER_TASK=1` (or 2) and cap `EVAL_TIMEOUT` (e.g. 300s) so a hung program frees
     its worker quickly.
4. **Pre-create the wandb project** (the resume logic queries the project BEFORE it exists, which
   errors with `Could not find project`). Either `wandb.Api().create_project(name, entity)` or a
   throwaway `wandb.init(project=..., name="_bootstrap"); wandb.finish()`. Existing project used
   on della: `ttt-discover-gptoss20b` under entity `sk7524-princeton-university` (use your own
   entity on neuronic).
5. **Fix the SLURM env in the runner** — `tpu/run_ttd_gptoss20b.sh` hardcodes della defaults
   (`TTD_SLURM_PARTITION=cpu`, `TTD_SLURM_ACCOUNT=zhuangl`). Change these for neuronic (or set
   `TTD_EVAL_BACKEND=local` and they're irrelevant).
6. **Run** (validation first, then scale):
   ```bash
   NUM_EPOCHS=3  TTD_EVAL_BACKEND=local NUM_CPUS_PER_TASK=1 EVAL_TIMEOUT=300 \
     bash tpu/run_ttd_gptoss20b.sh          # 3-step validation
   # then, once grading is fast and a step completes cleanly:
   NUM_EPOCHS=50 bash tpu/run_ttd_gptoss20b.sh
   ```
   Watch wandb for reward + the discovered-programs table (that is the actual output — we do NOT
   keep adapters).

## 6. Gotchas (do not re-learn these)
- **Stale `TINKER_API_KEY`** in the shell/tmux env shadows `.env` (`load_dotenv` does not
  override an already-set var) → the runner does `unset TINKER_API_KEY`. Keep it.
- **SSL race** — the entrypoint pre-inits OpenSSL on the main thread; don't remove it.
- **`SAVE_EVERY=0`** = no periodic AND no final checkpoint. User wants zero adapters persisted.
  The `save_checkpoint took Ns` log line at step start is just the mandatory sampler-weights
  export (`save_weights_and_get_sampling_client`), NOT a persisted checkpoint — verify by
  checking `.../tinker_log/<exp>/checkpoints.jsonl` is empty/absent.
- **Grading is the only slow part.** Don't chase sampling/training perf — get grading wide.

## 7. Key paths / names (della values — adjust for neuronic)
- Repo: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ttd-discover`  (branch `agent/ttd-discover-erdos`)
- Runner: `tpu/run_ttd_gptoss20b.sh`; entrypoint `tpu/run_ttd_smoke_gptoss20b.py`
- discover client: `third_party/discover` (venv `.venv-ttd-discover`, secrets `.env`)
- Run dir: `runs/ttd_gptoss20b_full`
- wandb: project `ttt-discover-gptoss20b`, entity `sk7524-princeton-university`
- Grader-node sbatch template: `tpu/ray_graders_slurm.sh` (partition/account/QOS at the top)
