# Phase 0 report — baseline importability + zone/quota (2026-08-06)

Executed via sbatch on a neuronic compute node (job **3645849**, log
`runs/pallas_arena/phase0-3645849.log`, raw results
`runs/pallas_arena/phase0-results.jsonl`). Runner:
`tpu/pallas_arena/phase0_import_gate.py` via
`tpu/pallas_arena/run_phase0.sbatch`. **GATE: GREEN — all six baselines
import at our pins.** Env: python 3.12.11, **jax 0.10.2 / jaxlib 0.10.2**
(uv.lock pin, CPU backend), `uv run --isolated`.

## Per-baseline results

| # | Baseline | Exact import path | Pinned version | Where it imports |
|---|----------|-------------------|----------------|------------------|
| 1 | Splash attention | `jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel` (`make_splash_mha`, `SplashAttentionKernel`, `BlockSizes`) + `...splash_attention_mask` (`CausalMask`) | jax 0.10.2 (pip, uv.lock) | login-CPU jax env (`--extra jax`) |
| 2 | vLLM-TPU ragged paged attention | `tpu_inference.kernels.ragged_paged_attention.v3.kernel.ragged_paged_attention`, path-imported from `third_party/tpu-inference` | checkout @ `0ae6bf7aa` (v0.23.0-7, matches our vllm==0.23.0 pin) | login-CPU jax env — but ONLY via parent-package stub import: the plain `import tpu_inference` fails on CPU (`ModuleNotFoundError: vllm`). The kernel module itself is pure jax. TPU-host venv (with vllm-tpu installed) imports it plainly. |
| 3 | Megablox grouped matmul | `jax.experimental.pallas.ops.tpu.megablox.gmm` | jax 0.10.2 (pip, uv.lock) | login-CPU jax env. MaxText's copy (`MaxText.kernels.megablox`) is a vendored variant of this same kernel; MaxText itself is a manual TPU-host install in this project (pip `maxtext` exists but does not expose `MaxText.*` on CPU — informational check, recorded FAIL, not gating). Arena baseline = the jax-bundled gmm. |
| 4 | RG-LRU Pallas scan | `recurrentgemma.jax.pallas` (`lru_pallas_scan`, `pallas_lru`, `linear_rnn_pallas_call`, `compute_pallas_kernel_spec`) and `recurrentgemma.jax.scan` (`lru_pallas_scan`, `lru_associative_scan`, `resolve_scan_type`) | recurrentgemma **1.0.1** (pip; resolves cleanly against jax 0.10.2) | CPU jax env **with `--with recurrentgemma` added** — NOT in uv.lock; the phase-2 judge venv must add it. |
| 5 | FLCE custom_vjp (ours) | `skyrl.backends.tunix_backend.TunixBackend._flce_target_logprobs` (commits 198f41fa / 2e85086f; working tree @ f2b522d1) | in-tree | login-CPU jax env with `--extra tunix` (transformers/cloudpathlib/optax); numeric check PASSED (fwd == log_softmax closed form, 13 tok × V=97). The judge uses a verbatim standalone copy in `judge/problems/flce.py` so graders never import skyrl (banned prefix for candidates). |
| 6 | XLA RMSNorm | pure `jnp` closed form under `jax.jit` | jax 0.10.2 | login-CPU jax env; tiny numeric check PASSED |

## Zone / quota check (read-only, gcloud, project `vision-mix`)

- **Quota**: `tpu.googleapis.com/queuedResources` effective limit for
  us-east5-b = **20** (project default; no zone override). Currently **13 QRs
  in the zone** (all `taiming-strategist-*`, another user's v6e-8 fleet: 3
  ACTIVE, 8 PROVISIONING, 1 SUSPENDING, 1 FAILED) → **7 free slots ≥ 5
  required.** QUOTA OK.
- **Accelerator types**: `v6e-1` … `v6e-8` (and up) listed in us-east5-b. OK.
- **CAPACITY WARNING**: one of the existing v6e-8 QRs FAILED with
  `"There is no more capacity in the zone us-east5-b"` and 8 more v6e-8 spot
  QRs are sitting in PROVISIONING. The zone is capacity-crunched for v6e-8
  **right now**; v6e-1 (our phase-2 shape) is much easier to land but not
  guaranteed. Phase-2 policy: attempt AT MOST ONE spot v6e-1 QR with a hard
  2h floundering timeout, delete on expiry.
- **Bucket**: `gs://sk7524-pallas-arena-us-east5` exists, regional US-EAST5,
  currently empty. OK.
- **Auth**: user creds (`sk7524@princeton.edu`) alive; the service-account
  IAM grant for the 24/7 keeper (PLAN.md phase-0 item) is still pending on an
  admin — unchanged, remains the availability blocker for phase 3, not for
  phases 1–2.

## Pin decision

Arena runs at **jax/jaxlib 0.10.2** (the uv.lock pin) for Mosaic-compile
determinism (PLAN.md risk item). tpu-inference stays pinned to the
`third_party/tpu-inference` checkout @ `0ae6bf7aa`; recurrentgemma pinned
**1.0.1** when the judge venv is provisioned.
