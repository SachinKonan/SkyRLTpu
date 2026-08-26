# Tunix Multi-Host Trainer — Overall Plan

Worktree: `/n/fs/vision-mix/sk7524/SkyRLTpu-multihost`, branch `agent/tunix-multihost`.
Splays off `agent/ttd-league-qwen-gemma` @ `29601970` (the SkyRLTpu-league worktree —
the live Erdős RL fleet). Submodule `third_party/discover` pinned at `c95227a`
(local-only commits; fetch it FROM the league checkout, not origin:
`git -C third_party/discover fetch /n/fs/vision-mix/sk7524/SkyRLTpu-league/third_party/discover`).

## Goal

Make the **tunix (MaxText) backend** of the tinker server train across multiple TPU
hosts, so 27–31B RL cells can run on v6e (Trillium: 32 GB HBM/chip — our v5p
per-chip fb arenas do not fit; multi-host sharding is the entry ticket, not an
optimization). The `jax` (tx) backend already has full multi-host machinery; the
user explicitly rejects it (efficiency) — tunix is the production backend
(`TINKER_BACKEND=tunix` in every RL cell).

## How a v5p cell runs TODAY (the system being extended)

One v5p-32 = 4 hosts: worker 0 = trainer + tinker API, workers 1–3 = vLLM serving.

1. `tpu/jobman/configs/stagea_<cell>.yaml` — jobman job (QR create, hooks, env).
   `jobman create <yaml>` starts a controller (`loop: true`, spot-preemption proof).
2. prepare hook → unpacks the code bundle from
   `gs://sk7524-tinker-tpu-us-east5/code-bundles/stagea-league.tar.gz`
   (BUILD: tar of this tree, see git log for the exclude list) and runs
   `tpu/jobman/ensure_orbax_ckpt.sh` (worker-0-gated orbax cache restore).
3. monitor hook → `tpu/jobman/cell_worker.sh` (w0 only; drives everything):
   per-model dispatch by cell prefix (`g-*`=gemma, `m-*`=muse, else qwen),
   sets `TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2,3`, tiles/nvt,
   then calls `tpu/start_colocated_vllm_tinker.sh`.
4. `start_colocated_vllm_tinker.sh` — brings up vLLM per serving host
   (`tpu/start_vllm_tpu.sh`; independent no-Ray servers, TP=4/host qwen+gemma,
   2×TP=2 muse), then the tinker server on w0 (tmux `skyrl-tinker`,
   `-m skyrl.tinker.api ... --backend tunix --backend-config <json>`).
   MULTI-HOST WIRING ALREADY EXISTS here for the jax backend only:
   `TRAIN_WORKERS` list → process bounds → per-host worker tmux sessions
   running `-m skyrl.backends.jax --coordinator-address ...` (lines ~780–822);
   the tunix branch is guarded single-host at line ~558.
5. The RL client (`third_party/discover` = ttt-discover fork) runs under jobman
   (`league_monitor.sh`/`league_launch.sh` family) against the tinker API;
   PUCT tree + GRPO/TTD advantage estimators live in
   `third_party/discover/ttt_discover/rl/train.py` (`compute_advantages`).
6. Training data path inside the server: engine (separate PROCESS, ONE serialized
   dispatch loop — `skyrl/tinker/engine.py:780`) → `AbstractBackend` methods →
   `skyrl/backends/tunix_backend.py` (2553 lines; MaxText model via
   `from_pretrained(mesh=None)`, LoRA via qwix, orbax base checkpoint).

Key env vars: `TUNIX_MAXTEXT_MODEL_NAME`, `TUNIX_MAXTEXT_KWARGS` (JSON pyconfig
overrides, e.g. `ici_fsdp_parallelism`), `TUNIX_MAXTEXT_CKPT_CACHE[_GCS]`,
`TRAIN_WORKERS`, `VLLM_WORKERS`, `JAX_COORD_PORT`, `TRAIN_TPU_PROCESS_BOUNDS`.

## Design (verified against code before writing)

The jax backend's multi-host layer is a generic RPC-over-broadcast, reusable as-is:
coordinator facade pickles `(method, kwargs)` and ships via
`multihost_utils.broadcast_one_to_all` (`skyrl/backends/jax.py:1343
_broadcast_command`, `RpcPayload:1330`); workers loop in `run_worker():~1467`.
The engine's single-threaded dispatch makes collective ORDER safe by construction.

### Work items

1. **Extract RPC layer** → `skyrl/backends/rpc.py`: `RpcPayload`,
   `_broadcast_command`, generic `run_worker(backend_registry)`. INIT payload
   gains `backend_name`; worker CLI stays `-m skyrl.backends.jax` (it learns the
   backend from INIT — zero launcher CLI change) — or a thin
   `-m skyrl.backends.worker`. jax.py re-imports for back-compat.
2. **`DistributedTunixBackend(TunixBackend)`** facade in `tunix_backend.py`:
   broadcast-wrap create_model / forward_backward / forward / optim_step /
   save_checkpoint / load_checkpoint / save_sampler_checkpoint / delete_model.
   `sample` local (external vLLM). `has_model` local (metadata). Config gains
   `coordinator_address/num_processes/active_worker_ids`. `engine.py:196`
   returns the facade (it no-ops distributed when coordinator_address is None,
   mirroring `JaxBackend.__init__:1372`).
3. **Tunix internal edits** (all in `skyrl/backends/tunix_backend.py`):
   - `__init__`: `jax.distributed.initialize(process_id=0,...)` before any JAX op
     when coordinator_address set. KEEP `skip_jax_distributed_system=True` in
     MaxText overrides (:600) — we own init.
   - ckpt-conversion subprocess (:540): `process_index()==0` only +
     `sync_global_devices` barrier.
   - `_place_microbatch` (:1095–1121): `jax.device_put(a, row_sharding)` →
     `jax.make_array_from_process_local_data` when `process_count()>1`
     (RPC broadcast guarantees identical batch on every process).
   - fb outputs (:1406): `jax.device_get(per_token...)` fails on non-addressable
     shards → `multihost_utils.process_allgather(..., tiled=True)`
     (in-tree precedent: `skyrl/tx/utils/generator.py:391`).
   - LoRA I/O (:2137 save, :2166 load, :2186 sampler ckpt, :2224 tar, :2421
     peft tensors): allgather LoRA state (collective — ALL processes call),
     then p0-only writes/uploads/vLLM notify, then barrier. load: p0 reads
     bytes → broadcast bytes → each process device_puts with its sharding
     (hosts do not share disks).
   - batch divisibility assert (:249) against global device_count.
   **RULE: no collective inside a process-gated branch** (deadlock).
4. **Launcher** `tpu/start_colocated_vllm_tinker.sh`: drop the :558 tunix
   single-host guard; add coordinator keys to the tunix cfg branch (mirror
   :590–598); worker scripts reuse the :780–822 generation unchanged.
5. **Hooks**: `tpu/jobman/ensure_orbax_ckpt.sh` gate `worker 0` →
   `worker ∈ TRAIN_WORKER_IDS` (sharded orbax restore reads on every trainer
   host). `cell_worker.sh`: allow per-cell `TRAIN_WORKERS`/`VLLM_WORKERS`
   override (default unchanged = single-host).
6. **Tests**: CPU 2-process fake-mesh
   (`XLA_FLAGS=--xla_force_host_platform_device_count=4`, localhost
   coordinator) driving the RPC layer with a STUB backend: ordering, kwargs
   round-trip, INIT dispatch, the no-gated-collective rule. Single-process
   facade == plain backend equivalence.

### Test gates on real hardware (after this lands)

- T0 v6e-8 single-host tunix smoke (config-only; validates Trillium wheels +
  fb footprint; OOM here = the multi-host business case, not failure).
- T1 v6e-16 two-host (`TRAIN_WORKERS="0,1"`, `ici_fsdp_parallelism=16`):
  identical LoRA-init hash across processes, 1 banked gradient, adapter
  published, sampler registered, loss parity vs v5p single-host @ same
  seed/batch to ~1e-3.
- T2 v6e-32 full cell: w0–1 trainer, w2–3 serving (per-host engines, TP=8 or
  2×TP=4 by the KV-head zero-padding rule; re-derive MAX_NUM_SEQS; fresh
  JAX_COMPILATION_CACHE_DIR prefix — topology-keyed, everything recompiles).
  v6e QR strings proven in `SkyRLTpu/tpu/rq2/fleet/yamls/…v6e8…east5b.yaml`:
  `accelerator: v6e-8, zone: us-east5-b, version: v2-alpha-tpuv6e`.

### Deliberately out of scope

Multi-host vLLM (exists behind `VLLM_RAY_EXECUTOR=1` in `start_vllm_tpu.sh:158`
for >single-host models; a KV-padding loss for our 27–31B fleet), gpt-oss/muse
specifics, meta-driver integration, any change to running v5p cells.

## Status log (append entries here)

- 2026-08-26: worktree created, plan written. Implementation not started.
