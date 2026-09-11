#!/usr/bin/env python3
"""Generate the v6e-32 (2 trainer VMs + 6 engine VMs) task yamls for the
matched-validity cells by porting the v5p-32 yamls in ../v5p32-cells.

The v5p bundles (v20 qwen, v21 gemma) predate tpu/swarm/run_v6e32_cell.sh, so
each yaml's `run` writes that wrapper from this repo into $HOME on the worker
and executes it against the installed bundle; everything the wrapper calls
(cell_worker.sh, ensure_orbax_ckpt.sh, reconcile/cleanup) is in the bundle.
GCS_RUN is unchanged on purpose: the v6e job RESUMES the v5p run.

    python3 tpu/swarm/examples/v6e32-cells/gen_matched_yamls.py
"""
from __future__ import annotations

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V5P = HERE.parent / "v5p32-cells"
WRAPPER = (ROOT / "tpu/swarm/run_v6e32_cell.sh").read_text()
BUCKET = "gs://sk7524-tinker-tpu-us-east5"

# Per-model v6e overrides. cell_worker.sh keeps the proven v5p per-model
# defaults; these are the "deployment overrides" it documents.
MODEL_ENV = {
    "qwen": {
        # v5p: TP4 over 4 x 95 GB chips, 4 rows of 18432 per fb call (BUDGET 73728).
        # v6e: TP8 over 8 x 32 GB chips, ONE row per fb call. Two rows still
        # overflowed: job 650 built its LoRA template and died loading jit_scan
        # ("reserve 6.87G, 2.68G free"). One row halves the activation and
        # scratch memory of that program; the step is unchanged (512 sequences
        # accumulated before one update, in twice as many calls).
        "TUNIX_TRAIN_TOKEN_BUDGET": "18432",
        # Engines: the v5p shape (128 seqs, 8192 batched tokens, 0.90 util) hit
        # CompileTimeHbmOom on the 32 GB v6e chip (31.37 of 31.25 GB, jobs
        # 631/632 at the 4096-token x 128-req prefill graph). Halve the
        # compiled batch shapes; KV pool stays at 0.90 (it, not max-num-seqs,
        # bounds concurrency: ~15 full 22k sequences per TP4 engine).
        # Second OOM (job 638, 15:55Z) at the 4096-token x 64-req graph with the
        # same 31.38 of 31.25 GB: the KV pool sized at 0.90 leaves only ~3 GB
        # of compile headroom on a 32 GB chip (v5p at 0.90 leaves ~9.5 GB).
        # Reserve 20% and halve the prefill chunk again; KV drops to ~8 GB per
        # chip (~11 full 22k sequences per engine, six engines).
        "VLLM_MAX_NUM_SEQS": '"64"',
        "VLLM_EXTRA_ARGS": '"--max-num-batched-tokens 2048 --gpu-memory-utilization 0.80"',
        "TUNIX_JAX_CACHE_GCS": f"{BUCKET}/jax-compile-cache-v6e-qwen35-tp8-fsdp1-r32-s18432-b18432-cells-v1",
        # The east5b pool's own qwen TP4 22k engine cache; a version miss just recompiles.
        "VLLM_XLA_CACHE_GCS": f"{BUCKET}/vllm-xla-cache-v6e-qwen35-tp4-s22528-v1",
    },
    "gemma": {
        # Gemma needs FOUR trainer VMs. Its create_model jit_scan asks for
        # 6.87 GB and finds 2.68 GB free on 8 chips, the same numbers at two
        # rows (job 650) and at one (job 668), so the overflow is the 7.8 GB of
        # bf16 weights per chip plus gemma's vocab workspace, not the batch.
        # 16 chips halve the weights; the cost is two of the six engines.
        # Qwen (27B, vocab tiling 64) fits on two VMs and keeps all six.
        "V6E_TRAINER_VMS": '"4"',
        "TRAIN_WORKERS": "0,1,2,3",
        "VLLM_WORKERS": "4,5,6,7",
        "TRAIN_FSDP_SIZE": '"2"',
        "TUNIX_ROW_SHARD": '"2"',
        "TRAIN_TPU_PROCESS_BOUNDS": "2,2,1",
        # FSDP 2 shards the rows two ways, so the call carries two of them.
        "TUNIX_TRAIN_TOKEN_BUDGET": "20480",
        # Gemma already serves 32 seqs at 16k; only the prefill chunk shrinks.
        "VLLM_MAX_NUM_SEQS": '"32"',
        "VLLM_EXTRA_ARGS": '"--max-num-batched-tokens 2048 --disable-chunked-mm-input --gpu-memory-utilization 0.80"',
        "TUNIX_JAX_CACHE_GCS": f"{BUCKET}/jax-compile-cache-v6e-gemma4-tp8-fsdp2-r32-s10240-b20480-cells-v1",
        "VLLM_XLA_CACHE_GCS": f"{BUCKET}/vllm-xla-cache-v6e-gemma4-31b-tp4-16k-v1",
    },
}

LAYOUT_ENV = {
    "ZONE": "us-east5-b",
    "TRAIN_WORKERS": "0,1",
    "VLLM_WORKERS": "2,3,4,5,6,7",
    "TRAIN_TP_SIZE": '"8"',
    "TRAIN_FSDP_SIZE": '"1"',
    "TUNIX_ROW_SHARD": '"1"',
    "TRAIN_TPU_PROCESS_BOUNDS": "2,1,1",
    "TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
    "VLLM_TP_SIZE": '"4"',
    "VLLM_ENGINES_PER_HOST": '"1"',
}

CELLS = [
    ("stageC-v32-ttd-n", "qwen"),
    ("stageC-v32-grpo-n", "qwen"),
    ("stageB2-g-v32-ttd-n", "gemma"),
    ("stageB2-g-v32-grpo-n", "gemma"),
]

RESOURCES = """resources:
  cloud: gcp
  accelerators: tpu-v6e-32
  accelerator_args:
    gcp_queued_resource: true
    runtime_version: v2-alpha-tpuv6e
  use_spot: true
  disk_size: 150
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 3
    recover_on_exit_codes: [33, 34]
"""


def port(name: str, model: str) -> str:
    src = (V5P / f"{name}.yaml").read_text()
    head, rest = src.split("\nresources:\n", 1)
    _, envs_and_run = rest.split("\nenvs:\n", 1)
    envs, run = envs_and_run.split("\nrun: |\n", 1)

    # Header: keep the experiment comment block, note the v6e layout.
    head = head.replace(f"name: {name}", f"name: {name}-v6e", 1)
    head = re.sub(
        r"# One complete four-host tpu-v5p-32 slice.*?\n(# .*\n)*?# Reusing GCS_RUN",
        "# One complete eight-VM tpu-v6e-32 slice from pool tpuswarm-v6e32-east5b-qwen35,\n"
        "# 2+6 layout: VMs 0-1 = TP8/FSDP1 trainer (VM 0 also client + grader Ray head),\n"
        "# VMs 2-7 = six TP4 vLLM engines (tpu/swarm/run_v6e32_cell.sh).\n"
        "# Reusing GCS_RUN",
        head,
        flags=re.S,
    )

    env_lines = envs.rstrip("\n").split("\n")
    out_env = []
    for line in env_lines:
        m = re.match(r"^  ([A-Z_]+): (.*)$", line)
        if m and m.group(1) == "TPUSWARM_BUNDLE_ID":
            line = f"  TPUSWARM_BUNDLE_ID: {m.group(2)}-v6e"
        if m and m.group(1) == "ZONE":
            line = "  ZONE: us-east5-b"
        out_env.append(line)
    out_env.append("")
    out_env.append("  # v6e-32 layout: two trainer VMs (8 chips, TP8, no FSDP), six engine VMs.")
    # A per-model override replaces the layout default in place; emitting both
    # would put duplicate keys in the yaml (gemma runs four trainer VMs).
    layout = {k: v for k, v in LAYOUT_ENV.items() if k != "ZONE"}
    model_env = dict(MODEL_ENV[model])
    for k in list(model_env):
        if k in layout:
            layout[k] = model_env.pop(k)
    for k, v in layout.items():
        out_env.append(f"  {k}: {v}")
    out_env.append("  # 32 GB chips: one sequence per forward/backward call per FSDP group (v5p packed four); v6e-specific compile caches.")
    for k, v in model_env.items():
        out_env.append(f"  {k}: {v}")
    envs_out = "\n".join(out_env) + "\n"

    # run: same bundle install, then write and exec the v6e wrapper.
    run = run.replace(
        '    test -r "$staging/tpu/swarm/run_v5p32_cell.sh"\n', ""
    )
    # One VM's transient GCS error during the bundle download fails the whole
    # eight-VM attempt with exit 1 and burns one of the three application
    # restarts (job 638 first attempt: rank 7 exit 1, ranks 0-6 fine). Retry
    # the copy a few times before giving up.
    run = run.replace(
        '    gcloud storage cp "$TPUSWARM_SKYRL_BUNDLE_URL" "$archive"\n',
        '    for _try in 1 2 3 4 5; do\n'
        '      gcloud storage cp "$TPUSWARM_SKYRL_BUNDLE_URL" "$archive" && break\n'
        '      echo "bundle download failed (attempt $_try); retrying in 30s" >&2; sleep 30\n'
        '    done\n',
    )
    assert "bundle download failed" in run, name
    run = run.replace(
        '  exec bash "$SKYRL_REPO_DIR/tpu/swarm/run_v5p32_cell.sh"\n',
        "  # The v5p bundles predate the v6e wrapper; install it from this task.\n"
        "  cat > \"$HOME/run_v6e32_cell.sh\" <<'V6E_WRAPPER'\n"
        # Every line of the `run: |` block carries the two-space YAML indent;
        # SkyPilot strips it, so the heredoc body and terminator come out flush.
        + "".join(("  " + l if l else "") + "\n" for l in WRAPPER.rstrip("\n").split("\n"))
        + "  V6E_WRAPPER\n"
        "  exec bash \"$HOME/run_v6e32_cell.sh\"\n",
    )
    assert "run_v6e32_cell.sh" in run, name
    return head + "\n" + RESOURCES + "\nenvs:\n" + envs_out + "\nrun: |\n" + run


def main() -> None:
    for name, model in CELLS:
        text = port(name, model)
        dst = HERE / f"{name}-v6e.yaml"
        dst.write_text(text)
        print("wrote", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
