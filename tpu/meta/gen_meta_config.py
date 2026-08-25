#!/usr/bin/env python3
"""Generate a jobman config for one meta-tree generation (one arm).

Clones the proven stageC cell config (template) and swaps: v5p-128 (16 hosts),
meta_* hook scripts, and the generation env (ARM/GEN/SPARE_FOR/NUM_EPOCHS +
optional per-member weights-carry INIT_SP_*/INIT_JSONL_GCS_*).

Usage:
  gen_meta_config.py --template <stageC yaml> --arm meta-wt16-fresh --gen 0 \
      [--spare-for qwen] [--steps 15] [--init qwen=STATE_PATH,JSONL_GCS ...] \
      --out cfg.yaml
"""
import argparse, copy, sys
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--gen", type=int, required=True)
ap.add_argument("--spare-for", default="qwen", choices=["qwen", "gemma", "muse"])
ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--init", action="append", default=[],
                help="tag=state_path,jsonl_gcs  (carry arms only)")
ap.add_argument("--accel", default="v5p-128")
ap.add_argument("--extra-env", action="append", default=[], help="K=V passed into resumable env")
ap.add_argument("--out", required=True)
args = ap.parse_args()

c = yaml.safe_load(open(args.template))
name = f"sk7524-{args.arm}-g{args.gen}"

# strip instance-specific fields the create flow re-derives
for k in ("id", "user", "dir"):
    c["job"].pop(k, None)
c["tpu"].pop("ips", None)
c["tpu"].pop("num_workers", None)

c["job"]["name"] = f"{name}_1"
c["job"]["worker_num"] = 1
c["tpu"]["accelerator"] = args.accel
c["tpu"]["name"] = f"{name}_1"

c["command"]["cmd"] = ('export PATH="$HOME/.local/bin:$PATH"\n'
                       'bash "$HOME/SkyRLTpu-league/tpu/jobman/meta_worker.sh"\n')
c["command"]["workers"] = "all"

r = c["resumable"]
r["run_id_prefix"] = name
r["completion_probe"]["command"] = 'bash "$HOME/SkyRLTpu-league/tpu/jobman/meta_probe.sh"\n'
r["monitor"]["command"] = ('export PATH="$HOME/.local/bin:$PATH"\n'
                           'bash "$HOME/SkyRLTpu-league/tpu/jobman/meta_monitor.sh"\n')
r["sync"]["command"] = 'bash "$HOME/SkyRLTpu-league/tpu/jobman/meta_sync.sh"\n'
# prepare (bundle download/extract) must land on EVERY host: trainer hosts
# w4/w8 run cell_worker there, vLLM hosts get code via SYNC_SKYRL anyway.
r["prepare"]["workers"] = "all"

env = r["env"]
for k in list(env):
    if k.startswith(("TTD_", "GROUPS", "GROUP_", "KL_", "LEARNING", "EXPERIMENT",
                     "RUN_DIR", "CELL")):
        del env[k]
env.update({
    "CELL": f"{args.arm}-g{args.gen}",
    "ARM": args.arm,
    "GEN": str(args.gen),
    "SPARE_FOR": args.spare_for,
    "NUM_EPOCHS": str(args.steps),
})
for spec in args.extra_env:
    k, v = spec.split("=", 1)
    env[k] = v
for spec in args.init:
    tag, rest = spec.split("=", 1)
    sp, jsonl = (rest.split(",", 1) + [""])[:2]
    env[f"INIT_SP_{tag.upper()}"] = sp
    if jsonl:
        env[f"INIT_JSONL_GCS_{tag.upper()}"] = jsonl

r["run_spec"] = {
    "experiment": "meta-tree",
    "arm": args.arm,
    "generation": args.gen,
    "members": "qwen3.5-27b,gemma4-31b,muse-glimmer-30b",
    "batch_shape": "16x32",
    "lr": "1.5e-4",
    "steps": args.steps,
    "stop": "fixed-15 + flatline(3x<1e-9, min4)",
    "spare_for": args.spare_for,
    "accelerator": args.accel,
}
yaml.safe_dump(c, open(args.out, "w"), sort_keys=False)
print(args.out)
