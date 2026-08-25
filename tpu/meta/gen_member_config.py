#!/usr/bin/env python3
"""Generate a jobman config for ONE meta member as a standalone v5p-64 cell.

The member IS the proven stage-cell unit -- cell_worker/launch_cell/cell_monitor
/cell_probe/cell_sync untouched -- on a 1-trainer + 7-inference v5p-64 (the
N-host generalization turns the extra hosts into vLLM engines automatically:
qwen/gemma 7xTP=4, muse 14x(2xTP=2)). The meta driver advances generations by
editing RUN_DIR_NAME/EXPERIMENT_NAME in the snapshot and restarting the
controller (runbook layer-2), so ONE job serves an arm's whole lifetime.

Usage: gen_member_config.py --template <stageC yaml> --arm A --gen 0 \
         --member qwen|gemma|muse [--steps 15] [--accel v5p-64] \
         [--init-sp PATH --init-jsonl GCS] --out cfg.yaml
"""
import argparse
import yaml

CELL_ALIAS = {"qwen": "meta-qwen", "gemma": "g-meta", "muse": "m-meta"}
ENV_SUFFIX = {"qwen": "QWEN", "gemma": "GEMMA", "muse": "MUSE"}

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--gen", type=int, required=True)
ap.add_argument("--member", required=True, choices=list(CELL_ALIAS))
ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--accel", default="v5p-64")
ap.add_argument("--init-sp", default="")
ap.add_argument("--init-jsonl", default="")
ap.add_argument("--out", required=True)
args = ap.parse_args()

c = yaml.safe_load(open(args.template))
run = f"{args.arm}-g{args.gen}-{args.member}"
name = f"sk7524-{args.arm}-{args.member}"   # stable across generations

for k in ("id", "user", "dir"):
    c["job"].pop(k, None)
c["tpu"].pop("ips", None)
c["tpu"].pop("num_workers", None)

# RUNBOOK section 8: rename ALL identity fields; assert what we set.
c["job"]["name"] = f"{name}_1"
c["tpu"]["name"] = f"{name}_1"
c["tpu"]["accelerator"] = args.accel
r = c["resumable"]
r["run_id_prefix"] = f"{args.arm}-{args.member}"

env = r["env"]
extra = ("TTD_FLATLINE_STOP=1 TTD_FLATLINE_EPS=1e-9 TTD_FLATLINE_CONSEC=3 "
         "TTD_FLATLINE_MIN_STEPS=4 TTD_RESTART_RATIO=0 TTD_RESTART_AT_STEP=-1")
if args.init_sp:
    extra += f" TTD_INIT_STATE_PATH_{ENV_SUFFIX[args.member]}={args.init_sp}"
env.update({
    "CELL": CELL_ALIAS[args.member],
    "RUN_DIR_NAME": run,
    "EXPERIMENT_NAME": run,
    "NUM_EPOCHS": str(args.steps),
    "LEARNING_RATE": "1.5e-4",
    "TTD_ELITE_SLOTS": "2",
    "GROUPS_PER_BATCH": "16",
    "GROUP_SIZE": "32",
    "KL_PENALTY_COEF": "0",
    "TTD_ADV_ESTIMATOR": "mean_baseline",
    "TTD_ENV": "erdos_min_overlap",
    "TTD_RESTART_RATIO": "0",
    "EXTRA_TTD_ENV": extra,
})
if args.init_jsonl:
    env["INIT_JSONL_GCS"] = args.init_jsonl   # fetched by launch via EXTRA_REREG_JSONL path
    env["EXTRA_REREG_JSONL"] = f"/home/sk7524_princeton_edu/init_prev_{args.member}.jsonl"

r["run_spec"] = {
    "experiment": "meta-tree",
    "arm": args.arm,
    "generation": args.gen,
    "member": args.member,
    "shape": f"{args.accel} 1 trainer + 7 vLLM",
    "batch_shape": "16x32",
    "lr": "1.5e-4",
    "steps": args.steps,
    "stop": "fixed-15 + flatline(3x<1e-9, min4)",
}

# assertions per runbook (a silent no-op rename caused a real incident)
assert c["job"]["name"] == c["tpu"]["name"] == f"{name}_1"
assert env["RUN_DIR_NAME"] == env["EXPERIMENT_NAME"] == run
assert r["run_id_prefix"] == f"{args.arm}-{args.member}"
yaml.safe_dump(c, open(args.out, "w"), sort_keys=False)
print(args.out)
