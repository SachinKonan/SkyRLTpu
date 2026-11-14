#!/usr/bin/env python3
"""Run A (snapshot-ensemble continuation), Muse seed 3 on v5p-16: trainer profile + three v5p Muse farm profiles.

Trainer  rebut-snapA-erdos-muse-s3: one normal single-run client + trainer on 10.202.15.201 (v5p-16 5nmb host 0),
         resuming the fork that tpu/runA_muse_setup.py writes (batch 11, K/V collapsed to 2).
  * Everything a candidate sees and how it is scored = the base run rebut-erdos-muse-seed3 (profile
    rebut-v6e128-erdos-muse-seed3 at b21e88b2): client env (env, estimator, seeds, prompts, grading caps,
    EVAL_TIMEOUT, open grading K 20), LR 4e-5, contexts 28672 / 22528, inference block (remote-only, discovery group
    rebut-muse, compatibility hash), grading (math 64, farm transport), trainer env, bucket us-east5.
  * Hardware = the v5p-16 Muse trainer of the co-search (rebut-cosearch-erdos-qm-seed3-muse at ccbd6c2f, branch
    agent/cosearch; ran batches 12-24 on hardware): trainer block (TP1/FSDP4, logical K/V 2, host offload,
    seq 28672), RAM caches (trainer 128 / inference 128 / reserve 192 GiB), trainer compile cache (same shapes).
    Its trainer-only/co-search keys are dropped.
  * Then rq4-local/rebut/frozen/runA_patches.py <profile> muse-s3 (unmodified) writes <profile>.ensemble.json: the
    snapshot-ensemble client env (lags 0/2/4/8, snapshots steps 9/7/3, CISPO, IS cap 2, NUM_EPOCHS 26,
    SKYRL_SNAPSHOT_KEEP 12) and external_pool_target_leases >= 4. The .ensemble.json is the one to launch.
Farms    rebut-v5p16-farm-muse-{5nmb-h1,de70-h0,de70-h1} on 10.202.15.200 / .204 / .203: one TP4 inference-only engine
         per host, copied from the v6e-128 Muse farm rebut-v6e128-farm-muse-w05 (b21e88b2) with the v5p-16 changes of
         rebut-v5p16-farm-qwen35-h0 (accelerator, node, v5p-only inference compile cache, reserve 280 GiB and 48 math +
         48 placement grading slots for a 440 GiB host).

    python tpu/runA_muse_profiles.py            # differences vs the base / source profiles, validation
    python tpu/runA_muse_profiles.py --write    # write the five profiles under tpu/swarm/ray_train/profiles
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = ROOT / "tpu/swarm/ray_train/profiles"
BUCKET = "gs://sk7524-tinker-tpu-us-east5"
RUN_ID = "rebut-snapA-erdos-muse-s3"
ARM = "muse-s3"
HOST = "10.202.15.201"
RESUME_BATCH = 11
# (git ref, profile name)
BASE = ("b21e88b2", "rebut-v6e128-erdos-muse-seed3")
HARDWARE = ("ccbd6c2f", "rebut-cosearch-erdos-qm-seed3-muse")   # agent/cosearch tip
FARM_SOURCE = ("b21e88b2", "rebut-v6e128-farm-muse-w05")
PATCHES = Path("/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/frozen/runA_patches.py")
NODE_5NMB = "tpuswarm-v5p16-east5a-5nmb-7bfcb694-head-770153ji-tpu"
NODE_DE70 = "tpuswarm-v5p16-east5a-de70-7bfcb694-head-1mxa9mje-tpu"
NODES = {"10.202.15.201": NODE_5NMB, "10.202.15.200": NODE_5NMB,
         "10.202.15.204": NODE_DE70, "10.202.15.203": NODE_DE70}
FARMS = {"rebut-v5p16-farm-muse-5nmb-h1": "10.202.15.200",
         "rebut-v5p16-farm-muse-de70-h0": "10.202.15.204",
         "rebut-v5p16-farm-muse-de70-h1": "10.202.15.203"}
# v5p-only compile cache: XLA programs of a TP4 Muse engine on v5p chips (v6e farms use rebut/muse-tp4).
FARM_COMPILE = f"{BUCKET}/rebut/muse-v5p-tp4/inference-compile"
# 440 GiB v5p host (rebut-v5p16-farm-qwen35-h0): reserve 280 GiB, 48 math + 48 placement slots of 2 CPUs / 4 GiB.
FARM_RESERVE_GIB = 280
FARM_GRADING = {"math": {"slots_per_host": 48, "cpus": 2, "memory_gib": 4},
                "placement": {"slots_per_host": 48, "cpus": 2, "memory_gib": 4}}
COSEARCH_KEYS = ("trainer_only", "cosearch_client_run", "client_member_spec")


def git_profile(ref: str, name: str) -> dict:
    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:tpu/swarm/ray_train/profiles/{name}.json"],
                          check=True, capture_output=True, text=True).stdout
    return json.loads(text)


def runA_patch(profile: dict) -> dict:
    """rq4-local/rebut/frozen/runA_patches.patch, unmodified, on a temporary copy of the profile."""
    spec = importlib.util.spec_from_file_location("runA_patches", PATCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{profile['run_id']}.json"
        path.write_text(json.dumps(profile, indent=2) + "\n")
        out = module.patch(str(path), ARM)
        return json.loads(Path(out).read_text())


def trainer() -> dict:
    base, hw = git_profile(*BASE), git_profile(*HARDWARE)
    p = copy.deepcopy(base)
    for key in COSEARCH_KEYS:
        p.pop(key, None)
    p.update(run_id=RUN_ID, root=f"~/.cache/{RUN_ID}", accelerator=hw["accelerator"], hosts=hw["hosts"],
             trainer=copy.deepcopy(hw["trainer"]), assigned_host_ips=[HOST], physical_node=NODES[HOST],
             resume_min_checkpoint_step=RESUME_BATCH)
    p["cache"] = dict(hw["cache"], inference_compile=f"{BUCKET}/rebut/{RUN_ID}/unused-inference-compile")
    p["client_env"] = dict(base["client_env"], SKYRL_RUN_ID=RUN_ID,
                           TTD_SICK_MARKER=f"/home/gcpuser/.cache/{RUN_ID}/runs/{RUN_ID}/ENGINE-SICK")
    return p


def farm(run_id: str, host: str) -> dict:
    p = copy.deepcopy(git_profile(*FARM_SOURCE))
    p.update(run_id=run_id, root=f"~/.cache/{run_id}", accelerator="tpu-v5p-16", assigned_host_ips=[host],
             physical_node=NODES[host])
    p["cache"].update(trainer_compile=f"{BUCKET}/rebut/{run_id}/unused-trainer-compile",
                      inference_compile=FARM_COMPILE, reserve_gib=FARM_RESERVE_GIB)
    p["grading"]["families"] = copy.deepcopy(FARM_GRADING)
    return p


def profiles() -> dict[str, dict]:
    t = trainer()
    out = {RUN_ID: t, f"{RUN_ID}.ensemble": runA_patch(t)}
    out.update({run_id: farm(run_id, host) for run_id, host in FARMS.items()})
    return out


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v and k not in ("maxtext_kwargs",):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def differences(new: dict, ref: dict) -> list[str]:
    a, b = flat(new), flat(ref)
    return [f"{k}: {b.get(k, '<absent>')!r} -> {a.get(k, '<absent>')!r}" for k in sorted(set(a) | set(b))
            if a.get(k, "<absent>") != b.get(k, "<absent>")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    x = ap.parse_args(argv)
    from tpu.swarm.ray_train.config import Config
    base, hw, w05 = git_profile(*BASE), git_profile(*HARDWARE), git_profile(*FARM_SOURCE)
    for name, p in profiles().items():
        config = Config.from_dict(p)
        print(f"== {name}: {config.accelerator} {config.assigned_host_ips} grading pool {config.grading_memory_gib} "
              f"GiB, services {config.grading_service_cpus} CPUs, families {config.grading_families}")
        if name.startswith(RUN_ID):
            print(f"  vs base {BASE[1]}@{BASE[0]}:")
            for line in differences(p, base):
                print("    ", line)
            print(f"  vs hardware {HARDWARE[1]}@{HARDWARE[0]} (trainer/cache/accelerator keys only):")
            for line in differences(p, hw):
                if line.split(":")[0].split(".")[0] in ("trainer", "cache", "accelerator", "hosts"):
                    print("    ", line)
        else:
            print(f"  vs {FARM_SOURCE[1]}@{FARM_SOURCE[0]}:")
            for line in differences(p, w05):
                print("    ", line)
        if x.write:
            text = json.dumps(p, indent=1) if name.endswith(".ensemble") else json.dumps(p, indent=2) + "\n"
            (PROFILES / f"{name}.json").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
