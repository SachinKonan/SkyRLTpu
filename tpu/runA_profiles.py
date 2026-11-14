#!/usr/bin/env python3
"""Run A (snapshot-ensemble continuation): trainer profiles of the arms forked by tpu/runA_setup.py, plus the three
v5p-16 Muse farm profiles.

Every trainer profile is "base run settings on the arm's hardware", then rq4-local/rebut/frozen/runA_patches.py
<profile> <arm> (unmodified) writes <profile>.ensemble.json: the snapshot-ensemble client env (lags 0/2/4/8,
snapshots steps 9/7/3, CISPO, IS cap 2, NUM_EPOCHS 26, SKYRL_SNAPSHOT_KEEP 12) and
external_pool_target_leases >= 4. The .ensemble.json is the one to launch. trainer.backward_warmup is false in
both (required with TTD_LOSS_FN=cispo; the live muse-s3 and qwen-s2 arms run that way).

muse-s3  rebut-snapA-erdos-muse-s3 on 10.202.15.201 (v5p-16 5nmb host 0). Client env, LR, contexts, inference
         block (remote-only, discovery group rebut-muse, compatibility hash), grading, trainer env and bucket =
         the base run rebut-v6e128-erdos-muse-seed3 (b21e88b2); trainer block (TP1/FSDP4, logical K/V 2, host
         offload), RAM caches and trainer compile cache = the co-search v5p-16 Muse trainer
         (rebut-cosearch-erdos-qm-seed3-muse, ccbd6c2f); co-search-only keys dropped.
muse-s2  rebut-snapA-erdos-muse-s2 on one v5p-16 host (default 10.202.15.204 = de70 host 0; --muse-s2-host).
         The muse-s3 recipe with the seed-2 base run's client env (rebut-od-erdos-muse-seed2, b21e88b2): the two
         base runs differ only in seed (TTD_M0_LORA_SEED 2), run id and hardware. Not carried over from the
         OD seed-2 profile: its OD hardware/region keys, its 8 farm leases (the patch makes it max(4, 8) = 8 and
         starves muse-s3; both Muse arms take 4 here) and TTD_ENGINE_DB_RETRY=1 (an OD-disk-thrash workaround
         the muse-s3 arm on the same v5p hosts does not use).
qwen-s3  rebut-snapA-erdos-qwen-s3 as a SkyPilot POOL JOB on the OD v4-64 (pool tpuswarm-v4-64-central2-od):
         ranks 0-5 = TPU workers 0-5, trainer on the 2x4x2 box workers {5,3,2,4} (bounds 1,2,2, TP8 x FSDP2,
         logical K/V 8), Ray head/client/ingress on worker 0 (default ports, apart from the SSH partition qwen-s2
         arm's 25xxx), worker 1 idle. Client env, LR, contexts, inference block, trainer env, bucket = the
         weights run rl28k-qwen35-erdos-s2-od16-r4 (TTD_M0_LORA_SEED 2); hardware (accelerator/hosts 6,
         trainer block, RAM caches, 64 math slots) = rl28k-qwen35-erdos-s1-od-r4, the Qwen pool job that ran on
         workers 2-5 next to s2-od16 on 0/1 (b21e88b2); rebuttal keys (discovery group rebut-qwen35, grading
         registry rebut-farms.json, open grading K 20) = the tree run rebut-v6e128-erdos-qwen-seed1-splice and the
         live qwen-s2 arm; 4 farm leases like qwen-s2 (the base's 8 would compete for the four Qwen farms). Own
         trainer compile prefix, seeded read-only from s1-od-r4's (same shapes).

    python tpu/runA_profiles.py            # differences vs the base / hardware profiles, validation
    python tpu/runA_profiles.py --write    # write the profiles under tpu/swarm/ray_train/profiles
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
EAST5 = "gs://sk7524-tinker-tpu-us-east5"
CENTRAL2 = "gs://sk7524-tinker-tpu-us-central2"
RESUME_BATCH = 11
REF = "b21e88b2"
PATCHES = Path("/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/frozen/runA_patches.py")
# Muse: (git ref, profile name)
MUSE_BASE = {"muse-s3": (REF, "rebut-v6e128-erdos-muse-seed3"), "muse-s2": (REF, "rebut-od-erdos-muse-seed2")}
MUSE_SETTINGS = MUSE_BASE["muse-s3"]                                  # region/inference/grading of a v5p Muse arm
MUSE_HARDWARE = ("ccbd6c2f", "rebut-cosearch-erdos-qm-seed3-muse")   # agent/cosearch tip
MUSE_DROPPED_ENV = ("TTD_ENGINE_DB_RETRY",)
FARM_SOURCE = (REF, "rebut-v6e128-farm-muse-w05")
# Qwen s3
QWEN_BASE = (REF, "rl28k-qwen35-erdos-s2-od16-r4")
QWEN_HARDWARE = (REF, "rl28k-qwen35-erdos-s1-od-r4")
QWEN_TREE_RUN = (REF, "rebut-v6e128-erdos-qwen-seed1-splice")
QWEN_PARTITION_KEYS = ("assigned_host_ips", "physical_node", "ports", "systemd_port", "storage_min_free_gib")
QWEN_REBUTTAL_ENV = {"SKYRL_GRADING_REGISTRY": "/home/gcpuser/.cache/skyrl/rebut-farms.json",
                     "SKYRL_OPEN_GRADING_K": "20"}
QWEN_DISCOVERY_GROUP = "rebut-qwen35"

NODE_5NMB = "tpuswarm-v5p16-east5a-5nmb-7bfcb694-head-770153ji-tpu"
NODE_DE70 = "tpuswarm-v5p16-east5a-de70-7bfcb694-head-1mxa9mje-tpu"
NODES = {"10.202.15.201": NODE_5NMB, "10.202.15.200": NODE_5NMB,
         "10.202.15.204": NODE_DE70, "10.202.15.203": NODE_DE70}
HOSTS = {"muse-s3": "10.202.15.201", "muse-s2": "10.202.15.204"}
RUN_IDS = {arm: f"rebut-snapA-erdos-{arm}" for arm in ("muse-s3", "muse-s2", "qwen-s3")}
FARMS = {"rebut-v5p16-farm-muse-5nmb-h1": "10.202.15.200",
         "rebut-v5p16-farm-muse-de70-h0": "10.202.15.204",
         "rebut-v5p16-farm-muse-de70-h1": "10.202.15.203"}
# v5p-only compile cache: XLA programs of a TP4 Muse engine on v5p chips (v6e farms use rebut/muse-tp4).
FARM_COMPILE = f"{EAST5}/rebut/muse-v5p-tp4/inference-compile"
# 440 GiB v5p host (rebut-v5p16-farm-qwen35-h0): reserve 280 GiB, 48 math + 48 placement slots of 2 CPUs / 4 GiB.
FARM_RESERVE_GIB = 280
FARM_GRADING = {"math": {"slots_per_host": 48, "cpus": 2, "memory_gib": 4},
                "placement": {"slots_per_host": 48, "cpus": 2, "memory_gib": 4}}
COSEARCH_KEYS = ("trainer_only", "cosearch_client_run", "client_member_spec")


def git_profile(ref: str, name: str) -> dict:
    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:tpu/swarm/ray_train/profiles/{name}.json"],
                          check=True, capture_output=True, text=True).stdout
    return json.loads(text)


def runA_patch(profile: dict, arm: str) -> dict:
    """rq4-local/rebut/frozen/runA_patches.patch, unmodified, on a temporary copy of the profile."""
    spec = importlib.util.spec_from_file_location("runA_patches", PATCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{profile['run_id']}.json"
        path.write_text(json.dumps(profile, indent=2) + "\n")
        out = module.patch(str(path), arm)
        return json.loads(Path(out).read_text())


def run_env(env: dict, run_id: str) -> dict:
    return dict(env, SKYRL_RUN_ID=run_id, TTD_SICK_MARKER=f"/home/gcpuser/.cache/{run_id}/runs/{run_id}/ENGINE-SICK")


def muse_trainer(arm: str, host: str) -> dict:
    settings, hw, seed_base = git_profile(*MUSE_SETTINGS), git_profile(*MUSE_HARDWARE), git_profile(*MUSE_BASE[arm])
    run_id = RUN_IDS[arm]
    p = copy.deepcopy(settings)
    for key in COSEARCH_KEYS:
        p.pop(key, None)
    p.update(run_id=run_id, root=f"~/.cache/{run_id}", accelerator=hw["accelerator"], hosts=hw["hosts"],
             trainer=dict(copy.deepcopy(hw["trainer"]), backward_warmup=False), assigned_host_ips=[host],
             physical_node=NODES[host], resume_min_checkpoint_step=RESUME_BATCH)
    p["cache"] = dict(hw["cache"], inference_compile=f"{EAST5}/rebut/{run_id}/unused-inference-compile")
    env = {k: v for k, v in seed_base["client_env"].items() if k not in MUSE_DROPPED_ENV}
    p["client_env"] = run_env(env, run_id)
    return p


def qwen_trainer() -> dict:
    base, hw = git_profile(*QWEN_BASE), git_profile(*QWEN_HARDWARE)
    run_id = RUN_IDS["qwen-s3"]
    p = copy.deepcopy(base)
    for key in QWEN_PARTITION_KEYS:
        p.pop(key, None)
    p.update(run_id=run_id, root=f"~/.cache/{run_id}", accelerator=hw["accelerator"], hosts=hw["hosts"],
             trainer=dict(copy.deepcopy(hw["trainer"]), backward_warmup=False),
             resume_min_checkpoint_step=RESUME_BATCH)
    p["cache"] = dict(hw["cache"], trainer_compile=f"{CENTRAL2}/rebut/{run_id}/trainer-compile",
                      trainer_compile_seed=hw["cache"]["trainer_compile"],
                      inference_compile=f"{CENTRAL2}/rebut/{run_id}/unused-inference-compile")
    p["grading"] = copy.deepcopy(hw["grading"])
    # 4 leases like the live qwen-s2 arm (one farm per adapter); the base's 8 would compete with qwen-s2 for the
    # four Qwen farms.
    p["inference"] = dict(base["inference"], external_pool_discovery_group=QWEN_DISCOVERY_GROUP,
                          external_pool_target_leases=4)
    p["client_env"] = run_env(dict(base["client_env"], **QWEN_REBUTTAL_ENV), run_id)
    return p


def farm(run_id: str, host: str) -> dict:
    p = copy.deepcopy(git_profile(*FARM_SOURCE))
    p.update(run_id=run_id, root=f"~/.cache/{run_id}", accelerator="tpu-v5p-16", assigned_host_ips=[host],
             physical_node=NODES[host])
    p["cache"].update(trainer_compile=f"{EAST5}/rebut/{run_id}/unused-trainer-compile",
                      inference_compile=FARM_COMPILE, reserve_gib=FARM_RESERVE_GIB)
    p["grading"]["families"] = copy.deepcopy(FARM_GRADING)
    return p


def trainer(arm: str, host: str | None = None) -> dict:
    return qwen_trainer() if arm == "qwen-s3" else muse_trainer(arm, host or HOSTS[arm])


def profiles(arms=tuple(RUN_IDS), muse_s2_host: str | None = None, farms: bool = True) -> dict[str, dict]:
    out = {}
    for arm in arms:
        t = trainer(arm, muse_s2_host if arm == "muse-s2" else None)
        out[RUN_IDS[arm]] = t
        out[f"{RUN_IDS[arm]}.ensemble"] = runA_patch(t, arm)
    if farms:
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
    ap.add_argument("--arm", action="append", choices=sorted(RUN_IDS), help="default: all")
    ap.add_argument("--muse-s2-host", default=HOSTS["muse-s2"], choices=sorted(NODES))
    ap.add_argument("--no-farms", action="store_true")
    x = ap.parse_args(argv)
    from tpu.swarm.ray_train.config import Config
    refs = {"muse-s3": [MUSE_BASE["muse-s3"], MUSE_HARDWARE], "muse-s2": [MUSE_BASE["muse-s2"], ("muse-s3", None)],
            "qwen-s3": [QWEN_BASE, QWEN_HARDWARE, QWEN_TREE_RUN]}
    built = profiles(tuple(x.arm or RUN_IDS), x.muse_s2_host, not x.no_farms)
    for name, p in built.items():
        config = Config.from_dict(p)
        print(f"== {name}: {config.accelerator} hosts {config.hosts} trainer {config.trainer.hosts}x TP"
              f"{config.trainer.tp}/FSDP{config.trainer.fsdp} K/V {config.trainer.logical_kv_heads} "
              f"{config.assigned_host_ips or 'pool job'} grading pool {config.grading_memory_gib} GiB, families "
              f"{config.grading_families}")
        arm = next((a for a, r in RUN_IDS.items() if name.startswith(r)), None)
        for ref in (refs[arm] if arm else [FARM_SOURCE]):
            other = trainer("muse-s3") if ref == ("muse-s3", None) else git_profile(*ref)
            label = "muse-s3 trainer" if ref == ("muse-s3", None) else f"{ref[1]}@{ref[0]}"
            print(f"  vs {label}:")
            for line in differences(p, other):
                print("    ", line)
        if x.write:
            text = json.dumps(p, indent=1) if name.endswith(".ensemble") else json.dumps(p, indent=2) + "\n"
            (PROFILES / f"{name}.json").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
