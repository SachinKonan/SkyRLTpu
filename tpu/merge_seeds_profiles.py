#!/usr/bin/env python3
"""Erdős multi-seed merge, round 2: trainer profiles of the four runs forked by tpu/merge_seeds_setup.py.

Each run is an ordinary single-model run (client + trainer + grading on ONE v5p-16 host, inference on leased
farms) that resumes at batch 12 and stops after aggregated step 20 (NUM_EPOCHS 20 = batches 12..19, then the
"final" save). Built as "the weights source run's profile on single-host v5p hardware":

  * everything from the WEIGHTS source run's own profile: client env (estimator, loss, LR, contexts, seeds incl.
    TTD_M0_LORA_SEED, TTD_WARMUP_FB, open-grading keys), inference block (remote-only farms, discovery group,
    compatibility hash, lease target), trainer env, grading (64 math slots), bucket/caches (us-east5):
      qwen-a  rebut-v6e128-erdos-qwen-seed1-splice  (rebut-erdos-qwen35-seed1-splice)
      qwen-b  rebut-v6e128-erdos-qwen-seed3         (= rl28k-qwen35-erdos-s2-od16-r4's rl28k client settings plus
                                                     the rebuttal keys: group rebut-qwen35, rebut-farms registry,
                                                     open grading K 20; checked against the rl28k profile below)
      muse-a  rebut-v6e128-erdos-muse-seed1
      muse-b  rebut-v6e128-erdos-muse-seed3
  * the trainer block, RAM caches and v5p trainer compile cache of the co-search v5p-16 trainers (agent/cosearch
    rebut-cosearch-erdos-qm-seed3 = Qwen, -seed3-muse = Muse): TP1/FSDP4, chip bounds 2,2,1, native logical K/V
    heads (Qwen 4, Muse 2; the forks' archives are collapsed to them), Muse parameter host offload;
    trainer.backward_warmup = the base run's value (true; it was only turned off for CISPO arms);
  * run identity: run id, root, SKYRL_RUN_ID, TTD_SICK_MARKER, unused inference-compile prefix; NUM_EPOCHS 20;
    resume_min_checkpoint_step 12; accelerator tpu-v5p-16, hosts 1, the assigned host and its physical node;
    the v6e pair's custom port keys dropped (dedicated host: default ports), co-search-only keys absent.

    python tpu/merge_seeds_profiles.py                       # field differences + Config validation
    python tpu/merge_seeds_profiles.py --write               # write tpu/swarm/ray_train/profiles/<run>.json
    python tpu/merge_seeds_profiles.py --host qwen-a=10.202.15.201 ...
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = ROOT / "tpu/swarm/ray_train/profiles"
EAST5 = "gs://sk7524-tinker-tpu-us-east5"
RESUME_BATCH = 12
NUM_EPOCHS = 20
COSEARCH_REF = "ccbd6c2f"                 # agent/cosearch tip: v5p-16 Erdős trainer blocks
HARDWARE = {"qwen": "rebut-cosearch-erdos-qm-seed3", "muse": "rebut-cosearch-erdos-qm-seed3-muse"}
NODE_5NMB = "tpuswarm-v5p16-east5a-5nmb-7bfcb694-head-770153ji-tpu"
NODE_DE70 = "tpuswarm-v5p16-east5a-de70-7bfcb694-head-1mxa9mje-tpu"
NODES = {"10.202.15.201": NODE_5NMB, "10.202.15.200": NODE_5NMB,
         "10.202.15.204": NODE_DE70, "10.202.15.203": NODE_DE70}
# arm -> (run id, model tag, weights source profile, default host)
ARMS = {
    "qwen-a": ("rebut-merge-erdos-qwen-a", "qwen", "rebut-v6e128-erdos-qwen-seed1-splice", "10.202.15.201"),
    "qwen-b": ("rebut-merge-erdos-qwen-b", "qwen", "rebut-v6e128-erdos-qwen-seed3", "10.202.15.200"),
    "muse-a": ("rebut-merge-erdos-muse-a", "muse", "rebut-v6e128-erdos-muse-seed1", "10.202.15.204"),
    "muse-b": ("rebut-merge-erdos-muse-b", "muse", "rebut-v6e128-erdos-muse-seed3", "10.202.15.203"),
}
# Also compared against (no keys taken from it): the rl28k profile the qwen-b weights were trained under.
RL28K = ("b21e88b2", "rl28k-qwen35-erdos-s2-od16-r4")
DROPPED = ("ports", "systemd_port", "trainer_only", "cosearch_client_run", "cosearch_peer", "client_member_spec")
TRAINER_KEYS = ("hosts", "tp", "fsdp", "process_bounds", "chip_bounds", "logical_kv_heads", "maxtext_kwargs",
                "tokamax_splash", "num_vocab_tiling", "flce_tile")
CACHE_KEYS = ("trainer_compile", "trainer_gib", "inference_gib", "reserve_gib")


def git_profile(ref: str, name: str) -> dict:
    text = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:tpu/swarm/ray_train/profiles/{name}.json"],
                          check=True, capture_output=True, text=True).stdout
    return json.loads(text)


def local_profile(name: str) -> dict:
    return json.loads((PROFILES / f"{name}.json").read_text())


def trainer(arm: str, host: str) -> dict:
    run_id, tag, base_name, _ = ARMS[arm]
    base, hw = local_profile(base_name), git_profile(COSEARCH_REF, HARDWARE[tag])
    p = copy.deepcopy(base)
    for key in DROPPED:
        p.pop(key, None)
    t = base["trainer"]
    # Sequence/budget/remat from the base run; mesh + memory layout from the v5p trainer; warm-up as the base.
    p["trainer"] = dict({k: t[k] for k in ("sequence_length", "max_target_length", "token_budget", "seq_buckets",
                                           "remat", "request_timeout")},
                        **{k: copy.deepcopy(hw["trainer"][k]) for k in TRAINER_KEYS if k in hw["trainer"]},
                        backward_warmup=t.get("backward_warmup", False))
    p.update(run_id=run_id, root=f"~/.cache/{run_id}", accelerator=hw["accelerator"], hosts=hw["hosts"],
             zone=hw["zone"], bucket=hw["bucket"], single_bucket=True, assigned_host_ips=[host],
             physical_node=NODES[host], resume_min_checkpoint_step=RESUME_BATCH)
    p["cache"] = dict(base["cache"], **{k: hw["cache"][k] for k in CACHE_KEYS},
                      inference_compile=f"{EAST5}/rebut/{run_id}/unused-inference-compile")
    p["client_env"] = dict(base["client_env"], NUM_EPOCHS=str(NUM_EPOCHS), SKYRL_RUN_ID=run_id,
                           TTD_SICK_MARKER=f"/home/gcpuser/.cache/{run_id}/runs/{run_id}/ENGINE-SICK")
    return p


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


def validate(arm: str, p: dict):
    """Invariants of a merge run, beyond Config.validate."""
    from tpu.swarm.ray_train.config import Config
    run_id, tag, base_name, _ = ARMS[arm]
    config = Config.from_dict(p)
    config.validate()
    base = local_profile(base_name)
    want_kv = {"qwen": 4, "muse": 2}[tag]
    checks = {
        "single v5p host": (config.accelerator, config.hosts, config.trainer.hosts, config.trainer.tp,
                            config.trainer.fsdp) == ("tpu-v5p-16", 1, 1, 1, 4),
        f"logical K/V {want_kv}": config.trainer.logical_kv_heads == want_kv,
        "backward_warmup as base": config.trainer.backward_warmup == base["trainer"].get("backward_warmup", False),
        "member tag": config.client_member_spec.rsplit(":", 1)[-1] == tag,
        "resume 12, stop after 20": (p["resume_min_checkpoint_step"], p["client_env"]["NUM_EPOCHS"]) == (12, "20"),
        "client env == base except run identity/epochs": {k: v for k, v in p["client_env"].items()
                                                         if k not in ("NUM_EPOCHS", "SKYRL_RUN_ID", "TTD_SICK_MARKER")}
        == {k: v for k, v in base["client_env"].items() if k not in ("NUM_EPOCHS", "SKYRL_RUN_ID", "TTD_SICK_MARKER")},
        "inference == base": p["inference"] == base["inference"],
        "LR/context == base": all(p[k] == base[k] for k in ("client_learning_rate", "client_context_window",
                                                             "client_phase1_max_tokens", "model_preset")),
        "us-east5 bucket": config.bucket == EAST5,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"{arm}: failed {bad}")
    return config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), help="default: all four")
    ap.add_argument("--host", action="append", default=[], metavar="ARM=IP", help="override the default host")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing profiles")
    x = ap.parse_args(argv)
    hosts = {arm: ARMS[arm][3] for arm in ARMS}
    for spec in x.host:
        arm, _, ip = spec.partition("=")
        if arm not in ARMS or ip not in NODES:
            ap.error(f"--host wants ARM=IP with ARM in {sorted(ARMS)} and IP in {sorted(NODES)}, got {spec!r}")
        hosts[arm] = ip
    arms = x.arm or sorted(ARMS)
    if len({hosts[a] for a in arms}) != len(arms):
        ap.error(f"two arms on one host: {hosts}")
    for arm in arms:
        run_id, tag, base_name, _ = ARMS[arm]
        p = trainer(arm, hosts[arm])
        config = validate(arm, p)
        print(f"== {run_id} ({arm}): {config.accelerator} host {config.assigned_host_ips} ({p['physical_node']}) "
              f"trainer TP{config.trainer.tp}/FSDP{config.trainer.fsdp} K/V {config.trainer.logical_kv_heads} "
              f"backward_warmup {config.trainer.backward_warmup}; member {config.client_member_spec}; NUM_EPOCHS "
              f"{p['client_env']['NUM_EPOCHS']}; leases {p['inference']['external_pool_target_leases']} from "
              f"{p['inference']['external_pool_discovery_group']}; grading {config.grading_families}")
        refs = [(f"{base_name} (weights source, working tree)", local_profile(base_name)),
                (f"{HARDWARE[tag]}@{COSEARCH_REF} (v5p trainer)", git_profile(COSEARCH_REF, HARDWARE[tag]))]
        if arm == "qwen-b":
            refs.append((f"{RL28K[1]}@{RL28K[0]} (rl28k original)", git_profile(*RL28K)))
        for label, ref in refs:
            print(f"  vs {label}:")
            for line in differences(p, ref):
                print("    ", line)
        if x.write:
            path = PROFILES / f"{run_id}.json"
            if path.exists() and not x.force:
                if json.loads(path.read_text()) == p:
                    print(f"  {path} unchanged")
                    continue
                print(f"refusing to overwrite {path} (pass --force)", file=sys.stderr)
                return 2
            path.write_text(json.dumps(p, indent=2) + "\n")
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
