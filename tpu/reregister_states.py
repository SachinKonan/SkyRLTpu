#!/usr/bin/env python3
"""Re-register durable GCS checkpoints in a fresh tinker server's sqlite DB.

A preempted slice loses the per-VM sqlite registry (models/checkpoints tables)
while checkpoint tarballs survive in GCS via the gcsfuse --checkpoints-base
mount. A fresh server therefore 404s ("Model not found") on
create_training_client_from_state even though the weights exist. This script
re-inserts the registry rows for every (model, checkpoint) whose tarball is
present, so resume-from-state works across slices.

Run ON the trainer host whose DB should learn the models:
  python3 reregister_states.py --base-model Qwen/Qwen3.5-27B \
      --entry model_bded5e7f:000003 [--entry model_x:000002 ...]
  python3 reregister_states.py --base-model X --jsonl <checkpoints.jsonl>
"""
import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

SESSION_ID = "session_reregister"
LORA_CONFIG = {
    "rank": 32,
    "alpha": 32.0,
    "seed": 0,
    "train_attn": True,
    "train_mlp": True,
    "train_unembed": False,
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def entries_from_jsonl(path):
    pat = re.compile(r"tinker://(model_[0-9a-f]+)/weights/(\d+)")
    out = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            m = pat.match(row.get("state_path") or "")
            if m:
                out.append((m.group(1), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/SkyRLTpu/skyrl/tinker/tinker.db"))
    ap.add_argument("--ckpt-root", default=os.path.expanduser("~/gcs/skyrl-checkpoints"))
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--entry", action="append", default=[], help="model_id:checkpoint_id")
    ap.add_argument("--jsonl", help="checkpoints.jsonl to harvest state_path rows from")
    args = ap.parse_args()

    entries = [tuple(e.split(":", 1)) for e in args.entry]
    if args.jsonl:
        entries += entries_from_jsonl(args.jsonl)
    if not entries:
        print("reregister: no entries")
        return

    db = sqlite3.connect(args.db)
    db.execute(
        "INSERT OR IGNORE INTO sessions (session_id, tags, user_metadata, sdk_version,"
        " status, created_at, heartbeat_count) VALUES (?, '[]', '{}', 'reregister',"
        " 'active', ?, 0)",
        (SESSION_ID, now()),
    )
    for model_id, ckpt in sorted(set(entries)):
        train_tar = os.path.join(args.ckpt_root, model_id, f"{ckpt}.tar.gz")
        if not os.path.exists(train_tar):
            print(f"skip {model_id}/{ckpt}: no tarball at {train_tar}")
            continue
        db.execute(
            "INSERT OR IGNORE INTO models (model_id, base_model, lora_config, status,"
            " request_id, session_id, created_at) VALUES (?, ?, ?, 'created', 0, ?, ?)",
            (model_id, args.base_model, json.dumps(LORA_CONFIG), SESSION_ID, now()),
        )
        db.execute(
            "INSERT OR IGNORE INTO checkpoints (model_id, checkpoint_id, checkpoint_type,"
            " status, created_at, completed_at) VALUES (?, ?, 'TRAINING', 'COMPLETED', ?, ?)",
            (model_id, ckpt, now(), now()),
        )
        print(f"registered {model_id}/{ckpt} TRAINING")
        samp_tar = os.path.join(args.ckpt_root, model_id, "sampler_weights", f"{ckpt}.tar.gz")
        if os.path.exists(samp_tar):
            db.execute(
                "INSERT OR IGNORE INTO checkpoints (model_id, checkpoint_id, checkpoint_type,"
                " status, created_at, completed_at) VALUES (?, ?, 'SAMPLER', 'COMPLETED', ?, ?)",
                (model_id, ckpt, now(), now()),
            )
            print(f"registered {model_id}/{ckpt} SAMPLER")
    db.commit()
    print("reregister: done")


if __name__ == "__main__":
    main()
