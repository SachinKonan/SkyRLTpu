#!/usr/bin/env python3
"""Run A (snapshot-ensemble continuation), Muse seed 3: fork rebut-erdos-muse-seed3 at batch 11 onto a v5p-16 host
(dry run by default).

Destination run rebut-snapA-erdos-muse-s3 (profiles: tpu/runA_muse_profiles.py), trainer on one v5p-16 host
(TP1/FSDP4, native K/V 2). The source trained on v6e-128 pairs (TP8, 8 tied logical K/V heads), so this does what
tpu/cosearch_setup.py and tpu/qubit_rebut_setup.py do for a resume in place on v5p:
  * verify the source ledger row (batch 11 -> tinker://model_92e8ec42/weights/000011) and the archive size;
  * collapse the repeated K/V LoRA heads 8 -> 2 in the weights AND their Adam moments
    (tpu/cosearch_collapse_kv.py: bit-identical replicas, exact round trip, counters kept) and store the result
    under a NEW checkpoint id v5p-000011; the trainer's start-up reconcile registers it from the ledger;
  * build a minimal trainer registry (tpu/science/circuit_weight_handoff.minimal_registry, unmodified: model,
    session, source checkpoint 000011; no futures). The source registry (5.0 GB) is reused from the co-search
    stage when its size matches the GCS object, else downloaded;
  * write the client log dir: one ledger row (batch 11, state_path only), the run's own step-11 tree as the ONLY
    snapshot (step field 11), global_step 11, the source metrics rows of batches < 11, and a provenance record.
Snapshot adapters (lags 2/4/8 = steps 9/7/3) are not part of this: they go to the trainer host with
rq4-local/rebut/frozen/copy_snapshots.sh, and the profile registers them (TTD_SNAPSHOT_REGISTER).

Modes: (default) print every step; --prepare also runs the reads, checks and local steps; --go also performs the
GCS writes (create-only, --if-generation-match=0). Refuses if the destination run exists.

    bash tpu/runA_muse_setup.sh [--prepare | --go]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = "gs://sk7524-tinker-tpu-us-east5/ray-training"
STAGE = Path("/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/cosearch_tmp/runA")
SOURCE_RUN = "rebut-erdos-muse-seed3"
DEST_RUN = "rebut-snapA-erdos-muse-s3"
TAG = "muse"
BATCH = 11
MODEL, SOURCE_CKPT = "model_92e8ec42", f"{BATCH:06d}"
STATE_PATH = f"tinker://{MODEL}/weights/{SOURCE_CKPT}"
NEW_CKPT = f"v5p-{BATCH:06d}"
CONVERT = "muse-glimmer-30b"
# The co-search seed-3 setup downloaded this run's final registry (same object: 5,039,845,376 bytes, 2026-11-12).
LOCAL_DB = Path("/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/cosearch_tmp/rebut-cosearch-erdos-qm-seed3/muse/"
                "source-tinker.db")
# Bytes (gcloud storage ls -l, 2026-11-13).
SIZES = {"archive": 2_331_471_244, "db": 5_039_845_376, "tree": 6_677_481, "metrics": 374_196}
LOG_FILES = [f"member_{TAG}/checkpoints.jsonl", f"puct_sampler_step_{BATCH:06d}.json", "global_step.json",
             "metrics.jsonl", "runA_setup.json"]


class Plan:
    def __init__(self, mode: str):
        self.mode = mode  # dry | prepare | go

    def _run(self, kind, argv, note):
        execute = self.mode == "go" or (self.mode == "prepare" and kind != "write")
        print(f"[{kind}{'' if execute else ', not run'}] {note}\n  " + " ".join(shlex.quote(str(a)) for a in argv),
              flush=True)
        if execute:
            subprocess.run([str(a) for a in argv], check=True)

    def read(self, argv, note):
        self._run("read", argv, note)

    def local(self, argv, note):
        self._run("local", argv, note)

    def write(self, argv, note):
        self._run("write", argv, note)

    def check(self, fn, note):
        print(f"[check{', not run' if self.mode == 'dry' else ''}] {note}", flush=True)
        if self.mode != "dry":
            fn()

    def file(self, path: Path, text_fn, note):
        execute = self.mode in ("prepare", "go")
        print(f"[local{'' if execute else ', not run'}] {note}: {path}", flush=True)
        if execute:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text_fn())


def gs(*args):
    return ["gcloud", "storage", *args]


def gcs_size(uri: str) -> int:
    out = subprocess.run(gs("ls", "-l", uri), check=True, capture_output=True, text=True).stdout
    return int(out.split()[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kept_metrics(text: str) -> str:
    rows = [line for line in text.splitlines(True) if line.strip()]
    kept = [line for line in rows if json.loads(line).get("progress/batch", 10**9) < BATCH]
    batches = [json.loads(line)["progress/batch"] for line in kept]
    if batches != list(range(BATCH)):
        raise SystemExit(f"source metrics batches < {BATCH} are {batches}, expected 0..{BATCH - 1}")
    return "".join(kept)


def setup(plan: Plan, stage: Path, py: str):
    d = stage / DEST_RUN / TAG
    src = f"{RUNS}/{SOURCE_RUN}"
    src_logs = f"{src}/client/tinker_log/{SOURCE_RUN}"
    src_archive = f"{src}/checkpoints/{MODEL}/{SOURCE_CKPT}.tar.gz"
    dest = f"{RUNS}/{DEST_RUN}"
    if plan.mode != "dry":
        d.mkdir(parents=True, exist_ok=True)

    def refuse():
        if subprocess.run(gs("ls", f"{dest}/"), capture_output=True).returncode == 0:
            raise SystemExit(f"{dest} exists; refusing")
    plan.check(refuse, f"destination run must not exist: {dest}")

    plan.read(gs("cp", f"{src_logs}/member_{TAG}/checkpoints.jsonl", d / "source_checkpoints.jsonl"),
              "source ledger")

    def verify_row():
        rows = [json.loads(l) for l in (d / "source_checkpoints.jsonl").read_text().splitlines() if l.strip()]
        hit = [r for r in rows if r.get("batch") == BATCH and r.get("state_path") and r.get("name") != "final"]
        if not hit or hit[-1]["state_path"] != STATE_PATH:
            raise SystemExit(f"ledger batch {BATCH} is {hit[-1:]} not {STATE_PATH}")
    plan.check(verify_row, f"ledger row batch {BATCH} == {STATE_PATH}")

    archive, converted, report = d / "source.tar.gz", d / f"{NEW_CKPT}.tar.gz", d / "convert.json"
    if converted.exists() and report.exists():
        def reuse():
            rep = json.loads(report.read_text())
            if sha256(converted) != rep["archive_sha256"] or rep.get("model") != CONVERT:
                raise SystemExit(f"{converted} does not match {report}; delete both and rerun")
        plan.check(reuse, f"reuse {converted} (sha256 == convert.json)")
    else:
        if not archive.exists():
            plan.read(gs("cp", src_archive, archive), f"source checkpoint batch {BATCH} (~2.3 GB)")

        def same_size():
            want = gcs_size(src_archive)
            if archive.stat().st_size != want or want != SIZES["archive"]:
                raise SystemExit(f"{archive} is {archive.stat().st_size} bytes, {src_archive} is {want}")
        plan.check(same_size, f"{archive} has the size of {src_archive}")
        plan.local([py, ROOT / "tpu/cosearch_collapse_kv.py", archive, converted, "--model", CONVERT,
                    "--report", report], "collapse repeated K/V heads 8 -> 2 (weights + Adam)")

    def kv_report():
        rep = json.loads(report.read_text())
        shapes = {(tuple(v["before"]), tuple(v["after"]), v["replicas_per_native_head"]) for v in rep["converted"].values()}
        if not rep["converted"] or any(r != 4 or b[-1] != 4 * a[-1] for b, a, r in shapes):
            raise SystemExit(f"unexpected K/V conversion {shapes}")
        if not (rep["optimizer_present"] and rep["layouts_dropped"]):
            raise SystemExit("conversion report lacks the optimizer or kept the layouts")
        print(f"  converted {len(rep['converted'])} K/V tensors {sorted(shapes)}; optimizer counters "
              f"{rep['optimizer_counters']}; {rep['bytes']} bytes, sha256 {rep['archive_sha256']}")
    plan.check(kv_report, "conversion report: every K/V B tensor 4 replicas -> 1, Adam kept, layouts dropped")

    if LOCAL_DB.exists():
        db = LOCAL_DB

        def db_size():
            want = gcs_size(f"{src}/tinker-backup.db")
            if db.stat().st_size != want:
                raise SystemExit(f"{db} is not the size of {src}/tinker-backup.db ({want}); delete it and rerun")
        plan.check(db_size, f"reuse the source registry {db} (size == GCS object)")
    else:
        db = d / "source-tinker.db"
        if not db.exists():
            plan.read(gs("cp", f"{src}/tinker-backup.db", db), "source registry (~5.0 GB)")
    registry = d / "tinker-backup.db"
    if not registry.exists():
        plan.local([py, "-c", "import json,sys; sys.path.insert(0, sys.argv[1]); "
                    "from tpu.science.circuit_weight_handoff import minimal_registry; "
                    "print(json.dumps(minimal_registry(*sys.argv[2:6])))",
                    ROOT, db, registry, MODEL, SOURCE_CKPT], f"minimal registry ({MODEL}, {SOURCE_CKPT})")

    def registry_rows():
        import sqlite3
        with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as con:
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("sessions", "models", "checkpoints", "futures")}
            row = con.execute("SELECT model_id, checkpoint_id, checkpoint_type, status FROM checkpoints").fetchall()
        if counts != {"sessions": 1, "models": 1, "checkpoints": 1, "futures": 0} or \
                row != [(MODEL, SOURCE_CKPT, "TRAINING", "COMPLETED")]:
            raise SystemExit(f"registry {registry}: {counts} {row}")
        print(f"  registry rows {counts}, checkpoint {row[0]}, {registry.stat().st_size} bytes")
    plan.check(registry_rows, "minimal registry: 1 session, 1 model, 1 COMPLETED training checkpoint, no futures")

    logs = d / "client_log"
    plan.read(gs("cp", f"{src_logs}/puct_sampler_step_{BATCH:06d}.json", d / "tree.json"),
              f"tree {SOURCE_RUN} step {BATCH} (~6.7 MB)")

    def renumber():
        tree = json.loads((d / "tree.json").read_text())
        if tree.get("step") != BATCH:
            raise SystemExit(f"source tree step field is {tree.get('step')}, expected {BATCH}")
        return json.dumps(tree)
    plan.file(logs / f"puct_sampler_step_{BATCH:06d}.json", renumber,
              f"tree as the only snapshot, step field {BATCH}")
    plan.read(gs("cp", f"{src_logs}/metrics.jsonl", d / "source_metrics.jsonl"), "source metrics")
    row = dict(name=f"{BATCH:06d}", batch=BATCH, state_path=f"tinker://{MODEL}/weights/{NEW_CKPT}",
               runA_source=STATE_PATH)
    plan.file(logs / f"member_{TAG}" / "checkpoints.jsonl", lambda: json.dumps(row) + "\n",
              f"ledger, one row: batch {BATCH} -> {row['state_path']}")
    plan.file(logs / "global_step.json", lambda: json.dumps({"global_step": BATCH}) + "\n",
              f"global step {BATCH}")
    plan.file(logs / "metrics.jsonl", lambda: kept_metrics((d / "source_metrics.jsonl").read_text()),
              f"source metrics rows of batches < {BATCH}")
    plan.file(logs / "runA_setup.json", lambda: json.dumps(dict(
        dest_run=DEST_RUN, member=TAG, source_run=SOURCE_RUN, source_state_path=STATE_PATH,
        source_archive=src_archive, state_path=row["state_path"], converted=CONVERT, kv_heads="8 -> 2",
        convert_report=str(report), convert_sha256=json.loads(report.read_text())["archive_sha256"],
        registry=dict(source=f"{src}/tinker-backup.db", model=MODEL, checkpoint=SOURCE_CKPT),
        tree=f"{src_logs}/puct_sampler_step_{BATCH:06d}.json (step {BATCH}, unchanged)",
        snapshots="muse-s3-step09/07/03 (rq4-local/rebut/frozen/adapters, TTD_SNAPSHOT_REGISTER)",
        time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1) + "\n", "provenance")

    plan.write(gs("cp", "--if-generation-match=0", converted, f"{dest}/checkpoints/{MODEL}/{NEW_CKPT}.tar.gz"),
               "converted checkpoint (~1.48 GB)")
    plan.write(gs("cp", "--if-generation-match=0", registry, f"{dest}/tinker-backup.db"), "minimal registry")
    for rel in LOG_FILES:
        plan.write(gs("cp", "--if-generation-match=0", logs / rel, f"{dest}/client/tinker_log/{DEST_RUN}/{rel}"),
                   f"client log {rel}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="run reads, checks and local steps; print the writes")
    mode.add_argument("--go", action="store_true", help="also perform the GCS writes (create-only)")
    ap.add_argument("--stage", default=str(STAGE))
    x = ap.parse_args(argv)
    plan = Plan("go" if x.go else "prepare" if x.prepare else "dry")
    print(f"# Run A Muse seed 3 setup: {SOURCE_RUN} batch {BATCH} -> {DEST_RUN}, stage {x.stage}, mode {plan.mode}")
    setup(plan, Path(x.stage), sys.executable)
    print("# then: copy the snapshot adapters to the trainer host (copy_snapshots.sh), register the run with "
          "rebut-discovery, and launch the .ensemble profile (see tpu/runA_muse_profiles.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
