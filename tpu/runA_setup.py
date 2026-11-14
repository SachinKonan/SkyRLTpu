#!/usr/bin/env python3
"""Run A (snapshot-ensemble continuation): fork a base run at batch 11 into a new run (dry run by default).

One arm per seed (ARMS): the trainer state (LoRA + Adam) of the weights run at batch 11, a PUCT tree at step 11,
written as a new run that resumes at batch 11 (profiles: tpu/runA_profiles.py):
  * verify the weights run's ledger row (batch 11 -> STATE_PATH) and the archive size;
  * K/V layout: the Muse arms trained with TP8 (8 tied logical K/V heads) and resume on one v5p-16 host (native
    2), so tpu/cosearch_collapse_kv.py collapses the K/V LoRA B tensors 8 -> 2 in the weights AND their Adam
    moments (bit-identical replicas, exact round trip, counters kept) into a NEW checkpoint id v5p-000011; the
    trainer's start-up reconcile registers it from the ledger. The Qwen arm trained with TP8 (logical 8) and
    resumes with TP8 (logical 8) on the OD v4-64 box, so the archive is only inspected (every K/V tensor 8
    logical heads, replicas identical, Adam present) and copied server-side under its own id 000011;
  * a minimal trainer registry (tpu/science/circuit_weight_handoff.minimal_registry, unmodified: model, session,
    source checkpoint 000011; no futures) instead of the multi-GB source registry, so no big restore at start;
  * the client log dir: one ledger row (batch 11, state_path only), the tree run's step-11 tree as the ONLY
    snapshot (step field 11; its best must equal the tree run's metrics row batch 10, the one-step lag), global
    step 11, the metrics rows of batches < 11 of the tree run (provenance; the pool-value series stays
    continuous with the tree), and a provenance record.
Snapshot adapters (lags 2/4/8 = steps 9/7/3) are not part of this: they go to the client host with
rq4-local/rebut/frozen/copy_snapshots.sh, and the profile registers them (TTD_SNAPSHOT_REGISTER).

Modes: (default) print every step; --prepare also runs the reads, checks and local steps; --go also performs the
GCS writes (create-only, --if-generation-match=0). Refuses if the destination run exists.

    bash tpu/runA_setup.sh --arm {muse-s3,muse-s2,qwen-s3} [--prepare | --go]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EAST5 = "gs://sk7524-tinker-tpu-us-east5"
CENTRAL2 = "gs://sk7524-tinker-tpu-us-central2"
STAGE = Path("/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/cosearch_tmp/runA")
BATCH = 11
SOURCE_CKPT = f"{BATCH:06d}"


@dataclass(frozen=True)
class Arm:
    source_bucket: str
    source_run: str           # weights run (trainer state at batch 11)
    tag: str                  # client member tag
    model: str                # tinker model id of the batch-11 checkpoint
    dest_bucket: str
    dest_run: str
    sizes: dict               # bytes (gcloud storage ls -l): archive, db, tree, metrics
    convert: str | None       # collapse_kv --model for a K/V conversion, None = copy unchanged
    inspect_model: str        # collapse_kv --model used to read the saved K/V layout
    logical_kv_heads: int     # the destination trainer's logical_kv_heads
    snapshots: str
    tree_bucket: str = ""     # default: the weights run's
    tree_run: str = ""
    local_db: str = ""        # a verified local copy of the source registry, if any

    @property
    def state_path(self):
        return f"tinker://{self.model}/weights/{SOURCE_CKPT}"

    @property
    def new_ckpt(self):
        return f"v5p-{SOURCE_CKPT}" if self.convert else SOURCE_CKPT

    @property
    def tree_src(self):
        return (self.tree_bucket or self.source_bucket, self.tree_run or self.source_run)


ARMS = {
    "muse-s3": Arm(
        EAST5, "rebut-erdos-muse-seed3", "muse", "model_92e8ec42", EAST5, "rebut-snapA-erdos-muse-s3",
        dict(archive=2_331_471_244, db=5_039_845_376, tree=6_677_481, metrics=374_196),
        "muse-glimmer-30b", "muse-glimmer-30b", 2, "muse-s3-step09/07/03",
        # The co-search seed-3 setup downloaded this run's final registry (same object, 2026-11-12).
        local_db="/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/cosearch_tmp/rebut-cosearch-erdos-qm-seed3/muse/"
                 "source-tinker.db"),
    # Trained on OD v4-64 workers 0/1 (TP8, logical K/V 8), stored in us-central2; resumes on a v5p-16 host in
    # us-east5 like muse-s3.
    "muse-s2": Arm(
        CENTRAL2, "rebut-erdos-muse-seed2", "muse", "model_9c88ba98", EAST5, "rebut-snapA-erdos-muse-s2",
        dict(archive=2_331_471_244, db=5_200_699_392, tree=7_369_140, metrics=406_434),
        "muse-glimmer-30b", "muse-glimmer-30b", 2, "muse-s2-step09/07/03"),
    # Weights: rl28k-qwen35-erdos-s2-od16-r4 batch 11 = model_9dab5505/000011, trained 2026-11-05 on OD v4-64
    # workers 0/1 (TP8, logical K/V 8) by the rl28k arm (steps 1-21). The run id also holds the rebuttal's
    # continuation (batches 22-37, v6e-128, us-east5 copy of the run); rows 1-21 of both ledgers are identical
    # and the east5 000011 archive is the same object (MD5 BlyVXm/dTbJ3TpKXzpDMLg==). The central2 run is the
    # rl28k original and its registry (4.45 GB, 2026-11-07) predates the continuation. Tree: seed 1's lottery
    # tree, rebut-erdos-qwen35-seed1-splice step 11 (us-east5).
    "qwen-s3": Arm(
        CENTRAL2, "rl28k-qwen35-erdos-s2-od16-r4", "qwen", "model_9dab5505", CENTRAL2,
        "rebut-snapA-erdos-qwen-s3",
        dict(archive=1_925_557_154, db=4_450_881_536, tree=11_530_854, metrics=357_664),
        None, "qwen3.5-27b", 8, "qwen-s3-step09/07/03",
        tree_bucket=EAST5, tree_run="rebut-erdos-qwen35-seed1-splice"),
}


def log_files(arm: Arm) -> list[str]:
    return [f"member_{arm.tag}/checkpoints.jsonl", f"puct_sampler_step_{BATCH:06d}.json", "global_step.json",
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


def tree_matches_metrics(tree: dict, metrics_text: str) -> float:
    """The step-11 tree's best is the metrics row of batch 10 (rows lag the newest tree by one step); a
    snapshot rewritten later by another life would not match."""
    best = max(s["value"] for s in tree["states"] if s.get("value") is not None)
    rows = {json.loads(l)["progress/batch"]: json.loads(l) for l in metrics_text.splitlines() if l.strip()}
    want = rows.get(BATCH - 1, {}).get("pool/best_value")
    if want != best:
        raise SystemExit(f"tree step {BATCH} best {best} != metrics batch {BATCH - 1} pool/best_value {want}")
    return best


def setup(plan: Plan, arm: Arm, stage: Path, py: str):
    d = stage / arm.dest_run / arm.tag
    src = f"{arm.source_bucket}/ray-training/{arm.source_run}"
    src_logs = f"{src}/client/tinker_log/{arm.source_run}"
    src_archive = f"{src}/checkpoints/{arm.model}/{SOURCE_CKPT}.tar.gz"
    tree_bucket, tree_run = arm.tree_src
    tree_logs = f"{tree_bucket}/ray-training/{tree_run}/client/tinker_log/{tree_run}"
    dest = f"{arm.dest_bucket}/ray-training/{arm.dest_run}"
    dest_archive = f"{dest}/checkpoints/{arm.model}/{arm.new_ckpt}.tar.gz"
    if plan.mode != "dry":
        d.mkdir(parents=True, exist_ok=True)

    def refuse():
        if subprocess.run(gs("ls", f"{dest}/"), capture_output=True).returncode == 0:
            raise SystemExit(f"{dest} exists; refusing")
    plan.check(refuse, f"destination run must not exist: {dest}")

    plan.read(gs("cp", f"{src_logs}/member_{arm.tag}/checkpoints.jsonl", d / "source_checkpoints.jsonl"),
              "source ledger")

    def verify_row():
        rows = [json.loads(l) for l in (d / "source_checkpoints.jsonl").read_text().splitlines() if l.strip()]
        hit = [r for r in rows if r.get("batch") == BATCH and r.get("state_path") and r.get("name") != "final"]
        if not hit or hit[-1]["state_path"] != arm.state_path:
            raise SystemExit(f"ledger batch {BATCH} is {hit[-1:]} not {arm.state_path}")
    plan.check(verify_row, f"ledger row batch {BATCH} == {arm.state_path}")

    archive, report = d / "source.tar.gz", d / ("convert.json" if arm.convert else "inspect.json")
    converted = d / f"{arm.new_ckpt}.tar.gz"

    def fetch_archive():
        if not archive.exists():
            plan.read(gs("cp", src_archive, archive), f"source checkpoint batch {BATCH} "
                      f"(~{arm.sizes['archive'] / 1e9:.1f} GB)")

        def same_size():
            want = gcs_size(src_archive)
            if archive.stat().st_size != want or want != arm.sizes["archive"]:
                raise SystemExit(f"{archive} is {archive.stat().st_size} bytes, {src_archive} is {want}")
        plan.check(same_size, f"{archive} has the size of {src_archive}")

    if arm.convert:
        if converted.exists() and report.exists():
            def reuse():
                rep = json.loads(report.read_text())
                if sha256(converted) != rep["archive_sha256"] or rep.get("model") != arm.convert:
                    raise SystemExit(f"{converted} does not match {report}; delete both and rerun")
            plan.check(reuse, f"reuse {converted} (sha256 == convert.json)")
        else:
            fetch_archive()
            plan.local([py, ROOT / "tpu/cosearch_collapse_kv.py", archive, converted, "--model", arm.convert,
                        "--report", report], "collapse repeated K/V heads 8 -> 2 (weights + Adam)")

        def kv_report():
            rep = json.loads(report.read_text())
            shapes = {(tuple(v["before"]), tuple(v["after"]), v["replicas_per_native_head"])
                      for v in rep["converted"].values()}
            if not rep["converted"] or any(r != 4 or b[-1] != 4 * a[-1] for b, a, r in shapes):
                raise SystemExit(f"unexpected K/V conversion {shapes}")
            if not (rep["optimizer_present"] and rep["layouts_dropped"]):
                raise SystemExit("conversion report lacks the optimizer or kept the layouts")
            print(f"  converted {len(rep['converted'])} K/V tensors {sorted(shapes)}; optimizer counters "
                  f"{rep['optimizer_counters']}; {rep['bytes']} bytes, sha256 {rep['archive_sha256']}")
        plan.check(kv_report, "conversion report: every K/V B tensor 4 replicas -> 1, Adam kept, layouts dropped")
    else:
        fetch_archive()
        if not report.exists():
            plan.local([py, ROOT / "tpu/cosearch_collapse_kv.py", archive, "--inspect", "--model",
                        arm.inspect_model, "--report", report], "read the saved K/V layout (no conversion)")

        def kv_layout():
            rep = json.loads(report.read_text())
            if rep["logical_kv_heads"] != [arm.logical_kv_heads] or not rep["kv"] or not rep["optimizer_present"]:
                raise SystemExit(f"archive K/V layout {rep['logical_kv_heads']} (want [{arm.logical_kv_heads}]), "
                                 f"optimizer {rep.get('optimizer_present')}")
            if rep["archive_sha256"] != sha256(archive):
                raise SystemExit(f"{report} is not of {archive}")
            shapes = sorted({tuple(v["shape"]) for v in rep["kv"].values()})
            print(f"  {len(rep['kv'])} K/V tensors {shapes}, logical K/V heads {rep['logical_kv_heads']} == trainer "
                  f"logical_kv_heads {arm.logical_kv_heads}; optimizer counters {rep['optimizer_counters']}; "
                  f"layouts saved {rep['layouts_present']}")
        plan.check(kv_layout, f"saved K/V layout == the trainer's logical_kv_heads {arm.logical_kv_heads}, Adam "
                              "present: copy unchanged")

    if arm.local_db and Path(arm.local_db).exists():
        db = Path(arm.local_db)

        def db_size():
            want = gcs_size(f"{src}/tinker-backup.db")
            if db.stat().st_size != want:
                raise SystemExit(f"{db} is not the size of {src}/tinker-backup.db ({want}); delete it and rerun")
        plan.check(db_size, f"reuse the source registry {db} (size == GCS object)")
    else:
        db = d / "source-tinker.db"
        if not db.exists():
            plan.read(gs("cp", f"{src}/tinker-backup.db", db), f"source registry (~{arm.sizes['db'] / 1e9:.1f} GB)")

        def db_size():
            if db.stat().st_size != arm.sizes["db"] or gcs_size(f"{src}/tinker-backup.db") != arm.sizes["db"]:
                raise SystemExit(f"{db} is not the {arm.sizes['db']}-byte {src}/tinker-backup.db")
        plan.check(db_size, f"{db} has the size of {src}/tinker-backup.db")
    registry = d / "tinker-backup.db"
    if not registry.exists():
        plan.local([py, "-c", "import json,sys; sys.path.insert(0, sys.argv[1]); "
                    "from tpu.science.circuit_weight_handoff import minimal_registry; "
                    "print(json.dumps(minimal_registry(*sys.argv[2:6])))",
                    ROOT, db, registry, arm.model, SOURCE_CKPT], f"minimal registry ({arm.model}, {SOURCE_CKPT})")

    def registry_rows():
        with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as con:
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("sessions", "models", "checkpoints", "futures")}
            row = con.execute("SELECT model_id, checkpoint_id, checkpoint_type, status FROM checkpoints").fetchall()
            text = json.dumps([con.execute(f"SELECT * FROM {t}").fetchall() for t in ("sessions", "models",
                                                                                     "checkpoints")])
        if counts != {"sessions": 1, "models": 1, "checkpoints": 1, "futures": 0} or \
                row != [(arm.model, SOURCE_CKPT, "TRAINING", "COMPLETED")]:
            raise SystemExit(f"registry {registry}: {counts} {row}")
        for needle in (arm.source_bucket, arm.source_run, "gs://"):
            if needle in text:
                raise SystemExit(f"registry {registry} rows mention {needle!r}")
        print(f"  registry rows {counts}, checkpoint {row[0]}, {registry.stat().st_size} bytes, no paths")
    plan.check(registry_rows, "minimal registry: 1 session, 1 model, 1 COMPLETED training checkpoint, no futures, "
                              "no bucket/run paths")

    logs = d / "client_log"
    plan.read(gs("cp", f"{tree_logs}/puct_sampler_step_{BATCH:06d}.json", d / "tree.json"),
              f"tree {tree_run} step {BATCH} (~{arm.sizes['tree'] / 1e6:.1f} MB)")
    plan.read(gs("cp", f"{tree_logs}/metrics.jsonl", d / "source_metrics.jsonl"), f"metrics of {tree_run}")

    def tree_checks():
        if (d / "tree.json").stat().st_size != arm.sizes["tree"]:
            raise SystemExit(f"tree is {(d / 'tree.json').stat().st_size} bytes, expected {arm.sizes['tree']}")
        tree = json.loads((d / "tree.json").read_text())
        if tree.get("step") != BATCH:
            raise SystemExit(f"source tree step field is {tree.get('step')}, expected {BATCH}")
        best = tree_matches_metrics(tree, (d / "source_metrics.jsonl").read_text())
        print(f"  tree step {tree['step']}: {len(tree['states'])} states, best {best!r} == metrics batch "
              f"{BATCH - 1} pool/best_value")
    plan.check(tree_checks, f"tree size, step field {BATCH}, best == metrics batch {BATCH - 1}")

    plan.file(logs / f"puct_sampler_step_{BATCH:06d}.json", lambda: (d / "tree.json").read_text(),
              f"tree as the only snapshot, step field {BATCH} (unchanged bytes)")
    row = dict(name=SOURCE_CKPT, batch=BATCH, state_path=f"tinker://{arm.model}/weights/{arm.new_ckpt}",
               runA_source=arm.state_path)
    plan.file(logs / f"member_{arm.tag}" / "checkpoints.jsonl", lambda: json.dumps(row) + "\n",
              f"ledger, one row: batch {BATCH} -> {row['state_path']}")
    plan.file(logs / "global_step.json", lambda: json.dumps({"global_step": BATCH}) + "\n",
              f"global step {BATCH}")
    plan.file(logs / "metrics.jsonl", lambda: kept_metrics((d / "source_metrics.jsonl").read_text()),
              f"{tree_run} metrics rows of batches < {BATCH}")

    def provenance():
        rep = json.loads(report.read_text())
        tree = json.loads((d / "tree.json").read_text())
        return json.dumps(dict(
            dest_run=arm.dest_run, dest=dest, member=arm.tag, source_run=arm.source_run, source_bucket=arm.source_bucket,
            source_state_path=arm.state_path, source_archive=src_archive, state_path=row["state_path"],
            checkpoint=dict(converted=arm.convert, kv_heads="8 -> 2" if arm.convert else
                            f"{arm.logical_kv_heads} (unchanged)", report=str(report),
                            archive_sha256=rep["archive_sha256"], dest_archive=dest_archive),
            registry=dict(source=f"{src}/tinker-backup.db", model=arm.model, checkpoint=SOURCE_CKPT),
            tree=f"{tree_logs}/puct_sampler_step_{BATCH:06d}.json (step {BATCH}, unchanged, "
                 f"{len(tree['states'])} states)",
            metrics=f"{tree_logs}/metrics.jsonl rows < {BATCH}",
            snapshots=f"{arm.snapshots} (rq4-local/rebut/frozen/adapters, TTD_SNAPSHOT_REGISTER)",
            time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1) + "\n"
    plan.file(logs / "runA_setup.json", provenance, "provenance")

    if arm.convert:
        plan.write(gs("cp", "--if-generation-match=0", converted, dest_archive),
                   "converted checkpoint (~1.48 GB)")
    else:
        plan.write(gs("cp", "--if-generation-match=0", src_archive, dest_archive),
                   f"checkpoint, unchanged (server-side copy, {arm.sizes['archive'] / 1e9:.2f} GB, same bucket)"
                   if arm.source_bucket == arm.dest_bucket else "checkpoint, unchanged")
    plan.write(gs("cp", "--if-generation-match=0", registry, f"{dest}/tinker-backup.db"), "minimal registry")
    for rel in log_files(arm):
        plan.write(gs("cp", "--if-generation-match=0", logs / rel,
                      f"{dest}/client/tinker_log/{arm.dest_run}/{rel}"), f"client log {rel}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="run reads, checks and local steps; print the writes")
    mode.add_argument("--go", action="store_true", help="also perform the GCS writes (create-only)")
    ap.add_argument("--stage", default=str(STAGE))
    x = ap.parse_args(argv)
    arm = ARMS[x.arm]
    plan = Plan("go" if x.go else "prepare" if x.prepare else "dry")
    tree_bucket, tree_run = arm.tree_src
    print(f"# Run A {x.arm} setup: weights {arm.source_bucket}/{arm.source_run} batch {BATCH}, tree "
          f"{tree_bucket}/{tree_run} step {BATCH} -> {arm.dest_bucket}/{arm.dest_run}, stage {x.stage}, "
          f"mode {plan.mode}")
    setup(plan, arm, Path(x.stage), sys.executable)
    print("# then: copy the snapshot adapters to the client host (copy_snapshots.sh), register the run with "
          "rebut-discovery, and launch the .ensemble profile (see tpu/runA_profiles.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
