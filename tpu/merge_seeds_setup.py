#!/usr/bin/env python3
"""Erdős multi-seed merge, round 2: fork four ordinary single-model runs at aggregated step 12 (dry run by default).

Per model, the top-2 seeds at aggregated step 12 continue as TWO independent ordinary training runs (normal loss,
no co-search, no ensemble). Each run gets the trainer state (LoRA + Adam) of ITS seed and its own copy of the SAME
merged PUCT tree = union of both seeds' step-12 trees (tpu/cosearch_merge_trees.py, unmodified):

  qwen-a  rebut-merge-erdos-qwen-a  weights rebut-erdos-qwen35-seed1-splice batch 5 (its own step 5 = aggregated 12)
  qwen-b  rebut-merge-erdos-qwen-b  weights rl28k-qwen35-erdos-s2-od16-r4 batch 12 (rl28k lineage model_9dab5505)
  muse-a  rebut-merge-erdos-muse-a  weights rebut-erdos-muse-seed1 batch 12
  muse-b  rebut-merge-erdos-muse-b  weights rebut-erdos-muse-seed3 batch 12
  tree    qwen: splice step-5 tree + rl28k s2-od16-r4 step-12 tree; muse: seed1 + seed3 step-12 trees

Every destination resumes at batch 12 (the first new sampling is aggregated step 13; NUM_EPOCHS 20 stops after
step 20). Per arm:
  * sources: each source's ledger row (batch -> state_path), its tree's step field and its tree best == its
    metrics row of batch K-1 (rows lag the newest tree by one step); for the rl28k Qwen source also the
    us-central2 original (same ledger row, archive md5 equal to the us-east5 copy);
  * weights: the single-host v5p trainer (TP1/FSDP4) instantiates native K/V heads (Qwen 4, Muse 2), so the TP8
    archive (logical 8) needs the K/V collapse WITH Adam (tpu/cosearch_collapse_kv.py). The co-search setup
    already produced exactly these four conversions (cosearch_tmp/rebut-cosearch-erdos-qm-seed{1,3}/{qwen,muse});
    they are reused read-only after sha256(converted) == convert.json and sha256(GCS source archive, streamed)
    == convert.json source_archive_sha256. Uploaded under the new id v5p-000012; the trainer's start-up
    reconcile registers it from the ledger (model registered, snapshot 12 present, archive present);
  * a minimal trainer registry (tpu/science/circuit_weight_handoff.minimal_registry, unmodified) from the
    source registry (co-search's local copy, size == GCS object): model, session, source checkpoint, no futures;
  * client log dir: one ledger row (batch 12 -> v5p-000012), the merged tree as puct_sampler_step_000012.json
    (the only snapshot; byte-identical for both arms of a model), global_step 12, merge_setup.json provenance.
    No metrics.jsonl (as the co-search forks): a merged tree has no single pre-fork series; the source series
    stay in the source runs and the provenance records their step-12 bests.
Merged tree statistics (cosearch_merge_trees rules): states/initial states = union (ids are UUIDs; the two seeds
share none), puct_n per id summed (= kept, the trees are disjoint), puct_m per id max (= kept), puct_T summed --
the union is treated as one tree that has seen both searches' expansions, so the exploration term sqrt(1+T) grows
~1.41x for every state. Elite slots are not stored: they are re-chosen each step over the union (top values,
one per lineage), so both may come from the better seed.

Modes: (default) print every step; --prepare runs the reads, checks and local steps and prints the writes; --go
also performs the GCS writes (create-only, --if-generation-match=0). Refuses if a destination run exists.

    bash tpu/merge_seeds_setup.sh [--arm qwen-a ...] [--prepare | --go]
"""
from __future__ import annotations

import argparse
import base64
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
sys.path.insert(0, str(ROOT))
EAST5 = "gs://sk7524-tinker-tpu-us-east5"
CENTRAL2 = "gs://sk7524-tinker-tpu-us-central2"
COSEARCH = Path("/scratch/gpfs/ZHUANGL/sk7524/rq4-local/rebut/cosearch_tmp")
STAGE = Path("/home/sk7524/.claude/jobs/055aca86/tmp/merge/stage")   # small files only (home quota)
BATCH = 12                      # destination batch: state after aggregated step 12
NEW_CKPT = f"v5p-{BATCH:06d}"
MAX_BUFFER = 1000               # PUCTSampler.max_buffer_size


@dataclass(frozen=True)
class Source:
    """One seed's state after aggregated step 12."""
    tag: str            # client member tag
    preset: str         # collapse_kv --model
    bucket: str
    run: str
    model: str
    batch: int          # ledger batch == tree step of the source run (own numbering)
    cosearch_dir: str   # co-search staging: cosearch-v5p-000012.tar.gz, convert.json, source-tinker.db
    archive_size: int
    original_bucket: str = ""   # a second copy of the run whose ledger/archive must agree

    @property
    def ckpt(self):
        return f"{self.batch:06d}"

    @property
    def state_path(self):
        return f"tinker://{self.model}/weights/{self.ckpt}"

    def uri(self, bucket=None):
        return f"{bucket or self.bucket}/ray-training/{self.run}"

    def logs(self, bucket=None):
        return f"{self.uri(bucket)}/client/tinker_log/{self.run}"

    def archive(self, bucket=None):
        return f"{self.uri(bucket)}/checkpoints/{self.model}/{self.ckpt}.tar.gz"


SOURCES = {
    # f+8 splice: own batch 5 = aggregated 12 (tree step 5 best == its metrics batch 4).
    "qwen-s1": Source("qwen", "qwen3.5-27b", EAST5, "rebut-erdos-qwen35-seed1-splice", "model_71dee643", 5,
                      str(COSEARCH / "rebut-cosearch-erdos-qm-seed1/qwen"), 1_925_557_154),
    # The run id holds two experiments; batch 12 of model_9dab5505 is the rl28k lineage (OD v4-64, 2026-11-05).
    # us-east5 is a copy of the us-central2 original (same ledger rows 1-21, same archive bytes): read from
    # us-east5 (destination region), verify against us-central2.
    "qwen-s3": Source("qwen", "qwen3.5-27b", EAST5, "rl28k-qwen35-erdos-s2-od16-r4", "model_9dab5505", 12,
                      str(COSEARCH / "rebut-cosearch-erdos-qm-seed3/qwen"), 1_925_557_154, original_bucket=CENTRAL2),
    "muse-s1": Source("muse", "muse-glimmer-30b", EAST5, "rebut-erdos-muse-seed1", "model_5b97def6", 12,
                      str(COSEARCH / "rebut-cosearch-erdos-qm-seed1/muse"), 2_331_471_244),
    "muse-s3": Source("muse", "muse-glimmer-30b", EAST5, "rebut-erdos-muse-seed3", "model_92e8ec42", 12,
                      str(COSEARCH / "rebut-cosearch-erdos-qm-seed3/muse"), 2_331_471_244),
}
# Merged tree per model: source order = state order in the merged file (seed 1 first).
MERGES = {"qwen": ("qwen-s1", "qwen-s3"), "muse": ("muse-s1", "muse-s3")}
# arm -> (weights source, destination run)
ARMS = {
    "qwen-a": ("qwen-s1", "rebut-merge-erdos-qwen-a"),
    "qwen-b": ("qwen-s3", "rebut-merge-erdos-qwen-b"),
    "muse-a": ("muse-s1", "rebut-merge-erdos-muse-a"),
    "muse-b": ("muse-s3", "rebut-merge-erdos-muse-b"),
}
DEST_BUCKET = EAST5
TP8_LOGICAL_KV = 8


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


def describe(uri: str) -> dict:
    out = subprocess.run(gs("objects", "describe", uri, "--format=json"), check=True, capture_output=True,
                         text=True).stdout
    return json.loads(out)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gcs_stream_hashes(uri: str) -> tuple[str, str, int]:
    """(sha256 hex, md5 base64, bytes) of a GCS object, streamed (nothing on disk)."""
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with subprocess.Popen(gs("cat", uri), stdout=subprocess.PIPE) as proc:
        for chunk in iter(lambda: proc.stdout.read(16 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    if proc.returncode:
        raise SystemExit(f"gcloud storage cat {uri} failed ({proc.returncode})")
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode(), n


def tree_best(tree: dict) -> float:
    return max(s["value"] for s in tree["states"] if s.get("value") is not None)


def ledger_row(path: Path, batch: int) -> dict | None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    hit = [r for r in rows if r.get("batch") == batch and r.get("state_path") and r.get("name") != "final"]
    return hit[-1] if hit else None


def check_source(plan: Plan, key: str, src: Source, stage: Path):
    """Ledger row, tree step field and the one-step-lag metrics match of one source; leaves tree.json."""
    d = stage / "sources" / key
    if plan.mode != "dry":
        d.mkdir(parents=True, exist_ok=True)
    plan.read(gs("cp", f"{src.logs()}/member_{src.tag}/checkpoints.jsonl", d / "checkpoints.jsonl"),
              f"{key}: ledger of {src.run}")
    plan.read(gs("cp", f"{src.logs()}/puct_sampler_step_{src.batch:06d}.json", d / "tree.json"),
              f"{key}: tree step {src.batch}")
    plan.read(gs("cp", f"{src.logs()}/metrics.jsonl", d / "metrics.jsonl"), f"{key}: metrics")

    def verify():
        row = ledger_row(d / "checkpoints.jsonl", src.batch)
        if not row or row["state_path"] != src.state_path:
            raise SystemExit(f"{key}: ledger batch {src.batch} is {row} not {src.state_path}")
        tree = json.loads((d / "tree.json").read_text())
        if tree.get("step") != src.batch:
            raise SystemExit(f"{key}: tree step field {tree.get('step')} != {src.batch}")
        best = tree_best(tree)
        rows = {json.loads(l)["progress/batch"]: json.loads(l)
                for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()}
        want = rows.get(src.batch - 1, {}).get("pool/best_value")
        if want != best:
            raise SystemExit(f"{key}: tree step {src.batch} best {best} != metrics batch {src.batch - 1} {want}")
        info = dict(states=len(tree["states"]), initial=len(tree.get("initial_states") or []),
                    puct_T=tree.get("puct_T"), best=best, metrics_row=src.batch - 1, ledger_row=row,
                    tree_sha256=sha256(d / "tree.json"))
        (d / "verified.json").write_text(json.dumps(info, indent=1) + "\n")
        print(f"  {key}: ledger batch {src.batch} -> {row['state_path']}; tree step {src.batch}: "
              f"{info['states']} states, {info['initial']} initial, T {info['puct_T']}, best {best!r} == metrics "
              f"batch {src.batch - 1} pool/best_value")
    plan.check(verify, f"{key}: ledger row batch {src.batch} == {src.state_path}; tree step field {src.batch}; "
                       f"tree best == metrics batch {src.batch - 1}")

    if src.original_bucket:
        o = d / "original"
        if plan.mode != "dry":
            o.mkdir(parents=True, exist_ok=True)
        plan.read(gs("cp", f"{src.logs(src.original_bucket)}/member_{src.tag}/checkpoints.jsonl",
                      o / "checkpoints.jsonl"), f"{key}: ledger of the original {src.uri(src.original_bucket)}")

        def original():
            a, b = ledger_row(d / "checkpoints.jsonl", src.batch), ledger_row(o / "checkpoints.jsonl", src.batch)
            if a != b:
                raise SystemExit(f"{key}: copy ledger row {a} != original {b}")
            x, y = describe(src.archive()), describe(src.archive(src.original_bucket))
            if (x.get("md5_hash"), x.get("size")) != (y.get("md5_hash"), y.get("size")) or not x.get("md5_hash"):
                raise SystemExit(f"{key}: archive differs between {src.bucket} and {src.original_bucket}: {x} {y}")
            print(f"  {key}: original ledger row identical; archive md5 {x['md5_hash']} ({x['size']} bytes) in both")
        plan.check(original, f"{key}: original lineage ({src.original_bucket}) has the same ledger row and archive")


def merged_tree(plan: Plan, model: str, stage: Path) -> Path:
    keys = MERGES[model]
    out = stage / "merged" / model / f"puct_sampler_step_{BATCH:06d}.json"

    def build():
        from tpu.cosearch_merge_trees import merge_snapshots, write_atomic
        trees = [(key, json.loads((stage / "sources" / key / "tree.json").read_text())) for key in keys]
        merged, report = merge_snapshots(trees, BATCH, MAX_BUFFER)
        n = sum(len(t["states"]) for _, t in trees)
        if report["shared_identical"] != {"states": 0, "initial_states": 0}:
            raise SystemExit(f"{model}: the seeds share state ids {report['shared_identical']}")
        if report["merged_states"] != n or report["over_buffer"] or report["shared_parent_ids"]:
            raise SystemExit(f"{model}: unexpected merge {report}")
        if report["merged_best_value"] != max(tree_best(t) for _, t in trees):
            raise SystemExit(f"{model}: merged best is not the max of the sources")
        if merged["puct_T"] != sum(int(t["puct_T"]) for _, t in trees):
            raise SystemExit(f"{model}: puct_T is not the sum")
        if out.exists():
            if json.loads(out.read_text()) != merged:
                raise SystemExit(f"{out} exists with other content; delete it and rerun")
        else:
            write_atomic(str(out), merged)
        report["sha256"] = sha256(out)
        (out.parent / "merge_report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(f"  {model}: merged {report['merged_states']} states ({' + '.join(str(report['sources'][k]['states']) for k in keys)}), "
              f"{report['merged_initial_states']} initial, T {report['merged_puct_T']}, best "
              f"{report['merged_best_value']!r} from {report['best_source']}, dangling parent refs "
              f"{report['dangling_parent_refs']}, <= {MAX_BUFFER}; sha256 {report['sha256']}")
    plan.check(build, f"{model}: merge {' + '.join(keys)} at step {BATCH} (tpu/cosearch_merge_trees.merge_snapshots): "
                      "disjoint ids, states = sum, best = max, T = sum, under the buffer cap")
    return out


def setup(plan: Plan, arm: str, stage: Path, py: str):
    key, dest_run = ARMS[arm]
    src = SOURCES[key]
    model_tag = src.tag
    d = stage / dest_run
    cos = Path(src.cosearch_dir)
    converted, report = cos / f"cosearch-v5p-{BATCH:06d}.tar.gz", cos / "convert.json"
    dest = f"{DEST_BUCKET}/ray-training/{dest_run}"
    dest_archive = f"{dest}/checkpoints/{src.model}/{NEW_CKPT}.tar.gz"
    if plan.mode != "dry":
        d.mkdir(parents=True, exist_ok=True)

    def refuse():
        if subprocess.run(gs("ls", f"{dest}/"), capture_output=True).returncode == 0:
            raise SystemExit(f"{dest} exists; refusing")
    plan.check(refuse, f"destination run must not exist: {dest}")

    native = {"qwen3.5-27b": 4, "muse-glimmer-30b": 2}[src.preset]
    replicas = TP8_LOGICAL_KV // native

    def weights():
        rep = json.loads(report.read_text())
        if rep.get("model") != src.preset:
            raise SystemExit(f"{report} converted for {rep.get('model')}, not {src.preset}")
        local_sha = sha256(converted)
        if local_sha != rep["archive_sha256"] or converted.stat().st_size != rep["bytes"]:
            raise SystemExit(f"{converted} does not match {report}")
        shapes = {(tuple(v["before"]), tuple(v["after"]), v["replicas_per_native_head"])
                  for v in rep["converted"].values()}
        if not rep["converted"] or any(r != replicas or b[-1] != replicas * a[-1] for b, a, r in shapes):
            raise SystemExit(f"unexpected K/V conversion {shapes} (want {replicas} replicas per native head)")
        if not (rep["optimizer_present"] and rep["layouts_dropped"]):
            raise SystemExit("conversion report lacks the optimizer or kept the layouts")
        meta = describe(src.archive())
        if int(meta["size"]) != src.archive_size:
            raise SystemExit(f"{src.archive()} is {meta['size']} bytes, expected {src.archive_size}")
        sha, md5, n = gcs_stream_hashes(src.archive())
        if sha != rep["source_archive_sha256"] or n != src.archive_size or \
                (meta.get("md5_hash") and meta["md5_hash"] != md5):
            raise SystemExit(f"{src.archive()} (sha256 {sha}) is not the archive {report} converted "
                             f"({rep['source_archive_sha256']})")
        info = dict(source_archive=src.archive(), source_sha256=sha, source_md5=md5, converted=str(converted),
                    converted_sha256=local_sha, converted_bytes=rep["bytes"], kv=f"{TP8_LOGICAL_KV} -> {native}",
                    kv_tensors=len(rep["converted"]), shapes=sorted(map(str, shapes)),
                    optimizer_counters=rep["optimizer_counters"], lora_config=rep.get("lora_config"))
        (d / "weights.json").write_text(json.dumps(info, indent=1) + "\n")
        print(f"  {arm}: {src.archive()} streamed sha256 {sha} == convert.json source; converted {converted.name} "
              f"sha256 {local_sha} ({rep['bytes']} bytes) == convert.json; {len(rep['converted'])} K/V tensors "
              f"{sorted(shapes)}; optimizer counters {rep['optimizer_counters']}")
    plan.check(weights, f"{arm}: reuse {converted} (K/V {TP8_LOGICAL_KV} -> {native}, Adam kept): sha256 == "
                        f"convert.json, GCS source archive (streamed) == convert.json source")

    db, registry = cos / "source-tinker.db", d / "tinker-backup.db"

    def db_size():
        want = int(describe(f"{src.uri()}/tinker-backup.db")["size"])
        if db.stat().st_size != want:
            raise SystemExit(f"{db} ({db.stat().st_size}) is not the size of {src.uri()}/tinker-backup.db ({want})")
    plan.check(db_size, f"{arm}: source registry {db} has the size of {src.uri()}/tinker-backup.db")
    if not registry.exists():
        plan.local([py, "-c", "import json,sys; sys.path.insert(0, sys.argv[1]); "
                    "from tpu.science.circuit_weight_handoff import minimal_registry; "
                    "print(json.dumps(minimal_registry(*sys.argv[2:6])))",
                    ROOT, db, registry, src.model, src.ckpt], f"{arm}: minimal registry ({src.model}, {src.ckpt})")

    def registry_rows():
        with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as con:
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("sessions", "models", "checkpoints", "futures")}
            row = con.execute("SELECT model_id, checkpoint_id, checkpoint_type, status FROM checkpoints").fetchall()
            text = json.dumps([con.execute(f"SELECT * FROM {t}").fetchall() for t in ("sessions", "models",
                                                                                     "checkpoints")])
        if counts != {"sessions": 1, "models": 1, "checkpoints": 1, "futures": 0} or \
                row != [(src.model, src.ckpt, "TRAINING", "COMPLETED")]:
            raise SystemExit(f"registry {registry}: {counts} {row}")
        for needle in (src.bucket, src.run, "gs://"):
            if needle in text:
                raise SystemExit(f"registry {registry} rows mention {needle!r}")
        print(f"  {arm}: registry rows {counts}, checkpoint {row[0]}, {registry.stat().st_size} bytes, no paths")
    plan.check(registry_rows, f"{arm}: minimal registry: 1 session, 1 model, 1 COMPLETED training checkpoint, no "
                              "futures, no bucket/run paths")

    tree = merged_tree(plan, model_tag, stage)
    logs = d / "client_log"
    plan.file(logs / tree.name, lambda: tree.read_text(), f"{arm}: merged {model_tag} tree as the only snapshot "
                                                          f"(step field {BATCH}, same bytes for both {model_tag} arms)")
    row = dict(name=f"{BATCH:06d}", batch=BATCH, state_path=f"tinker://{src.model}/weights/{NEW_CKPT}",
               merge_source=src.state_path)
    plan.file(logs / f"member_{src.tag}" / "checkpoints.jsonl", lambda: json.dumps(row) + "\n",
              f"{arm}: ledger, one row: batch {BATCH} -> {row['state_path']} (no sampler_path: the client exports one)")
    plan.file(logs / "global_step.json", lambda: json.dumps({"global_step": BATCH}) + "\n", f"global step {BATCH}")

    def provenance():
        merge = json.loads((tree.parent / "merge_report.json").read_text())
        sources = {k: dict(run=SOURCES[k].run, bucket=SOURCES[k].bucket, batch=SOURCES[k].batch,
                           state_path=SOURCES[k].state_path,
                           tree=f"{SOURCES[k].logs()}/puct_sampler_step_{SOURCES[k].batch:06d}.json",
                           **json.loads((stage / 'sources' / k / 'verified.json').read_text()))
                   for k in MERGES[model_tag]}
        return json.dumps(dict(
            arm=arm, dest_run=dest_run, dest=dest, member=src.tag, batch=BATCH,
            design="multi-seed merge round 2: ordinary single-model training from this seed's step-12 trainer "
                   "state (LoRA + Adam) on the union of both seeds' step-12 PUCT trees; steps 13..20",
            weights=dict(source=key, source_state_path=src.state_path, state_path=row["state_path"],
                         dest_archive=dest_archive, **json.loads((d / "weights.json").read_text())),
            registry=dict(source=f"{src.uri()}/tinker-backup.db", model=src.model, checkpoint=src.ckpt),
            tree=dict(file=f"puct_sampler_step_{BATCH:06d}.json", sha256=merge["sha256"], sources=sources,
                      merged_states=merge["merged_states"], merged_initial_states=merge["merged_initial_states"],
                      merged_puct_T=merge["merged_puct_T"], merged_best_value=merge["merged_best_value"],
                      best_source=merge["best_source"], dangling_parent_refs=merge["dangling_parent_refs"],
                      rules="states/initial_states union (disjoint ids); puct_n summed, puct_m max (disjoint: "
                            "kept); puct_T summed; elite slots re-chosen each step over the union"),
            metrics="none: a merged tree has no single pre-fork series (source series stay in the source runs)",
            time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1) + "\n"
    plan.file(logs / "merge_setup.json", provenance, f"{arm}: provenance")

    plan.write(gs("cp", "--if-generation-match=0", converted, dest_archive),
               f"{arm}: converted checkpoint ({src.model}/{NEW_CKPT}, ~{converted.stat().st_size / 1e9:.2f} GB)"
               if converted.exists() else f"{arm}: converted checkpoint")
    plan.write(gs("cp", "--if-generation-match=0", registry, f"{dest}/tinker-backup.db"), f"{arm}: minimal registry")
    for rel in (f"member_{src.tag}/checkpoints.jsonl", tree.name, "global_step.json", "merge_setup.json"):
        plan.write(gs("cp", "--if-generation-match=0", logs / rel, f"{dest}/client/tinker_log/{dest_run}/{rel}"),
                   f"{arm}: client log {rel}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), help="default: all four")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="run reads, checks and local steps; print the writes")
    mode.add_argument("--go", action="store_true", help="also perform the GCS writes (create-only)")
    ap.add_argument("--stage", default=str(STAGE))
    x = ap.parse_args(argv)
    arms = x.arm or sorted(ARMS)
    plan = Plan("go" if x.go else "prepare" if x.prepare else "dry")
    stage = Path(x.stage)
    print(f"# multi-seed merge setup: arms {arms}, batch {BATCH}, new checkpoint {NEW_CKPT}, stage {stage}, "
          f"mode {plan.mode}")
    needed = sorted({k for arm in arms for k in MERGES[SOURCES[ARMS[arm][0]].tag]})
    for key in needed:
        check_source(plan, key, SOURCES[key], stage)
    for arm in arms:
        setup(plan, arm, stage, sys.executable)
    print("# then: launch each profile tpu/swarm/ray_train/profiles/<dest run>.json with assigned_launch "
          "(see tpu/merge_seeds_profiles.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
