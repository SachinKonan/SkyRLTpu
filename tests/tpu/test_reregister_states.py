"""Unit tests for tpu/reregister_states.py -- the resume-after-preemption linchpin.

A preempted slice keeps its checkpoint tarballs (the tinker server's
--checkpoints-base is a gcsfuse mount) but loses the per-VM sqlite registry, so
create_training_client_from_state 404s and the member silently restarts from
base weights. reregister_states.py re-inserts the registry rows. These tests
pin the properties the durability story depends on:

  * only states whose tarball actually exists are registered
  * running it twice is a no-op (it runs on every launch)
  * SAMPLER rows appear only when sampler weights exist
  * a malformed jsonl line never aborts the run
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tpu" / "reregister_states.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reregister_states", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load_module()


# The columns the script INSERTs into; mirrors the live tinker.db schema.
SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY, tags TEXT, user_metadata TEXT,
    sdk_version TEXT, status TEXT, created_at TEXT, heartbeat_count INTEGER
);
CREATE TABLE models (
    model_id TEXT PRIMARY KEY, base_model TEXT, lora_config TEXT, status TEXT,
    request_id INTEGER, session_id TEXT, created_at TEXT
);
CREATE TABLE checkpoints (
    model_id TEXT, checkpoint_id TEXT, checkpoint_type TEXT, status TEXT,
    created_at TEXT, completed_at TEXT,
    PRIMARY KEY (model_id, checkpoint_id, checkpoint_type)
);
"""


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "tinker.db"
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    return p


@pytest.fixture()
def ckpt_root(tmp_path: Path) -> Path:
    return tmp_path / "skyrl-checkpoints"


def write_jsonl(path: Path, state_paths: list[str], extra_lines: list[str] = ()) -> Path:
    with open(path, "w") as f:
        for sp in state_paths:
            f.write(json.dumps({"name": "step", "state_path": sp}) + "\n")
        for line in extra_lines:
            f.write(line + "\n")
    return path


def make_tarball(root: Path, model_id: str, ckpt: str, sampler: bool = False) -> None:
    (root / model_id).mkdir(parents=True, exist_ok=True)
    (root / model_id / f"{ckpt}.tar.gz").write_bytes(b"x")
    if sampler:
        (root / model_id / "sampler_weights").mkdir(parents=True, exist_ok=True)
        (root / model_id / "sampler_weights" / f"{ckpt}.tar.gz").write_bytes(b"x")


def run(monkeypatch, db: Path, root: Path, jsonl: Path, base_model="Qwen/Qwen3.5-27B"):
    argv = ["reregister_states.py", "--db", str(db), "--ckpt-root", str(root),
            "--base-model", base_model, "--jsonl", str(jsonl)]
    monkeypatch.setattr(sys, "argv", argv)
    rr.main()


def counts(db: Path) -> tuple[int, int]:
    con = sqlite3.connect(db)
    m = con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    c = con.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    con.close()
    return m, c


# --------------------------------------------------------------------------
# entries_from_jsonl
# --------------------------------------------------------------------------

def test_parses_model_and_checkpoint_ids(tmp_path: Path):
    p = write_jsonl(tmp_path / "c.jsonl", [
        "tinker://model_c0ca4366/weights/000006",
        "tinker://model_7446f2bc/weights/000012",
    ])
    assert rr.entries_from_jsonl(p) == [
        ("model_c0ca4366", "000006"),
        ("model_7446f2bc", "000012"),
    ]


def test_malformed_lines_are_skipped_not_fatal(tmp_path: Path):
    """A truncated final line is normal -- the sidecar can copy mid-write."""
    p = write_jsonl(
        tmp_path / "c.jsonl",
        ["tinker://model_aaaa1111/weights/000001"],
        extra_lines=['{"name": "partial", "state_pa', "", "not json at all"],
    )
    assert rr.entries_from_jsonl(p) == [("model_aaaa1111", "000001")]


def test_rows_without_state_path_are_ignored(tmp_path: Path):
    """Sampler-only rows share the file and must not be treated as resumable."""
    p = tmp_path / "c.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"name": "sampler-only"}) + "\n")
        f.write(json.dumps({"name": "s", "state_path": "tinker://model_b/weights/000002"}) + "\n")
    assert rr.entries_from_jsonl(p) == [("model_b", "000002")]


# --------------------------------------------------------------------------
# registration behaviour
# --------------------------------------------------------------------------

def test_registers_only_states_whose_tarball_exists(monkeypatch, db_path, ckpt_root, tmp_path):
    """A registry row pointing at a missing tarball would 404 at resume time --
    worse than no row, because it looks like a healthy lineage."""
    make_tarball(ckpt_root, "model_aaa11111", "000003")
    jsonl = write_jsonl(tmp_path / "c.jsonl", [
        "tinker://model_aaa11111/weights/000003",
        "tinker://model_bbb22222/weights/000004",
    ])
    run(monkeypatch, db_path, ckpt_root, jsonl)

    con = sqlite3.connect(db_path)
    models = {r[0] for r in con.execute("SELECT model_id FROM models")}
    con.close()
    assert models == {"model_aaa11111"}


def test_is_idempotent(monkeypatch, db_path, ckpt_root, tmp_path):
    """Runs on every launch; a second run must not duplicate or error."""
    make_tarball(ckpt_root, "model_ccc33333", "000001")
    make_tarball(ckpt_root, "model_ccc33333", "000002")
    jsonl = write_jsonl(tmp_path / "c.jsonl", [
        "tinker://model_ccc33333/weights/000001",
        "tinker://model_ccc33333/weights/000002",
    ])
    run(monkeypatch, db_path, ckpt_root, jsonl)
    first = counts(db_path)
    run(monkeypatch, db_path, ckpt_root, jsonl)
    assert counts(db_path) == first == (1, 2)


def test_sampler_row_only_when_sampler_weights_present(monkeypatch, db_path, ckpt_root, tmp_path):
    make_tarball(ckpt_root, "model_ddd44444", "000001", sampler=False)
    make_tarball(ckpt_root, "model_eee55555", "000001", sampler=True)
    jsonl = write_jsonl(tmp_path / "c.jsonl", [
        "tinker://model_ddd44444/weights/000001",
        "tinker://model_eee55555/weights/000001",
    ])
    run(monkeypatch, db_path, ckpt_root, jsonl)

    con = sqlite3.connect(db_path)
    rows = set(con.execute("SELECT model_id, checkpoint_type FROM checkpoints"))
    con.close()
    assert ("model_ddd44444", "TRAINING") in rows
    assert ("model_ddd44444", "SAMPLER") not in rows
    assert ("model_eee55555", "TRAINING") in rows
    assert ("model_eee55555", "SAMPLER") in rows


def test_preserves_existing_rows(monkeypatch, db_path, ckpt_root, tmp_path):
    """INSERT OR IGNORE must never clobber a live server's own registration."""
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO models (model_id, base_model, lora_config, status, request_id,"
        " session_id, created_at) VALUES ('model_fff66666','ORIGINAL','{}','created',0,'sess','t')"
    )
    con.commit()
    con.close()

    make_tarball(ckpt_root, "model_fff66666", "000001")
    jsonl = write_jsonl(tmp_path / "c.jsonl", ["tinker://model_fff66666/weights/000001"])
    run(monkeypatch, db_path, ckpt_root, jsonl, base_model="SOMETHING/ELSE")

    con = sqlite3.connect(db_path)
    base = con.execute("SELECT base_model FROM models WHERE model_id='model_fff66666'").fetchone()[0]
    con.close()
    assert base == "ORIGINAL"


def test_empty_jsonl_is_a_noop(monkeypatch, db_path, ckpt_root, tmp_path):
    """An arm with no prior lineage is legitimate -- must not raise."""
    jsonl = write_jsonl(tmp_path / "c.jsonl", [])
    run(monkeypatch, db_path, ckpt_root, jsonl)
    assert counts(db_path) == (0, 0)
