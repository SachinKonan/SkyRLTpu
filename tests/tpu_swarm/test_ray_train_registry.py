"""Durable Tinker registry: WAL-folded backups and checkpoint re-registration (jobs 578/596)."""
import json
import shutil
import sqlite3

from tpu.swarm.ray_train.registry import backup_database, harvest_entries, register_rows

SCHEMA = [
    "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, tags TEXT, user_metadata TEXT, sdk_version TEXT,"
    " status TEXT, created_at TEXT, last_heartbeat_at TEXT, heartbeat_count INTEGER)",
    "CREATE TABLE models (model_id TEXT PRIMARY KEY, base_model TEXT, lora_config TEXT, status TEXT,"
    " request_id INTEGER, session_id TEXT, created_at TEXT)",
    "CREATE TABLE checkpoints (model_id TEXT, checkpoint_id TEXT, checkpoint_type TEXT, status TEXT,"
    " created_at TEXT, completed_at TEXT, error_message TEXT, PRIMARY KEY (model_id, checkpoint_id, checkpoint_type))",
]


def _db(path, wal=True):
    c = sqlite3.connect(path)
    if wal:
        c.execute("PRAGMA journal_mode=WAL")
    for stmt in SCHEMA:
        c.execute(stmt)
    c.commit()
    return c


def test_backup_folds_wal_into_main_file(tmp_path):
    live = _db(tmp_path / "tinker.db")
    live.execute("INSERT INTO models VALUES ('model_8b4c841f','openai/gpt-oss-120b','{}','created',0,'s','t')")
    live.commit()  # rows sit in tinker.db-wal; the live server keeps the connection open
    backup = tmp_path / "tinker-backup.db"
    backup_database(tmp_path / "tinker.db", backup)
    assert not (tmp_path / "tinker-backup.db-wal").exists()
    # what the writeback uploads is the main file alone
    shutil.copy(backup, tmp_path / "uploaded.db")
    got = sqlite3.connect(tmp_path / "uploaded.db").execute("SELECT model_id FROM models").fetchall()
    assert got == [("model_8b4c841f",)]
    live.close()


def test_harvest_entries_reads_state_paths(tmp_path):
    log = tmp_path / "checkpoints.jsonl"
    log.write_text("\n".join([
        json.dumps({"name": "000001", "state_path": "tinker://model_8b4c841f/weights/000001",
                    "sampler_path": "tinker://model_8b4c841f/000001"}),
        "not json",
        json.dumps({"name": "000002", "state_path": "tinker://model_8b4c841f/weights/000002"}),
    ]))
    assert harvest_entries([log]) == [("model_8b4c841f", "000001"), ("model_8b4c841f", "000002")]


def test_register_rows_only_for_present_tarballs(tmp_path):
    db = tmp_path / "tinker.db"
    _db(db, wal=False).close()
    present_set = {("model_8b4c841f", "000001", ""), ("model_8b4c841f", "000001", "sampler_weights")}
    registered = register_rows(db, [("model_8b4c841f", "000001"), ("model_8b4c841f", "000002")],
                               "openai/gpt-oss-120b", {"rank": 32}, lambda m, c, s: (m, c, s) in present_set)
    assert registered == ["model_8b4c841f/000001 TRAINING", "model_8b4c841f/000001 SAMPLER"]
    c = sqlite3.connect(db)
    assert c.execute("SELECT model_id, base_model FROM models").fetchall() == [("model_8b4c841f", "openai/gpt-oss-120b")]
    assert sorted(c.execute("SELECT checkpoint_id, checkpoint_type, status FROM checkpoints").fetchall()) == [
        ("000001", "SAMPLER", "COMPLETED"), ("000001", "TRAINING", "COMPLETED")]
    # idempotent
    assert register_rows(db, [("model_8b4c841f", "000001")], "x", {}, lambda *a: True) == [
        "model_8b4c841f/000001 TRAINING", "model_8b4c841f/000001 SAMPLER"]
    assert c.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 2


def test_ray_cpus_profile_field():
    from tpu.swarm.ray_train.config import Config
    assert Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json").ray_cpus == 32
    assert Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v5p_32_grpo.json").ray_cpus == 120
