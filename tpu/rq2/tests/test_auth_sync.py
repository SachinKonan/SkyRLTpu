"""Unit tests for the codex auth-rotation plumbing in loop.py.

Background: write_codex_home COPIES ~/.codex/auth.json into every cell. The OAuth refresh token
inside is single-use, so the first cell to refresh rotates it and every other copy 401s with
`refresh_token_reused`. That silently emptied the memory of 10 RQ2 cells.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fleet"))
import loop  # noqa: E402


@pytest.fixture
def auth(tmp_path, monkeypatch):
    src = tmp_path / "home" / ".codex" / "auth.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps({"tokens": {"refresh_token": "v1"}}))
    monkeypatch.setattr(loop, "AUTH_SRC", src)
    monkeypatch.setattr(loop, "AUTH_LOCK", tmp_path / "home" / ".codex" / ".lock")
    return src


def _home(tmp_path, name, token):
    ch = tmp_path / name
    ch.mkdir(parents=True, exist_ok=True)
    (ch / "auth.json").write_text(json.dumps({"tokens": {"refresh_token": token}}))
    return ch


def test_sync_in_pulls_current_credential(auth, tmp_path):
    ch = _home(tmp_path, "cellA", "stale")
    loop._auth_sync(ch, "in")
    assert json.loads((ch / "auth.json").read_text())["tokens"]["refresh_token"] == "v1"


def test_sync_out_publishes_a_rotation(auth, tmp_path):
    ch = _home(tmp_path, "cellA", "v2-rotated")
    loop._auth_sync(ch, "out")
    assert json.loads(auth.read_text())["tokens"]["refresh_token"] == "v2-rotated"


def test_rotation_propagates_between_cells(auth, tmp_path):
    """Cell A rotates; cell B must pick the new token up on its next sync-in."""
    a = _home(tmp_path, "cellA", "v1")
    b = _home(tmp_path, "cellB", "v1")
    (a / "auth.json").write_text(json.dumps({"tokens": {"refresh_token": "v2"}}))
    loop._auth_sync(a, "out")
    loop._auth_sync(b, "in")
    assert json.loads((b / "auth.json").read_text())["tokens"]["refresh_token"] == "v2"


def test_sync_is_a_noop_without_a_source(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "AUTH_SRC", tmp_path / "nope.json")
    ch = _home(tmp_path, "cellA", "keepme")
    loop._auth_sync(ch, "in")
    assert json.loads((ch / "auth.json").read_text())["tokens"]["refresh_token"] == "keepme"


def test_reuse_error_detection(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text('{"type":"turn.completed"}\n')
    bad = tmp_path / "bad.jsonl"
    bad.write_text('ERROR ...: {"code":"refresh_token_reused"}\n{"type":"turn.failed"}\n')
    assert loop._auth_reuse_error(good) is False
    assert loop._auth_reuse_error(bad) is True
    assert loop._auth_reuse_error(tmp_path / "missing.jsonl") is False


def test_run_codex_retries_then_gives_up(auth, tmp_path, monkeypatch):
    """A cell that keeps losing the rotation race must retry, resync, and eventually return."""
    ch = _home(tmp_path, "cellA", "v1")
    wd = tmp_path / "wd"
    wd.mkdir()
    calls = {"n": 0}

    def fake_once(model, effort, prompt, w, mcp, wall, tag, home):
        calls["n"] += 1
        (Path(w) / f"{tag}.jsonl").write_text('{"code":"refresh_token_reused"}')
        return 1

    monkeypatch.setattr(loop, "_run_codex_once", fake_once)
    monkeypatch.setattr(loop.time, "sleep", lambda *_: None)
    loop.run_codex("m", "medium", "p", wd, None, 10, "mem0", ch)
    assert calls["n"] == loop.AUTH_RETRIES


def test_run_codex_stops_retrying_once_it_succeeds(auth, tmp_path, monkeypatch):
    ch = _home(tmp_path, "cellA", "v1")
    wd = tmp_path / "wd"
    wd.mkdir()
    calls = {"n": 0}

    def fake_once(model, effort, prompt, w, mcp, wall, tag, home):
        calls["n"] += 1
        body = '{"code":"refresh_token_reused"}' if calls["n"] == 1 else '{"type":"turn.completed"}'
        (Path(w) / f"{tag}.jsonl").write_text(body)
        return 0

    monkeypatch.setattr(loop, "_run_codex_once", fake_once)
    monkeypatch.setattr(loop.time, "sleep", lambda *_: None)
    rc = loop.run_codex("m", "medium", "p", wd, None, 10, "mem0", ch)
    assert calls["n"] == 2 and rc == 0
