"""The orbax CHECKPOINT_COMPLETE preflight must find a marker *object*.

Job 411 (gpt-oss 120B on v5p-32) failed because the check listed the marker's
own path with a "/**" suffix, which can never match a file. The preflight now
lists the checkpoint tree and looks for the marker among its objects.
"""
from tpu.swarm.ray_train.cache import Object
from tpu.swarm.ray_train.host import orbax_marker_present

ORBAX = "gs://bucket/skyrl-maxtext-ckpts-gptoss120b-bf16-d388/gpt-oss-120b"


def _obj(relative):
    return Object(ORBAX + "/" + relative, relative, 1, "1", "md5", None)


class FakeGCS:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def list(self, prefix, allow_empty=False):
        self.calls.append((prefix, allow_empty))
        return [o for o in self.objects if o.uri.startswith(prefix.rstrip("/") + "/")]


def test_marker_found_when_present_at_tree_root():
    gcs = FakeGCS([_obj("CHECKPOINT_COMPLETE"), _obj("0/_METADATA"), _obj("0/params/checkpoint")])
    assert orbax_marker_present(gcs, ORBAX)
    # One tree listing, tolerant of an absent tree (the caller reports that separately).
    assert gcs.calls == [(ORBAX, True)]


def test_marker_missing_or_nested_is_rejected():
    assert not orbax_marker_present(FakeGCS([_obj("0/_METADATA")]), ORBAX)
    assert not orbax_marker_present(FakeGCS([_obj("0/CHECKPOINT_COMPLETE")]), ORBAX)
    assert not orbax_marker_present(FakeGCS([]), ORBAX)


# --- private RAM cache hygiene across runs (job 411 follow-ups) -------------
import json as _json

from tpu.swarm.ray_train.cache import CacheStore


class _Events:
    def __init__(self, tmp_path):
        self.events = tmp_path / "events.jsonl"

    def list(self, prefix, allow_empty=False):
        return []


def test_restore_tree_evicts_other_models_trees(tmp_path):
    root = tmp_path / "ram"
    (root / "orbax" / "qwen3.5-27b" / "0").mkdir(parents=True)
    (root / "orbax" / "qwen3.5-27b" / "0" / "x").write_bytes(b"1")
    (root / "orbax" / "gpt-oss-120b").mkdir()
    store = CacheStore(root, _Events(tmp_path))
    store.prune_siblings("orbax/gpt-oss-120b")
    assert sorted(p.name for p in (root / "orbax").iterdir()) == ["gpt-oss-120b"]
    events = [_json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "tree_evicted" and events[-1]["tree"] == "orbax/qwen3.5-27b"


def test_scope_compile_discards_cache_when_prefix_changes(tmp_path):
    root = tmp_path / "ram"
    (root / "compile").mkdir(parents=True)
    (root / "compile" / "jit_a-cache").write_bytes(b"x")
    store = CacheStore(root, _Events(tmp_path))
    store.scope_compile("gs://b/qwen-compile")
    # First scoping of a pre-existing cache is a change of prefix (None -> qwen): cleared.
    assert not (root / "compile").exists()
    (root / "compile").mkdir()
    (root / "compile" / "jit_a-cache").write_bytes(b"x")
    store.scope_compile("gs://b/qwen-compile")
    assert (root / "compile" / "jit_a-cache").exists()
    store.scope_compile("gs://b/gptoss-compile")
    assert not (root / "compile").exists()
    assert (root / "compile.prefix").read_text() == "gs://b/gptoss-compile"
