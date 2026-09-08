import base64
import hashlib
from pathlib import Path
import zlib

import pytest

from tpu.swarm.ray_train.cache import (
    CacheStore, Object, completed, complete_compile_file, objects_from_listing, safe_relative, valid_file,
)
from tpu.swarm.ray_train.config import Config


def config_dict():
    return dict(run_id="test-v5p", accelerator="tpu-v5p-32", hosts=4,
                bucket="gs://test", base_bundle="gs://test/code/base.tar.gz",
                base_bundle_sha256="a" * 64,
                cache=dict(hf="gs://test/hf", orbax="gs://test/orbax",
                           trainer_compile="gs://test/train", inference_compile="gs://test/infer"))


def test_default_profile_is_the_legacy_v5p_cell_split():
    config = Config.from_dict(config_dict())
    assert (config.trainer.hosts, config.inference_hosts) == (1, 3)
    assert (config.trainer.tp, config.trainer.fsdp, config.trainer.remat) == (1, 4, "full")
    # Defaults follow the legacy v5p-32 qwen cell since 2026-09-08: 128 sequences.
    assert (config.inference.max_sequences, config.inference.memory_utilization) == (128, 0.9)
    assert Config.from_dict(config.to_dict()) == config


def test_v5p_profile_reuses_controller_contract():
    raw = config_dict()
    raw.update(accelerator="tpu-v5p-32", hosts=4,
               trainer=dict(hosts=1, tp=4, fsdp=1, process_bounds="1,1,1", logical_kv_heads=4))
    config = Config.from_dict(raw)
    assert config.inference_hosts == 3


@pytest.mark.parametrize("field,value", [("ray", 6380), ("ray", 16379), ("engine", 42001),
                                         ("engine", 19800), ("worker_max", 41999)])
def test_reject_conflicting_ports(field, value):
    raw = config_dict()
    raw["ports"] = {field: value}
    with pytest.raises(ValueError):
        Config.from_dict(raw)


@pytest.mark.parametrize("field,value", [("trainer_gib", 2), ("reserve_gib", 0),
                                         ("thread_count", 0), ("sync_seconds", 0)])
def test_reject_unsafe_memory_and_concurrency(field, value):
    raw = config_dict()
    raw["cache"][field] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


@pytest.mark.parametrize("name", ["../other", "/home/user", "a/../../b", "", "a\nb", "a\\b"])
def test_reject_cache_traversal(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_orbax_hidden_metadata_is_not_transfer_debris():
    assert completed("params/.zarray")
    assert not completed("params/0_.gstmp")
    assert not completed("params/staging.partial")


def object_for(data, name="step-cache"):
    return Object("gs://test/compile/" + name, name, len(data), "123",
                  base64.b64encode(hashlib.md5(data).digest()).decode(), None)


def compile_bytes(payload=b"executable"):
    return zlib.compress((12).to_bytes(4, "big") + payload)


def test_checksum_rejects_same_size_corruption(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"wrong")
    assert not valid_file(path, object_for(b"right"))
    path.write_bytes(b"right")
    assert valid_file(path, object_for(b"right"))


def test_cleanup_stays_inside_owned_namespace(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    other = tmp_path / "other-agent"
    other.mkdir()
    (other / "weights").write_text("keep")
    (private / "hf").symlink_to(other)
    with pytest.raises(ValueError):
        CacheStore(private, None).clear("hf")
    assert (other / "weights").read_text() == "keep"


def test_compile_upload_is_additive_and_verifies_completion(tmp_path):
    cache = tmp_path / "compile"
    cache.mkdir()
    (cache / "old-cache").write_bytes(b"old")
    new = compile_bytes(b"new")
    (cache / "new-cache").write_bytes(new)
    (cache / "bad-cache_.gstmp").write_bytes(b"partial")
    (cache / "other-atime").write_bytes(b"123")

    class GCS:
        events = tmp_path / "events.jsonl"
        calls = []
        remote = [object_for(b"old", "old-cache")]

        def list(self, prefix, **kwargs):
            return self.remote

        def transfer(self, args, label, **kwargs):
            self.calls.append(args)
            assert args[0:2] == ["cp", "--no-clobber"]
            assert args[2:-1] == [str(tmp_path / "compile-upload/new-cache")]
            assert Path(args[2]).read_bytes() == new
            self.remote.append(object_for(new, "new-cache"))

    gcs = GCS()
    store = CacheStore(tmp_path, gcs)
    assert store.publish_compile("gs://test/compile") == 1
    assert store.publish_compile("gs://test/compile") == 0
    assert len(gcs.calls) == 1
    assert (cache / "old-cache").read_bytes() == b"old"
    assert not (tmp_path / "compile-upload").exists()


def test_compile_upload_failure_does_not_claim_durability(tmp_path):
    (tmp_path / "compile").mkdir()
    (tmp_path / "compile/step-cache").write_bytes(compile_bytes())

    class GCS:
        events = tmp_path / "events.jsonl"

        def list(self, *args, **kwargs):
            return []

        def transfer(self, *args, **kwargs):
            pass

    with pytest.raises(RuntimeError, match="verification failed"):
        CacheStore(tmp_path, GCS()).publish_compile("gs://test/compile")


@pytest.mark.parametrize("codec", ["zlib", "zstd"])
def test_compile_frame_must_be_complete(tmp_path, codec):
    data = (3).to_bytes(4, "big") + b"executable" * 1000
    if codec == "zstd":
        import zstandard
        encoded = zstandard.ZstdCompressor().compress(data)
    else:
        encoded = zlib.compress(data)
    path = tmp_path / "step-cache"
    path.write_bytes(encoded)
    assert complete_compile_file(path)
    for broken in (encoded[:-1], encoded[:4], encoded + b"trailing", b""):
        path.write_bytes(broken)
        assert not complete_compile_file(path)


def test_cold_cache_upload_retry_and_warm_restore(tmp_path):
    class GCS:
        events = tmp_path / "events.jsonl"
        data = {}
        fail = True

        def list(self, *args, **kwargs):
            return [object_for(data, name) for name, data in self.data.items()]

        def transfer(self, args, label, destination=None, **kwargs):
            if label == "compile-publish":
                if self.fail:
                    self.fail = False
                    raise RuntimeError("temporary upload failure")
                for source in args[2:-1]:
                    path = Path(source)
                    self.data.setdefault(path.name, path.read_bytes())
            else:
                assert label == "compile-restore"
                for source in args[1:-1]:
                    name = source.split("#")[0].rsplit("/", 1)[1]
                    (Path(args[-1]) / name).write_bytes(self.data[name])

    gcs = GCS()
    store = CacheStore(tmp_path, gcs)
    cache = store.restore_compile("gs://test/compile")
    assert list(cache.iterdir()) == []
    payload = compile_bytes()
    (cache / "step-cache").write_bytes(payload)
    (cache / "incomplete-cache").write_bytes(payload[:-1])
    with pytest.raises(RuntimeError, match="temporary upload failure"):
        store.publish_compile("gs://test/compile")
    assert (cache / "step-cache").read_bytes() == payload
    assert not (tmp_path / "compile-upload").exists()
    assert store.publish_compile("gs://test/compile") == 1
    assert set(gcs.data) == {"step-cache"}
    # An unfinished frame is deferred, not removed; a later completed write uploads.
    (cache / "incomplete-cache").write_bytes(payload)
    assert store.publish_compile("gs://test/compile") == 1
    warm = CacheStore(tmp_path / "another-host", gcs).restore_compile("gs://test/compile")
    assert (warm / "step-cache").read_bytes() == payload
    assert (warm / "incomplete-cache").read_bytes() == payload


def test_upload_snapshot_is_independent_of_live_cache(tmp_path):
    cache = tmp_path / "compile"
    cache.mkdir()
    payload = compile_bytes()
    (cache / "step-cache").write_bytes(payload)

    class GCS:
        events = tmp_path / "events.jsonl"
        remote = []

        def list(self, *args, **kwargs):
            return self.remote

        def transfer(self, args, *unused, **kwargs):
            (cache / "step-cache").write_bytes(b"concurrent writer")
            assert Path(args[2]).read_bytes() == payload
            self.remote = [object_for(Path(args[2]).read_bytes())]

    assert CacheStore(tmp_path, GCS()).publish_compile("gs://test/compile") == 1


def test_concurrent_compile_upload_accepts_only_verified_remote_copy(tmp_path):
    cache = tmp_path / "compile"
    cache.mkdir()
    payload = compile_bytes()
    (cache / "step-cache").write_bytes(payload)

    class GCS:
        events = tmp_path / "events.jsonl"
        remote = []
        corrupt = False

        def list(self, *args, **kwargs):
            return self.remote

        def transfer(self, *args, **kwargs):
            self.remote = [object_for(b"wrong" if self.corrupt else payload)]
            raise RuntimeError("HTTPError 412 precondition failed")

    gcs = GCS()
    assert CacheStore(tmp_path, gcs).publish_compile("gs://test/compile") == 1
    gcs.remote, gcs.corrupt = [], True
    with pytest.raises(RuntimeError, match="412"):
        CacheStore(tmp_path, gcs).publish_compile("gs://test/compile")


def test_listing_requires_integrity_and_prefix():
    listing = [{"type": "cloud_object", "metadata": dict(bucket="test", name="compile/x-cache",
                size="3", generation="12", crc32c="AAAAAA==")}]
    assert objects_from_listing(listing, "gs://test/compile")[0].size == 3
    with pytest.raises(ValueError):
        objects_from_listing(listing, "gs://test/wrong")
    del listing[0]["metadata"]["crc32c"]
    with pytest.raises(ValueError):
        objects_from_listing(listing, "gs://test/compile")
