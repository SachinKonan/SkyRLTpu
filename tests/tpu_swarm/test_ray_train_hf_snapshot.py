"""Both deployed HF mirror layouts must retain integrity and warm reuse."""
import base64
import hashlib
import json
from pathlib import Path

import pytest

from tpu.swarm.ray_train.cache import CacheStore, Object


class Mirror:
    revision = "a" * 40
    source = "gs://test/hf/models--org--model"

    def __init__(self, tmp_path, layout):
        self.events = tmp_path / "events.jsonl"
        self.layout = layout
        self.calls = []
        self.failures = 0
        self.corrupt = False
        self.manifest_error = None
        self.files = {"config.json": b"{}", "tokenizer.json": b"{}",
                      "nested/template.jinja": b"template",
                      "model.safetensors.index.json": json.dumps({
                          "weight_map": {"weight": "model-01.safetensors"}}).encode(),
                      "model-01.safetensors": b"weights"}

    def objects(self):
        prefix = self.source + ("/blobs" if self.layout == "manifest"
                                else "/snapshots/" + self.revision)
        return {name: Object(prefix + "/" + key, key, len(data), "123",
                            base64.b64encode(hashlib.md5(data).digest()).decode(), None)
                for name, data in self.files.items()
                for key in [hashlib.sha256(data).hexdigest() if self.layout == "manifest" else name]}

    def metadata(self, verb, uri, **kwargs):
        assert verb == "cat"
        if uri.endswith("/refs/main"):
            return self.revision
        assert uri.endswith("/trees/" + self.revision + ".json")
        assert kwargs["allow_empty"]
        if self.manifest_error:
            raise RuntimeError(self.manifest_error)
        if self.layout == "snapshot":
            return "[]"
        return json.dumps({"files": {name: {"lfs_sha256": obj.relative, "size": obj.size}
                                     for name, obj in self.objects().items()}})

    def list(self, prefix):
        objects = self.objects()
        assert all(obj.uri.startswith(prefix + "/") for obj in objects.values())
        return list({obj.relative: obj for obj in objects.values()}.values())

    def transfer(self, args, label, destination):
        assert label == "hf-restore"
        self.calls.append(args)
        data = {obj.uri + "#123": self.files[name] for name, obj in self.objects().items()}
        for uri in args[1:-1]:
            target = Path(args[-1]) / uri.split("#")[0].rsplit("/", 1)[1]
            target.write_bytes(b"broken" if self.corrupt else data[uri])
        if self.failures:
            self.failures -= 1
            raise RuntimeError("interrupted download")


@pytest.mark.parametrize("layout", ["manifest", "snapshot"])
@pytest.mark.parametrize("weights", [False, True])
def test_hf_layouts_validate_and_reuse(tmp_path, layout, weights):
    gcs = Mirror(tmp_path, layout)
    store = CacheStore(tmp_path / "ram", gcs)
    snapshot = store.restore_hf("gs://test/hf", "org/model", weights)
    for name, data in gcs.files.items():
        if weights or not name.endswith(".safetensors"):
            assert (snapshot / name).read_bytes() == data
        else:
            assert not (snapshot / name).exists()
    calls = len(gcs.calls)
    store.restore_hf("gs://test/hf", "org/model", weights)
    assert len(gcs.calls) == calls
    (snapshot / "config.json").write_bytes(b"XX")
    store.restore_hf("gs://test/hf", "org/model", weights)
    assert (snapshot / "config.json").read_bytes() == b"{}"
    assert len(gcs.calls) > calls


def test_snapshot_retry_and_incomplete_cleanup(tmp_path):
    gcs = Mirror(tmp_path, "snapshot")
    gcs.failures = 1
    store = CacheStore(tmp_path / "ram", gcs)
    snapshot = store.restore_hf("gs://test/hf", "org/model", True)
    assert (snapshot / "model-01.safetensors").read_bytes() == b"weights"
    store.clear("hf")
    gcs.corrupt = True
    with pytest.raises(RuntimeError, match="checksum"):
        store.restore_hf("gs://test/hf", "org/model", True)
    assert not (tmp_path / "ram/hf").exists()


@pytest.mark.parametrize("problem", ["weight", "metadata", "temporary", "access"])
def test_snapshot_does_not_bypass_missing_or_invalid_cache(tmp_path, problem):
    gcs = Mirror(tmp_path, "snapshot")
    if problem == "weight":
        del gcs.files["model-01.safetensors"]
    elif problem == "metadata":
        del gcs.files["tokenizer.json"]
    elif problem == "temporary":
        gcs.files["weights_.gstmp"] = b"partial"
    else:
        gcs.manifest_error = "403 access denied"
    with pytest.raises((RuntimeError, ValueError)):
        CacheStore(tmp_path / "ram", gcs).restore_hf("gs://test/hf", "org/model", True)
    assert not (tmp_path / "ram/hf/.complete.json").exists()
