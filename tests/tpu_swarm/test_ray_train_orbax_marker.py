"""The orbax CHECKPOINT_COMPLETE preflight must find a marker *object*.

Job 411 (gpt-oss 120B on v5p-32) failed because the check listed the marker's
own path with a "/**" suffix, which can never match a file. The preflight now
lists the checkpoint tree and looks for the marker among its objects.
"""
from pathlib import Path
import json
import pytest

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


# --- gpt-oss trainer extra pins (job 413: drjax) ----------------------------
from tpu.swarm.ray_train.config import Config


def _cfg(preset, **trainer):
    return Config.from_dict({"run_id": "r", "accelerator": "tpu-v5p-32", "hosts": 4,
                             "bucket": "gs://b", "base_bundle": "gs://b/c.tar.gz",
                             "base_bundle_sha256": "0" * 64, "model_preset": preset,
                             "cache": {"hf": "gs://b/hf", "orbax": "gs://b/orbax",
                                       "trainer_compile": "gs://b/tc", "inference_compile": "gs://b/ic"},
                             "trainer": dict(hosts=2, tp=4, fsdp=2, process_bounds="1,1,2", **trainer),
                             "inference": {"backend": "ray_serve", "tp": 4}})


def test_gptoss_preset_pins_drjax_and_qwen_does_not():
    assert _cfg("gpt-oss-120b").trainer.extra_pins == ["drjax==0.2.1"]
    assert _cfg("qwen3.5-27b").trainer.extra_pins == []


def test_extra_pins_must_be_exact():
    with pytest.raises(ValueError, match="extra_pins"):
        _cfg("gpt-oss-120b", extra_pins=["drjax>=0.1.4"])


# --- v6e-32 (8 hosts x 4 chips, 32 GiB HBM): the validated legacy 4+4 split --
from tpu.swarm.ray_train.build import build


def test_gptoss_v6e_profile_uses_the_validated_four_host_block(tmp_path):
    import yaml
    path = Path("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json")
    cfg = Config.load(path)
    assert (cfg.accelerator, cfg.hosts, cfg.trainer.hosts, cfg.inference_hosts) == ("tpu-v6e-32", 8, 4, 4)
    assert (cfg.trainer.tp, cfg.trainer.fsdp, cfg.trainer.process_bounds) == (8, 2, "2,2,1")
    assert cfg.effective_zone == "us-east5-b"
    assert cfg.bucket.endswith("us-east5") and "us-east5" in cfg.cache.hf
    _, _, task_path = build(path, tmp_path)
    task = yaml.safe_load(task_path.read_text())
    assert task["resources"]["zone"] == "us-east5-b"
    assert task["resources"]["accelerator_args"]["runtime_version"] == "v2-alpha-tpuv6e"
    assert task["resources"]["accelerators"] == "tpu-v6e-32"


def test_v6e_rejects_other_trainer_splits_and_unknown_zones():
    raw = json.loads(Path("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json").read_text())
    bad = dict(raw); bad["trainer"] = dict(raw["trainer"], hosts=2, tp=4, fsdp=2, process_bounds="1,1,2")
    with pytest.raises(ValueError, match="v6e-32"):
        Config.from_dict(bad)
    bad = dict(raw); bad["zone"] = "us-east5-a"
    with pytest.raises(ValueError, match="zone"):
        Config.from_dict(bad)
    ok = dict(raw); ok["zone"] = "asia-northeast1-b"
    assert Config.from_dict(ok).effective_zone == "asia-northeast1-b"
    default = dict(raw); default.pop("zone")
    assert Config.from_dict(default).effective_zone == "asia-northeast1-b"


# --- pipeline-parallel engine pairs on the executor's Ray cluster (v6e-32) ----
from tpu.swarm.ray_train.commands import inference_command, inference_environment


def _v6e():
    return Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json")


def test_v6e_profile_pairs_hosts_into_two_engines():
    cfg = _v6e()
    assert (cfg.inference.hosts_per_engine, cfg.inference.tp, cfg.inference_hosts) == (2, 4, 4)
    ips = ["10.0.0.5", "10.0.0.6", "10.0.0.7", "10.0.0.8"]
    assert cfg.engine_groups(ips) == [["10.0.0.5", "10.0.0.6"], ["10.0.0.7", "10.0.0.8"]]


def test_pair_engine_command_and_environment(tmp_path):
    cfg = _v6e()
    group = ["10.0.0.5", "10.0.0.6"]
    cmd = inference_command(cfg, tmp_path, tmp_path / "src", tmp_path / "snap", tmp_path / "run", group=group)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert cmd[cmd.index("--pipeline-parallel-size") + 1] == "2"
    assert cmd[cmd.index("--distributed-executor-backend") + 1] == "ray"
    assert cmd[cmd.index("--skyrl-ray-placement-hosts") + 1] == "10.0.0.5,10.0.0.6"
    env = inference_environment(cfg, tmp_path, tmp_path / "run", head="10.0.0.1", group=group)
    assert env["TPU_MULTIHOST_BACKEND"] == "ray" and env["VLLM_USE_RAY_EXECUTOR"] == "1"
    assert env["RAY_ADDRESS"] == "10.0.0.1:19679" and env["SKYRL_RAY_PLACEMENT_HOSTS"] == "10.0.0.5,10.0.0.6"
    # tpu-inference sets the per-host TPU process variables itself in Ray mode.
    assert not any(k in env for k in ("TPU_PROCESS_BOUNDS", "TPU_PROCESS_ADDRESSES", "TPU_VISIBLE_CHIPS", "CLOUD_TPU_TASK_ID"))
    assert env["MOE_REQUANTIZE_WEIGHT_DTYPE"] == "fp8"
    # Single-host engines are untouched.
    single = inference_environment(cfg, tmp_path, tmp_path / "run", head="10.0.0.1", group=["10.0.0.5"])
    assert single["TPU_PROCESS_BOUNDS"] == "1,1,1" and "TPU_MULTIHOST_BACKEND" not in single
    assert "--pipeline-parallel-size" not in inference_command(cfg, tmp_path, tmp_path / "src", tmp_path / "snap", tmp_path / "run")


def test_hosts_per_engine_validation():
    raw = json.loads(Path("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json").read_text())
    bad = dict(raw); bad["inference"] = dict(raw["inference"], hosts_per_engine=3)
    with pytest.raises(ValueError, match="hosts_per_engine"):
        Config.from_dict(bad)
    v5p = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v5p_32_grpo.json")
    assert v5p.inference.hosts_per_engine == 1 and v5p.engine_groups(["a", "b"]) == [["a"], ["b"]]


def test_deploy_creates_one_pinned_engine_per_pair(monkeypatch):
    pytest.importorskip("ray.serve")
    from unittest.mock import Mock
    from tpu.swarm.ray_train import serving
    cfg = _v6e()
    engine, ingress, serve = Mock(), Mock(), Mock()
    serve.run_many.return_value = ["handle"]
    monkeypatch.setattr(serving, "Engine", engine)
    monkeypatch.setattr(serving, "Ingress", ingress)
    monkeypatch.setattr(serving, "serve", serve)
    prepared = {}
    for ip, group in (("10.0.0.5", ["10.0.0.5", "10.0.0.6"]), ("10.0.0.6", ["10.0.0.5", "10.0.0.6"]),
                      ("10.0.0.7", ["10.0.0.7", "10.0.0.8"]), ("10.0.0.8", ["10.0.0.7", "10.0.0.8"])):
        prepared[ip] = {"role": "inference", "group": group}
    assert serving.deploy(cfg, prepared, Mock(), "10.0.0.1") == "handle"
    calls = engine.options.call_args_list
    assert len(calls) == 2
    for call, head in zip(calls, ("10.0.0.5", "10.0.0.7")):
        opts = call.kwargs
        assert opts["num_replicas"] == 1
        assert opts["ray_actor_options"] == {"num_cpus": 8, "resources": {f"node:{head}": 0.01}}
    # Ingress addresses the two engine heads only.
    assert ingress.options.return_value.bind.call_args.args[-1] == ["10.0.0.5", "10.0.0.7"]
