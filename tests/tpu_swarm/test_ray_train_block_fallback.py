"""CPU-only: a rejected trainer block must not leave probes running on its
hosts, or the next candidate is refused with `owned process already active`
(job 562, v6e east5b, 2026-09-09)."""
from types import SimpleNamespace

import pytest

pytest.importorskip("ray.serve")
from tpu.swarm.ray_train import controller as module
from tpu.swarm.ray_train.config import Config


def controller(tmp_path, outcomes):
    cfg = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json").to_dict()
    cfg["root"] = str(tmp_path)
    instance = module.Controller(Config.from_dict(cfg), [f"10.0.0.{i}" for i in range(8)])
    instance.events = []
    instance.report = lambda event, **fields: instance.events.append((event, fields))
    instance.calls = []

    def host(rank):
        def probe(ranks, port, subset):
            ref = ("probe", rank, tuple(ranks))
            instance.calls.append(ref)
            return ref

        def stop_process(name):
            ref = ("stop", rank, name)
            instance.calls.append(ref)
            return ref
        return SimpleNamespace(probe=SimpleNamespace(remote=probe),
                               stop_process=SimpleNamespace(remote=stop_process))

    instance.hosts = [host(r) for r in range(8)]

    def get(refs, timeout=None):
        if isinstance(refs, list):
            return [get(r) for r in refs]
        if refs[0] == "probe":
            result = outcomes.get(refs[1:], "ok")
            if result != "ok":
                raise RuntimeError(result)
        return result if refs[0] == "probe" else 0

    module_ray = SimpleNamespace(
        get=get,
        wait=lambda refs, num_returns=1, timeout=None: (list(refs)[:num_returns], list(refs)[num_returns:]),
        exceptions=module.ray.exceptions)
    return instance, module_ray


def test_rejected_block_probes_are_stopped_before_next_candidate(tmp_path, monkeypatch):
    first, second = ((0, 5, 3, 4), (1, 2, 6, 7)), ((1, 2, 6, 7), (0, 5, 3, 4))
    outcomes = {(0, (0, 5, 3, 4)): "topology-subset exited 250"}
    instance, fake_ray = controller(tmp_path, outcomes)
    monkeypatch.setattr(module, "ray", fake_ray)
    instance.stopping = SimpleNamespace(is_set=lambda: False)
    instance.failure = None

    chosen = instance.validated_block([(list(first[0]), list(first[1])), (list(second[0]), list(second[1]))])

    assert chosen == (list(second[0]), list(second[1]))
    stops = [c for c in instance.calls if c[0] == "stop"]
    assert sorted(c[1] for c in stops) == [0, 3, 4, 5]
    assert all(c[2] == "topology-subset" for c in stops)
    probes = [c for c in instance.calls if c[0] == "probe"]
    # the second candidate's probes are issued only after the stops
    assert instance.calls.index(stops[-1]) < instance.calls.index(probes[4])
    assert [e for e, _ in instance.events] == ["topology_block_rejected"]


def test_all_blocks_rejected_raises_with_every_detail(tmp_path, monkeypatch):
    outcomes = {(0, (0, 5, 3, 4)): "boom-a", (1, (1, 2, 6, 7)): "boom-b"}
    instance, fake_ray = controller(tmp_path, outcomes)
    monkeypatch.setattr(module, "ray", fake_ray)
    instance.stopping = SimpleNamespace(is_set=lambda: False)
    instance.failure = None
    with pytest.raises(RuntimeError, match="boom-a.*boom-b"):
        instance.validated_block([([0, 5, 3, 4], [1, 2, 6, 7]), ([1, 2, 6, 7], [0, 5, 3, 4])])
    assert len([c for c in instance.calls if c[0] == "stop"]) == 8


def test_readiness_counts_engines_not_hosts():
    """Pair engines register one Serve replica per two hosts (job 563 sat at the
    readiness gate with 2 replicas while the gate wanted 4)."""
    v6e = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json")
    assert (v6e.inference_hosts, v6e.inference.hosts_per_engine, v6e.engine_count) == (4, 2, 2)
    v5p = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v5p_32_grpo.json")
    assert (v5p.inference_hosts, v5p.engine_count) == (2, 2)



def test_pair_engine_command_disables_async_scheduling(tmp_path):
    """Async scheduling only works when every rank sees the sampled tokens;
    the first stage of a two-host engine does not (job 568)."""
    from tpu.swarm.ray_train.commands import inference_command
    cfg = Config.load("tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_grpo.json")
    root = tmp_path
    pair = inference_command(cfg, root, root / "source", root / "model", root / "run", group=["10.0.0.3", "10.0.0.4"])
    assert pair[pair.index("--pipeline-parallel-size") + 1] == "2"
    assert "--no-async-scheduling" in pair
    single = inference_command(cfg, root, root / "source", root / "model", root / "run")
    assert "--no-async-scheduling" not in single and "--pipeline-parallel-size" not in single
