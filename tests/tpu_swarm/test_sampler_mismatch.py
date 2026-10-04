"""Trainer vs sampler logprob mismatch: backend summary, farm attribution, parity tool."""
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from skyrl.backends import sampler_mismatch as backend

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / 'tpu/swarm/ray_train/profiles/remote-only-v5p32-qwen-ac2-pilot-20260924.json'


def test_backend_summary_counts_only_sampled_tokens():
    train = [-1.0, -0.50, -2.0, -0.3, -7.0]
    sampler = [0.0, -0.52, -1.9, -1e4, -7.0]  # prompt, sampled, sampled, vLLM NaN sentinel, sampled
    s = np.asarray(backend.summarize(train, sampler), dtype=np.float64)
    assert s.shape == (backend.LENGTH,) and s[0] == backend.VERSION
    d = np.array([0.02, -0.1, 0.0])
    assert s[1] == 3
    assert s[2] == pytest.approx(d.sum(), abs=1e-6)
    assert s[3] == pytest.approx(np.abs(d).sum(), abs=1e-6)
    assert s[4] == pytest.approx(np.square(d).sum(), abs=1e-6)
    assert s[5] == pytest.approx(0.1, abs=1e-6)
    assert (s[6], s[7]) == (pytest.approx(-0.1, abs=1e-6), pytest.approx(0.02, abs=1e-6))
    assert s[8] == pytest.approx((np.expm1(d) - d).sum(), abs=1e-6)
    histogram = s[len(backend.FIELDS):]
    assert histogram.sum() == 3
    assert histogram[0] == 1  # |d| = 0 lands below 1e-4
    assert histogram[backend.ABS_EDGES.index(1e-2) + 1] == 1  # 0.02 in [1e-2, 3e-2)
    assert histogram[backend.ABS_EDGES.index(1e-1) + 1] == 1  # 0.1 in [0.1, 0.3)
    empty = backend.summarize([-1.0], [0.0])
    assert empty[0] == backend.VERSION and not any(empty[1:])
    assert backend.summarize(None, [0.5])[1] == 0


def _trajectory(prompt, action, logprobs, served_by):
    import tinker
    from ttt_discover.rl.types import Trajectory, Transition
    from ttt_discover.tinker_utils.completers import TokensWithLogprobs
    ob = tinker.ModelInput.from_ints(prompt)
    ac = TokensWithLogprobs(tokens=action, maybe_logprobs=logprobs, served_by=served_by)
    return Trajectory(transitions=[Transition(ob=ob, ac=ac, reward=1.0, episode_done=True)],
                      final_ob=tinker.ModelInput.from_ints(prompt + action))


def test_client_attributes_each_datum_to_its_farm(monkeypatch):
    pytest.importorskip('tinker')
    from ttt_discover.rl import sampler_mismatch as client
    from ttt_discover.rl.data_processing import trajectory_to_data
    assert (client.VERSION, client.ABS_EDGES, client.LENGTH) == (backend.VERSION, backend.ABS_EDGES, backend.LENGTH)
    v5p = dict(run_id='farm-v5p', accelerator='tpu-v5p-32', tp=4, engine='10.0.0.1:19801')
    v6e = dict(run_id='farm-v6e', accelerator='tpu-v6e-8', tp=8, engine='10.0.0.2')
    trajectories = [_trajectory([1, 2, 3], [10, 11], [-0.5, -0.25], v5p),
                    _trajectory([1, 2, 4], [12, 13, 14], [-1.0, -0.5, -2.0], v6e),
                    _trajectory([1, 2, 5], [15], [-0.75], None)]
    group = SimpleNamespace(trajectories_G=trajectories)
    data = [d for t in trajectories for d in trajectory_to_data(t, 1.0)]
    routes = client.datum_routes([group], data)
    assert routes == [('farm-v5p', 'tpu-v5p-32-tp4'), ('farm-v6e', 'tpu-v6e-8-tp8'), ('unknown', 'unknown')]

    # The trainer's logprobs differ from each farm's by a known offset.
    offsets = {0: 0.01, 1: -0.2, 2: 0.0}
    outputs = []
    for i, datum in enumerate(data):
        sampled = datum.loss_fn_inputs['logprobs'].to_torch().tolist()
        train = [v + offsets[i] if v else -3.0 for v in sampled]
        summary = backend.summarize(train, sampled)
        outputs.append({'sampler_mismatch': SimpleNamespace(to_torch=lambda s=summary: __import__('torch').tensor(s))})
    future = object()
    client.track(future, routes, 'scope-a')
    client.settle(object(), outputs)  # an untracked future is ignored
    client.settle(future, outputs)
    metrics = client.drain('scope-a')
    assert metrics['sampler_mismatch/farm/farm-v5p/tokens'] == 2
    assert metrics['sampler_mismatch/farm/farm-v5p/mean_diff'] == pytest.approx(0.01, abs=1e-6)
    assert metrics['sampler_mismatch/hw/tpu-v6e-8-tp8/tokens'] == 3
    assert metrics['sampler_mismatch/hw/tpu-v6e-8-tp8/mean_abs'] == pytest.approx(0.2, abs=1e-6)
    assert metrics['sampler_mismatch/hw/tpu-v6e-8-tp8/frac_abs_ge_0.1'] == 1.0
    assert metrics['sampler_mismatch/hw/tpu-v6e-8-tp8/ratio_min'] == pytest.approx(math.exp(-0.2), rel=1e-5)
    assert metrics['sampler_mismatch/farm/unknown/tokens'] == 1
    assert metrics['sampler_mismatch/all/tokens'] == 6
    assert metrics['sampler_mismatch/all/max_abs'] == pytest.approx(0.2, abs=1e-6)
    assert metrics['sampler_mismatch/all/p99_abs_le'] == pytest.approx(0.3)
    assert client.drain('scope-a') == {}

    # Older backends return no summary; that is counted, never raised.
    future = object()
    client.track(future, routes, 'scope-b')
    client.settle(future, [{'logprobs': None}] * len(routes))
    assert client.drain('scope-b') == {'sampler_mismatch/datums_without_summary': 3.0}


def test_mixed_and_ambiguous_routes():
    pytest.importorskip('tinker')
    from ttt_discover.rl import sampler_mismatch as client
    from ttt_discover.rl.data_processing import trajectory_to_data
    a = _trajectory([1, 2], [3], [-0.1], dict(run_id='a', accelerator='x', tp=4))
    b = _trajectory([1, 2], [3], [-0.1], dict(run_id='b', accelerator='y', tp=8))
    data = trajectory_to_data(a, 1.0)
    assert client.datum_routes([SimpleNamespace(trajectories_G=[a, b])], data) == [('ambiguous', 'ambiguous')]
    assert client.route_labels({'run_id': 'f', 'accelerator': 'tpu-v5p-32', 'tp': 4}) == ('f', 'tpu-v5p-32-tp4')


def test_engine_stamps_every_choice_with_its_farm():
    pytest.importorskip('ray.serve')
    from tpu.swarm.ray_train import serving

    async def run():
        async def transport(request):
            return httpx.Response(200, json={'choices': [{'token_ids': [1]}, {'token_ids': [2]}]})

        async def ensure_adapter(model):
            pass

        cls = serving.Engine.func_or_class
        engine = cls.__new__(cls)
        engine.retiring, engine.key = False, '10.0.0.3:19801'
        engine.config = SimpleNamespace(run_id='farm-v5p32', accelerator='tpu-v5p-32', inference=SimpleNamespace(tp=4))
        engine.url = 'http://engine:19801'
        engine.ensure_adapter = ensure_adapter
        engine.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        try:
            result = await engine.generate({'model': 'base', 'prompt': 'x'})
        finally:
            await engine.http.aclose()
        stamp = dict(run_id='farm-v5p32', accelerator='tpu-v5p-32', tp=4, engine='10.0.0.3:19801')
        assert [c['served_by'] for c in result['choices']] == [stamp, stamp]

    asyncio.run(run())


def test_overlay_ships_both_halves_of_the_metric(tmp_path):
    from tpu.swarm.ray_train.config import Config
    from tpu.swarm.ray_train.overlay import install, manifest
    records = manifest(REPO, Config.load(PILOT))
    assert {'skyrl/backends/tunix_backend.py', 'skyrl/backends/sampler_mismatch.py',
            'third_party/discover/ttt_discover/rl/sampler_mismatch.py',
            'third_party/discover/ttt_discover/rl/ensemble.py',
            'third_party/discover/ttt_discover/rl/train.py'} <= records.keys()
    overlay = tmp_path / 'overlay'
    for name in records:
        target = overlay / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    (overlay / 'manifest.json').write_text(json.dumps(records, sort_keys=True))
    install(overlay, tmp_path / 'installed')


def _record(pid, tokens, logprobs):
    return dict(id=pid, tokens=tokens, logprobs=logprobs)


def test_parity_compares_only_the_common_greedy_prefix():
    from tpu.swarm.ray_train.logprob_parity import analyze, compare
    same = compare(_record('p', [1, 2, 3], [-0.1, -0.2, -0.3]), _record('p', [1, 2, 3], [-0.1, -0.2, -0.3]))
    assert same['identical'] and same['prefix'] == 3 and same['diverged_at'] is None and same['diffs'] == [0, 0, 0]
    split = compare(_record('p', [1, 2, 3, 4], [-0.1, -0.2, -0.3, -0.4]),
                    _record('p', [1, 2, 9, 9], [-0.15, -0.2, -5.0, -5.0]))
    assert split['prefix'] == 2 and split['diverged_at'] == 2
    assert split['diffs'] == pytest.approx([0.05, 0.0])
    farm = lambda shift: dict(
        sequential=[[_record('p', [1, 2], [-0.1 + shift, -0.2 + shift])],
                    [_record('p', [1, 2], [-0.1 + shift, -0.2 + shift])]],
        batched=[_record('p', [1, 2], [-0.1 + shift + 1e-3, -0.2 + shift])])
    report = analyze({'a': farm(0.0), 'b': farm(0.01), 'down': dict(error='lease refused')})
    assert report['repeat a']['mean_abs'] == 0 and report['repeat a']['identical'] == 1
    assert report['batch a']['max_abs'] == pytest.approx(1e-3)
    assert report['cross a | b']['mean_abs'] == pytest.approx(0.01)
    assert report['cross a | b']['mean_diff'] == pytest.approx(-0.01)
    assert not any('down' in name for name in report)


def test_parity_never_requests_prompt_logprobs():
    from tpu.swarm.ray_train import logprob_parity
    # vLLM TPU's EngineCore dies on prompt_logprobs; the probe must not send it.
    assert 'prompt_logprobs' not in logprob_parity.REMOTE
    assert "'temperature': 0.0" in logprob_parity.REMOTE
