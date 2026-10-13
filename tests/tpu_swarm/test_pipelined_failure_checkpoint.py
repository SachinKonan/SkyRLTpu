"""Exercise the production async finish function without optional client packages."""
import ast
import asyncio
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def runner(*, fail_at=None, submission_errors=0):
    path = Path('third_party/discover/ttt_discover/rl/ensemble.py')
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == '_pipelined_member_finish')
    events = []

    async def backward(future):
        events.append(('backward', future))
        if fail_at == future:
            raise RuntimeError('backward shape mismatch')

    async def enqueue(*args):
        events.append(('optimizer', 'enqueue'))
        return 'optimizer'

    async def consume(future):
        events.append(('optimizer', 'consume'))
        if fail_at == 'optimizer':
            raise RuntimeError('optimizer failed')
        return SimpleNamespace(metrics={})

    async def refresh(*args):
        events.append(('checkpoint', 'refresh'))
        return 'sampler'

    ns = dict(EnsembleConfig=object, asyncio=asyncio, time=time, os=os,
              logger=logging.getLogger(__name__), sampler_mismatch=None,
              _is_frozen=lambda cfg: False, consume_forward_backward=backward,
              enqueue_optim_step=enqueue, consume_optim_step=consume,
              _refresh_or_keep=refresh)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    member = SimpleNamespace(tag='qwen', training_client=object(),
                             train_cfg=SimpleNamespace(learning_rate=1.5e-4))
    plan = dict(member=member, fb_futs=[1, 2], stream_fb_errors=submission_errors,
                table=([], []), gen_rows=[])
    cfg = SimpleNamespace(pooled_multi_lora=False, save_every=1)
    return lambda: asyncio.run(ns[node.name](cfg, [plan], 2, {}, None)), events


def test_success_waits_for_all_backward_before_optimizer_and_checkpoint():
    run, events = runner()
    run()
    assert events == [('backward', 1), ('backward', 2), ('optimizer', 'enqueue'),
                      ('optimizer', 'consume'), ('checkpoint', 'refresh')]


@pytest.mark.parametrize('failure', [1, 2])
def test_backward_failure_never_enqueues_optimizer_or_checkpoint(failure):
    run, events = runner(fail_at=failure)
    with pytest.raises(RuntimeError, match='shape mismatch'):
        run()
    assert not any(kind in ('optimizer', 'checkpoint') for kind, _ in events)


def test_submission_failure_never_enqueues_optimizer_or_checkpoint():
    run, events = runner(submission_errors=1)
    with pytest.raises(RuntimeError, match='submission failed'):
        run()
    assert events == []


def test_optimizer_failure_never_saves_checkpoint():
    run, events = runner(fail_at='optimizer')
    with pytest.raises(RuntimeError, match='optimizer failed'):
        run()
    assert not any(kind == 'checkpoint' for kind, _ in events)


def test_circuit_tp1_retains_eight_kv_heads(tmp_path):
    from tpu.swarm.ray_train.commands import trainer_backend_config
    from tpu.swarm.ray_train.config import Config
    cfg = Config.load('tpu/swarm/ray_train/profiles/od-qwen38-trainer-20261010.json')
    backend = trainer_backend_config(cfg, tmp_path, '127.0.0.1', ['127.0.0.1'])
    kwargs = backend['maxtext_kwargs']
    assert kwargs['ici_tensor_parallelism'] == 1
    assert kwargs['ici_fsdp_parallelism'] == 4
    assert kwargs['override_model_config'] is True
    assert kwargs['base_num_kv_heads'] == cfg.trainer.logical_kv_heads == 8


def test_failed_sampler_export_cannot_silently_keep_old_policy(monkeypatch):
    path=Path('third_party/discover/ttt_discover/rl/ensemble.py')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_refresh_or_keep')
    async def fail(*args):raise RuntimeError('sampler export failed')
    ns=dict(MemberRuntime=object,os=os,logger=logging.getLogger(__name__),save_checkpoint_and_get_sampling_client=fail)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    m=SimpleNamespace(tag='qwen',sampling_client='old-policy',training_client=object(),train_cfg=SimpleNamespace(log_path='/tmp'))
    monkeypatch.setenv('TTD_RESUME_STRICT','1')
    with pytest.raises(RuntimeError,match='sampler export failed'):asyncio.run(ns[node.name](m,2,1))
    assert m.sampling_client=='old-policy'
    monkeypatch.setenv('TTD_RESUME_STRICT','0')
    assert asyncio.run(ns[node.name](m,2,1))=='old-policy'
