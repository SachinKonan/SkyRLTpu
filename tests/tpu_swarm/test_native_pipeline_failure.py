"""A native contract error aborts the sampling phase and cancels other groups."""
import ast
import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('kind', ['native', 'grader'])
def test_parity_failure_aborts_pipeline_and_cancels_sibling(kind):
    source = ROOT / 'third_party/discover/ttt_discover'
    exception = next(n for n in ast.parse((source / 'tinker_utils/completers.py').read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == 'NativeCompletionError')
    phase = next(n for n in ast.parse((source / 'rl/ensemble.py').read_text()).body
                 if isinstance(n, ast.AsyncFunctionDef) and n.name == '_pipelined_sampling_phase')
    ns = dict(asyncio=asyncio, timed=lambda *a: nullcontext(),
              logger=SimpleNamespace(error=lambda *a: None))
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), exception, phase], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'native_pipeline', 'exec'), ns)
    class GraderInfrastructureError(RuntimeError):
        abort_training_step = True
    failure = ns['NativeCompletionError'] if kind == 'native' else GraderInfrastructureError

    async def run():
        sibling_started = asyncio.Event()
        cancelled = []

        async def sample(client, builder, **kwargs):
            if builder == 'bad':
                await sibling_started.wait()
                raise failure('missing required completion or grading evidence')
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(builder)

        ns['do_group_rollout_and_filter_constant_reward'] = sample
        cfg = SimpleNamespace(pooled_multi_lora=False, temperature=1, cross_import_weight=0)
        member = SimpleNamespace(tag='qwen', sampling_client=None, train_cfg=SimpleNamespace(
            model_name='qwen', phase1_max_tokens=16384, context_window=22528,
            completion_max_tokens=None))
        plan = dict(member=member, builders=['bad', 'sibling'])
        with pytest.raises(failure, match='missing required'):
            await asyncio.wait_for(ns['_pipelined_sampling_phase'](cfg, [plan], 0, {}, None), 2)
        assert cancelled == ['sibling']
        assert plan['n_failed'] == 0  # It was fatal, not counted as a dropped group.

    asyncio.run(run())
