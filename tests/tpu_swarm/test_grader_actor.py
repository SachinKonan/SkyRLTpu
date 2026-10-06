import asyncio
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tpu'))
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.swarm.ray_train.grader_actor import Grader


def config(model='qwen'):
    suffix='dq3-003' if model=='qwen' else '002'
    raw=json.loads((ROOT/f'tpu/swarm/ray_train/profiles/smoke9-v5p-{model}-rglru-{suffix}.json').read_text())
    raw['arena_grader_rank']=1
    raw['client_env'].pop('ARENA_QUEUE_URL',None)
    raw['inference']['tp']=2 if model=='muse' else 4
    return raw


@pytest.mark.parametrize('model,engines',[('qwen',2),('muse',4)])
def test_dedicated_config_and_transport(model,engines):
    c=Config.from_dict(config(model))
    assert c.inference_hosts==2
    assert len(c.engine_slots(['10.0.0.3','10.0.0.4']))==engines
    assert workload_resources(c,1)=={'TPU':4, 'arena_grader':4, 'arena_pregate':2}
    env=client_environment(c,Path('/root'),'10.0.0.1')
    assert not env['ARENA_RAY_ACTOR'] and not env['ARENA_QUEUE_URL']
    assert env['ARENA_RAY_TASKS']=='1' and env['ARENA_RAY_ROOT']=='/root'
    assert env['RAY_NAMESPACE']==c.run_id


@pytest.mark.parametrize('rank',[0,2,3,4,True])
def test_bad_grader_rank_rejected(rank):
    raw=config();raw['arena_grader_rank']=rank
    with pytest.raises(ValueError):Config.from_dict(raw)


def test_external_url_rejected_for_owned_actor():
    raw=config();raw['client_env']['ARENA_QUEUE_URL']='http://old-judge:8791'
    with pytest.raises(ValueError):Config.from_dict(raw)


def grader(tmp_path,monkeypatch):
    import ray
    monkeypatch.setattr(ray,'get_runtime_context',lambda:SimpleNamespace(get_accelerator_ids=lambda:{'TPU':['0','1','2','3']}))
    return Grader(tmp_path,'test')


def payload(g):return dict(problem='rg_lru',cases=g.cases,code='test',tag='candidate')


def test_concurrent_requests_do_not_share_chips(tmp_path,monkeypatch):
    async def run():
        g=grader(tmp_path,monkeypatch);busy=set();seen=[]
        async def child(mode,p,tag,chip=None,case=None):
            (g.run/tag).mkdir(exist_ok=True)
            if mode=='pregate':return {'passed':True}
            assert chip not in busy
            busy.add(chip);seen.append((chip,case))
            try:
                await asyncio.sleep(.01)
                return dict(passed=True,score=2,grad_ok=True,grad_scores={case:2},task_noise_floor=.05)
            finally:busy.remove(chip)
        monkeypatch.setattr(g,'_child',child)
        results=await asyncio.gather(*(g.grade(payload(g)) for _ in range(5)))
        assert all(r['passed'] for r in results)
        assert len(seen)==5*len(g.cases)
        assert (await g.status())['free_chips']==4
        assert (await g.status())['completed']==5
    asyncio.run(run())


def test_timeout_releases_chips(tmp_path,monkeypatch):
    async def run():
        g=grader(tmp_path,monkeypatch)
        async def child(mode,*a,**k):
            if mode=='pregate':return {'passed':True}
            await asyncio.sleep(60)
        monkeypatch.setattr(g,'_child',child)
        with pytest.raises(TimeoutError):await g.grade(payload(g),timeout_s=.05)
        assert (await g.status())['free_chips']==4
        assert not g.requests
    asyncio.run(run())


def test_close_cancels_active_requests(tmp_path,monkeypatch):
    async def run():
        g=grader(tmp_path,monkeypatch)
        async def child(mode,*a,**k):
            if mode=='pregate':return {'passed':True}
            await asyncio.sleep(60)
        monkeypatch.setattr(g,'_child',child)
        task=asyncio.create_task(g.grade(payload(g)))
        await asyncio.sleep(.02)
        state=await g.close()
        assert task.cancelled() and state['free_chips']==4 and state['active']==0
        with pytest.raises(RuntimeError):await g.grade(payload(g))
    asyncio.run(run())


def test_candidate_failure_is_zero_and_stops_siblings(tmp_path,monkeypatch):
    async def run():
        g=grader(tmp_path,monkeypatch)
        async def child(mode,p,tag,chip=None,case=None):
            (g.run/tag).mkdir(exist_ok=True)
            if mode=='pregate':return {'passed':True}
            if case==g.cases[0]:return {'passed':False,'gate':'correctness','violations':['wrong output']}
            await asyncio.sleep(60)
        monkeypatch.setattr(g,'_child',child)
        result=await asyncio.wait_for(g.grade(payload(g)),1)
        assert not result['passed'] and result['gate']=='correctness'
        assert (await g.status())['free_chips']==4
    asyncio.run(run())


def test_child_crash_stays_infrastructure_failure(tmp_path,monkeypatch):
    from pallas_arena.rl.task import translate_verdict,ArenaInfrastructureError
    async def run():
        g=grader(tmp_path,monkeypatch)
        async def child(mode,p,tag,chip=None,case=None):
            (g.run/tag).mkdir(exist_ok=True)
            if mode=='pregate':return {'passed':True}
            raise RuntimeError('device unavailable')
        monkeypatch.setattr(g,'_child',child)
        result=await g.grade(payload(g))
        with pytest.raises(ArenaInfrastructureError):translate_verdict(result)
        assert len(result['excluded_cases'])==len(g.cases)
    asyncio.run(run())


def test_cancel_waits_for_actual_child_exit(tmp_path,monkeypatch):
    async def run():
        g=grader(tmp_path,monkeypatch)
        spawn=asyncio.create_subprocess_exec;processes=[]
        async def fake(*args,**kw):
            assert kw['env']['TPU_VISIBLE_CHIPS']=='2'
            assert 'RAY_ADDRESS' not in kw['env']
            proc=await spawn(sys.executable,'-c','import time; time.sleep(60)',**kw)
            processes.append(proc);return proc
        monkeypatch.setattr(asyncio,'create_subprocess_exec',fake)
        task=asyncio.create_task(g._child('case',payload(g),'cancel',chip='2',case=g.cases[0]))
        while not processes:await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert processes[0].returncode is not None and not g.processes
    asyncio.run(run())

@pytest.mark.parametrize('failed',[False,True])
def test_client_routes_to_actor_and_propagates_failure(tmp_path,monkeypatch,failed):
    import ast
    from pallas_arena.rl.task import public_contract,ArenaInfrastructureError
    tree=ast.parse((ROOT/'tpu/pallas_arena/rl/env.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RecurrentGemmaRewardEvaluator')
    ns=dict(BaseRewardEvaluator=object,State=object,Path=Path,os=__import__('os'),uuid=__import__('uuid'),json=json,
            threading=threading,
            public_contract=public_contract,ArenaInfrastructureError=ArenaInfrastructureError,
            translate_verdict=lambda r:r)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'evaluator','exec'),ns)
    calls=[];ref=object()
    def remote(p,**kw):calls.append((p,kw));return ref
    def get(r,**kw):
        assert r is ref
        if failed:raise RuntimeError('actor died')
        return {'passed':True,'test':True}
    fake=SimpleNamespace(is_initialized=lambda:True,get_actor=lambda name,namespace:SimpleNamespace(grade=SimpleNamespace(remote=remote)),get=get,cancel=lambda r:calls.append('cancelled'))
    monkeypatch.setitem(sys.modules,'ray',fake)
    monkeypatch.setenv('ARENA_RAY_ACTOR','rglru-grader');monkeypatch.setenv('RAY_NAMESPACE','test-run')
    evaluator=ns['RecurrentGemmaRewardEvaluator']('rg_lru',tmp_path)
    if failed:
        with pytest.raises(ArenaInfrastructureError):evaluator.get_reward('code',SimpleNamespace(id='parent'))
        assert calls[-1]=='cancelled'
    else:assert evaluator.get_reward('code',SimpleNamespace(id='parent'))['test']
    assert calls[0][0]['cases']==[n for n,_ in public_contract()[1]]


def test_noisy_case_is_remeasured_without_repeating_healthy_cases(tmp_path, monkeypatch):
    from collections import Counter
    from pallas_arena.rl.task import translate_verdict
    async def run():
        g = grader(tmp_path, monkeypatch)
        attempts = Counter()
        async def child(mode, p, tag, chip=None, case=None):
            (g.run / tag).mkdir(exist_ok=True)
            if mode == 'pregate':
                return {'passed': True}
            attempts[case] += 1
            # Exact production failure: a valid case gets excluded because
            # its baseline calibration had a 14.92 noise floor.
            floor = 14.92 if case == g.cases[0] and attempts[case] == 1 else .05
            return dict(passed=True, score=2, grad_ok=True,
                        grad_scores={case: 2}, task_noise_floor=floor)
        monkeypatch.setattr(g, '_child', child)
        verdict = await g.grade(payload(g))
        assert translate_verdict(verdict)['correctness'] == 1
        assert not verdict['excluded_cases']
        assert attempts[g.cases[0]] == 2
        assert all(attempts[c] == 1 for c in g.cases[1:])
        assert verdict['case_attempts'][g.cases[0]] == 2
        assert (await g.status())['free_chips'] == 4
    asyncio.run(run())


def test_persistent_noise_is_bounded_and_never_a_partial_pass(tmp_path, monkeypatch):
    from collections import Counter
    from pallas_arena.rl.task import translate_verdict, ArenaInfrastructureError
    async def run():
        g = grader(tmp_path, monkeypatch)
        attempts = Counter()
        async def child(mode, p, tag, chip=None, case=None):
            (g.run / tag).mkdir(exist_ok=True)
            if mode == 'pregate':
                return {'passed': True}
            attempts[case] += 1
            return dict(passed=True, score=2, grad_ok=True, grad_scores={case: 2},
                        task_noise_floor=1.42 if case == g.cases[0] else .05)
        monkeypatch.setattr(g, '_child', child)
        verdict = await g.grade(payload(g))
        assert verdict['passed'] is False and verdict['judge_fault']
        assert attempts[g.cases[0]] == 3
        with pytest.raises(ArenaInfrastructureError):
            translate_verdict(verdict)
        assert (await g.status())['free_chips'] == 4
    asyncio.run(run())


def test_compile_failure_is_not_hidden_by_noisy_calibration(tmp_path, monkeypatch):
    from pallas_arena.rl.task import translate_verdict
    async def run():
        g = grader(tmp_path, monkeypatch)
        async def child(mode, p, tag, chip=None, case=None):
            (g.run / tag).mkdir(exist_ok=True)
            if mode == 'pregate':
                return {'passed': True}
            return dict(passed=False, gate='aot_export', task_noise_floor=1.42,
                        violations=['unsupported dimension semantics'])
        monkeypatch.setattr(g, '_child', child)
        verdict = await g.grade(payload(g))
        assert verdict['gate'] == 'aot_export'
        assert translate_verdict(verdict)['reward'] == 0
        assert all(n == 1 for n in verdict['case_attempts'].values())
    asyncio.run(run())


def test_concurrent_first_completions_initialize_ray_once(tmp_path, monkeypatch):
    import ast
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pallas_arena.rl.task import public_contract, ArenaInfrastructureError
    tree = ast.parse((ROOT / 'tpu/pallas_arena/rl/env.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'RecurrentGemmaRewardEvaluator')
    ns = dict(BaseRewardEvaluator=object, State=object, Path=Path,
              os=__import__('os'), uuid=__import__('uuid'), json=json,
              threading=threading, public_contract=public_contract,
              ArenaInfrastructureError=ArenaInfrastructureError, translate_verdict=lambda r: r)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'evaluator', 'exec'), ns)
    initialized = False
    init_calls = []
    def init(**kw):
        nonlocal initialized
        init_calls.append(kw)
        time.sleep(.02)  # Widen the race that occurred with 32 completions.
        assert not initialized
        initialized = True
    fake = SimpleNamespace(is_initialized=lambda: initialized, init=init,
        get_actor=lambda *a, **kw: SimpleNamespace(grade=SimpleNamespace(remote=lambda *a, **kw: object())),
        get=lambda *a, **kw: {'passed': True}, cancel=lambda r: None)
    monkeypatch.setitem(sys.modules, 'ray', fake)
    monkeypatch.setenv('ARENA_RAY_ACTOR', 'rglru-grader')
    monkeypatch.setenv('RAY_NAMESPACE', 'test')
    monkeypatch.setenv('RAY_ADDRESS', '10.0.0.1:24679')
    def grade(_):
        evaluator = ns['RecurrentGemmaRewardEvaluator']('rg_lru', tmp_path)
        return evaluator.get_reward('code', SimpleNamespace(id='parent'))
    with ThreadPoolExecutor(max_workers=32) as pool:
        assert len(list(pool.map(grade, range(32)))) == 32
    assert len(init_calls) == 1
