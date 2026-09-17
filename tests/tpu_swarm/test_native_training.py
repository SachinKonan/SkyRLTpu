"""Native rollout parity, fail-closed masks, and TP2 deployment boundaries."""
import asyncio
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_native_thinking_budget import existing_completers, compat_settings, api, ROOT
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.commands import inference_environment, inference_command, client_environment

spec = importlib.util.spec_from_file_location('native_validator', ROOT/'skyrl/backends/native_completion.py')
validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)


@pytest.mark.parametrize('model', ['qwen3.5-27b','gemma4-31b','muse-glimmer-30b'])
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('route', ['external', 'forwarding'])
def test_complete_transport_matches_two_phase(existing_completers, monkeypatch, model, grouped, route):
    monkeypatch.setenv('TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE','0')
    f=json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    cls=existing_completers[model]
    for example in f['compat_examples'].values():
        first,tail=example['tokens'], f['answer_tail']
        policy=cls(); policy.phase1_max_tokens=100+len(first)
        policy.context_window=policy.phase1_max_tokens+len(f['transition'])+len(tail)+50
        policy.context_buffer=50;policy.temperature=.8;policy.min_think_tokens=0
        class Tok:
            def decode(self,ids):
                assert ids==first
                return example['decoded']
            def encode(self,text,**kw):
                assert text==cls.THINK_CLOSE+cls.ANSWER_CUE
                return f['transition']
        policy.tokenizer=Tok(); calls=[]
        async def sample(chunks, stop, max_tokens):
            calls.append(max_tokens)
            tok=tail if grouped or len(calls)>1 else first
            return tok,[-1.25]*len(tok)
        async def phase1(**kw):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=first,logprobs=[-1.25]*len(first))])
        policy._sample=sample;policy.sampling_client=SimpleNamespace(sample_async=phase1)
        prompt=SimpleNamespace(length=100,chunks=[])
        monkeypatch.setenv('TTD_NATIVE_THINKING_BUDGET','0')
        expected=(asyncio.run(policy.sample_group(prompt,['STOP'],1))[0] if grouped else asyncio.run(policy(prompt,['STOP'])))
        total=policy.context_window-100-50
        native_calls=[]
        async def native(**kw):
            native_calls.append(kw)
            assert kw['sampling_params'].thinking_token_budget==len(first)
            assert kw['sampling_params'].max_tokens==total
            choice=dict(token_ids=expected.tokens,logprobs={'token_logprobs':[-1.25]*len(expected.tokens)})
            api.annotate_response({'choices':[choice]},compat_settings(f,len(first)),f['end'])
            from test_native_external_transport import roundtrip
            choice['finish_reason'] = 'stop'
            return await roundtrip(route, [choice], len(first), total)
        policy.sampling_client=SimpleNamespace(sample_async=native)
        monkeypatch.setenv('TTD_NATIVE_THINKING_BUDGET','1')
        actual=(asyncio.run(policy.sample_group(prompt,['STOP'],1))[0] if grouped else asyncio.run(policy(prompt,['STOP'])))
        assert actual.tokens==expected.tokens
        assert actual.maybe_logprobs==expected.maybe_logprobs
        assert actual.maybe_mask==expected.maybe_mask
        assert len(native_calls)==1


def choice():
    return dict(token_ids=[5,6,7],loss_mask=[1.,0.,1.],forced_token_positions=[1],
        thinking_budget=dict(enforced=True,forced=True,budget_basis='phase1_generated_tokens',counted_phase1_tokens=1),
        logprobs={'token_logprobs':[-.4,-.5,-.6]})


@pytest.mark.parametrize('bad', ['loss_mask','thinking_budget','forced_token_positions','logprobs'])
def test_missing_transport_evidence_rejected(bad):
    c=choice();c.pop(bad)
    with pytest.raises(ValueError):validator.validate_choice(c,1,3)


@pytest.mark.parametrize('mutation',[lambda c:c['loss_mask'].__setitem__(1,1),lambda c:c['logprobs']['token_logprobs'].__setitem__(0,float('nan')),lambda c:c['thinking_budget'].__setitem__('counted_phase1_tokens',2)])
def test_inconsistent_transport_rejected(mutation):
    c=choice();mutation(c)
    with pytest.raises(ValueError):validator.validate_choice(c,1,3)


def muse_config():
    raw=json.loads((ROOT/'tpu/swarm/ray_train/profiles/smoke9-v5p-muse-ac2-002.json').read_text())
    raw['inference'].update(tp=2,native_thinking_budget=True)
    return Config.from_dict(raw)


def test_tp2_chip_isolation_and_all_six_endpoints():
    c=muse_config();ips=['10.0.0.2','10.0.0.3','10.0.0.4']
    slots=c.engine_slots(ips)
    assert len(slots)==len({s['key'] for s in slots})==6
    root=Path('/runtime'); envs=[]
    for slot in (0,1):
        env=inference_environment(c,root,root/'run',slot=slot);envs.append(env)
        assert env['TPU_CHIPS_PER_PROCESS_BOUNDS']=='1,2,1'
        cmd=inference_command(c,root,root/'source',root/'snapshot',root/'run',slot=slot)
        assert cmd[cmd.index('--tensor-parallel-size')+1]=='2'
        assert cmd[cmd.index('--max-num-seqs')+1]=='16'
        assert cmd[cmd.index('--port')+1]==str(c.ports.engine+slot)
    assert envs[0]['TPU_VISIBLE_CHIPS']=='0,1'
    assert envs[1]['TPU_VISIBLE_CHIPS']=='2,3'
    assert envs[0]['TPU_PROCESS_PORT']!=envs[1]['TPU_PROCESS_PORT']
    assert client_environment(c,root,'10.0.0.1')['TTD_NATIVE_THINKING_BUDGET']=='1'


def test_reject_unsupported_native_min_thinking_and_tp2_model():
    raw=muse_config().to_dict();raw['client_env']['TTD_MIN_THINK_TOKENS']='1'
    with pytest.raises(ValueError,match='min_think'):Config.from_dict(raw)
    raw=muse_config().to_dict();raw['model_preset']='qwen3.5-27b'
    with pytest.raises(ValueError,match='TP2'):Config.from_dict(raw)

@pytest.mark.parametrize('model', ['qwen3.5-27b','gemma4-31b','muse-glimmer-30b'])
def test_native_mixed_group_early_stop_and_forced_transition(existing_completers,monkeypatch,model):
    f=json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    cls=existing_completers[model];p=cls();p.phase1_max_tokens=110;p.context_window=512
    p.context_buffer=50;p.min_think_tokens=0;p.temperature=1
    p.tokenizer=SimpleNamespace(encode=lambda *a,**k:f['transition'], decode=lambda ids:'reasoning')
    rows=[[7,8],[7]*10+f['transition']+f['answer_tail']]
    calls=[]
    async def sample(**kw):
        calls.append(kw);seqs=[]
        for tokens in rows:
            c={'token_ids':tokens,'logprobs':{'token_logprobs':[-.4]*len(tokens)}}
            api.annotate_response({'choices':[c]},compat_settings(f,10),[])
            validator.validate_choice(c,10,362)
            seqs.append(SimpleNamespace(tokens=tokens,logprobs=c['logprobs']['token_logprobs'],loss_mask=c['loss_mask'],thinking_budget=c['thinking_budget']))
        return SimpleNamespace(sequences=seqs)
    p.sampling_client=SimpleNamespace(sample_async=sample)
    monkeypatch.setenv('TTD_NATIVE_THINKING_BUDGET','1')
    monkeypatch.setenv('TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE','0')
    out=asyncio.run(p.sample_group(SimpleNamespace(length=100),['STOP'],2))
    assert len(calls)==1 and calls[0]['num_samples']==2
    assert out[0].maybe_mask==[1.,1.]
    assert out[1].maybe_mask==[1.]*10+[0.]*len(f['transition'])+[1.]*len(f['answer_tail'])


def test_native_response_uses_json_when_protobuf_cannot_preserve_mask():
    tree=ast.parse((ROOT/'skyrl/tinker/api.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='retrieve_future')
    # Execute the actual completed-response branch with both formats accepted.
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=='future.status == RequestStatus.COMPLETED')
    nodes=branch.body
    test_fn=ast.FunctionDef(name='dispatch',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=nodes,decorator_list=[])
    data={'sequences':[dict(tokens=[1],loss_mask=[0.],thinking_budget={'enforced':True})]}
    ns={'future':SimpleNamespace(result_data=data)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test_fn],type_ignores=[])),'api-response','exec'),ns)
    assert ns['dispatch']() is data


def serving_namespace():
    tree=ast.parse((ROOT/'tpu/swarm/ray_train/serving.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Catalog','Ingress','deploy')]
    for n in nodes:
        n.decorator_list=[]
        if isinstance(n,ast.ClassDef):
            for f in n.body:
                if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):f.decorator_list=[]
    ns={'asyncio':asyncio,'Config':Config,'Path':Path,'time':SimpleNamespace(time=lambda:0)}
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),'serving','exec'),ns)
    return ns


def test_split_catalog_restart_and_request_routing():
    ns=serving_namespace();c=muse_config();slots=c.engine_slots(['a','b','c'])
    catalog=ns['Catalog']([s['key'] for s in slots],0)
    for s in slots:
        catalog.claim(s['key']);assert catalog.register(s['key'],s['key'],[])
    assert len(catalog.snapshot()['replicas'])==6
    with pytest.raises(RuntimeError,match='restart'):catalog.claim(slots[0]['key'])
    assert slots[1]['key'] not in catalog.snapshot()['exhausted']
    ingress=object.__new__(ns['Ingress']);ingress.engines=list(range(6));ingress.engine_models=None;ingress.next_engine=0
    assert [ingress.select_engine('muse') for _ in range(16)]==list(range(6))*2+[0,1,2,3]


def test_split_deploy_reserves_disjoint_two_chip_slots():
    ns=serving_namespace();c=muse_config();records=[]
    class Engine:
        @staticmethod
        def options(**kw):
            def bind(*args,**kwargs):records.append((kw,args,kwargs));return len(records)
            return SimpleNamespace(bind=bind)
    ns.update(Engine=Engine,serve=SimpleNamespace(start=lambda **kw:None,RunTarget=lambda **kw:kw,run_many=lambda *a,**kw:['handle']))
    class Ingress:
        @staticmethod
        def options(**kw):return SimpleNamespace(bind=lambda *a,**kw:None)
    ns['Ingress']=Ingress
    prepared={ip:{'role':'inference'} for ip in ['a','b','c']}
    assert ns['deploy'](c,prepared,None,'head')=='handle'
    assert len(records)==6
    for options,args,kwargs in records:
        assert options['num_replicas']==1 and options['ray_actor_options']['resources']['TPU']==2
    assert [r[2]['slot'] for r in records]==[0,1,0,1,0,1]


def test_native_source_path_fits_filesystem_name_limit_and_tracks_all_inputs():
    import hashlib
    tree=ast.parse((ROOT/'tpu/swarm/ray_train/host.py').read_text())
    init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    node=next(n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source_name' for t in n.targets))
    names=[]
    for suffix in ('a'*64,'b'*64):
        identity='a'*64+'-'+'b'*64+'-thinking-'+'c'*64+'-sdk-'+suffix
        assert len(identity)>255
        ns={'hashlib':hashlib,'self':SimpleNamespace(source_identity=identity,config=SimpleNamespace(inference=SimpleNamespace(native_thinking_budget=True)))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'host-path','exec'),ns)
        names.append(ns['source_name']);assert len(names[-1])<=255
    assert names[0]!=names[1]


@pytest.mark.parametrize('model', ['qwen3.5-27b', 'gemma4-31b', 'muse-glimmer-30b'])
@pytest.mark.parametrize('n', [1, 32])
@pytest.mark.parametrize('already_answering', [False, True])
def test_stop_exactly_at_cap_continues_same_prefix_like_legacy(existing_completers, monkeypatch, model, n, already_answering):
    f = json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    cls = existing_completers[model]
    first = [7] + (f['end'] if already_answering else [8, 99])
    tail = f['answer_tail']; cap = len(first)
    p = cls(); p.phase1_max_tokens = 100 + cap; p.context_window = 512
    p.context_buffer = 50; p.min_think_tokens = 0; p.temperature = 1
    decoded = cls.THINK_CLOSE_MARKER if already_answering else 'reasoning EOS'
    p.tokenizer = SimpleNamespace(encode=lambda *a, **k: f['transition'], decode=lambda ids: decoded)
    prompt = SimpleNamespace(length=100, chunks=[])
    old_calls = []
    async def legacy_sample(chunks, stop, max_tokens):
        old_calls.append((chunks, stop, max_tokens))
        tokens = first if len(old_calls) == 1 else tail
        return tokens, [-0.5] * len(tokens)
    p._sample = legacy_sample
    expected = asyncio.run(p._two_phase(prompt, ['STOP']))
    native_calls = []
    async def native_sample(**kwargs):
        native_calls.append(kwargs)
        assert kwargs['num_samples'] == n
        c = dict(token_ids=first, logprobs={'token_logprobs': [-0.5] * cap})
        api.annotate_response({'choices':[c]}, compat_settings(f, cap), [])
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=first, logprobs=c['logprobs']['token_logprobs'],
            loss_mask=c['loss_mask'], thinking_budget=c['thinking_budget']) for _ in range(n)])
    continuation_calls = []
    async def continuation(chunks, stop, max_tokens):
        continuation_calls.append((chunks, stop, max_tokens))
        assert chunks[-1].tokens == old_calls[1][0][-1].tokens
        assert (stop, max_tokens) == old_calls[1][1:]
        return tail, [-0.5] * len(tail)
    p._sample = continuation; p.sampling_client = SimpleNamespace(sample_async=native_sample)
    monkeypatch.setenv('TTD_NATIVE_THINKING_BUDGET', '1')
    monkeypatch.setenv('TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE', '0')
    out = asyncio.run(p.sample_group(prompt, ['STOP'], n))
    assert len(native_calls) == 1 and len(continuation_calls) == n
    for actual in out:
        assert actual.tokens == expected.tokens
        assert actual.maybe_mask == expected.maybe_mask
        assert actual.maybe_logprobs == expected.maybe_logprobs


@pytest.mark.parametrize('model', ['qwen3.5-27b', 'gemma4-31b', 'muse-glimmer-30b'])
def test_native_insufficient_headroom_uses_original_wall_behavior(existing_completers, model):
    f = json.loads((ROOT/'tests/tpu_swarm/fixtures/thinking_markers.json').read_text())[model]
    cls = existing_completers[model]; p = cls()
    p.phase1_max_tokens = 104; p.context_window = 154 + len(f['transition'])
    p.context_buffer = 50; p.min_think_tokens = 0; p.temperature = 1
    p.tokenizer = SimpleNamespace(encode=lambda *a, **kw: f['transition'], decode=lambda ids: 'reasoning')
    calls = []
    async def sample(chunks, stop, max_tokens):
        calls.append(max_tokens)
        assert max_tokens == 4
        return [7]*4, [-0.5]*4
    p._sample = sample
    prompt = SimpleNamespace(length=100, chunks=[])
    expected = asyncio.run(p._two_phase(prompt, ['STOP']))
    out = asyncio.run(p._native_group(prompt, ['STOP'], 2))
    assert len(calls) == 3
    for actual in out:
        assert vars(actual) == vars(expected)


def test_native_invalid_cap_is_a_fatal_contract_error(existing_completers):
    cls = existing_completers['qwen3.5-27b']; p = cls()
    p.phase1_max_tokens = 100; p.context_window = 512
    p.context_buffer = 50; p.min_think_tokens = 0
    p.tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1])
    with pytest.raises(Exception, match='positive thinking headroom') as error:
        asyncio.run(p._native_group(SimpleNamespace(length=100), [], 1))
    assert type(error.value).__name__ == 'NativeCompletionError'


def test_extracted_two_phase_algorithm_is_unchanged_from_discover_base():
    import subprocess
    path = 'ttt_discover/tinker_utils/completers.py'
    old = subprocess.check_output(['git', 'show', '1d662eb47971d868d12b03ecc5c050fadf7a6b44:' + path],
        cwd=ROOT/'third_party/discover', text=True)
    def body(source, method, skip):
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name == 'QwenTwoPhaseTokenCompleter')
        fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == method)
        return [ast.dump(n, include_attributes=False) for n in fn.body[skip:]]
    current = (ROOT/'third_party/discover'/path).read_text()
    assert body(old, '__call__', 2) == body(current, '_two_phase', 1)
