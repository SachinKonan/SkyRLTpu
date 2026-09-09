import asyncio
from types import SimpleNamespace

import httpx
import pytest
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.db_models import FutureDB, ModelDB, SessionDB, RequestStatus
from skyrl.tinker.engine import TinkerEngine


def batch():
    return types.ForwardBackwardInput(data=[types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])]),
        loss_fn_inputs=types.LossFnInputs(target_tokens=types.TensorData(data=[2, 3]),
            weights=types.TensorData(data=[1., 1.]), advantages=types.TensorData(data=[0., 1.]),
            logprobs=types.TensorData(data=[0., -2.])))], loss_fn='cispo')


def request():
    return types.MultiLoraTrainingRequest(model_ids=['a', 'b'], forward_backward_input=batch())


def engine():
    e = TinkerEngine.__new__(TinkerEngine)
    e.db_engine = create_engine('sqlite:///:memory:')
    SQLModel.metadata.create_all(e.db_engine)
    e.config = SimpleNamespace(backend='tunix')
    return e


def enqueue(session, kind, model=None, data=None):
    f = FutureDB(request_type=kind, model_id=model, request_data=data or {})
    session.add(f); session.commit(); session.refresh(f)
    return f


def test_bundle_is_a_queue_barrier_on_both_sides():
    e = engine()
    with Session(e.db_engine) as s:
        first = enqueue(s, types.RequestType.OPTIM_STEP, 'b')
        bundle = enqueue(s, types.RequestType.MULTI_LORA_TRAINING, data=request().model_dump())
        later = enqueue(s, types.RequestType.FORWARD_BACKWARD, 'a', batch().model_dump())
        step = enqueue(s, types.RequestType.OPTIM_STEP, 'b')
        assert e.find_batchable_model_passes(s, types.RequestType.FORWARD_BACKWARD) == {}
        assert list(e.find_single_requests(s)) == [str(first.request_id)]
        first.status = RequestStatus.COMPLETED; s.add(first); s.commit()
        assert list(e.find_single_requests(s)) == [str(bundle.request_id)]
        bundle.status = RequestStatus.COMPLETED; s.add(bundle); s.commit()
        assert list(e.find_batchable_model_passes(s, types.RequestType.FORWARD_BACKWARD)) == [str(later.request_id)]
        assert list(e.find_single_requests(s)) == [str(step.request_id)]


def test_earlier_forward_must_drain_before_bundle():
    e = engine()
    with Session(e.db_engine) as s:
        first = enqueue(s, types.RequestType.FORWARD_BACKWARD, 'b', batch().model_dump())
        enqueue(s, types.RequestType.MULTI_LORA_TRAINING, data=request().model_dump())
        assert list(e.find_batchable_model_passes(s, types.RequestType.FORWARD_BACKWARD)) == [str(first.request_id)]
        assert e.find_single_requests(s) == {}


def test_sequential_execution_and_failed_bundle_discards_all_accumulations():
    e = engine(); calls = []; aborted = []
    e.backend = SimpleNamespace(has_model=lambda name: name in ('a','b'),
                               abort_multi_lora_training=lambda names: aborted.append(names))
    def forward(requests):
        model, data = requests['shared']; calls.append((model, data))
        return {'shared': types.ForwardBackwardOutput(loss_fn_output_type='scalar', loss_fn_outputs=[], metrics={})}
    e.process_forward_backward = forward
    req = request()
    result = e.process_multi_lora_training(req)
    assert [x[0] for x in calls] == ['a','b']
    assert calls[0][1] is calls[1][1]
    assert set(result.results) == {'a','b'} and result.metrics['training_token_evaluations'] == 4
    assert not aborted
    def fail(requests):
        if requests['shared'][0] == 'b': raise RuntimeError('second adapter failed')
        return forward(requests)
    e.process_forward_backward = fail
    with pytest.raises(RuntimeError, match='second adapter'):
        e.process_multi_lora_training(req)
    assert aborted == [['a','b']]
    calls.clear()
    req.model_ids = ['a','missing']
    with pytest.raises(ValueError, match='not loaded'):e.process_multi_lora_training(req)
    assert not calls


def test_http_endpoint_persists_one_shared_payload_and_rejects_bad_targets(tmp_path):
    async def run():
        db = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path/'api.db'))
        async with db.begin() as conn: await conn.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(db) as s:
            s.add(SessionDB(session_id='s',sdk_version='test'))
            for name in ('a','b'):
                s.add(ModelDB(model_id=name,base_model='qwen',lora_config={},status='ready',request_id=1,session_id='s'))
            await s.commit()
        async def session():
            async with AsyncSession(db) as s: yield s
        api.app.dependency_overrides[api.get_session] = session
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test') as c:
                payload=request().model_dump()
                r=await c.post('/api/v1/multi_lora_training',json=payload)
                assert r.status_code==200,r.text
                async with AsyncSession(db) as s:
                    f=await s.get(FutureDB,int(r.json()['request_id']))
                    assert f.model_id is None and f.request_type==types.RequestType.MULTI_LORA_TRAINING
                    assert f.request_data['model_ids']==['a','b']
                    assert len(f.request_data['forward_backward_input']['data'])==1
                payload['model_ids']=['a','a']
                assert (await c.post('/api/v1/multi_lora_training',json=payload)).status_code==422
                payload['model_ids']=['a','absent']
                assert (await c.post('/api/v1/multi_lora_training',json=payload)).status_code==404
        finally:
            api.app.dependency_overrides.clear();await db.dispose()
    asyncio.run(run())
