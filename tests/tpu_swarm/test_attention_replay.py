"""The regression probe must test both shapes and never optimize failed work."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from tpu.swarm.ray_train import attention_replay as probe


@pytest.mark.parametrize('failed_bucket', [18432, 22528])
def test_backward_failure_prevents_optimizer(monkeypatch, tmp_path, failed_bucket):
    calls = []
    fixture = {'source_run': 'recorded', 'cases': [dict(bucket=b, length=b-1,
        request_id=i, datum_index=0, datum=b, loss_fn='importance_sampling', loss_fn_config=None)
        for i,b in enumerate((18432,22528))]}
    (tmp_path/'gemma_attention_replay.json').write_text(json.dumps(fixture))
    monkeypatch.setattr(probe, '__file__', str(tmp_path/'attention_replay.py'))
    monkeypatch.setattr(probe, 'restore_datum', lambda raw, sdk: raw)
    monkeypatch.setenv('TTD_RUN_DIR', str(tmp_path/'output'))
    monkeypatch.setenv('TINKER_BASE_URL', 'http://unused')
    class Client:
        model_id='test'
        def forward_backward(self, data, **kwargs):
            calls.append(data[0])
            def result():
                if data[0] == failed_bucket:raise RuntimeError('CompileTimeScopedVmemOom')
                return SimpleNamespace(metrics={'loss':1.0})
            return SimpleNamespace(result=result)
        def optim_step(self, *args):
            pytest.fail('optimizer was submitted after a failed backward')
    sdk=SimpleNamespace(ServiceClient=lambda **kwargs:SimpleNamespace(create_lora_training_client=lambda **kw:Client()))
    monkeypatch.setitem(sys.modules, 'tinker', sdk)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    with pytest.raises(RuntimeError, match='no optimizer update'):
        probe.run(SimpleNamespace(source=str(tmp_path), model='gemma', learning_rate=1e-4))
    assert set(calls)=={18432,22528}
    assert calls.count(failed_bucket)==1


def test_qwen_replay_preserves_recorded_batch_and_enables_tied_kv():
    from tpu.swarm.ray_train.config import Config
    from tpu.swarm.ray_train.commands import maxtext_kwargs, trainer_backend_config
    from tpu.swarm.ray_train.overlay import manifest, REPEATED_KV_FILES
    root = Path(__file__).resolve().parents[2]
    package = root/'tpu/swarm/ray_train'
    cfg = Config.load(package/'profiles/science-qwen-v4-tp8-fsdp2-replay-001.json')
    assert (cfg.trainer.tp, cfg.trainer.fsdp, cfg.trainer.logical_kv_heads) == (8, 2, 8)
    assert cfg.trainer.token_budget == 2*22528
    assert cfg.client_context_window == 22528 and cfg.client_phase1_max_tokens == 16384
    assert maxtext_kwargs(cfg, root)['base_num_kv_heads'] == 8
    assert REPEATED_KV_FILES <= manifest(root, cfg).keys()
    fixture = json.loads((package/cfg.attention_replay_fixture).read_text())
    case = fixture['cases'][0]
    assert fixture['model'] == cfg.model
    assert case['loss_fn'] == 'importance_sampling' and len(case['datums']) == 8
    assert case['lengths'] == [20498,19019,13742,13425,11105,19184,12436,13119]
    class SDK:
        TensorData = staticmethod(lambda **kw: kw)
        Datum = staticmethod(lambda **kw: kw)
        ModelInput = SimpleNamespace(from_ints=lambda x: x)
    for raw, length in zip(case['datums'], case['lengths']):
        restored = probe.restore_datum(raw, SDK)
        assert len(restored['model_input']) == length
        for key, tensor in raw['loss_fn_inputs'].items():
            assert restored['loss_fn_inputs'][key]['data'] == tensor['data']
