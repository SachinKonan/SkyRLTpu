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
