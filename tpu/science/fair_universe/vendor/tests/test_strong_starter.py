import json
from pathlib import Path
import numpy as np
from fairdiscover.data import stage, MapStore
from fairdiscover.trial import trial
from fairdiscover.submission import export_submission
from scripts.make_toy_data import make_toy
from scripts.create_agent_prompt import build


def test_v2_hybrid_and_ensemble_end_to_end(tmp_path):
    root=Path(__file__).resolve().parents[1]
    raw=make_toy(tmp_path/'toy',n_cosmo=4,n_real=16)
    staged=tmp_path/'stage'
    stage(str(raw),str(staged),mode='sample')
    for model_name in ('hybrid_physics_jax.py','diverse_ensemble_jax.py'):
        file=root/'candidates'/model_name
        run=tmp_path/model_name
        result=trial(str(file),str(staged),str(run),steps=4,batch_size=4,
                     max_reward_ind=8,train_timeout=180,reward_timeout=180)
        assert not result.get('failed'), result
        assert np.isfinite(result['reward'])
        assert 0<=result['reward']<=1
        store=MapStore(staged)
        ids=store.ids('val')[:3]
        compressed=np.asarray(store.maps[ids[:,0],ids[:,1]])
        test=tmp_path/(model_name+'.npy')
        np.save(test,compressed)
        result_zip=tmp_path/(model_name+'.zip')
        package=export_submission(file,run,staged,test,result_zip,
                                  batch_size=2,expected_count=3)
        assert package['count']==3
        import zipfile
        with zipfile.ZipFile(result_zip) as zf:
            assert zf.namelist()==['result.json']
            saved=json.loads(zf.read('result.json'))
            assert len(saved['ood_scores'])==3
            assert np.isfinite(saved['ood_scores']).all()


def test_prompt_contains_contract_not_real_test(tmp_path):
    s=build()
    assert 'fit(train, val, config, out_dir)' in s
    assert 'predict(model, maps, mask, config)' in s
    assert 'already noisy' in s
    j=tmp_path/'previous.json'
    j.write_text(json.dumps({'reward':.16,'reward_is_proxy_not_official':True,
                             'per_family':{'amplitude':.04,'smoothing':.3}}))
    t=build(j)
    assert '0.16' in t
    assert 'hidden FAIR test' in t
