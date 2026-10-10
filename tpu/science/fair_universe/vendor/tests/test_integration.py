import json
import subprocess
import sys
from pathlib import Path
import zipfile
import numpy as np
from fairdiscover.data import stage, MapStore
from fairdiscover.trial import trial
from fairdiscover.submission import export_submission


def test_train_reward_and_packaging(tmp_path):
    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / 'raw'
    subprocess.run([sys.executable, str(root / 'scripts/make_toy_data.py'),
                    '--out', str(raw), '--n-cosmo', '4', '--n-real', '16'],
                   check=True, capture_output=True, text=True)
    staged = tmp_path / 'stage'
    stage(str(raw), str(staged), mode='sample')
    candidate = root / 'candidates/baseline_jax.py'
    run_dir = tmp_path / 'run'
    result = trial(str(candidate), str(staged), str(run_dir), steps=2,
                   batch_size=4, max_reward_ind=8,
                   train_timeout=120, reward_timeout=120)
    assert not result.get('failed'), result
    assert 0 <= result['reward'] <= 1
    assert (run_dir / 'reward.txt').is_file()
    assert len(result['per_family']) == 4
    store = MapStore(staged)
    test_ids = store.ids('val')[:4]
    compressed = np.asarray(store.maps[test_ids[:,0],test_ids[:,1]])
    test_file = tmp_path / 'test.npy'
    np.save(test_file, compressed)
    zip_path = tmp_path / 'submission.zip'
    pack = export_submission(str(candidate), run_dir, staged, test_file, zip_path,
                             batch_size=2, expected_count=4)
    assert pack['count'] == 4
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ['result.json']
        payload = json.loads(zf.read('result.json'))
        assert len(payload['ood_scores']) == 4
        assert np.isfinite(payload['ood_scores']).all()
