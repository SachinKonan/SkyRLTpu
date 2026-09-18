"""Trusted helper identity, prompt selection, and all-host bootstrap gates."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from tpu.science.fast_proxy_deployment import verified_mounts
from tpu.swarm.ray_train.config import Config


def test_helper_mounts_are_minimal_and_checksum_checked(tmp_path):
    folder = tmp_path / 'tpu/science/fast_proxy'
    folder.mkdir(parents=True)
    record = {}
    for name in ('__init__.py', 'congestion.c', 'libproxy.so'):
        (folder / name).write_bytes(name.encode())
        record[name] = hashlib.sha256(name.encode()).hexdigest()
    (folder / 'build-manifest.json').write_text(json.dumps(record))
    mounts, actual = verified_mounts(tmp_path)
    assert actual == record
    assert [target for _, target in mounts] == ['/fast_proxy/__init__.py', '/fast_proxy/libproxy.so']
    (folder / 'libproxy.so').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='checksum mismatch: libproxy.so'):
        verified_mounts(tmp_path)


def test_helper_prompt_is_opt_in_and_starter_has_no_helper_import():
    from tpu.science.training_env import task_prompt
    with patch.dict('os.environ', SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_HELPER='none'):
        old = task_prompt('placement')
    with patch.dict('os.environ', SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_HELPER='fast_proxy_v1'):
        new = task_prompt('placement')
    assert 'Evaluator is already imported' not in old
    assert 'Evaluator is already imported' in new
    assert 'with Evaluator(problem, pos)' in new
    assert 'from fast_proxy' not in new
    assert 'Scores do NOT certify legality' in new


def test_circuit_draft_only_bootstrap_requires_all_hosts_and_helper():
    path = Path('tpu/swarm/ray_train/profiles/science-circuit-v4-qwen-cpu-seeded-train-001.json')
    data = json.loads(path.read_text())
    data.update(bootstrap_only=True, bootstrap_layers=1, bootstrap_all_hosts=True, seed_pool_sha256='')
    data['client_env']['SCIENCE_PLACEMENT_HELPER'] = 'fast_proxy_v1'
    Config.from_dict(data)
    for key,value in [('bootstrap_layers', 2), ('bootstrap_all_hosts', False)]:
        bad=dict(data, **{key:value})
        with pytest.raises(ValueError, match='bootstrap-only'):
            Config.from_dict(bad)
    data['client_env']['SCIENCE_PLACEMENT_HELPER'] = 'none'
    with pytest.raises(ValueError, match='bootstrap-only'):
        Config.from_dict(data)


def test_pair_handoff_rejects_incomplete_grading_and_preserves_pool(tmp_path):
    from tpu.science.bootstrap import identity, save
    from tpu.science.circuit_seed_pair import verify
    base = Path('tpu/swarm/ray_train/profiles')
    source = Config.load(base/'science-circuit-v4-qwen-helper-seed-20260918.json')
    targets = [Config.load(base/f'science-circuit-v4-qwen-{arm}-20260918.json') for arm in ('grpo','pwc')]
    root = Path('tpu/science')
    contract = dict(config=source.to_dict(), prompt=(root/'prompts/placement-fast-proxy-cpu-v1.txt').read_text(),
        implementation_sha256=hashlib.sha256((root/'bootstrap.py').read_bytes()).hexdigest(),
        policy='frozen-base-no-adapter', groups=16,size=32)
    folder=tmp_path/'bootstrap'
    save(folder/'contract.json',dict(contract=contract,sha256=identity(contract)))
    roots=[dict(id=f'root-{i}') for i in range(16)]
    save(folder/'roots.json',roots)
    save(folder/'layer-0-plan.json',[dict(root_id=r['id'],repair_parent_id=None) for r in roots])
    for i in range(16):
        for j in range(32):
            save(folder/f'layer-0/group-{i:03d}/grade-{j:03d}.json',dict(
                id=f'{i}-{j}',root_id=f'root-{i}',repair_parent_id=None,layer=0,
                correctness=int(i==j==0),reward=.5 if i==j==0 else 0,code='pass'))
    pool=dict(step=0,states=[dict(id='0-0',code='pass',value=.5)])
    path=tmp_path/'tinker_log'/source.run_id/'puct_sampler_step_000000.json';save(path,pool)
    summary=dict(contract_sha256=identity(contract),layers=1,optimizer_steps=0,total=512,valid=1,retained=1,pool_sha256=identity(pool))
    save(folder/'complete.json',summary);save(folder/'layer-0-summary.json',dict(groups=16,total=512,valid=1))
    assert verify(tmp_path,source,targets)==(path,summary)
    (folder/'layer-0/group-015/grade-031.json').unlink()
    with pytest.raises(ValueError,match='incomplete draft'):
        verify(tmp_path,source,targets)
