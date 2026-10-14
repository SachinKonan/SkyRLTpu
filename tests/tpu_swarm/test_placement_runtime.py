"""Missing trusted dependencies cannot silently become candidate rewards."""
import hashlib
import json
from pathlib import Path

import pytest

from tpu.science.placement_runtime import (validate_runtime, reject_infrastructure_result,
                                         PlacementInfrastructureError, check_reference)
from tpu.science.challenge_contract import CASES, SUITE


def prepared(root):
    for venv, lock in [('candidate-venv', 'requirements-placement-candidate.lock'),
                       ('venv-placement', 'requirements-challenge-pilot.lock')]:
        p = root / '.science' / venv / 'bin/python';p.parent.mkdir(parents=True);p.write_text('#!/bin/sh\n');p.chmod(0o755)
        q = root / 'tpu/science' / lock;q.parent.mkdir(parents=True, exist_ok=True);q.write_text('pinned')
        (p.parent.parent / '.placement-lock').write_text(hashlib.sha256(q.read_bytes()).hexdigest())
    for name in ['tpu/science/challenge_candidate_child.py', 'tpu/science/challenge_score_child.py',
                 '.science/challenge-probe/macro_place/loader.py', '.science/challenge-probe/macro_place/objective.py']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    manifest={'benchmark_suite':SUITE,'cases':{}}
    for case in CASES:
        folder=root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case;folder.mkdir(parents=True)
        hashes={}
        for name in ('netlist.pb.txt','initial.plc'):
            (folder/name).write_text('input');hashes[name]=hashlib.sha256(b'input').hexdigest()
        p=root/'.science/placement-inputs'/f'{case}-problem.npz';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'input')
        manifest['cases'][case]={'valid':True,'native_sha256':hashes,'problem_sha256':hashlib.sha256(b'input').hexdigest(),'metrics':{'proxy_cost':1.0}}
    (root/'.science/placement-inputs/manifest.json').write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('missing', ['.science/candidate-venv/bin/python', '.science/venv-placement/bin/python',
    '.science/placement-inputs/ibm01-problem.npz', '.science/challenge-probe/macro_place/loader.py',
    '.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt'])
def test_missing_dependencies_are_infrastructure(tmp_path, missing):
    prepared(tmp_path);(tmp_path/missing).unlink()
    # A generic routing readiness marker must not bless placement.
    (tmp_path/'.science/ready.json').write_text('{}')
    with pytest.raises(PlacementInfrastructureError) as e:validate_runtime(tmp_path,'ibm01')
    assert e.value.abort_training_step


def test_modified_inputs_and_locks_rejected(tmp_path):
    prepared(tmp_path);validate_runtime(tmp_path)
    (tmp_path/'.science/candidate-venv/.placement-lock').write_text('wrong')
    with pytest.raises(PlacementInfrastructureError):validate_runtime(tmp_path)


def test_import_failure_is_infrastructure(tmp_path, monkeypatch):
    import subprocess
    prepared(tmp_path)
    def broken(*a,**kw):raise subprocess.CalledProcessError(1,a[0],stderr=b'ModuleNotFoundError: jax')
    monkeypatch.setattr(subprocess,'run',broken)
    with pytest.raises(PlacementInfrastructureError, match='jax'):validate_runtime(tmp_path,imports=True)


def test_old_farm_missing_interpreter_zero_is_rejected():
    result={'reward':0,'correctness':0,'msg':"FileNotFoundError: '/runtime/.science/candidate-venv/bin/python'",'metrics':{}}
    with pytest.raises(PlacementInfrastructureError):reject_infrastructure_result(result)
    reject_infrastructure_result({'reward':0,'correctness':0,'msg':'wrong shape or nonfinite placement','metrics':{}})


def test_reference_requires_expected_score(tmp_path):
    manifest=prepared(tmp_path)
    check_reference({'correctness':1,'metrics':{'proxy_cost':1}},manifest,'ibm01')
    with pytest.raises(PlacementInfrastructureError):check_reference({'correctness':0,'metrics':{}},manifest,'ibm01')
    with pytest.raises(PlacementInfrastructureError):check_reference({'correctness':1,'metrics':{'proxy_cost':2}},manifest,'ibm01')


def test_assigned_farm_uses_both_family_preparations():
    from tpu.swarm.ray_train.config import Config
    from tpu.science.package_training import science_preparation
    c=Config.load('tpu/swarm/ray_train/profiles/od-qwen38-farm-20261010.json')
    env,run=science_preparation(c)
    assert 'prepare_cpu_host.sh' in run and 'prepare_placement_host' in run
    assert env['SCIENCE_PLACEMENT_VENV']=='venv-placement'
    assert env['PLACEMENT_TPU_RANKS']=='0'
    assert env['SCIENCE_CPU_BUNDLE'].startswith(c.bucket+'/')


def test_portfolio_reference_uses_checksums_and_pinned_npz_scores(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    manifest=prepared(tmp_path);manifest['schema']='abuplace-xplace-three-starts-v1'
    for row in manifest['cases'].values():row.pop('metrics')
    (tmp_path/'.science/placement-inputs/manifest.json').write_text(json.dumps(manifest))
    def probe(args,**kwargs):
        return SimpleNamespace(stdout=json.dumps({case:1.25 for case in CASES}) if 'starting_scores' in args[-1] else '')
    monkeypatch.setattr(subprocess,'run',probe)
    result=validate_runtime(tmp_path,imports=True)
    check_reference({'correctness':1,'metrics':{'proxy_cost':1.25}},result,'ibm01')
    with pytest.raises(PlacementInfrastructureError):
        check_reference({'correctness':1,'metrics':{'proxy_cost':1.0}},result,'ibm01')


def test_grading_overlay_contains_infrastructure_error_type():
    from tpu.swarm.ray_train.overlay import manifest
    from tpu.swarm.ray_train.config import Config
    root=Path(__file__).resolve().parents[2]
    cfg=Config.load(root/'tpu/swarm/ray_train/profiles/od-qwen38-trainer-20261010.json')
    assert 'tpu/science/placement_runtime.py' in manifest(root,cfg)


def test_runtime_start_avoids_snap_login_scope():
    from tpu.swarm.ray_train.build import runtime_start
    from tpu.swarm.ray_train.config import Config
    cfg=Config.load('tpu/swarm/ray_train/profiles/od-qwen38-farm-20261010.json')
    script=runtime_start(cfg,'profile.json')
    assert script.index('/snap/google-cloud-cli/current/bin:$PATH') < script.index('gcloud storage cp')


def test_hard_overlap_is_a_candidate_error_not_infrastructure(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import numpy as np
    from tpu.science import challenge_score_child

    class Tensor:
        def __init__(self, a): self.a, self.shape = a, a.shape
        def numpy(self): return self.a
    bench = SimpleNamespace(macro_positions=Tensor(np.zeros((3, 2), np.float32)),
                            macro_fixed=Tensor(np.array([False, False, False])))
    fakes = {'torch': SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None,
                                      from_numpy=Tensor),
             'macro_place': ModuleType('macro_place'),
             'macro_place.loader': SimpleNamespace(load_benchmark_from_dir=lambda d: (bench, None)),
             'macro_place.objective': SimpleNamespace(
                 compute_proxy_cost=lambda t, b, plc: {'proxy_cost': 1.0, 'overlap_count': 2}),
             'macro_place.utils': SimpleNamespace(validate_placement=lambda t, b: (True, []))}
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)
    positions, result = tmp_path / 'positions.npy', tmp_path / 'score.json'
    np.save(positions, np.ones((3, 2), np.float32))
    monkeypatch.setattr(sys, 'argv', ['x', '--root', str(tmp_path), '--case', 'ibm01',
                                      '--positions', str(positions), '--result', str(result)])
    challenge_score_child.main()  # must not raise: a raise exits 1 and placement_task calls that infrastructure
    assert json.loads(result.read_text()) == {'candidate_error': 'nonfinite proxy or hard overlap'}
