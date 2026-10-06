import json

import numpy as np
import pytest

from tpu.science.abuplace_starts import ABUPLACE_COMMIT, VARIANTS, variant_environment
from tpu.science import placement_start_portfolio as portfolio
from tpu.science import placement_warm_start as warm


@pytest.fixture
def completed(tmp_path):
    folder = tmp_path / warm.DESTINATION
    folder.mkdir(parents=True)
    queue = tmp_path / 'queue'
    manifest = dict(schema=warm.SCHEMA, benchmark_suite=warm.SUITE, cases={})
    for case in warm.CASES:
        native = tmp_path/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case
        native.mkdir(parents=True)
        for name in ('netlist.pb.txt', 'initial.plc'):
            (native/name).write_text(case)
        path = folder/f'{case}-problem.npz'
        np.savez(path, initial_positions=np.zeros((2, 2)), fixed=np.array([True, False]))
        manifest['cases'][case] = dict(valid=True, problem_sha256=warm.digest(path),
            native_sha256={p.name:warm.digest(p) for p in native.iterdir()})
        for index, variant in enumerate(VARIANTS):
            attempt = queue/'tasks'/f'xplace-{variant}-{case}'
            attempt.mkdir(parents=True)
            positions = np.array([[0., 0.], [index+1., index+1.]])
            np.save(attempt/'positions.npy', positions)
            (attempt/'gp.json').write_text(json.dumps(dict(source_commit=ABUPLACE_COMMIT,
                variant=variant, stage='global_placement_only')))
            (attempt/'report.json').write_text(json.dumps(dict(valid=True, case=case,
                method='xplace-abu', variant=variant, repository_commit=ABUPLACE_COMMIT,
                metrics={k:float(index+1) for k in portfolio.SCORE_COLUMNS}, candidate_wall_seconds=10)))
            (attempt/'done.json').write_text(json.dumps(dict(report=str(attempt/'report.json'))))
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, queue, tmp_path/'published'


def test_exact_three_configs():
    assert variant_environment('off')['XP_CONG_LOSS'] == '0'
    for variant in ('rudy', 'rudy_hv'):
        assert variant_environment(variant) == dict(XP_CONG_LOSS='1', XP_CONG_LOSS_KERNEL=variant)
    with pytest.raises(ValueError):
        variant_environment('random')


def test_publish_preserves_fallback_and_verifies_portfolio(completed):
    root, queue, out = completed
    portfolio.publish(root, queue, out)
    paths = warm.verified_inputs(root, folder=out)
    assert len(paths) == 17
    with np.load(paths['ibm01'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['initial_positions'], np.zeros((2, 2)))
        assert data['starting_layouts'].shape == (3, 2, 2)
        assert data['starting_names'].tolist() == list(VARIANTS)
    with pytest.raises(FileExistsError):
        portfolio.publish(root, queue, out)


def test_invalid_variant_prevents_partial_publication(completed):
    root, queue, out = completed
    report = queue/'tasks/xplace-rudy_hv-ibm18/report.json'
    record = json.loads(report.read_text())
    record['valid'] = False
    report.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='unverified GP'):
        portfolio.publish(root, queue, out)
    assert not out.exists()


def test_missing_variant_prevents_publication(completed):
    root, queue, out = completed
    (queue/'tasks/xplace-rudy-ibm01/done.json').unlink()
    with pytest.raises(FileNotFoundError):
        portfolio.publish(root, queue, out)
    assert not out.exists()


def test_portfolio_prompt_for_initial_and_parent_programs():
    from tpu.science.training_env import task_prompt
    env = dict(SCIENCE_PLACEMENT_BACKEND='cpu', SCIENCE_PLACEMENT_HELPER='fast_proxy_v1',
               SCIENCE_PLACEMENT_RUNTIME='cpu300-4g-v1', SCIENCE_PLACEMENT_STARTS=portfolio.SCHEMA)
    for starter in (False, True):
        prompt = task_prompt('placement', include_starter=starter, environment=env)
        assert 'ONE shared 300-second' in prompt and 'starting_layouts' in prompt
        assert 'off, rudy, rudy_hv' in prompt
