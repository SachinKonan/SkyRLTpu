"""Audit paired experiments, budgets, and model-specific regional assets."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.frozen_benchmark import payload

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / 'tpu/swarm/ray_train/campaigns/qwen35-qwen38-20261006.json').read_text())
RUNS = MANIFEST['runs']


def test_complete_two_seed_matrix():
    assert len(RUNS) == 8
    assert {(r['model'], r['problem'], r['seed']) for r in RUNS} == {
        (model, problem, seed)
        for model in ('Qwen/Qwen3.8-27B',)
        for problem in ('erdos', 'ac1', 'circuit', 'qubit') for seed in (0, 1)
    }


@pytest.mark.parametrize('run', RUNS, ids=lambda r: r['run_id'])
def test_science_and_runtime_contract(run):
    c = Config.load(ROOT / run['profile'])
    source = MANIFEST['budget_sources'][run['problem']]
    p = ROOT / 'tpu/swarm/ray_train/profiles' / source['profile']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == source['sha256']
    historical = Config.load(p)
    assert c.client_context_window == historical.client_context_window
    assert c.client_phase1_max_tokens == historical.client_phase1_max_tokens
    assert c.model == run['model']
    assert c.bootstrap_seed == 0
    assert c.client_env['TTD_M0_LORA_SEED'] == str(run['seed'])
    assert c.client_env['TTD_EXPLICIT_LORA_SEED'] == '1'
    assert c.bootstrap_fixed_budget and not c.bootstrap_require_full_pool
    assert (c.bootstrap_max_drafts, c.bootstrap_target_valid) == (1024, 512)
    assert not c.seed_pool_sha256 and not c.bootstrap_reuse_pool_sha256
    assert c.bootstrap_layers == 1
    hybrid = run['pool'] == 'tpuswarm-v4-64-central2-qwen35-erdos'
    assert c.bootstrap_all_hosts == hybrid
    if hybrid:
        assert c.accelerator == 'tpu-v4-64' and c.hosts == 8 and c.trainer.hosts == 4
        assert (c.trainer.tp, c.trainer.fsdp) == (8, 2)
        assert c.inference_hosts == 4 and c.trainer.token_budget == 45056
        assert not c.inference.remote_only and not c.inference.external_pool_require_initial
        assert c.inference.external_pool_dynamic_engines
        assert c.inference.external_pool_target_leases == 1
    else:
        assert c.accelerator == 'tpu-v6e-8' and c.hosts == c.trainer.hosts == 1
        assert (c.trainer.tp, c.trainer.fsdp) == (8, 1)
        assert c.trainer.process_bounds == '1,1,1' and c.trainer.chip_bounds == '2,4,1'
        assert c.inference.external_pool_target_leases == 2
        assert c.inference.remote_only and c.inference.external_pool_require_initial
    assert c.inference.external_pool_updates and c.inference.external_pool_attestation
    model_tag = 'qwen35' if run['model'] == 'Qwen/Qwen3.5-27B' else 'qwen38'
    assert c.inference.external_pool_discovery_group == 'qcmp-cross-region-20261006'
    assert c.inference.native_thinking_budget and not c.inference.prefix_caching
    assert c.trainer.ckpt_require_marker == (model_tag == 'qwen38')
    assert c.client_env['TTD_ADV_ESTIMATOR'] == 'piecewise_valid_entropic_centered_adaptive'
    assert c.client_env['TTD_ADV_PIECEWISE_RHO'] == '0.5'
    assert c.client_env['TTD_LOSS_FN'] == 'importance_sampling'
    assert (c.client_env['GROUPS_PER_BATCH'], c.client_env['GROUP_SIZE'], c.client_env['NUM_EPOCHS']) == ('16', '32', '10')
    assert not c.training_smoke
    assert c.single_bucket and c.zone == ('us-central2-b' if hybrid else 'us-east5-b')
    for value in (c.base_bundle, c.bucket, *vars(c.cache).values()):
        if isinstance(value, str) and value.startswith('gs://'):
            assert value == c.bucket or value.startswith(c.bucket + '/')
    if run['problem'] == 'qubit':
        assert c.client_env['SCIENCE_ROUTING_SUITE'] == 'full'
        assert c.science_routing_evaluator == 'parallel-v2'
    if run['problem'] == 'circuit':
        assert c.client_env['SCIENCE_PLACEMENT_SUITE'] == 'ibm17-proxy-v1'
        assert c.client_env['SCIENCE_PLACEMENT_STARTS'] == 'abuplace-xplace-three-starts-v1'
    # Actual request budgeting: phase one includes the prompt, plus 50 safety tokens.
    for prompt_length in (100, 4096, 10000):
        req = payload(c, SimpleNamespace(prompt=[1]*prompt_length, stop_ids=[2]), n=16)
        assert req['thinking_token_budget'] + prompt_length == source['prompt_plus_thinking']
        assert req['max_tokens'] + prompt_length + 50 == source['context']
        assert 'seed' not in req


@pytest.mark.parametrize('problem', ('erdos', 'ac1', 'circuit', 'qubit'))
@pytest.mark.parametrize('seed', (0, 1))
def test_science_settings_match_prepared_qwen35_baseline(problem, seed):
    pair = [r for r in RUNS if r['problem'] == problem and r['seed'] == seed]
    pair.append({'profile': f'tpu/swarm/ray_train/profiles/qcmp-qwen35-{problem}-pwc05-s{seed}-20261006.json'})
    normalized = []
    for r in pair:
        d = copy.deepcopy(json.loads((ROOT / r['profile']).read_text()))
        env = d['client_env']
        env.pop('TTD_SICK_MARKER')
        normalized.append({key:d[key] for key in ('client_env', 'client_learning_rate',
                          'client_context_window', 'client_phase1_max_tokens',
                          'bootstrap_max_drafts', 'bootstrap_target_valid',
                          'bootstrap_fixed_budget')})
    assert normalized[0] == normalized[1]

@pytest.mark.parametrize('problem', ('circuit', 'qubit'))
def test_shared_pool_producer_stops_before_training(problem):
    producer=next(r for r in MANIFEST['bootstrap_runs'] if r['problem']==problem)
    c=Config.load(ROOT/producer['profile'])
    assert c.bootstrap_only and c.bootstrap_all_hosts and c.bootstrap_fixed_budget
    assert not c.bootstrap_require_full_pool and c.bootstrap_seed==0
    assert (c.bootstrap_max_drafts,c.bootstrap_target_valid)==(1024,512)
    assert c.accelerator=='tpu-v4-64' and c.inference.tp==4
    assert not c.inference.external_pool_require_initial
    assert all(r['bootstrap_pool_source']==producer['run_id'] for r in RUNS if r['problem']==problem)
    assert all(r.get('requires_shared_pool') for r in RUNS if r['problem']==problem)
