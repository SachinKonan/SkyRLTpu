"""Prepare a shared AC2 winner for fresh-model follow-up runs.

No generated program is executed. Only its saved construction is evaluated with
our trusted AC2 verifier. Job completion is checked by the campaign frontend.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def best_completed_candidate(snapshots):
    """Require three completed, final pools and verify the winning construction."""
    if {r['model'] for r in snapshots} != {'qwen', 'gemma', 'muse'} or len(snapshots) != 3:
        raise ValueError('need exactly one completed pool per model')
    candidates = []
    for row in snapshots:
        if row['status'] != 'SUCCEEDED' or row['checkpoint'] != 10 or row['pool'].get('step') != 10:
            raise ValueError('AC2 first round is not complete')
        states = row['pool'].get('states', [])
        if isinstance(states, dict):
            states = list(states.values())
        valid = [s for s in states if s.get('code') and isinstance(s.get('construction'), list)
                 and s['construction'] and type(s.get('value')) in (float, int)
                 and math.isfinite(s['value']) and 0 < s['value'] <= 1]
        if not valid:
            raise ValueError('completed pool contains no valid AC2 candidates')
        candidates.extend((s['value'], row['run_id'], s['id'], row, s) for s in valid)
    _, _, _, source, winner = max(candidates, key=lambda x: x[:3])
    root = Path(__file__).resolve().parents[2]
    path = root/'third_party/discover/ttt_discover/tinker_utils/ac_helpers.py'
    spec = importlib.util.spec_from_file_location('ac2_trusted_verifier', path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    measured = float(verifier.evaluate_sequence_ac2(winner['construction']))
    if not math.isfinite(measured) or not math.isclose(measured, winner['value'], rel_tol=0, abs_tol=1e-10):
        raise ValueError('winning construction does not reproduce its saved AC2 score')
    raw = json.dumps(winner, sort_keys=True).encode()
    provenance = dict(source_run=source['run_id'], source_model=source['model'],
                      source_state=winner['id'], source_uri=source['uri'],
                      source_generation=source['generation'], candidate_sha256=hashlib.sha256(raw).hexdigest(),
                      score=measured, verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    state = deepcopy(winner)
    state.update(id='handoff-'+provenance['candidate_sha256'][:24], timestep=0,
                 parents=[], parent_values=[], origin=None)
    pool = dict(step=0, states=[state], initial_states=[deepcopy(state)], puct_n={}, puct_m={}, puct_T=0)
    provenance['seed_pool_sha256'] = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
    return pool, provenance


def recipient_profile(original, run_id, zone, bucket):
    """Retain the model recipe while isolating fresh training and cache writes."""
    if original['client_env'].get('TTD_PROBLEM_TYPE') != 'ac2':
        raise ValueError('recipient must use the AC2 environment')
    if run_id == original['run_id']:
        raise ValueError('handoff requires a new run ID')
    config = deepcopy(original)
    config.update(run_id=run_id, root='~/.cache/'+run_id, zone=zone, bucket=bucket,
                  bootstrap_layers=0, bootstrap_all_hosts=False, bootstrap_only=False,
                  bootstrap_max_drafts=0, bootstrap_target_valid=0, seed_pool_sha256='',
                  bootstrap_reuse_contract_sha256='', bootstrap_reuse_pool_sha256='',
                  checkpoint_resume=True, resume_min_checkpoint_step=0, systemd_runtime=True)
    config['client_env'].update(NUM_EPOCHS='10',
        TTD_SICK_MARKER=f'/home/gcpuser/.cache/{run_id}/runs/{run_id}/ENGINE-SICK')
    config['inference'].update(external_pool_require_initial=False, external_pool_initial_wait_seconds=300)
    for role in ('trainer', 'inference'):
        config['cache'][role+'_compile_seed'] = original['cache'][role+'_compile']
        config['cache'][role+'_compile'] = bucket+'/ac2-shared-best-20260921/'+run_id+'/'+role
    return config
