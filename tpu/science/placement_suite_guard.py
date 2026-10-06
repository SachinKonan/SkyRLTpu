"""Reject cached subset rewards when restoring the mandatory IBM17 objective."""
import json
from pathlib import Path
from .challenge_contract import CASES, SUITE


def validate_state(state):
    if not state.get('code'):
        return
    try:
        feedback = json.loads(state['observation'])
        metrics = feedback['metrics']
        cases = metrics['cases']
        if (metrics.get('benchmark_suite') != SUITE or metrics.get('case_count') != len(CASES)
                or len(cases) != len(CASES) or {r['case'] for r in cases} != set(CASES)
                or any(r.get('valid') is not True or r.get('overlap_count') != 0 for r in cases)):
            raise ValueError('incomplete suite')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Circuit state was not graded on all 17 IBM cases; regrade the pool in a new run before resuming') from exc


def validate_restored_pools(directory):
    directory = Path(directory)
    paths = sorted(directory.glob('puct_sampler_step_*.json'))
    if not paths:
        paths = list(directory.glob('puct_sampler.json'))
    if paths:
        data = json.loads(paths[-1].read_text())
        for state in data.get('states', []) + data.get('initial_states', []):
            validate_state(state)
    return paths[-1] if paths else None
