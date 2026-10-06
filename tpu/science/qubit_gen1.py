"""Validate the exact donor boundary and import routing programs for Gen1."""
from copy import deepcopy
import json
import math
import re
from .routing_suite import manifest


def checkpoint_at_step(rows, metrics, step=20):
    """Never substitute step19 or a later snapshot for the selected boundary."""
    matches = [r for r in rows if r.get('batch') == step]
    completed = [r for r in metrics if r.get('step') == step]
    if not matches or not completed:
        return None
    if any(r.get('gemma/train_error') or r.get('train/consec_all_member_errors', 0)
           for r in completed):
        raise ValueError('donor boundary has training errors')
    ck = matches[-1]
    if not re.fullmatch(r'tinker://[A-Za-z0-9_-]+/weights/[A-Za-z0-9_-]+', ck.get('state_path', '')):
        raise ValueError('invalid donor checkpoint path')
    return ck


def transfer_pool(source, step=20):
    if source.get('step') != step:
        raise ValueError('donor is not the exact requested step')
    pool = deepcopy(source)
    specs = manifest()['cases']
    expected = {s['id']: s for s in specs}
    programs = [s for s in pool.get('states', []) if s.get('code')]
    if not programs:
        raise ValueError('empty donor program pool')
    ids = [s['id'] for s in pool['states']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate state identity')
    for s in programs:
        obs = json.loads(s['observation'])
        metrics = obs['metrics']
        rows = metrics['cases']
        if metrics.get('case_columns'):
            rows = [dict(zip(metrics['case_columns'], row)) for row in rows]
        by_id = {r['case']: r for r in rows}
        if len(rows) != 72 or set(by_id) != set(expected):
            raise ValueError('incomplete or duplicate 72-case feedback')
        baseline = candidate = 0.0
        for key, spec in expected.items():
            swaps = by_id[key]['swaps']
            if type(swaps) is not int or swaps < 0:
                raise ValueError('invalid SWAP count')
            baseline += spec['weight'] * spec['original_cnot_added']
            candidate += spec['weight'] * 3 * swaps
        reward = baseline / (baseline + candidate)
        if not math.isfinite(s['value']) or not math.isclose(s['value'], reward, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('saved reward disagrees with full-suite counts')
        if not math.isclose(obs['reward'], reward, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('feedback reward disagrees with full-suite counts')
    # Preserve all programs, feedback, scores, identity and ancestry. Visits and
    # accumulated search values belong to the recipient, even when its adapter
    # is restored from a trained checkpoint. This is a new branch, not recovery
    # of an interrupted run; ordinary recovery must restore its own PUCT state.
    for s in pool['states'] + pool.get('initial_states', []):
        s['timestep'] = 0
    pool.update(step=0, puct_n={}, puct_m={}, puct_T=0)
    return pool
