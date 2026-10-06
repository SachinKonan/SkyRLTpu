"""Transfer the complete verified Gemma circuit pool to fresh-model Gen-1 runs."""
from copy import deepcopy
import math
from .placement_suite_guard import validate_state


def transfer_pool(source, step=15):
    if source.get('step') != step:
        raise ValueError('donor has not reached the required step')
    pool = deepcopy(source)
    states = pool.get('states', [])
    if not states or not any(s.get('code') for s in states):
        raise ValueError('donor has no programs')
    for state in states + pool.get('initial_states', []):
        validate_state(state)
        if state.get('code') and (not math.isfinite(state['value']) or not 0 < state['value'] <= 1):
            raise ValueError('invalid saved reward')
        # Keep program identities, ancestry, constructions, and full feedback.
        # All transferred states precede the recipient's first training step.
        state['timestep'] = 0
    pool.update(step=0, puct_n={}, puct_m={}, puct_T=0)
    return pool
