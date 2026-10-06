"""Fresh process entrypoint; no generated code ever runs in the Ray actor."""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys


def main():
    # If the owning actor/guardian dies unexpectedly, terminate this child.
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)
    if os.getppid() != parent or parent == 1:
        raise RuntimeError('Grader owner died during child startup')
    request, output = map(Path, sys.argv[1:])
    data = json.loads(request.read_text())
    from pallas_arena.judge.ray_pool import grade_case, stage0_pregate
    if data['mode'] == 'pregate':
        result = stage0_pregate('rg_lru', data['payload']['code'])
    else:
        chip = data['chip']
        if chip not in ('0', '1', '2', '3') or os.environ.get('TPU_VISIBLE_CHIPS') != chip:
            raise RuntimeError('Missing exclusive chip assignment')
        result = grade_case('rg_lru', data['case'], data['payload'], dict(
            allocated_chips=chip, baseline='all', cache=None, compile_cache_dir=data['cache'],
            timing_pairs=20, compile_budget_s=180, grade_budget_s=900))
    output.write_text(json.dumps(result))


if __name__ == '__main__':
    main()
