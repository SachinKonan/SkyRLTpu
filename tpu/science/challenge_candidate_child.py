"""Candidate-only runner. Invoke inside isolation; never from a controller."""
import ast
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ALLOWED = {'numpy', 'scipy', 'networkx', 'math', 'random', 'time', 'itertools',
           'functools', 'collections', 'heapq', 'bisect', 'array', 'statistics',
           'dataclasses', 'typing', 'enum', 'copy', 'operator', '__future__'}
use_tpu = len(sys.argv) > 2 and sys.argv[2] == 'tpu'
use_cpu_jax = len(sys.argv) > 2 and sys.argv[2] == 'cpu-jax'
if use_tpu or use_cpu_jax:
    ALLOWED.update({'jax', 'optax'})
started = time.monotonic()
device_info = {}
if use_tpu or use_cpu_jax:
    if use_cpu_jax:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    device_info = {'sandbox_tpu_visible_chips': os.environ.get('TPU_VISIBLE_CHIPS'),
                   'sandbox_device_paths': sorted(str(p) for p in Path('/dev').glob('accel*')) +
                       sorted(str(p) for p in Path('/dev/vfio').glob('*') if p.name != 'vfio')}
    platform = 'tpu' if use_tpu else 'cpu'
    print(json.dumps({'event':platform + '_initialization_started', **device_info}),flush=True)
    import jax
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != platform:
        raise RuntimeError(f'expected exactly one {platform} device, found {devices}')
    if use_cpu_jax and device_info['sandbox_device_paths']:
        raise RuntimeError('CPU placement sandbox exposes accelerator devices')
    device_info.update({'jax_version': jax.__version__, 'devices': [str(d) for d in devices],
                   'device_kind': devices[0].device_kind, 'device_count': len(devices),
                   'initialization_seconds': time.monotonic()-started})
    print(json.dumps({'event':platform + '_initialization_complete',**device_info}),flush=True)
source = Path('/candidate.py').read_text()
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names = [alias.name.split('.')[0] for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            raise ValueError('relative candidate imports are not allowed')
        names = [(node.module or '').split('.')[0]]
    else:
        continue
    if not set(names) <= ALLOWED:
        raise ValueError('disallowed candidate import: ' + ', '.join(names))
# This AST check expresses a library policy. Namespace isolation supplies the
# boundary; this is deliberately not advertised as a Python security sandbox.
with np.load('/problem.npz', allow_pickle=False) as data:
    problem = {k: data[k].copy() for k in data.files}
for key in ('num_hard', 'wirelength_normalizer', 'congestion_smoothing_range', 'schema_version'):
    problem[key] = problem[key].item()
namespace = {'__name__': 'candidate'}
exec(compile(tree, '/candidate.py', 'exec'), namespace)
remaining = float(sys.argv[1]) - (time.monotonic()-started)
result = namespace['place'](problem, 42, time_budget_s=max(0, remaining))
if not isinstance(result, dict) or set(result) != {'positions'}:
    raise ValueError('expected exactly {positions: numeric array}')
positions = np.asarray(result['positions'], dtype=np.float32)
if positions.shape != problem['initial_positions'].shape or not np.isfinite(positions).all():
    raise ValueError('wrong shape or nonfinite placement')
np.save('/output/positions.npy', positions, allow_pickle=False)
Path('/output/child.json').write_text(json.dumps({'candidate_seconds': time.monotonic()-started, **device_info}))
