"""Trusted placement runtime checks; setup failures must never become rewards."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


class PlacementInfrastructureError(RuntimeError):
    abort_training_step = True


REFERENCE_SOURCE = "def place(problem, seed, *, time_budget_s):\n    positions = problem['starting_layouts'][0] if 'starting_layouts' in problem else problem['initial_positions']\n    return {'positions': positions.copy()}\n"


def validate_runtime(root, case=None, *, imports=False):
    from .placement_resources import grader_python
    root = Path(root)
    candidate = root / '.science/candidate-venv/bin/python'
    grader = grader_python(root)
    try:
        for executable, lock in [(candidate, 'requirements-placement-candidate.lock'),
                                 (grader, 'requirements-challenge-pilot.lock')]:
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError(f'missing executable: {executable}')
            expected = hashlib.sha256((root / 'tpu/science' / lock).read_bytes()).hexdigest()
            if (executable.parent.parent / '.placement-lock').read_text().strip() != expected:
                raise ValueError(f'dependency lock mismatch: {executable}')
        for rel in ['tpu/science/challenge_candidate_child.py', 'tpu/science/challenge_score_child.py',
                    '.science/challenge-probe/macro_place/loader.py',
                    '.science/challenge-probe/macro_place/objective.py']:
            if not (root / rel).is_file():
                raise ValueError(f'missing trusted runtime file: {root / rel}')
        manifest = json.loads((root / '.science/placement-inputs/manifest.json').read_text())
        from .challenge_contract import CASES, SUITE
        if manifest.get('benchmark_suite') != SUITE or set(manifest.get('cases', {})) != set(CASES):
            raise ValueError('incomplete placement suite manifest')
        for name in ([case] if case else CASES):
            item = manifest['cases'][name]
            paths = {root / '.science/placement-inputs' / f'{name}-problem.npz': item['problem_sha256']}
            native = root / '.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04' / name
            paths.update({native / n: h for n, h in item['native_sha256'].items()})
            if set(item['native_sha256']) != {'netlist.pb.txt', 'initial.plc'} or item['valid'] is not True:
                raise ValueError(f'invalid reference manifest: {name}')
            for path, expected in paths.items():
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'input checksum mismatch: {path}')
        if imports:
            env = dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            for executable, program in [
                (candidate, "import sys,numpy,scipy,networkx,jax,optax; assert sys.version_info[:2] == (3,12)"),
                (grader, "import sys,numpy,torch; sys.path.insert(0,'.science/challenge-probe'); from macro_place.loader import load_benchmark_from_dir; from macro_place.objective import compute_proxy_cost")]:
                subprocess.run([str(executable), '-c', program], cwd=root, env=env,
                               check=True, timeout=90, capture_output=True)
            # Portfolio manifests pin the scores inside each checksummed NPZ;
            # older single-start manifests store the score in JSON instead.
            if manifest.get('schema') == 'abuplace-xplace-three-starts-v1':
                probe = """import json,numpy as np
from pathlib import Path
scores={}
for p in Path('.science/placement-inputs').glob('*-problem.npz'):
 with np.load(p,allow_pickle=False) as data:
  assert data['starting_score_columns'].tolist()[0]=='proxy_cost'
  scores[p.name.removesuffix('-problem.npz')]=float(data['starting_scores'][0,0])
print(json.dumps(scores,allow_nan=False))
"""
                result = subprocess.run([str(candidate), '-c', probe], cwd=root, env=env,
                                        check=True, timeout=90, capture_output=True, text=True)
                manifest['reference_costs'] = json.loads(result.stdout)
                if set(manifest['reference_costs']) != set(CASES):
                    raise ValueError('incomplete reference scores')
    except Exception as exc:
        detail = (getattr(exc, 'stderr', b'') or b'')
        if isinstance(detail, bytes): detail = detail.decode(errors='replace')
        raise PlacementInfrastructureError(f'placement infrastructure: {exc}; {detail[-1000:]}') from exc
    return manifest


def check_reference(result, manifest, case):
    import math
    expected = (manifest['reference_costs'][case] if 'reference_costs' in manifest
                else manifest['cases'][case]['metrics']['proxy_cost'])
    actual = (result.get('metrics') or {}).get('proxy_cost')
    if (result.get('correctness') != 1 or actual is None or
            not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)):
        raise PlacementInfrastructureError(f'placement infrastructure: reference failed for {case}: '
                                           f'expected {expected}, actual {actual}; {result.get("msg")}')


def reject_infrastructure_result(result):
    """Protect clients even while an older farm is being drained/replaced."""
    if not isinstance(result, dict):
        raise PlacementInfrastructureError('grading infrastructure: non-dict result')
    message = str(result.get('msg', ''))
    if ((result.get('metrics') or {}).get('failure_phase') == 'infrastructure' or
            ('FileNotFoundError' in message and any(p in message for p in
              ('candidate-venv/bin/python', 'venv-placement/bin/python', 'placement-inputs/', 'challenge-probe/')))):
        raise PlacementInfrastructureError('grading infrastructure: ' + message)
