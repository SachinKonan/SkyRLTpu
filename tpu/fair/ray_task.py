"""FAIR feature-family grading as farm Ray tasks: one TPU chip per grade, fresh child process per candidate.

Same pattern as the placement and RG-LRU TPU graders: Ray assigns exactly one chip (resources
{'TPU': 1, 'grading_fair': 1}, which exist only on the farm's non-engine hosts), the candidate runs in a
fresh child under process.py's guard holding a per-chip lock, and no generated code runs in the Ray worker.

Guards around the generated code (defense in depth; there is no OS sandbox):
  - a static check before anything runs: imports limited to jax/numpy/math/functools/itertools/collections/
    typing, no file / OS / network / callback / dynamic-code access, no paths in string literals;
  - the child gets a minimal environment (no cloud credentials, no Ray or TPU coordination variables);
  - the grader jit-compiles `features`, so per-call Python state cannot exist after tracing.

Failure classes:
  - static-check rejections, contract failures, candidate exceptions and candidate timeouts are candidate
    failures: a result with ok=False and correctness 0 (reward 0);
  - a child that exits before writing started.json (imports, TPU init, data or stack loading failed) raises,
    which the farm reports as an infrastructure error, so the trainer retries or aborts the step.

Host preparation (`prepare_host`) stages the grader's Python env and frozen inputs from the run's
same-region bucket (FAIR_FACTORY_GCS, default gs://sk7524-tinker-tpu-us-east5/fair-factory/v1) into
/dev/shm/fair (manifest.json with sha256 per file), once per host under a file lock.

Every full result (including the hidden-shift block, which never reaches the trainer) is kept in
/dev/shm/fair/factory/results/<host>.jsonl and copied to FAIR_RESULTS_GCS (default
gs://sk7524-tinker-tpu-us-east5/fair-factory/results/<host>/) for the final acceptance step.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

import ray

SHM = Path('/dev/shm/fair')
FACTORY = SHM / 'factory'
GCS = os.environ.get('FAIR_FACTORY_GCS', 'gs://sk7524-tinker-tpu-us-east5/fair-factory/v1')
RESULTS_GCS = os.environ.get('FAIR_RESULTS_GCS', 'gs://sk7524-tinker-tpu-us-east5/fair-factory/results')
CODE = Path(__file__).resolve().parent          # tpu/fair: tpu_grader/ is vendored next to this file
GRADE_TIMEOUT_S = 600                            # in-grader limit (candidate failure)
CHILD_TIMEOUT_S = 900                            # outer kill (the grader is stuck)

# ------------------------------------------------------------------------------------------ static check
ALLOWED_IMPORTS = {'jax', 'numpy', 'math', 'functools', 'itertools', 'collections', 'typing'}
BAD_NAMES = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals', 'vars', 'getattr', 'setattr',
             'delattr', 'input', 'breakpoint', 'os', 'sys', 'subprocess', 'socket', 'pathlib', 'importlib',
             'shutil', 'pickle', 'builtins', '__builtins__', '__file__', '__loader__', '__spec__', 'memoryview'}
BAD_ATTRS = {'load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt', 'genfromtxt', 'fromfile', 'tofile',
             'memmap', 'open_memmap', 'fromregex', 'DataSource', 'ctypeslib', 'lib', 'from_file', 'system', 'popen',
             'distributed', 'profiler', 'experimental', 'callback', 'pure_callback', 'io_callback', 'debug',
             'extend', 'export', 'config', 'interpreters', 'tools', 'device_put_sharded'}
SOURCE_DENY = [(re.compile(p), why) for p, why in (
    (r'\bopen\s*\(', 'open('), (r'\bos\s*\.', 'os.'), (r'\bsubprocess\b', 'subprocess'), (r'\bsocket\b', 'socket'),
    (r'\bpathlib\b', 'pathlib'), (r'\bsys\b', 'sys'), (r'__import__', '__import__'), (r'\bimportlib\b', 'importlib'),
    (r'\b(?:eval|exec)\s*\(', 'eval/exec'))]


def _docstrings(tree):
    ids = set()
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef,
                                                                      ast.AsyncFunctionDef))]:
        body = getattr(node, 'body', [])
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            ids.add(id(body[0].value))
    return ids


def static_check(code):
    """(ok, reason) for a candidate module source."""
    if len(code) > 60000:
        return False, 'code longer than 60k characters'
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f'syntax error: {e}'
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'features']
    if not fns:
        return False, 'no top-level def features(maps, mask)'
    if len(fns[0].args.args) < 2:
        return False, 'features must take (maps, mask)'
    docs = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] not in ALLOWED_IMPORTS:
                    return False, f'import {a.name} not allowed'
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or '').split('.')[0] not in ALLOWED_IMPORTS:
                return False, f'from {node.module} import not allowed'
            for a in node.names:
                if a.name in BAD_ATTRS or a.name in BAD_NAMES:
                    return False, f'from {node.module} import {a.name} not allowed'
        elif isinstance(node, ast.Name) and node.id in BAD_NAMES:
            return False, f'name {node.id} not allowed'
        elif isinstance(node, ast.Attribute) and (node.attr in BAD_ATTRS or node.attr.startswith('__')):
            return False, f'attribute .{node.attr} not allowed'
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            s = node.value.lower()
            if any(t in s for t in ('/', '\\', '.np', '.pt', '.pkl', '.json', 'shm', 'gs:')):
                return False, f'string literal {node.value[:40]!r} not allowed'
    stripped = '\n'.join(line.split('#', 1)[0] for line in code.splitlines())
    for rx, why in SOURCE_DENY:
        if rx.search(stripped):
            return False, f'{why} not allowed'
    return True, None


# ---------------------------------------------------------------------------------------------- staging
def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def _gcloud(*args, timeout=3600):
    env = dict(os.environ, CLOUDSDK_STORAGE_CHECK_HASHES='never')   # v5p images lack crc32c; we check sha256
    subprocess.run(['gcloud', 'storage', *args], check=True, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=timeout)


def stage():
    """Download and verify the grader env + frozen inputs once per host. Returns the READY record."""
    SHM.mkdir(parents=True, exist_ok=True)
    ready = SHM / 'READY.json'
    with open(SHM / '.stage.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready.is_file():
            return json.loads(ready.read_text())
        t0 = time.time()
        _gcloud('cp', f'{GCS}/manifest.json', str(SHM / 'manifest.json'))
        manifest = json.loads((SHM / 'manifest.json').read_text())
        for item in manifest['files']:
            dest = SHM / item['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(3):
                if dest.is_file() and _sha256(dest) == item['sha256']:
                    break
                _gcloud('cp', f'{GCS}/{item["path"]}', str(dest))
            else:
                if not (dest.is_file() and _sha256(dest) == item['sha256']):
                    raise RuntimeError(f'sha256 mismatch after 3 downloads: {item["path"]}')
        for item in manifest.get('untar', []):
            subprocess.run(['tar', '-xzf', str(SHM / item['archive']), '-C', str(SHM / item['into'])], check=True)
        record = dict(staged_s=round(time.time() - t0, 1), files=len(manifest['files']), version=manifest.get('version'))
        ready.write_text(json.dumps(record))
        return record


@ray.remote(num_cpus=0.1, max_retries=0)
def prepare_host():
    """Staging probe for one grading host (scheduled with node affinity by the farm's warmup)."""
    try:
        record = stage()
        if not (SHM / 'venv/bin/python').is_file():
            return dict(ok=False, detail='grader venv missing after staging')
        return dict(ok=True, host=socket.gethostname(), **record)
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, detail=f'{type(exc).__name__}: {exc}'[:500])


def reference_source():
    """A valid, known-neutral candidate (the prompt's API example) for the farm's readiness grade."""
    return (CODE / 'tpu_grader' / 'examples' / 'psband.py').read_text()


# ---------------------------------------------------------------------------------------------- grading
def _chip():
    """Ray's single assigned TPU chip, never a driver-selected ID or an inherited env var."""
    ids = ray.get_runtime_context().get_accelerator_ids().get('TPU', [])
    if not isinstance(ids, (list, tuple)) or len(ids) != 1 or str(ids[0]) not in ('0', '1', '2', '3'):
        raise RuntimeError(f'expected exactly one Ray-assigned TPU chip, got {ids!r}')
    return int(ids[0])


def _archive(host, record):
    """Keep the full result on the host and copy it to the evaluator bucket (best effort)."""
    results = FACTORY / 'results'
    results.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record)
    with open(results / f'{host}.jsonl', 'a') as fh:
        fh.write(line + '\n')
    name = f"{int(record['t'])}-{record['sha']}-{record['mode']}-{record['stack']}.json"
    path = results / 'pending' / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(line)
    try:
        _gcloud('cp', str(path), f'{RESULTS_GCS}/{host}/{name}', timeout=120)
        path.unlink()
        return True
    except Exception:  # noqa: BLE001 - kept in results/pending for a later sync
        return False


def grade(spec):
    """Grade one candidate on this task's Ray-assigned chip; returns the public result dict."""
    host = socket.gethostname()
    src, mode, stack = spec['source'], spec['mode'], spec.get('stack', 'v2')
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    ok, why = static_check(src)
    if not ok:
        return dict(ok=False, reward=None, sha=sha, mode=mode, stack=stack, correctness=0.0,
                    contract=dict(fail=f'static check: {why}'), metrics=dict(host=host))
    stage()
    chip = _chip()
    (FACTORY / 'work').mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='grade-', dir=FACTORY / 'work'))
    cand = work / 'candidate.py'
    cand.write_text(src)
    out = work / 'out'
    env = dict(PATH='/usr/bin:/bin', HOME=str(work), LANG='C.UTF-8', PYTHONPATH=str(CODE),
               PYTHONDONTWRITEBYTECODE='1', FAIR_FACTORY_ROOT=str(FACTORY), JAX_PLATFORMS='tpu',
               TPU_VISIBLE_CHIPS=str(chip), TPU_PROCESS_BOUNDS='1,1,1', TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1',
               TPU_PROCESS_PORT=str(8476 + chip), TPU_PROCESS_ADDRESSES=f'localhost:{8476 + chip}',
               CLOUD_TPU_TASK_ID='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='16')
    child = [str(SHM / 'venv/bin/python'), '-m', 'tpu_grader.grade', str(cand), '--out', str(out),
             '--mode', mode, '--stack', stack, '--timeout', str(GRADE_TIMEOUT_S)]
    locks = SHM / 'chip-locks'
    locks.mkdir(exist_ok=True)
    process_py = Path(__file__).resolve().parents[1] / 'swarm' / 'ray_train' / 'process.py'
    command = [sys.executable, str(process_py), 'guard', str(os.getpid()), json.dumps(child), '15',
               str(locks / f'{chip}.lock')]
    t0 = time.time()
    log = work / 'grade.log'
    with log.open('wb') as fh:
        proc = subprocess.Popen(command, env=env, cwd=str(work), stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=35)
            rc = 'timeout'
    wall = round(time.time() - t0, 1)
    tail = log.read_bytes()[-2000:].decode(errors='replace')
    if (out / 'result.json').is_file():
        result = json.loads((out / 'result.json').read_text())
    elif (out / 'started.json').is_file():
        # The grader was up and the candidate killed or hung it: the candidate's failure.
        result = dict(ok=False, reward=None, sha=sha, mode=mode, stack=stack,
                      contract=dict(fail=f'candidate crashed or hung the grader (exit {rc})'))
    else:
        raise RuntimeError(f'FAIR grader failed before the candidate started (exit {rc}): {tail}')
    archived = _archive(host, dict(t=time.time(), sha=sha, mode=mode, stack=stack, chip=chip, source=src,
                                   result=result))
    public = {k: v for k, v in result.items() if k not in ('hidden', 'traceback')}
    public['correctness'] = 1.0 if result.get('ok') else 0.0
    public['metrics'] = dict(public.get('metrics') or {}, host=host, chip=chip, wall_s=wall, archived=archived)
    return public


@ray.remote(max_retries=0)
def grade_fair(spec):
    return grade(spec)
