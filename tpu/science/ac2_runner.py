"""Trusted AC2 sandbox runner inside a systemd-limited CPU unit.

Executes one prepared candidate program (verifier already injected by the
trainer) with the same sandbox semantics as the discover evaluator's injected
child script: spawn start method, CPU affinity inherited by children, capped
process pools with a read-only-filesystem initializer, shared-memory unlink
guard and BLAS thread caps. Runs in ``envs/controller`` (stdlib + numpy);
never imports ``ttt_discover``.

Contract (``request.json`` -> ``result.json``):
  request: program_code, function_name, cpus (list of CPU ids),
           eval_timeout_seconds, stdout_limit_bytes, memory_gib, systemd (bool)
  result:  {"result": <json-safe list|number|null>, "error": str|null,
            "stdout": str, "metrics": {...}}
``started.json`` is written before the candidate starts: a unit that exits
without it failed for infrastructure reasons, not because of the candidate.
Candidate failures (timeout, crash, exception, non-finite output) are encoded
in ``error`` with exit code 0.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import signal
import shutil
import subprocess
import sys
import time

CHILD_TEMPLATE = r'''
import sys
import os
import pickle
import traceback
import importlib.util as _il

try:
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
except Exception:
    pass

_CPU_LIST_STR = "__PROGRAM_CORES__"
if _CPU_LIST_STR:
    try:
        cores = sorted({int(c) for c in _CPU_LIST_STR.split(",") if c.strip() != ""})
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cores))
    except Exception:
        pass


def _sandbox_worker():
    try:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
    except Exception:
        pass
    try:
        import builtins
        _orig_open = builtins.open

        def _ro_open(file, mode='r', *args, **kwargs):
            if any(ch in mode for ch in ('w', 'a', '+', 'x')):
                raise PermissionError("File writes are disabled in sandboxed workers")
            return _orig_open(file, mode, *args, **kwargs)
        builtins.open = _ro_open

        def _blocked(*args, **kwargs):
            raise PermissionError("Filesystem mutation disabled in sandboxed workers")
        for _name in ("remove", "unlink", "rename", "replace", "rmdir", "mkdir",
                      "makedirs", "chmod", "chown", "link", "symlink"):
            if hasattr(os, _name):
                setattr(os, _name, _blocked)
    except Exception:
        pass


def _compose_initializers(a, b):
    if a is None:
        return b

    def _combo():
        try:
            a()
        finally:
            b()
    return _combo


def _install_capped_executor(cap):
    import multiprocessing as mp
    import concurrent.futures as _cf
    import concurrent.futures.process as _cfp
    _Orig = _cfp.ProcessPoolExecutor
    _ctx = mp.get_context("spawn")

    class _Capped(_Orig):
        def __init__(self, max_workers=None, *args, **kwargs):
            mw = max_workers if max_workers is not None else (os.cpu_count() or 1)
            try:
                mw = max(1, min(int(mw), int(cap)))
            except Exception:
                mw = int(cap)
            init = kwargs.get("initializer", None)
            kwargs["initializer"] = _compose_initializers(init, _sandbox_worker)
            kwargs.setdefault("mp_context", _ctx)
            super().__init__(max_workers=mw, *args, **kwargs)

    _cfp.ProcessPoolExecutor = _Capped
    _cf.ProcessPoolExecutor = _Capped


try:
    import atexit
    import weakref
    from multiprocessing import shared_memory as _sm
    _orig_SharedMemory = _sm.SharedMemory
    _created_names = set()

    class _PatchedSharedMemory(_orig_SharedMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kwargs.get("create", False):
                _created_names.add(self.name)
                weakref.finalize(self, lambda n=self.name: _safe_unlink(n))

        def unlink(self):
            try:
                super().unlink()
            except Exception:
                pass

    def _safe_unlink(name):
        try:
            _orig_SharedMemory(name=name).unlink()
        except Exception:
            pass

    _sm.SharedMemory = _PatchedSharedMemory

    @atexit.register
    def _cleanup_shm():
        for n in list(_created_names):
            _safe_unlink(n)
except Exception:
    pass

_target_program_path = "__PROGRAM_PATH__"
_target_function_name = "__FUNCTION_NAME__"
_results_path = "__RESULTS_PATH__"
_max_cpus = int("__MAX_CPUS__")
sys.path.insert(0, os.path.dirname(_target_program_path))
try:
    _install_capped_executor(_max_cpus)
    spec = _il.spec_from_file_location("program", _target_program_path)
    program = _il.module_from_spec(spec)
    spec.loader.exec_module(program)
    sys.modules["program"] = program
    func = getattr(program, _target_function_name)
    result = func()
    with open(_results_path, "wb") as f:
        pickle.dump(result, f)
except BaseException as e:
    try:
        with open(_results_path, "wb") as f:
            pickle.dump({"error": f"{type(e).__name__}: {e}"}, f)
    except Exception:
        pass
    traceback.print_exc()
'''


def json_safe(value):
    """Return (json_value, error) for the candidate's return value."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                try:
                    import numpy as np
                    if isinstance(item, np.generic) and math.isfinite(float(item)):
                        out.append(float(item))
                        continue
                except ImportError:
                    pass
                return None, 'result is not finite numeric JSON'
            out.append(float(item))
        return out, None
    if isinstance(value, bool):
        return None, 'result is not finite numeric JSON'
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value), None
    return None, 'result is not finite numeric JSON'


def tail(path, limit):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ''
    return data[-limit:].decode(errors='replace')


def kill_tree(process, pgid, hard):
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL if hard else signal.SIGTERM)
        except Exception:
            pass
    if shutil.which('pkill'):
        try:
            subprocess.run(['pkill', '-KILL' if hard else '-TERM', '-P', str(process.pid)], check=False)
        except Exception:
            pass


def run_candidate(request, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    program = work / 'program.py'
    program.write_text(request['program_code'])
    results_path = work / 'results.pickle'
    stdout_path = work / 'candidate.stdout'
    cpus = [int(c) for c in request['cpus']]
    child = CHILD_TEMPLATE
    for key, value in (('__PROGRAM_PATH__', str(program)), ('__FUNCTION_NAME__', request['function_name']),
                       ('__RESULTS_PATH__', str(results_path)), ('__MAX_CPUS__', str(max(1, len(cpus)))),
                       ('__PROGRAM_CORES__', ','.join(map(str, cpus)))):
        child = child.replace(key, value)
    script = work / 'child.py'
    script.write_text(child)
    env = os.environ.copy()
    threads = str(max(1, len(cpus)))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env.setdefault(key, threads)
    timeout = float(request['eval_timeout_seconds'])
    limit = int(request.get('stdout_limit_bytes') or 16384)
    started = time.monotonic()
    with stdout_path.open('wb') as out:
        process = subprocess.Popen([sys.executable, str(script)], stdout=out, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, cwd=str(work))
        try:
            pgid = os.getpgid(process.pid)
        except Exception:
            pgid = None
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(process, pgid, hard=False)
            try:
                process.wait(timeout=1)
            except Exception:
                pass
            kill_tree(process, pgid, hard=True)
            try:
                process.wait(timeout=.5)
            except Exception:
                pass
            return dict(result=None, error=f'Process timed out after {timeout:.0f} seconds',
                        stdout=tail(stdout_path, limit), candidate_seconds=time.monotonic() - started)
        kill_tree(process, pgid, hard=False)
        kill_tree(process, pgid, hard=True)
    elapsed = time.monotonic() - started
    stdout = tail(stdout_path, limit)
    if code != 0:
        return dict(result=None, error=f'Process exited with code {code}', stdout=stdout, candidate_seconds=elapsed)
    if not results_path.exists():
        return dict(result=None, error='Results file not found', stdout=stdout, candidate_seconds=elapsed)
    try:
        with results_path.open('rb') as handle:
            value = pickle.load(handle)
    except Exception as exc:
        return dict(result=None, error=f'unreadable result: {type(exc).__name__}', stdout=stdout,
                    candidate_seconds=elapsed)
    if isinstance(value, dict) and 'error' in value:
        return dict(result=None, error=f'Program execution failed: {value["error"]}', stdout=stdout,
                    candidate_seconds=elapsed)
    safe, error = json_safe(value)
    return dict(result=safe, error=error, stdout=stdout, candidate_seconds=elapsed)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--owner-pid', type=int)
    parser.add_argument('--owner-start')
    args = parser.parse_args(argv)
    if args.owner_pid is not None and args.owner_start:
        from tpu.science.worker import watch_owner
        watch_owner(args.owner_pid, args.owner_start)
    request = json.loads(Path(args.request).read_text())
    work = Path(args.result).parent
    work.mkdir(parents=True, exist_ok=True)
    (work / 'started.json').write_text(json.dumps(dict(pid=os.getpid(), time=time.time())))
    metrics = dict(cpu_affinity=sorted(os.sched_getaffinity(0)), cpus=list(request['cpus']))
    memory_group = None
    if request.get('systemd', True):
        from tpu.science.cgroup_limits import envelope, metrics as cgroup_metrics
        memory_group, limit = envelope(int(request.get('memory_gib') or 4))
        metrics['hard_memory_bytes'] = limit
    started = time.monotonic()
    outcome = run_candidate(request, work / 'candidate')
    metrics['worker_seconds'] = time.monotonic() - started
    metrics['candidate_seconds'] = outcome.pop('candidate_seconds', None)
    if memory_group is not None:
        metrics['allocation_memory'] = cgroup_metrics(memory_group)
    payload = dict(result=outcome['result'], error=outcome['error'], stdout=outcome['stdout'], metrics=metrics)
    Path(args.result).write_text(json.dumps(payload, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
