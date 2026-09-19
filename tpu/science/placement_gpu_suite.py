"""Run pinned public placers on CUDA; preserve raw output and grade independently.

Xplace uses Archgen's published LEF/DEF bridge and route-aware settings. Its
raw global placement is saved before our deterministic legalization. Human
submissions are graded as returned, without silently repairing their outputs.
"""
import argparse
import contextlib
import glob
import io
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time


def load_module(path):
    spec = importlib.util.spec_from_file_location('public_placer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_finite_xplace_log(log):
    # DEF export/parser can turn a failed numeric trajectory into finite
    # coordinates. Reject the failed optimization before legalization.
    if re.search(r'(?:masked_hpwl|exact HPWL|density_weight|obj):\s*[-+]?(?:nan|inf)\b', log, re.I):
        raise ValueError('Xplace reported nonfinite optimization metrics; exported coordinates are not a valid successful start')


def child(args):
    import numpy as np
    import torch
    from macro_place.loader import load_benchmark_from_dir
    started = time.monotonic()
    assert torch.cuda.is_available(), 'CUDA required; CPU fallback is not a reproduction'
    torch.set_num_threads(args.cpus)
    torch.set_num_interop_threads(1)
    Path('/work/external').symlink_to('/eval/external', target_is_directory=True)
    b, plc = load_benchmark_from_dir('/eval/external/MacroPlacement/Testcases/ICCAD04/' + args.case)
    out = Path('/output')
    if args.method == 'xplace':
        sys.path.insert(0, str(Path(args.repository)/'submissions'))
        from bookshelf_to_lefdef import run_xplace_lefdef
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                raw = run_xplace_lefdef(
                    b, plc=plc, work_dir=out/'xplace', xplace_root=args.xplace_root,
                    inner_iter=1200, target_density=.8, use_route_force=True,
                    use_cell_inflate=False, route_weight=.01, congest_weight=.01,
                    num_route_iter=20, num_bin_x=128, num_bin_y=128,
                    timeout_s=900, verbose=True, extra_args=['--seed=42'])
        finally:
            print(captured.getvalue(), flush=True)
            (out/'xplace.log').write_text(captured.getvalue())
        require_finite_xplace_log(captured.getvalue())
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != tuple(b.macro_positions.shape) or not np.isfinite(raw).all():
            raise ValueError('invalid Xplace output')
        np.save(out/'raw_positions.npy', raw, allow_pickle=False)
        from tpu.science.challenge_contract import problem_from_native
        from tpu.science.challenge_seed_jax import legalize
        p = problem_from_native(b, plc)
        p['initial_positions'] = raw.copy()
        p['initial_positions'][p['fixed']] = b.macro_positions.numpy()[p['fixed']]
        positions = legalize(p, 42, time_budget_s=args.legalization_seconds)['positions']
    else:
        source, cls = (('submissions/final_placer.py', 'OptimalPlacer')
                       if args.method == 'archgen' else ('abuplace/placer.py', 'XplacePlacer'))
        placer = getattr(load_module(Path(args.repository)/source), cls)()
        positions = placer.place(b)
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
    torch.cuda.synchronize()
    np.save(out/'positions.npy', np.asarray(positions, dtype=np.float32), allow_pickle=False)
    (out/'candidate.json').write_text(json.dumps({
        'seconds': time.monotonic()-started, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['xplace', 'archgen', 'abuplace'], required=True)
    p.add_argument('--case', choices=[f'ibm{i:02d}' for i in range(1, 19) if i != 5], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repository', required=True)
    p.add_argument('--xplace-root', required=True)
    p.add_argument('--seconds', type=int, default=3450)
    p.add_argument('--cpus', type=int, default=16)
    p.add_argument('--legalization-seconds', type=float, default=120)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.child:
        return child(args)
    from isolation import command, python_mounts
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise FileExistsError('Use a new output directory for a new attempt')
    repo = out/'repository'
    shutil.copytree(Path(args.repository).resolve(), repo, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', 'build', 'results', 'output'), symlinks=True)
    xp = Path(args.xplace_root).resolve()
    if args.method == 'abuplace':
        # Keep AbuPlace's own Xplace Python code; use its separately built binaries.
        shutil.copytree(xp/'cpp_to_py/cpybin', repo/'abuplace/Xplace/cpp_to_py/cpybin', dirs_exist_ok=True)
        # These C helpers are compiled with -march=native. A cached helper
        # from a different CPU can SIGILL; rebuild within this private copy.
        for cached in (repo/'abuplace/extensions').glob('*.so'):
            cached.unlink()
    env = {
        'PATH': str(Path(sys.executable).parent)+':/usr/local/cuda-12.6/bin:/usr/bin:/bin',
        'PYTHONPATH': '/eval:/source:'+str(repo),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        'CUDA_HOME': '/usr/local/cuda-12.6', 'MPLCONFIGDIR': '/tmp/matplotlib',
        'MPLBACKEND': 'Agg', 'XDG_CACHE_HOME': '/tmp/cache',
        'CUDA_CACHE_PATH': '/tmp/cuda-cache', 'TORCH_HOME': '/tmp/torch',
        'LD_LIBRARY_PATH': str(xp/'cpp_to_py/cpybin'),
        'OMP_NUM_THREADS': str(args.cpus), 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'XPLACE_PYTHON': sys.executable,
        'PYTHONUNBUFFERED': '1',
        'EVO_XRA_XPLACE_ROOT': str(xp), 'EVO_XRA_ENABLE': '1',
        'EVO_FORCE_FULL_TIME': '1', 'EVO_TARGET_TOTAL_TIME_S': '3150',
        'AUTODMP_TOTAL_TIME_LIMIT': '3150', 'EVO_UNDER60_DEADLINE_S': '3300',
        'EVO_UNDER60_SCORE_SAFETY_S': '120', 'EVO_RETURN_GUARD_S': '10',
    }
    mounts = python_mounts(sys.executable)+[
        (root/'.science/challenge-probe', '/eval'), (root/'tpu', '/source/tpu'),
        (Path(__file__), '/runner.py'), ('/usr/local', '/usr/local'), ('/sys', '/sys')]
    for f in ['/etc/alternatives/ld', '/etc/ld.so.cache', '/etc/passwd', '/etc/group']:
        if Path(f).exists(): mounts.append((f, f))
    argv = [sys.executable, '/runner.py', '--child', '--method', args.method,
            '--case', args.case, '--output', '/output', '--repository', str(repo),
            '--xplace-root', str(xp), '--legalization-seconds', str(args.legalization_seconds),
            '--cpus', str(args.cpus)]
    cmd = command(argv, readonly=mounts, writable=[(out, '/output'), (repo, repo), (xp, xp)],
                  env=env, cwd='/work')
    devices = []
    for dev in glob.glob('/dev/nvidia*'):
        selected = os.environ.get('PLACEMENT_GPU_NODE')
        if selected and Path(dev).name.removeprefix('nvidia').isdigit() and dev != selected:
            continue
        devices += ['--dev-bind', dev, dev]
    idx = cmd.index('--dev')+2
    cmd[idx:idx] = devices+['--tmpfs', '/dev/shm']
    report = dict(method=args.method, case=args.case, host=platform.node(),
                  job=os.environ.get('SLURM_JOB_ID'), seconds_limit=args.seconds, cpus=args.cpus,
                  environment=env, valid=False, reward=0.0,
                  postprocess='deterministic CPU legalization' if args.method=='xplace' else 'none')
    for label, path in [('repository', args.repository), ('xplace', args.xplace_root)]:
        result = subprocess.run(['git','-C',path,'rev-parse','HEAD'],capture_output=True,text=True)
        report[label+'_commit'] = result.stdout.strip()
    (out/'launch.json').write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    with (out/'candidate.log').open('wb') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={'PATH':'/usr/bin:/bin'})
        try:
            code = proc.wait(timeout=args.seconds)
            report['exit_code'] = code
            if code: raise RuntimeError('candidate failed; see candidate.log')
            report['candidate'] = json.loads((out/'candidate.json').read_text())
            if 'ModuleNotFoundError' in (out/'candidate.log').read_text(errors='replace'):
                report['reproduction_complete'] = False
                raise RuntimeError('placer skipped work after a missing dependency; see candidate.log')
            report['reproduction_complete'] = True
        except BaseException as exc:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            report['error'] = f'{type(exc).__name__}: {exc}'
    report['candidate_wall_seconds'] = time.monotonic()-started
    if 'error' not in report:
        cmd = [sys.executable, str(root/'tpu/science/challenge_score_child.py'), '--root', str(root),
               '--case', args.case, '--positions', str(out/'positions.npy'), '--result', str(out/'metrics.json')]
        try:
            with (out/'grader.log').open('wb') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4'))
            report['metrics'] = json.loads((out/'metrics.json').read_text())
            report.update(valid=True, reward=1/(1+report['metrics']['proxy_cost']))
        except Exception as exc:
            report['grading_error'] = str(exc)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
