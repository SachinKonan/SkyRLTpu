"""One Archgen CUDA run with a 3600s external deadline and separate grading."""
import argparse
import contextlib
import glob
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def child(source, output):
    started = time.monotonic()
    import numpy as np
    import torch
    assert torch.cuda.is_available(), 'CUDA required; do not silently run CPU fallback'
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    from macro_place.loader import load_benchmark_from_dir
    Path('/work/external').symlink_to('/eval/external', target_is_directory=True)
    b, _ = load_benchmark_from_dir('/eval/external/MacroPlacement/Testcases/ICCAD04/ibm18')
    spec = importlib.util.spec_from_file_location('archgen_cuda_submission', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    positions = module.OptimalPlacer(seed=42).place(b)
    torch.cuda.synchronize()
    positions = positions.detach().cpu().numpy().astype(np.float32)
    np.save(Path(output)/'positions.npy', positions, allow_pickle=False)
    (Path(output)/'candidate.json').write_text(json.dumps({
        'seconds': time.monotonic()-started, 'torch':torch.__version__,
        'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
    }, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.child:
        child(args.source, args.output)
        return
    from isolation import command, python_mounts
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output).resolve()
    probe = root/'.science/challenge-probe'
    repo = root/'.science/archgen-cuda-run'
    xplace = root/'.science/xplace-cuda'
    out.mkdir(parents=True, exist_ok=True)
    env = {
        'PATH':str(Path(sys.executable).parent)+':/usr/local/cuda-12.6/bin:/usr/bin:/bin',
        'PYTHONPATH':'/eval:'+str(repo), 'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES','0'),
        'CUDA_HOME':'/usr/local/cuda-12.6', 'MPLCONFIGDIR':'/tmp/matplotlib', 'MPLBACKEND':'Agg',
        'XDG_CACHE_HOME':'/tmp/cache','XDG_CONFIG_HOME':'/tmp/config',
        'CUDA_CACHE_PATH':'/tmp/cuda-cache','TORCH_HOME':'/tmp/torch',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'EVO_XRA_XPLACE_ROOT':str(xplace),'XPLACE_PYTHON':sys.executable,
        'EVO_XRA_ENABLE':'1','EVO_FORCE_FULL_TIME':'1','EVO_TARGET_TOTAL_TIME_S':'3300',
        'AUTODMP_TOTAL_TIME_LIMIT':'3300','EVO_UNDER60_DEADLINE_S':'3420',
        'EVO_UNDER60_SCORE_SAFETY_S':'120','EVO_RETURN_GUARD_S':'10',
    }
    mounts = python_mounts(sys.executable)+[(probe,'/eval'),(Path(__file__),'/runner.py'),
              ('/usr/local','/usr/local'), ('/sys','/sys')]
    if Path('/etc/alternatives/ld').exists():
        mounts.append(('/etc/alternatives/ld','/etc/alternatives/ld'))
    for system_file in ['/etc/ld.so.cache','/etc/passwd','/etc/group']:
        if Path(system_file).exists():
            mounts.append((system_file,system_file))
    cmd = command([sys.executable,'/runner.py','--child','--source',str(repo/'submissions/final_placer.py'),
                   '--output','/output'], readonly=mounts,
                  writable=[(repo,repo),(xplace,xplace),(out,'/output')],env=env,cwd='/work')
    devices = []
    selected_node = os.environ.get('PLACEMENT_GPU_NODE')
    for dev in glob.glob('/dev/nvidia*'):
        if selected_node and Path(dev).name.removeprefix('nvidia').isdigit() and dev != selected_node:
            continue
        devices += ['--dev-bind',dev,dev]
    # Add devices after the private /dev mount, which would otherwise hide them.
    device_index = cmd.index('--dev') + 2
    cmd[device_index:device_index] = devices + ['--tmpfs','/dev/shm']
    report = {'case':'ibm18','profile':'one_cuda_run_challenge_time_limit',
              'candidate_limit_seconds':3600,'cpu_cores':16,'host_memory_gib':100,
              'environment':env,'valid':False,'reward':0.0}
    for label, checkout in [('archgen',repo),('xplace',xplace)]:
        report[label+'_commit'] = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    started = time.monotonic()
    with (out/'candidate.log').open('wb') as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                env={'PATH':'/usr/bin:/bin'})
        try:
            code = proc.wait(timeout=3600)
            report['candidate_exit_code'] = code
            if code:
                raise RuntimeError(f'candidate exited {code}; see candidate.log')
            report['candidate'] = json.loads((out/'candidate.json').read_text())
        except BaseException as exc:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            report['error'] = f'{type(exc).__name__}: {exc}'
    report['candidate_wall_seconds'] = time.monotonic()-started
    if 'error' not in report:
        grade_start = time.monotonic()
        import numpy as np
        import torch
        sys.path.insert(0,str(probe))
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement
        with (out/'grader.log').open('w') as log, contextlib.redirect_stdout(log):
            b, plc = load_benchmark_from_dir(str(probe/'external/MacroPlacement/Testcases/ICCAD04/ibm18'))
            positions = torch.from_numpy(np.load(out/'positions.npy',allow_pickle=False))
            valid, errors = validate_placement(positions,b)
            report.update(valid=valid,violations=errors)
            if valid:
                metrics = {k:float(v) for k,v in compute_proxy_cost(positions,b,plc).items()}
                report['metrics'] = metrics
                report['valid'] = all(np.isfinite(v) for v in metrics.values()) and metrics['overlap_count']==0
                if report['valid']:
                    report['reward'] = max(1e-6,1/(1+metrics['proxy_cost']))
        report['grading_seconds'] = time.monotonic()-grade_start
    report['total_seconds'] = time.monotonic()-started
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
