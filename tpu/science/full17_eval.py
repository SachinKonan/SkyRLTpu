"""Local full-suite audit; separate from the four-case training environment.

One GPU pipeline and four CPU case workers. Sources are frozen before launch.
No partial-suite mean is presented as a qualifying score.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = tuple(f'ibm{i:02d}' for i in range(1, 19) if i != 5)
MODELS = ('gemma', 'qwen', 'muse')
CUDA = ROOT / '.science/venv-cuda/bin/python'
JAX = ROOT / '.science/venv-jax-local/bin/python'


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_case(out, case):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / '.science/challenge-probe'))
    from macro_place.loader import load_benchmark_from_dir
    from tpu.science.challenge_contract import problem_from_native
    torch.set_num_threads(4)
    native = ROOT / '.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04' / case
    benchmark, plc = load_benchmark_from_dir(str(native))
    problem = problem_from_native(benchmark, plc)
    old = ROOT / 'tpu/science/results/placement/xplace-start-v1' / f'{case}-problem.npz'
    if old.exists():
        from tpu.science.placement_warm_start import verified_inputs
        old = verified_inputs(ROOT)[case]
        with np.load(old, allow_pickle=False) as data:
            positions = data['initial_positions'].copy()
        source = ROOT / 'tpu/science/results/placement/gpu-suite-20260916' / f'xplace-{case}'
        provenance = 'existing immutable training start'
    else:
        source = out / f'xplace-{case}'
        if not json.loads((source / 'report.json').read_text())['valid']:
            raise ValueError('Xplace start failed independent validation')
        positions = np.load(source / 'positions.npy', allow_pickle=False)
        provenance = 'new full-suite Xplace start'
    if not np.array_equal(positions[problem['fixed']], problem['initial_positions'][problem['fixed']]):
        raise ValueError('fixed positions changed in start')
    problem['initial_positions'] = positions
    target = out / 'inputs'; target.mkdir(exist_ok=True)
    np.savez_compressed(target / f'{case}.npz', **problem)
    np.save(target / f'{case}-start.npy', positions, allow_pickle=False)
    report = json.loads((source / 'report.json').read_text())
    save(target / f'{case}.json', dict(case=case, provenance=provenance,
         xplace_report=str(source / 'report.json'), xplace_seconds=report['candidate_wall_seconds'],
         problem_sha256=digest(target / f'{case}.npz'),
         native_sha256={name:digest(native/name) for name in ('netlist.pb.txt','initial.plc')}))


def rss_tree(pid):
    """Sample actual resident memory, not JAX's large virtual address space."""
    seen = set(); pending = [pid]; total = 0
    while pending:
        current = pending.pop()
        if current in seen: continue
        seen.add(current)
        try:
            for line in Path(f'/proc/{current}/status').read_text().splitlines():
                if line.startswith('VmRSS:'): total += int(line.split()[1]) * 1024
            pending.extend(map(int, Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except (FileNotFoundError, ProcessLookupError): pass
    return total


def cpu_case(out, case, model):
    from tpu.science.isolation import command, python_mounts
    from tpu.science.fast_proxy_deployment import verified_mounts
    folder = out / f'{model}-{case}'; folder.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(method=model, case=case, valid=False, candidate_limit_seconds=180,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), memory_watchdog_gib=8,
                  memory_enforcement='250ms aggregate descendant RSS watchdog; not a cgroup hard cap',
                  source_sha256=digest(out/f'{model}.py'), backend='jax-cpu', seed=42)
    try:
        helper, hashes = verified_mounts(ROOT)
        mounts = python_mounts(JAX) + helper + [
            (out/f'{model}.py','/candidate.py'), (out/'inputs'/f'{case}.npz','/problem.npz'),
            (ROOT/'tpu/science/challenge_candidate_child.py','/runner.py')]
        env = dict(JAX_PLATFORMS='cpu', JAX_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='4',
                   TF_NUM_INTEROP_THREADS='1', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4',
                   XDG_CACHE_HOME='/tmp/cache')
        cmd = command([str(JAX),'/runner.py','170','cpu-jax','fast_proxy_v1'],
                      readonly=mounts,writable=[(folder,'/output')],env=env)
        with (folder/'candidate.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            peak = 0
            try:
                while proc.poll() is None:
                    peak = max(peak, rss_tree(proc.pid))
                    if peak > 8*1024**3: raise MemoryError('candidate process tree exceeded 8 GiB RSS')
                    if time.monotonic()-started > 180: raise TimeoutError('candidate exceeded 180 seconds')
                    time.sleep(.25)
                if proc.returncode: raise RuntimeError(f'candidate exit {proc.returncode}')
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                report.update(candidate_wall_seconds=time.monotonic()-started, peak_tree_rss_bytes=peak)
        with (folder/'grader.log').open('wb') as log:
            subprocess.run([str(CUDA),'-m','tpu.science.challenge_score_child','--root',str(ROOT),
                '--case',case,'--positions',str(folder/'positions.npy'),'--result',str(folder/'metrics.json')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        report.update(valid=True, metrics=json.loads((folder/'metrics.json').read_text()),helper_hashes=hashes,
                      positions_sha256=digest(folder/'positions.npy'))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['total_seconds'] = time.monotonic()-started
    start_meta=json.loads((out/'inputs'/f'{case}.json').read_text())
    report['xplace_precomputation_seconds']=start_meta['xplace_seconds']
    report['end_to_end_seconds']=report['total_seconds']+start_meta['xplace_seconds']
    save(folder/'report.json',report)


def summarize(out):
    summary = {}
    for method in (*MODELS,'abuplace','archgen'):
        rows = {case:json.loads((out/f'{method}-{case}'/'report.json').read_text())
                for case in CASES if (out/f'{method}-{case}'/'report.json').exists()}
        valid = {case:r for case,r in rows.items() if r.get('valid')}
        complete = len(rows)==len(CASES)
        eligible = complete and len(valid)==len(CASES)
        summary[method] = dict(completed=len(rows),valid=len(valid),required=17,eligible=eligible,
            mean_proxy_cost=sum(r['metrics']['proxy_cost'] for r in valid.values())/17 if eligible else None,
            cases={c:dict(valid=r.get('valid'),proxy_cost=r.get('metrics',{}).get('proxy_cost'),
                          error=r.get('error',r.get('grading_error'))) for c,r in rows.items()})
    save(out/'summary.json',summary)


def supervise(out):
    lock=(out/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=(ROOT/'.science/full17-a40-gpu0.lock').open('w')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=sorted(os.sched_getaffinity(0)); assert len(cpus)>=48
    save(out/'supervisor.json',dict(pid=os.getpid(),started_unix=time.time(),host=os.uname().nodename,
         gpu=0,cpu_case_workers=4,gpu_cpus=cpus[:16],cpu_worker_cpus=cpus[32:48]))
    slots=queue.Queue()
    for i in range(4): slots.put(cpus[32+i*4:36+i*4])
    output_lock=threading.Lock()

    def command_run(cmd,log,affinity,timeout,env=None):
        with Path(log).open('ab') as stream:
            return subprocess.run(['taskset','-c',','.join(map(str,affinity)),*map(str,cmd)],
                stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,env=env).returncode

    def gpu(method,case):
        folder=out/f'{method}-{case}'
        if (folder/'report.json').exists(): return
        repo=ROOT/('.science/abuplace' if method=='abuplace' else '.science/archgen-cuda-run')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PLACEMENT_GPU_NODE='/dev/nvidia0',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='16',MKL_NUM_THREADS='1')
        command_run([CUDA,ROOT/'tpu/science/placement_gpu_suite.py','--method',method,'--case',case,
            '--repository',repo,'--xplace-root',ROOT/'.science/xplace-cuda','--output',folder,
            '--seconds',1100 if method=='xplace' else 3450],out/'gpu-driver.log',cpus[:16],3800,env)
        with output_lock: summarize(out)

    def run_case(case):
        affinity=slots.get()
        try:
            if not (out/'inputs'/f'{case}.json').exists():
                code=command_run([CUDA,'-m','tpu.science.full17_eval','--out',out,'--mode','input','--case',case],
                                 out/f'prepare-{case}.log',affinity,180)
                if code: raise RuntimeError(f'input preparation failed: {case}')
                code=command_run([CUDA,'-m','tpu.science.challenge_score_child','--root',ROOT,'--case',case,
                    '--positions',out/'inputs'/f'{case}-start.npy','--result',out/'inputs'/f'{case}-score.json'],
                    out/f'prepare-{case}.log',affinity,120)
                if code: raise RuntimeError(f'start validation failed: {case}')
            if not (out/'inputs'/f'{case}-score.json').exists(): raise RuntimeError('unverified start')
            for model in MODELS:
                if (out/f'{model}-{case}'/'report.json').exists(): continue
                code=command_run([CUDA,'-m','tpu.science.full17_eval','--out',out,'--mode','cpu',
                    '--case',case,'--model',model],out/f'{model}-{case}-driver.log',affinity,330)
                if code: raise RuntimeError(f'CPU driver failed: {model}/{case}')
                with output_lock: summarize(out)
        finally: slots.put(affinity)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[]
        initial=('ibm01','ibm04','ibm08','ibm18')
        futures.extend(pool.submit(run_case,case) for case in initial)
        for case in CASES:
            if case not in initial:
                gpu('xplace',case)
                futures.append(pool.submit(run_case,case))
        for case in CASES:
            for method in ('abuplace','archgen'): gpu(method,case)
        errors=[]
        for future in futures:
            try: future.result()
            except Exception as exc: errors.append(str(exc))
        save(out/'completion.json',dict(finished_unix=time.time(),errors=errors))
        summarize(out)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['supervise','input','cpu'],default='supervise')
    p.add_argument('--case',choices=CASES);p.add_argument('--model',choices=MODELS)
    a=p.parse_args();out=Path(a.out).resolve()
    if a.mode=='input': input_case(out,a.case)
    elif a.mode=='cpu': cpu_case(out,a.case,a.model)
    else: supervise(out)
