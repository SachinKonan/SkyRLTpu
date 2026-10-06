"""Bounded A10 preprocessing, using pinned AbuPlace GP and canonical scoring.

Run with modal run tpu/science/modal_xplace.py. Outputs are downloaded to GPFS.
Only curated sources/benchmarks are shipped; no local credentials or homes.
"""
import io
import json
import os
from pathlib import Path
import tarfile
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/source')
PAYLOAD = ROOT / '.science/modal-xplace/payload.tar.gz'
app = modal.App('xplace-three-a10-20260921')
image = (
    modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11')
    .apt_install('build-essential', 'git', 'libcairo2-dev', 'pkg-config', 'libboost-all-dev', 'libgomp1')
    .pip_install('torch==2.5.1', 'torchvision==0.20.1', index_url='https://download.pytorch.org/whl/cu124')
    .pip_install('numpy==2.2.6', 'scipy', 'tqdm', 'absl-py', 'matplotlib', 'cmake==3.31.6', 'ninja', 'pandas', 'numba', 'cairocffi', 'opencv-python-headless', 'seaborn', 'pulp', 'igraph')
    .apt_install('bison', 'flex', 'zlib1g-dev')
    .add_local_file(str(PAYLOAD), '/payload.tar.gz', copy=True)
    .run_commands('tar xzf /payload.tar.gz -C / && rm /payload.tar.gz',
                  'cmake -S /xplace -B /xplace/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DPYTHON_EXECUTABLE=$(which python)',
                  'timeout 2400 cmake --build /xplace/build --parallel 4',
                  'cmake --install /xplace/build',
                  'cp -a /xplace/cpp_to_py/cpybin /repository/abuplace/Xplace/cpp_to_py/',
                  'mkdir -p /source/.science /work && ln -s /eval /source/.science/challenge-probe')
    .env({'PYTHONPATH':'/source:/eval:/repository', 'OMP_NUM_THREADS':'8',
          'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLBACKEND':'Agg',
          'XPLACE_PYTHON':'/usr/local/bin/python', 'CUDA_HOME':'/usr/local/cuda',
          'LD_LIBRARY_PATH':'/repository/abuplace/Xplace/cpp_to_py/cpybin:/usr/local/cuda/lib64'})
)

@app.function(image=image, gpu='A10', cpu=(8,8), memory=(16384,16384),
              timeout=1980, retries=0, max_containers=4, scaledown_window=2)
def case_variant(case: str, variant: str):
    import subprocess
    import sys
    import time
    import shutil
    import hashlib
    if case not in [f'ibm{i:02d}' for i in range(1,19) if i != 5] or variant not in ('off','rudy','rudy_hv'):
        raise ValueError('unknown task')
    out = Path('/output')
    if out.exists(): shutil.rmtree(out)
    out.mkdir()
    link=Path('/work/external')
    if link.is_symlink(): link.unlink()
    started=time.monotonic()
    report={'case':case,'variant':variant,'method':'xplace-abu','valid':False,
            'repository_commit':'a24087c45588f1873eb2dc18293e407e5477041d',
            'executor':'modal-a10','seconds_limit':1800,'cpus':8}
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu','--format=csv','-l','1'],stdout=(out/'gpu-memory.csv').open('w'))
    try:
        with (out/'candidate.log').open('w') as log:
            subprocess.run([sys.executable,'/source/tpu/science/placement_gpu_suite.py','--child',
                '--method','xplace-abu','--variant',variant,'--case',case,'--output','/output',
                '--repository','/repository','--xplace-root','/xplace','--cpus','8'],
                cwd='/work',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,
                env=dict(os.environ,XPLACE_PYTHON=sys.executable))
        report['candidate_wall_seconds']=time.monotonic()-started
        report['candidate']=json.loads((out/'candidate.json').read_text())
        with (out/'grader.log').open('w') as log:
            subprocess.run([sys.executable,'/source/tpu/science/challenge_score_child.py',
                '--root','/source','--case',case,'--positions',str(out/'positions.npy'),
                '--result',str(out/'metrics.json')],check=True,timeout=120,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        report['metrics']=json.loads((out/'metrics.json').read_text())
        report.update(valid=True,reward=1/(1+report['metrics']['proxy_cost']))
    except Exception as exc:
        report['error']=repr(exc)
    finally:
        monitor.terminate(); monitor.wait()
    report['total_seconds']=time.monotonic()-started
    report['native_sha256']={name:hashlib.sha256((Path('/eval/external/MacroPlacement/Testcases/ICCAD04')/case/name).read_bytes()).hexdigest() for name in ('netlist.pb.txt','initial.plc')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as archive:
        for path in out.iterdir():
            if path.is_file(): archive.add(path,arcname=path.name)
    return report,payload.getvalue()

@app.local_entrypoint()
def main(full: bool=False):
    dest=ROOT/'.science/modal-xplace/results'
    dest.mkdir(parents=True,exist_ok=True)
    pilot=[('ibm01',v) for v in ('off','rudy','rudy_hv')]+[('ibm18','rudy_hv'),('ibm10','rudy_hv')]
    if full:
        for case,variant in pilot:
            prior=json.loads((dest/f'{case}-{variant}'/'report.json').read_text())
            if not prior['valid']: raise RuntimeError('pilot not validated')
        tasks=[(f'ibm{i:02d}',v) for i in range(1,19) if i!=5 for v in ('off','rudy','rudy_hv')]
    else: tasks=pilot
    # Durable reservation before submission; failed/uncertain calls require review,
    # never blindly retried, bounding total paid attempts across invocations.
    todo=[]
    for case,variant in tasks:
        target=dest/f'{case}-{variant}'
        if (target/'report.json').exists(): continue
        target.mkdir(exist_ok=True)
        with (target/'submitted.json').open('x') as f:
            json.dump({'case':case,'variant':variant,'max_seconds':1980},f)
        todo.append((case,variant))
    failures=[]
    for report,data in case_variant.starmap(todo,order_outputs=False):
        target=dest/f"{report['case']}-{report['variant']}"
        (target/'artifacts.tar.gz').write_bytes(data)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name != member.name: raise ValueError('unexpected artifact')
                (target/member.name).write_bytes(archive.extractfile(member).read())
        print(json.dumps(report),flush=True)
        if not report['valid']: failures.append(str(target))
    for case,variant in tasks:
        path=dest/f'{case}-{variant}'/'report.json'
        if not path.exists() or not json.loads(path.read_text())['valid']:
            failures.append(str(path))
    if failures: raise RuntimeError('Failed tasks: '+', '.join(failures))
