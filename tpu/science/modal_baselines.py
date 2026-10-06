"""Full pinned public baselines on Modal A10; separate from Xplace seed generation."""
import io
import json
import os
from pathlib import Path
import tarfile
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/source')
OUT = ROOT / '.science/modal-baselines'
CASES = [f'ibm{i:02d}' for i in range(1, 19) if i != 5]
app = modal.App('circuit-full-baselines-a10-20260922')
if modal.is_local():
    from tpu.science.modal_xplace import image as xplace_image
    image = (xplace_image.pip_install('setuptools==80.9.0')
        .add_local_file(str(OUT/'overlay.tar.gz'), '/baseline-overlay.tar.gz', copy=True)
        .run_commands('tar xzf /baseline-overlay.tar.gz -C / && rm /baseline-overlay.tar.gz',
                      'find /repository/abuplace/extensions -name "*.so" -delete')
        .env({'EVO_XRA_XPLACE_ROOT':'/xplace', 'EVO_XRA_ENABLE':'1',
              'EVO_FORCE_FULL_TIME':'1', 'EVO_TARGET_TOTAL_TIME_S':'3150',
              'AUTODMP_TOTAL_TIME_LIMIT':'3150', 'EVO_UNDER60_DEADLINE_S':'3300',
              'EVO_UNDER60_SCORE_SAFETY_S':'120', 'EVO_RETURN_GUARD_S':'10',
              'PYTHONUNBUFFERED':'1', 'TORCH_EXTENSIONS_DIR':'/tmp/torch_extensions',
              'TRITON_CACHE_DIR':'/tmp/triton/cache', 'XDG_CACHE_HOME':'/tmp/cache'}))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='A10', cpu=(8,8), memory=(16384,16384),
              timeout=3540, retries=0, max_containers=4, scaledown_window=2)
def evaluate(method: str, case: str):
    import hashlib
    import shutil
    import subprocess
    import sys
    import time
    assert method in ('archgen','abuplace') and case in CASES
    out = Path('/output')
    if out.exists(): shutil.rmtree(out)
    out.mkdir()
    link = Path('/work/external')
    if link.is_symlink(): link.unlink()
    # Rebuild native CPU helpers on the assigned CPU, including reused containers.
    for p in Path('/repository/abuplace/extensions').glob('*.so'): p.unlink()
    started = time.monotonic()
    report = dict(method=method,case=case,valid=False,reward=0.,executor='modal-a10',
                  candidate_limit_seconds=3300,scoring_limit_seconds=180,cpus=8,
                  source_commit={'archgen':'6b3661bad55d81c00fad151d81ec1e74ef0271e0',
                                 'abuplace':'a24087c45588f1873eb2dc18293e407e5477041d'}[method])
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu','--format=csv','-l','5'],stdout=(out/'gpu-memory.csv').open('w'))
    try:
        with (out/'candidate.log').open('w') as log:
            subprocess.run([sys.executable,'/source/tpu/science/placement_gpu_suite.py','--child',
                '--method',method,'--case',case,'--output','/output','--repository','/repository',
                '--xplace-root','/xplace','--cpus','8'],cwd='/work',stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=3300)
        report['candidate_seconds'] = time.monotonic()-started
        report['candidate'] = json.loads((out/'candidate.json').read_text())
        text = (out/'candidate.log').read_text(errors='replace')
        if 'ModuleNotFoundError' in text or 'ImportError:' in text:
            raise RuntimeError('Public baseline skipped a dependency; see candidate.log')
        score_started = time.monotonic()
        with (out/'grader.log').open('w') as log:
            subprocess.run([sys.executable,'/source/tpu/science/challenge_score_child.py',
                '--root','/source','--case',case,'--positions',str(out/'positions.npy'),
                '--result',str(out/'metrics.json')],stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4'))
        report['scoring_seconds'] = time.monotonic()-score_started
        report['metrics'] = json.loads((out/'metrics.json').read_text())
        report.update(valid=True,reward=1/(1+report['metrics']['proxy_cost']))
    except Exception as exc:
        report['error'] = repr(exc)
    finally:
        monitor.terminate(); monitor.wait()
    report['total_seconds'] = time.monotonic()-started
    report['native_sha256'] = {name:hashlib.sha256((Path('/eval/external/MacroPlacement/Testcases/ICCAD04')/case/name).read_bytes()).hexdigest() for name in ('netlist.pb.txt','initial.plc')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for p in out.iterdir():
            if p.is_file(): archive.add(p,arcname=p.name)
    return report,buf.getvalue()

@app.local_entrypoint()
def main():
    # Includes all previously reserved calls even if billing is delayed or a client died.
    plan=json.loads((OUT/'budget.json').read_text())
    assert plan['approved_total_cap_usd']==100 and plan['reserved_total_usd']<90
    dest=OUT/'results';dest.mkdir(exist_ok=True)
    todo=[]
    for case in CASES:
        for method in ['archgen','abuplace']:
            target=dest/f'{method}-{case}';target.mkdir(exist_ok=True)
            if (target/'report.json').exists():continue
            with (target/'submitted.json').open('x') as f:
                json.dump(dict(method=method,case=case,max_seconds=3540),f)
            todo.append((method,case))
    for report,data in evaluate.starmap(todo,order_outputs=False):
        target=dest/f"{report['method']}-{report['case']}"
        (target/'artifacts.tar.gz').write_bytes(data)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name!=member.name:raise ValueError('unexpected artifact')
                (target/member.name).write_bytes(archive.extractfile(member).read())
        print(json.dumps(report),flush=True)
    summary={}
    for method in ['archgen','abuplace']:
        rows=[json.loads((dest/f'{method}-{c}'/'report.json').read_text()) for c in CASES]
        valid=[r for r in rows if r['valid']]
        mean=sum(r['metrics']['proxy_cost'] for r in valid)/17 if len(valid)==17 else None
        summary[method]=dict(valid=len(valid),required=17,mean_proxy_cost=mean,reward=1/(1+mean) if mean is not None else None)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
