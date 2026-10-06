import json,os,subprocess,sys,time,hashlib
from pathlib import Path
from tpu.science.isolation import run,Limits,python_mounts
root=Path.cwd();base=root/'.science/placement-assessment-20260917';model,case=sys.argv[1:3];out=base/'qwen-gemma-parity/cpu'/model/case;out.mkdir(parents=True,exist_ok=False)
python=root/'.science/venv-jax-local/bin/python';problem=next((base/'muse-coordinate-check/files').rglob(case+'-problem.npz'));candidate=base/(model+'-best.py')
manifest=json.loads((root/'tpu/science/results/placement/xplace-start-v1/manifest.json').read_text());assert hashlib.sha256(problem.read_bytes()).hexdigest()==manifest['cases'][case]['problem_sha256']
report=dict(model=model,case=case,backend='cpu',source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(problem.read_bytes()).hexdigest(),cpu_affinity=sorted(os.sched_getaffinity(0)))
try:
 report['candidate_wall_seconds']=run([str(python),'/runner.py','170','cpu'],limits=Limits(seconds=180,memory_gib=16,cpus=4),log=out/'candidate.log',readonly=python_mounts(python)+[(base/'cpu-probe/runner.py','/runner.py'),(candidate,'/candidate.py'),(problem,'/problem.npz')],writable=[(out,'/output')],env={'JAX_PLATFORMS':'cpu','XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4'})
 report['child']=json.loads((out/'child.json').read_text())
 cmd=[str(root/'.science/venv/bin/python'),'-m','tpu.science.challenge_score_child','--root',str(root),'--case',case,'--positions',str(out/'positions.npy'),'--result',str(out/'score.json')]
 with (out/'grader.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=90,env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
 if r.returncode:raise RuntimeError((out/'grader.log').read_text()[-1600:])
 report['score']=json.loads((out/'score.json').read_text());report['valid']=True
except Exception as e:report.update(valid=False,error=str(e))
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
