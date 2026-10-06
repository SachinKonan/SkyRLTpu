import sys
from pathlib import Path
from tpu.science.isolation import run,Limits,python_mounts
root=Path.cwd();p=root/'.science/placement-assessment-20260917/qwen-gemma-parity';name=sys.argv[1];out=p/'cpu-gradients'/name;out.mkdir(parents=True,exist_ok=False)
python=root/'.science/venv-jax-local/bin/python';problem=next((p.parent/'muse-coordinate-check/files').rglob('ibm01-problem.npz'))
run([str(python),'/runner.py','170','cpu'],limits=Limits(seconds=180,memory_gib=16,cpus=4),log=out/'candidate.log',readonly=python_mounts(python)+[(p.parent/'cpu-probe/runner.py','/runner.py'),(p/(name+'.py'),'/candidate.py'),(problem,'/problem.npz')],writable=[(out,'/output')],env={'JAX_PLATFORMS':'cpu'})
print(name);print((out/'candidate.log').read_text()[-2500:])
