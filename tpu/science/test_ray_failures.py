"""Opt-in integration checks on the prepared pilot head; never launches TPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from .ray_cpu import grade


def main():
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--namespace',required=True)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);output=Path(a.output)
    if output.exists():raise RuntimeError('do not overwrite prior test evidence')
    ray.init(address=a.address,namespace=a.namespace,log_to_driver=False,
        runtime_env={'env_vars':{'PYTHONPATH':str(root)}})
    node=ray.get_runtime_context().get_node_id()
    worker=grade.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node,soft=False))
    rows=[]
    for name,source in [
        ('nonfinite_action','import numpy as np\ndef fit(*args, **kwargs): return None\ndef act(model, obs): return np.full(len(obs["current_weights"]), np.nan)\n'),
        ('prohibited_import','import jax\n'),
        ('invalid_rust_wrapper','RUST_CODE = str(1)\n')]:
        result=ray.get(worker.remote('routing' if name=='invalid_rust_wrapper' else 'portfolio',source,str(root)),timeout=60)
        assert result['reward']==0 and result['correctness']==0,result
        rows.append(dict(test=name,passed=True,result=result))
    for force in (False,True):
        source=f'# cancellation test {uuid.uuid4().hex}\nimport time\ndef fit(*args, **kwargs):\n    while True: time.sleep(.1)\ndef act(model,obs): return obs["current_weights"]\n'
        ref=worker.remote('portfolio',source,str(root));unit=None
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            for file in (root/'.science/ray-jobs').glob('*/candidate.py'):
                if file.read_text()==source:unit='science-grade-'+file.parent.name;break
            if unit:
                state=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True)
                if state.stdout.strip()=='active':break
            time.sleep(.2)
        else:raise RuntimeError('cancellation test unit never started')
        details=subprocess.check_output(['systemctl','show',unit,'-p','MemoryMax','-p','CPUQuotaPerSecUSec',
            '-p','AllowedCPUs','-p','TasksMax','-p','RuntimeMaxUSec','-p','ControlGroup'],text=True)
        group=next(x.split('=',1)[1] for x in details.splitlines() if x.startswith('ControlGroup='))
        started=time.monotonic();ray.cancel(ref,force=force)
        while time.monotonic()-started<15:
            state=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True)
            if state.stdout.strip() not in ('active','deactivating','activating') and not Path('/sys/fs/cgroup'+group).exists():break
            time.sleep(.2)
        else:raise AssertionError(f'{unit} survived Ray cancellation')
        rows.append(dict(test='force_cancel' if force else 'cooperative_cancel',passed=True,
            cleanup_seconds=time.monotonic()-started,enforced_properties=details))
    output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2),flush=True)
    ray.shutdown()


if __name__=='__main__':main()
