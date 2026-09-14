"""Run native bootstrap plus a rank-one pilot driver within the same pool job."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    profile=sys.argv[1];rank=int(os.environ['SKYPILOT_NODE_RANK'])
    from tpu.swarm.ray_train.config import Config
    config=Config.load(profile)
    if not config.inference_only:
        raise ValueError('routing_model_host runs sampling pilots only')
    proc=subprocess.Popen([sys.executable,'-m','tpu.swarm.ray_train.bootstrap',profile])
    driver=None
    try:
        if rank==1:
            python=Path(config.root).expanduser()/'envs/controller/bin/python'
            deadline=time.monotonic()+1800
            while not python.exists():
                if proc.poll() is not None:raise RuntimeError('bootstrap exited before controller preparation')
                if time.monotonic()>deadline:raise TimeoutError('controller Python missing')
                time.sleep(5)
            # Ray may not yet listen. Wait for its local bootstrap registration.
            import socket
            head=os.environ['SKYPILOT_NODE_IPS'].split()[0]
            while True:
                try:
                    with socket.create_connection((head,config.ports.ray),timeout=2):break
                except OSError:
                    if proc.poll() is not None or time.monotonic()>deadline:raise TimeoutError('Ray startup')
                    time.sleep(5)
            with open('model-driver.log','wb') as log:
                driver=subprocess.Popen([str(python),'-m','tpu.science.routing_model_driver','--profile',profile],stdout=log,stderr=subprocess.STDOUT)
            while driver.poll() is None:
                if proc.poll() is not None:raise RuntimeError('native bootstrap exited during pilot')
                time.sleep(5)
            code=driver.returncode
            proc.wait(timeout=180)
            raise SystemExit(code)
        raise SystemExit(proc.wait())
    finally:
        for child in (driver,proc):
            if child and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
