import json,os,subprocess,sys,uuid
from pathlib import Path
root=Path('/home/gcpuser/.cache/skyrl-ray-code/7ab6a72ce38b94369298f3c0e940722cbc2211cb3b15a7c55e14029d45278fbb')
sys.path.insert(0,str(root))
from tpu.science.placement_slots import chip_lock,chip_cpus
from tpu.science.worker import process_identity
spec=json.load(sys.stdin);chip=spec['chip'];folder=root/'.science/backend-investigation'/spec['name'];folder.mkdir(parents=True,exist_ok=False)
(folder/'candidate.py').write_text(spec['source'])
request=dict(source=str(folder/'candidate.py'),case=spec.get('case','ibm01'),root=str(root),work=str(folder/'evaluation'),backend='tpu',tpu_ids=[str(chip)],accelerator='tpu-v4-64')
(folder/'request.json').write_text(json.dumps(request))
unit='placement-diag-'+uuid.uuid4().hex
cmd=['sudo','-n','systemd-run','--unit='+unit,'--uid='+str(os.getuid()),'--gid='+str(os.getgid()),'--wait','--collect','--pipe','--quiet','--property=MemoryMax=16G','--property=MemorySwapMax=0','--property=LimitMEMLOCK=infinity','--property=CPUQuota=400%','--property=AllowedCPUs='+','.join(map(str,chip_cpus(chip))),'--property=TasksMax=1024','--property=RuntimeMaxSec=300','--property=KillMode=control-group','--property=TimeoutStopSec=2','--working-directory='+str(root),str(root/'.science/venv/bin/python'),'-m','tpu.science.placement_task','--request',str(folder/'request.json'),'--result',str(folder/'result.json'),'--owner-pid',str(os.getpid()),'--owner-start',process_identity(os.getpid())]
with chip_lock(chip):
 try:
  with (folder/'worker.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=310)
 finally:subprocess.run(['sudo','-n','systemctl','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
print(json.dumps(dict(name=spec['name'],returncode=r.returncode,folder=str(folder),result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None,log=(folder/'worker.log').read_text()[-3000:]),allow_nan=False))
