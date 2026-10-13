"""Scoped two-host bundles from the exact running single-host archives."""
import ast,copy,hashlib,io,json,os,subprocess,sys,tarfile,tempfile
from pathlib import Path
from launch_watch import ROOT,save,HERE
from od_deploy import OUT
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap_reuse import validate_assigned_migration,validate_reuse
plan=json.loads((OUT/'deployment.json').read_text());only=os.environ.get('PREPARE_PROBLEM');records=[r for r in json.loads((OUT/'twohost-bundles.json').read_text()) if r['new']['problem']!=only] if only else []
for problem,physical,ips,external,step in [
 ('circuit',[1,3],['10.128.1.95','10.128.1.93'],['34.136.9.17','35.222.180.160'],3),
 ('ac1',[2,0],['10.128.1.91','10.128.1.94'],['34.122.223.241','34.58.61.190'],12)]:
 if only and problem!=only:continue
 old=next(r for r in plan['rows'] if r['role']=='trainer' and r.get('problem','circuit')==problem)
 if (OUT/'twohost-bundles.json').exists():old=next(r['old'] for r in json.loads((OUT/'twohost-bundles.json').read_text()) if r['new']['problem']==problem)
 with tarfile.open(old['archive']) as src:
  config=Config.from_dict(json.load(src.extractfile(old['profile'])));raw=config.to_dict()
  raw.update(hosts=2,assigned_host_ips=ips,resume_min_checkpoint_step=step)
  raw['trainer'].update(hosts=2,tp=4,fsdp=2,process_bounds='1,1,2' if problem=='circuit' else '1,2,1')
  raw['cache']['trainer_compile_seed']=raw['cache']['trainer_compile']
  raw['cache']['trainer_compile']=raw['bucket']+'/'+raw['run_id']+'/trainer-compile-od-tp4-fsdp2-kv8'
  farms=['10.128.1.92','10.128.1.89'] if problem=='circuit' else ['10.128.1.90','10.128.1.88']
  raw['inference'].update(external_pool_target_leases=2,external_pool_urls={raw['model']:['http://'+ip+':24800' for ip in farms]})
  if problem=='ac1':
   doc=json.loads((OUT/'ac1-twohost-contract.json').read_text());summary=json.loads((OUT/'ac1-twohost-complete.json').read_text())
   raw.update(bootstrap_reuse_contract_sha256=doc['sha256'],bootstrap_reuse_pool_sha256=summary['pool_sha256'])
  target=Config.from_dict(raw);target.validate();validate_assigned_migration(config,target)
  if target.bootstrap_reuse_contract_sha256:
   proof=HERE/'single-seed-east5-20261010/circuit' if problem=='circuit' else OUT
   doc=proof/('contract.json' if problem=='circuit' else problem+'-twohost-contract.json');summary=proof/('complete.json' if problem=='circuit' else problem+'-twohost-complete.json')
   validate_reuse(target,json.loads(doc.read_text()),json.loads(summary.read_text()))
  profile='tpu/swarm/ray_train/profiles/od-qwen38-'+problem+'-twohost-20261013.json'
  (ROOT/profile).write_text(json.dumps(raw,indent=2)+'\n')
  replacements={profile:(ROOT/profile).read_bytes()};manifest=json.load(src.extractfile('tpu/swarm/ray_train/source_overlay/manifest.json'))
  for module in ['config','controller','bootstrap_reuse','assigned_topology']:
   name='tpu/swarm/ray_train/'+module+'.py';data=(ROOT/name).read_bytes();compile(data,name,'exec');replacements[name]=data
   if name in manifest:manifest[name]=hashlib.sha256(data).hexdigest();replacements['tpu/swarm/ray_train/source_overlay/'+name]=data
  # The same strict export failure gate already running in Circuit also protects AC1.
  if problem=='ac1':
   name='third_party/discover/ttt_discover/rl/ensemble.py';local=(ROOT/name).read_text();fn=next(n for n in ast.parse(local).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_refresh_or_keep');replacement=ast.get_source_segment(local,fn)
   path='tpu/swarm/ray_train/source_overlay/'+name;code=src.extractfile(path).read().decode();node=next(n for n in ast.parse(code).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_refresh_or_keep');code=code.replace(ast.get_source_segment(code,node),replacement);compile(code,name,'exec');manifest[name]=hashlib.sha256(code.encode()).hexdigest();replacements[path]=code.encode()
  replacements['tpu/swarm/ray_train/source_overlay/manifest.json']=json.dumps(manifest,sort_keys=True).encode();hashes={n:hashlib.sha256(d).hexdigest() for n,d in replacements.items()};dest=OUT/(problem+'-twohost-20261013.tar.gz')
  with tarfile.open(dest,'w:gz') as dst:
   for member in src:
    m=copy.copy(member)
    if m.isfile():data=replacements.pop(m.name,src.extractfile(m).read());m.size=len(data);dst.addfile(m,io.BytesIO(data))
    else:dst.addfile(m)
   for name,data in replacements.items():m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o644;dst.addfile(m,io.BytesIO(data))
 with tempfile.TemporaryDirectory(dir=OUT,prefix='twohost-verify-') as tmp:
  with tarfile.open(dest) as t:t.extractall(tmp,filter='data')
  command="from pathlib import Path; from tpu.swarm.ray_train.overlay import install; from tpu.swarm.ray_train.config import Config; install(Path('tpu/swarm/ray_train/source_overlay'),Path('verified-source')); Config.load("+repr(profile)+").validate(); print('Exact two-host bundle verified')"
  subprocess.run([sys.executable,'-c',command],cwd=tmp,env=dict(os.environ,PYTHONPATH=tmp),check=True)
 sha=hashlib.sha256(dest.read_bytes()).hexdigest();new=old|dict(problem=problem,profile=profile,archive=str(dest),sha256=sha,uri='gs://sk7524-tinker-tpu-us-central1/code-bundles/od-'+problem+'-twohost-'+sha+'.tar.gz',ips=ips,external_ips=external,physical_ranks=physical,ip=ips[0],external_ip=external[0],resume_min_checkpoint_step=step)
 records.append(dict(old=old,new=new,files=hashes));print(problem,sha,flush=True)
save(OUT/'twohost-bundles.json',records)
