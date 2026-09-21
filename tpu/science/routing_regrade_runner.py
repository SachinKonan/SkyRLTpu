"""Regrade a frozen archive without drawing samples or touching trainer state."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .routing_regrade import digest,save
from .routing_resources import contract


def run(manifest_path,output,destination,slots):
    from .routing_benchmark import evaluate_one
    root=Path.cwd();manifest=json.loads(Path(manifest_path).read_text())
    if manifest['sha256']!=digest({k:v for k,v in manifest.items() if k!='sha256'}):
        raise ValueError('source manifest checksum mismatch')
    rank=int(os.environ['SKYPILOT_NODE_RANK']);hosts=len(os.environ['SKYPILOT_NODE_IPS'].split())
    if hosts!=8 or slots not in range(1,11):raise ValueError('expected eight-host slice, 1-10 grading slots')
    evaluator=hashlib.sha256(b''.join(p.name.encode()+b'\0'+p.read_bytes()
        for p in sorted((root/'tpu/science').glob('*.py')))).hexdigest()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    selected=[p for i,p in enumerate(sorted(manifest['programs'],key=lambda p:p['source_sha256'])) if i%hosts==rank]
    def upload(path,uri):
        subprocess.run(['gcloud','storage','cp',str(path),uri,'--if-generation-match=0'],check=True,
                       stdout=subprocess.DEVNULL,timeout=180)
    def grade(program):
        sha=program['source_sha256'];path=output/(sha+'.json');uri=destination+'/verdicts/'+sha+'.json'
        # Durable completed results are reused; never choose a best duplicate replay.
        if not path.exists():
            probe=subprocess.run(['gcloud','storage','cp',uri,str(path)],capture_output=True,text=True,timeout=180)
            if probe.returncode and 'matched no objects' not in (probe.stderr+probe.stdout).lower() and '404' not in probe.stderr:
                raise RuntimeError('unable to inspect durable verdict: '+probe.stderr[-800:])
        if path.exists():
            result=json.loads(path.read_text())
            if (result['metrics'].get('regrade_evaluator_sha256')!=evaluator or
                    result['metrics'].get('source_sha256')!=sha):raise ValueError('resume identity mismatch')
            return result
        folder=output/'evaluations';folder.mkdir(exist_ok=True)
        # The isolated child is the same production worker used by Ray grading.
        result=evaluate_one(root,folder,program['code'],'production',1900,sha,slots=slots)
        m=result['metrics']
        if m.get('grader_sha256')!=evaluator or m.get('resource_contract')!=contract():
            raise ValueError('worker evaluator/resource mismatch')
        m.update(regrade_evaluator_sha256=evaluator,source_manifest_sha256=manifest['sha256'])
        save(path,result);upload(path,uri)
        print(json.dumps(dict(source=sha,correctness=result['correctness'],reward=result['reward'],
                              completed=m.get('case_count',0),rank=rank)),flush=True)
        return result
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as ex:results=list(ex.map(grade,selected))
    summary=dict(rank=rank,source_manifest_sha256=manifest['sha256'],evaluator_sha256=evaluator,
        programs=len(selected),valid=sum(r['correctness']==1 for r in results),wall_seconds=time.monotonic()-start,
        cpu_seconds=sum(r['metrics'].get('process_cpu_seconds',0) for r in results),slots_per_host=slots)
    path=output/'complete.json';save(path,summary);upload(path,destination+'/rank-'+str(rank)+'-complete.json')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--slots',type=int,default=10);a=p.parse_args()
    print(json.dumps(run(a.manifest,a.output,a.destination,a.slots)),flush=True)
