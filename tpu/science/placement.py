"""Isolated macro placement with native PLC grading and a bounded helper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from .cgroup_limits import envelope, metrics as memory_metrics
from .contracts import placement as validate
from .isolation import Limits, Session, python_mounts
from .placement_cost import score, COMPLETION
from .prepare_placement import PLC_SHA256, digest
from .resources import COMPUTATIONAL_STDLIB, library_environment
from .rewards import placement as reward, valid

ALLOWED=set(COMPUTATIONAL_STDLIB)|{'numpy','scipy','networkx','placement_api'}


def check_imports(source):
    for node in ast.walk(ast.parse(source)):
        names=[]
        if isinstance(node,ast.Import):names=[a.name for a in node.names]
        if isinstance(node,ast.ImportFrom):
            if node.level:raise ValueError('relative imports prohibited')
            names=[node.module or '']
        if any(n.split('.')[0] not in ALLOWED for n in names):
            raise ValueError('import outside placement allowlist: '+','.join(names))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('open','exec','eval','compile','__import__'):
            raise ValueError('dynamic code and filesystem access prohibited')


def evaluate(source, *, data, binary, work, python=sys.executable):
    check_imports(source)
    group,limit=envelope(16)
    if digest(binary)!=PLC_SHA256:raise ValueError('PLC binary checksum mismatch')
    data,work=Path(data).resolve(),Path(work).resolve()
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['completion']!=COMPLETION or manifest['binary_sha256']!=PLC_SHA256:
        raise ValueError('baseline and grader configuration mismatch')
    work.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+600;cases=[];costs=[]
    runner=Path(__file__).with_name('placement_child.py').resolve()
    for row in manifest['cases']:
        case=data/row['name'];folder=work/row['name'];folder.mkdir();inputs=folder/'input';inputs.mkdir()
        for file,sha in row['sha256'].items():
            if digest(case/file)!=sha:raise ValueError('case checksum mismatch')
        problem=json.loads((case/'problem.json').read_text())
        shutil.copyfile(case/'problem.json',inputs/'problem.json');(inputs/'candidate.py').write_text(source)
        solve_start=time.monotonic();helper_seconds=0.;calls=0
        session=Session([str(python),'/runner.py'],limits=Limits(min(60,deadline-time.monotonic()),16,4),
            log=folder/'candidate.log',readonly=python_mounts(python)+[(runner,'/runner.py'),(inputs,'/input')],
            env=library_environment(4))
        try:
            while True:
                response=session.receive()
                if not isinstance(response,dict) or set(response)!={'kind','placement'}:
                    raise ValueError('invalid candidate protocol')
                if response['kind']=='result':
                    result=validate(problem,response['placement']);session.finish();break
                if response['kind']!='evaluate':raise ValueError('unknown helper operation')
                calls+=1;helper_start=time.monotonic()
                try:
                    placement=validate(problem,response['placement'])
                    measured,_=score(None,binary,case/'netlist.pb.txt',case/'initial.plc',problem,placement,
                        timeout=min(session.deadline,deadline)-time.monotonic())
                    normalized=float(np.array(measured)/np.array(row['baseline'])@np.array([.5,.4,.1]))
                    session.send(dict(metrics=dict(wirelength=measured[0],congestion=measured[1],density=measured[2],normalized_cost=normalized)))
                except ValueError as exc:session.send(dict(error=str(exc)))
                helper_seconds+=time.monotonic()-helper_start
        finally:session.close()
        solve_seconds=time.monotonic()-solve_start
        measured,grading_seconds=score(None,binary,case/'netlist.pb.txt',case/'initial.plc',problem,result,
            timeout=min(90,deadline-time.monotonic()))
        costs.append(measured)
        cases.append(dict(name=row['name'],costs=measured,candidate_seconds=solve_seconds,
            helper_calls=calls,helper_seconds=helper_seconds,grading_seconds=grading_seconds))
        (folder/'placement.json').write_text(json.dumps(result)+'\n')
    score_value,metrics=reward([r['baseline'] for r in manifest['cases']],costs)
    if time.monotonic()>deadline:raise TimeoutError('placement overall deadline exhausted')
    metrics.update(cases=cases,hardware='cpu',cpus=4,memory_gib=16,total_seconds=time.monotonic()-start,
        allocation_memory=memory_metrics(group),memory_limit_bytes=limit,binary_sha256=PLC_SHA256,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),seed=1)
    result=valid(score_value,metrics);(work/'verdict.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--data',required=True)
    p.add_argument('--binary',required=True);p.add_argument('--work',required=True);a=p.parse_args()
    print(json.dumps(evaluate(Path(a.source).read_text(),data=a.data,binary=a.binary,work=a.work),indent=2))
