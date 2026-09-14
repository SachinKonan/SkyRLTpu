"""Freeze a legal four-netlist suite using the real pinned native PLC engine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .placement_data import parse
from .placement_cost import connection, score, COMPLETION
from .contracts import placement

PLC_SHA256='86fe9a2841fc21d3c18bb838d93fff128ceb51f82490d561e22985caab00c9b3'
CASES=('sample_clustered','simple_grouped_with_coords_with_blockage','toy_macro_stdcell','ariane')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legalize(repo,binary,netlist,initial,problem):
    cols,rows=problem['num_cols'],problem['num_rows']
    cells=np.arange(cols*rows)
    xs=(cells%cols+.5)*problem['width']/cols
    ys=(cells//cols+.5)*problem['height']/rows
    result=dict(cell_ids=[],orientations=[m['orientation'] for m in problem['movable_macros']])
    locations={}
    with connection(repo,binary,netlist,initial,problem) as plc:
        for macro in problem['movable_macros']:
            i=macro['id']
            if plc.get_node_name(i)!=problem['nodes'][i]['name']:
                raise ValueError('parser/native node index mismatch')
            plc.unfix_node_coord(i);plc.unplace_node(i)
        for macro in sorted(problem['movable_macros'],key=lambda m:(-m['width']*m['height'],m['id'])):
            order=np.argsort((xs-macro['x'])**2+(ys-macro['y'])**2,kind='stable')
            for cell in order:
                if plc.can_place_node(macro['id'],int(cell)):
                    plc.place_node(macro['id'],int(cell));locations[macro['id']]=int(cell)
                    break
            else:
                raise ValueError(f'cannot legalize macro {macro["id"]}')
    result['cell_ids']=[locations[m['id']] for m in problem['movable_macros']]
    return placement(problem,result)


def build(repo,binary,output):
    repo,binary,output=map(lambda p:Path(p).resolve(),(repo,binary,output))
    if digest(binary)!=PLC_SHA256:raise ValueError('unexpected PLC binary checksum')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(version='macro_placement_v1',binary_sha256=PLC_SHA256,
        binary_source_commit='9ff931bc2728222c2767d5a8cd05ba083b5f11f5',
        circuit_training_commit='c417a3a13f40867b649c719c03daaf1b39a909bc',
        completion=COMPLETION,cpus=4,memory_gib=16,candidate_seconds=60,total_seconds=600,
        weights=[.5,.4,.1],cases=[])
    for name in CASES:
        start=time.monotonic();source=repo/'circuit_training/environment/test_data'/name
        dest=output/name;dest.mkdir()
        for file in ('netlist.pb.txt','initial.plc'):shutil.copyfile(source/file,dest/file)
        netlist,initial=dest/'netlist.pb.txt',dest/'initial.plc'
        problem=parse(netlist,initial,require_legal_grid=False)
        if not problem['movable_macros']:raise ValueError('suite case needs hard macros')
        problem['initial_placement']=legalize(repo,binary,netlist,initial,problem)
        (dest/'problem.json').write_text(json.dumps(problem,allow_nan=False)+'\n')
        scores=[score(repo,binary,netlist,initial,problem,problem['initial_placement']) for _ in range(3)]
        baseline=np.array([x[0] for x in scores])
        if not (baseline>0).all():raise ValueError('nonpositive baseline denominators')
        if not np.allclose(baseline,baseline[0],rtol=0,atol=1e-9):
            raise ValueError(f'nonrepeatable completion for {name}: {baseline.tolist()}')
        row=dict(name=name,baseline=baseline[0].tolist(),native_seconds=[x[1] for x in scores],
            macros=len(problem['movable_macros']),clusters=len(problem['standard_cell_clusters']),
            nets=len(problem['nets']),pins=len(problem['pins']),seconds=time.monotonic()-start,
            sha256={file:digest(dest/file) for file in ('netlist.pb.txt','initial.plc','problem.json')})
        manifest['cases'].append(row)
        print(json.dumps(row),flush=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--binary',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    build(args.repo,args.binary,args.output)
