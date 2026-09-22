"""Verify downloaded Modal provenance and expose results to portfolio publisher."""
import argparse
import hashlib
import json
from pathlib import Path

from .abuplace_starts import ABUPLACE_COMMIT, VARIANTS
from .placement_warm_start import CASES


def import_results(root, results, queue, source):
    records=[]
    # Validate the complete suite before creating any queue completion records.
    for case in CASES:
        for variant in VARIANTS:
            folder=results/f'{case}-{variant}'
            report=json.loads((folder/'report.json').read_text())
            gp=json.loads((folder/'gp.json').read_text())
            if not (report.get('valid') is True and report.get('case')==case
                    and report.get('variant')==variant and report.get('method')=='xplace-abu'
                    and report.get('repository_commit')==ABUPLACE_COMMIT
                    and gp.get('source_commit')==ABUPLACE_COMMIT
                    and gp.get('stage')=='global_placement_only' and gp.get('variant')==variant):
                raise ValueError(f'invalid result {folder}')
            for name,digest in report['native_sha256'].items():
                path=root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case/name
                if name not in ('netlist.pb.txt','initial.plc') or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'input mismatch: {case}/{name}')
            if set(report['native_sha256'])!={'netlist.pb.txt','initial.plc'}:
                raise ValueError('incomplete native hashes')
            for name in ('placer.py','Xplace/src/calculator.py','Xplace/src/run_placement_nesterov.py'):
                if hashlib.sha256((source/'abuplace'/name).read_bytes()).hexdigest()!=gp['source_sha256'][name]:
                    raise ValueError(f'GP source mismatch: {name}')
            records.append((f'xplace-{variant}-{case}',folder/'report.json'))
    for name,report in records:
        dest=queue/'tasks'/name
        dest.mkdir(parents=True,exist_ok=True)
        with (dest/'done.json').open('x') as f:
            json.dump({'report':str(report.resolve()),'executor':'modal-a10'},f,indent=2)
    return len(records)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    print(import_results(Path(__file__).resolve().parents[2],a.results,a.queue,a.source))
