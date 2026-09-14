"""Candidate-only runner: problem input and restricted helper RPC, no grader."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np


def main():
    transport=sys.stdout;sys.stdout=sys.stderr
    problem=json.loads(Path('/input/problem.json').read_text())

    def clean(result):
        if not isinstance(result,dict) or set(result)!={'cell_ids','orientations'}:
            raise ValueError('return cell_ids and orientations only')
        cells=np.asarray(result['cell_ids'])
        if cells.dtype.kind not in 'iu':raise ValueError('cell_ids must be integers')
        return dict(cell_ids=cells.tolist(),orientations=list(result['orientations']))

    def evaluate(supplied_problem,placement):
        if supplied_problem is not problem:raise ValueError('helper takes the original supplied problem')
        transport.write(json.dumps(dict(kind='evaluate',placement=clean(placement)),allow_nan=False)+'\n')
        transport.flush()
        response=json.loads(sys.stdin.readline())
        if 'error' in response:raise ValueError(response['error'])
        return response['metrics']

    api=types.ModuleType('placement_api');api.evaluate=evaluate;sys.modules['placement_api']=api
    spec=importlib.util.spec_from_file_location('candidate','/input/candidate.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    result=module.place(problem,1)
    transport.write(json.dumps(dict(kind='result',placement=clean(result)),allow_nan=False)+'\n')
    transport.flush()


if __name__=='__main__':main()
