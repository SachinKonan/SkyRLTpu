"""Fetch only pinned Python scorer files and selected IBM inputs, not 40 GB of flows."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

CHALLENGE = '996193c83eaad151e5ae3fb166cedbd83372d060'
TILOS = '45a721d01dfe56fc4800b95da52e9b4193b76d04'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Fetch all 17 public IBM inputs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]/'.science/challenge-probe'
    from .challenge_contract import CASES
    cases = list(CASES)
    files = []
    for name in ['__init__.py','_plc.py','benchmark.py','loader.py','objective.py','utils.py']:
        path = 'macro_place/' + name
        files.append((path, f'https://raw.githubusercontent.com/partcleda/macro-place-challenge-2026/{CHALLENGE}/{path}'))
    paths = ['CodeElements/Plc_client/plc_client_os.py']
    paths += [f'Testcases/ICCAD04/{case}/{name}' for case in cases for name in ['netlist.pb.txt','initial.plc']]
    for path in paths:
        files.append(('external/MacroPlacement/' + path,
                      f'https://raw.githubusercontent.com/partcleda/MacroPlacement/{TILOS}/{path}'))
    manifest = {'challenge_commit':CHALLENGE,'evaluator_commit':TILOS,'cases':cases,'files':[]}
    for rel, url in files:
        path = root/rel
        content = urllib.request.urlopen(url, timeout=60).read()
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'existing file differs from pinned source: {path}')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        manifest['files'].append({'path':rel,'url':url,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
        print(rel, len(content), flush=True)
    (root/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
