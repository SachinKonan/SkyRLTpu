"""One legal starting layout from upstream Xplace global placement (no challenge entry's code).

Steps, all inside one task:
  1. load the IBM case with the official challenge loader and build our problem contract;
  2. export it as Bookshelf (our own writer below) and run upstream Xplace (cuhk-eda/Xplace)
     global placement only, with the given settings, on one GPU;
  3. legalize hard macros with our deterministic seed legalizer (challenge_seed_jax.legalize);
  4. score with the official validate_placement + compute_proxy_cost.
Writes positions.npy, raw_positions.npy and report.json into --out.

Usage: xplace_start.py --case ibm01 --config td075 --out DIR --xplace DIR --probe DIR
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DBU = 1000          # Bookshelf coordinates in nm (canvas is in microns)

CONFIGS = {
    # name: Xplace global-placement settings (upstream flags). All use GP only: Xplace's own
    # legalization and detailed placement are off; we legalize macros ourselves.
    'td075': dict(target_density=0.75, num_bin_x=128, num_bin_y=128, inner_iter=2000),
    'td090': dict(target_density=0.90, num_bin_x=128, num_bin_y=128, inner_iter=2000),
    'td060': dict(target_density=0.60, num_bin_x=128, num_bin_y=128, inner_iter=2000),
    'td075b64': dict(target_density=0.75, num_bin_x=64, num_bin_y=64, inner_iter=2000),
}


def write_bookshelf(problem, folder, name):
    """Bookshelf export: hard macros h*, soft macros s*, ports p* (terminal_NI)."""
    p = problem
    folder.mkdir(parents=True, exist_ok=True)
    n, nh = len(p['sizes']), int(p['num_hard'])
    ports = np.asarray(p['port_positions'], dtype=np.float64).reshape(-1, 2)
    fixed = np.asarray(p['fixed'], dtype=bool)
    W, H = (int(round(v * DBU)) for v in p['canvas'])
    names = [f'h{i}' if i < nh else f's{i}' for i in range(n)] + [f'p{k}' for k in range(len(ports))]
    size = np.maximum(1, np.round(np.asarray(p['sizes']) * DBU)).astype(np.int64)
    pos = np.asarray(p['initial_positions'], dtype=np.float64) * DBU
    nterm = int(fixed.sum()) + len(ports)
    with open(folder / f'{name}.nodes', 'w') as f:
        f.write(f'UCLA nodes 1.0\n\nNumNodes : {n + len(ports)}\nNumTerminals : {nterm}\n')
        for i in range(n):
            f.write(f'{names[i]} {size[i,0]} {size[i,1]}' + (' terminal' if fixed[i] else '') + '\n')
        for k in range(len(ports)):
            f.write(f'{names[n+k]} 1 1 terminal_NI\n')
    with open(folder / f'{name}.pl', 'w') as f:
        f.write('UCLA pl 1.0\n\n')
        for i in range(n):
            ll = pos[i] - size[i] / 2
            f.write(f'{names[i]} {ll[0]:.3f} {ll[1]:.3f} : N' + (' /FIXED' if fixed[i] else '') + '\n')
        for k in range(len(ports)):
            x, y = ports[k] * DBU
            f.write(f'{names[n+k]} {x:.3f} {y:.3f} : N /FIXED_NI\n')
    start, owner, off = p['net_offsets'], p['pin_owner'], np.asarray(p['pin_offset']) * DBU
    with open(folder / f'{name}.nets', 'w') as f:
        f.write(f'UCLA nets 1.0\n\nNumNets : {len(start) - 1}\nNumPins : {int(start[-1])}\n')
        for j in range(len(start) - 1):
            a, z = int(start[j]), int(start[j + 1])
            f.write(f'NetDegree : {z - a} n{j}\n')
            for q in range(a, z):
                o = int(owner[q])
                ox, oy = (0.0, 0.0) if o >= n else off[q]
                f.write(f'  {names[o]} {"O" if q == a else "I"} : {ox:.3f} {oy:.3f}\n')
    with open(folder / f'{name}.wts', 'w') as f:
        f.write('UCLA wts 1.0\n')
    row_h = max(1, int(round(H / 256)))
    rows = H // row_h
    with open(folder / f'{name}.scl', 'w') as f:
        f.write(f'UCLA scl 1.0\n\nNumRows : {rows}\n')
        for r in range(rows):
            f.write(f'CoreRow Horizontal\n  Coordinate : {r * row_h}\n  Height : {row_h}\n  Sitewidth : 1\n'
                    f'  Sitespacing : 1\n  Siteorient : N\n  Sitesymmetry : Y\n  SubrowOrigin : 0 NumSites : {W}\nEnd\n')
    (folder / f'{name}.aux').write_text(
        f'RowBasedPlacement : {name}.nodes {name}.nets {name}.wts {name}.pl {name}.scl\n')
    return folder / f'{name}.aux', names, size


def read_pl(path, names, size, n):
    """Xplace GP .pl (lower-left, float) -> centers in microns for the n blocks."""
    index = {nm: i for i, nm in enumerate(names[:n])}
    out = np.full((n, 2), np.nan)
    for line in Path(path).read_text().splitlines()[1:]:
        tok = line.split()
        if len(tok) >= 3 and tok[0] in index:
            i = index[tok[0]]
            out[i] = (np.array([float(tok[1]), float(tok[2])]) + size[i] / 2) / DBU
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--config', required=True, choices=sorted(CONFIGS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--xplace', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--legalize-seconds', type=float, default=120)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.probe); sys.path.insert(0, str(ROOT))
    import torch
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    from tpu.science.challenge_contract import problem_from_native
    from tpu.science.challenge_seed_jax import legalize
    report = dict(case=a.case, config=a.config, settings=CONFIGS[a.config], seed=a.seed, valid=False)
    t0 = time.time()
    b, plc = load_benchmark_from_dir(str(Path(a.probe) / 'external/MacroPlacement/Testcases/ICCAD04' / a.case))
    problem = problem_from_native(b, plc)
    n = len(problem['sizes'])
    aux, names, size = write_bookshelf(problem, out / 'bookshelf', a.case)
    cfg = CONFIGS[a.config]
    cmd = [sys.executable, 'main.py', '--custom_path', f'aux:{aux},benchmark:ibm,design_name:{a.case}',
           '--load_from_raw', 'True', '--global_placement', 'True', '--legalization', 'False',
           '--detail_placement', 'False', '--write_placement', 'True', '--write_global_placement', 'True',
           '--draw_placement', 'False', '--final_route_eval', 'False', '--seed', str(a.seed),
           '--num_threads', str(a.threads), '--result_dir', str(out / 'xplace'), '--exp_id', 'run',
           '--gpu', '0'] + [x for k, v in cfg.items() for x in (f'--{k}', str(v))]
    t1 = time.time()
    with open(out / 'xplace.log', 'w') as log:
        subprocess.run(cmd, cwd=a.xplace, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1500)
    report['xplace_seconds'] = time.time() - t1
    pls = sorted((out / 'xplace').rglob(f'*{a.case}_gp.pl'))
    if not pls:
        raise FileNotFoundError('Xplace wrote no global placement .pl')
    raw = read_pl(pls[-1], names, size, n)
    fixed = np.asarray(problem['fixed'], dtype=bool)
    if not np.isfinite(raw).all():
        raise ValueError('global placement missing blocks')
    raw[fixed] = problem['initial_positions'][fixed]
    np.save(out / 'raw_positions.npy', raw.astype(np.float32))
    problem['initial_positions'] = raw.astype(np.float32)
    t2 = time.time()
    legal = legalize(problem, 42, time_budget_s=a.legalize_seconds)['positions']
    report['legalize_seconds'] = time.time() - t2
    legal[fixed] = b.macro_positions.numpy()[fixed]
    np.save(out / 'positions.npy', legal.astype(np.float32))
    tensor = torch.from_numpy(legal.astype(np.float32))
    ok, errors = validate_placement(tensor, b)
    report.update(valid=bool(ok), errors=[str(e) for e in errors][:5])
    report['metrics'] = {k: float(v) for k, v in compute_proxy_cost(tensor, b, plc).items()}
    report['raw_metrics'] = {k: float(v) for k, v in compute_proxy_cost(torch.from_numpy(raw.astype(np.float32)), b, plc).items()}
    report['total_seconds'] = time.time() - t0
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('case', 'config', 'valid', 'total_seconds')} | {'proxy': report['metrics']['proxy_cost']}))


if __name__ == '__main__':
    main()
