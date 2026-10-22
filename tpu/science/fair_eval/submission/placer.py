"""MOMO placer for the Partcl/HRT Macro Placement Challenge harness (macro_place.evaluate).

Pipeline, all inside place():
  1. reload the netlist with the official loader (as submissions/will_seed does) and build the
     numeric problem (pin offsets and net weights from the PLC);
  2. global placement with upstream Xplace (cuhk-eda/Xplace, unmodified algorithm) under a few
     settings, each legalized by a deterministic greedy legalizer;
  3. the search program MOMO discovered (momo_search.py, unchanged; sha256 d4832342...), starting
     from those layouts and scoring moves with momo_proxy, our evaluator written from the
     official scorer.

Environment:
  MOMO_XPLACE_ROOT        built Xplace tree (required)
  MOMO_XPLACE_PYTHON      Python with Torch for Xplace (default: this interpreter)
  MOMO_XPLACE_CONFIGS     comma-separated settings from XPLACE_CONFIGS (default: td075,td090,td060)
  MOMO_SEARCH_SECONDS     search budget in seconds, or 'fill' to use what remains of MOMO_TOTAL_SECONDS
  MOMO_TOTAL_SECONDS      whole-placer budget for 'fill' (default 3000)
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DBU = 1000          # Bookshelf coordinates in nm (the canvas is in microns)
QUANTUM = 10        # block sizes snapped up to 10 nm for Xplace's site grid

XPLACE_CONFIGS = {
    'td075': dict(target_density=0.75, num_bin_x=128, num_bin_y=128, inner_iter=2000),
    'td090': dict(target_density=0.90, num_bin_x=128, num_bin_y=128, inner_iter=2000),
    'td060': dict(target_density=0.60, num_bin_x=128, num_bin_y=128, inner_iter=2000),
    'td075b64': dict(target_density=0.75, num_bin_x=64, num_bin_y=64, inner_iter=2000),
}


# ---------- netlist ----------

def load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path('external/MacroPlacement/Testcases/ICCAD04') / name
    if root.exists():
        return load_benchmark_from_dir(str(root))[1]
    ng45 = {'ariane133_ng45': 'ariane133', 'ariane136_ng45': 'ariane136',
            'nvdla_ng45': 'nvdla', 'mempool_tile_ng45': 'mempool_tile'}
    ng45.update({v: v for v in list(ng45.values())})
    if name in ng45:
        base = Path('external/MacroPlacement/Flows/NanGate45') / ng45[name] / 'netlist' / 'output_CT_Grouping'
        return load_benchmark(str(base / 'netlist.pb.txt'), str(base / 'initial.plc'))[1]
    raise FileNotFoundError(f'no netlist for {name}')


def build_problem(b, plc):
    """Numeric problem: blocks (hard then soft), ports, nets as CSR of pins with offsets."""
    indices = list(b.hard_macro_indices) + list(b.soft_macro_indices) + list(plc.port_indices)
    owners = {plc.modules_w_pins[idx].get_name(): i for i, idx in enumerate(indices)}
    nodes = {mod.get_name(): mod for mod in plc.modules_w_pins}
    offsets, pin_owner, pin_offset, weights = [0], [], [], []
    for driver, sinks in plc.nets.items():
        for name in [driver, *sinks]:
            node = nodes[name]
            if name in owners:
                owner, offset = owners[name], [0.0, 0.0]
            else:
                owner = owners[node.get_macro_name()]
                offset = [float(node.x_offset), float(node.y_offset)]
            pin_owner.append(owner)
            pin_offset.append(offset)
        weights.append(float(nodes[driver].get_weight()))
        offsets.append(len(pin_owner))
    return {
        'canvas': np.array([b.canvas_width, b.canvas_height], dtype=np.float64),
        'initial_positions': b.macro_positions.numpy().astype(np.float32).copy(),
        'sizes': b.macro_sizes.numpy().copy(),
        'fixed': b.macro_fixed.numpy().copy(),
        'num_hard': int(b.num_hard_macros),
        'port_positions': b.port_positions.numpy().copy().reshape(-1, 2),
        'net_offsets': np.array(offsets, dtype=np.int64),
        'pin_owner': np.array(pin_owner, dtype=np.int64),
        'pin_offset': np.array(pin_offset, dtype=np.float64).reshape(-1, 2),
        'net_weights': np.array(weights, dtype=np.float64),
        'wirelength_normalizer': float(plc.net_cnt) * (b.canvas_width + b.canvas_height),
        'grid_shape': np.array([b.grid_rows, b.grid_cols], dtype=np.int64),
        'routes_per_micron': np.array([plc.hroutes_per_micron, plc.vroutes_per_micron]),
        'macro_routing_allocation': np.array([plc.hrouting_alloc, plc.vrouting_alloc]),
        'congestion_smoothing_range': int(plc.smooth_range),
    }


# ---------- greedy legalizer (hard macros; soft macros only clipped to the canvas) ----------

def legalize(p, time_budget_s):
    deadline = time.monotonic() + time_budget_s
    pos = np.array(p['initial_positions'], dtype=np.float64)
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    hard = int(p['num_hard'])
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo, hi = sizes / 2 + gap, canvas - sizes / 2 - gap
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])
    occupied = list(np.flatnonzero(fixed[:hard]))
    for i in sorted(np.flatnonzero(~fixed[:hard]), key=lambda i: -float(np.prod(sizes[i]))):
        if time.monotonic() > deadline:
            raise TimeoutError('legalizer budget exhausted')
        xs, ys = [lo[i, 0], hi[i, 0], pos[i, 0]], [lo[i, 1], hi[i, 1], pos[i, 1]]
        for j in occupied:
            d = (sizes[i] + sizes[j]) / 2 + gap
            xs += [pos[j, 0] - d[0], pos[j, 0] + d[0]]
            ys += [pos[j, 1] - d[1], pos[j, 1] + d[1]]
        xx, yy = np.meshgrid(np.unique(np.clip(xs, lo[i, 0], hi[i, 0])), np.unique(np.clip(ys, lo[i, 1], hi[i, 1])))
        cand = np.column_stack([xx.ravel(), yy.ravel()])
        order = np.argsort(np.sum((cand - pos[i]) ** 2, axis=1))
        chosen = None
        for s in range(0, len(order), 256):
            batch = cand[order[s:s + 256]]
            if occupied:
                delta = np.abs(batch[:, None, :] - pos[occupied][None, :, :])
                bad = (delta < (sizes[i] + sizes[occupied])[None, :, :] / 2 + gap / 2).all(axis=2).any(axis=1)
            else:
                bad = np.zeros(len(batch), dtype=bool)
            if (~bad).any():
                chosen = batch[np.flatnonzero(~bad)[0]]
                break
        if chosen is None:
            raise ValueError('legalizer could not fit a hard macro')
        pos[i] = chosen
        occupied.append(i)
    pos[fixed] = np.asarray(p['initial_positions'], dtype=np.float64)[fixed]
    return pos.astype(np.float32)


# ---------- upstream Xplace global placement via Bookshelf ----------

def write_bookshelf(p, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    n, nh = len(p['sizes']), int(p['num_hard'])
    ports = np.asarray(p['port_positions'], dtype=np.float64).reshape(-1, 2)
    fixed = np.asarray(p['fixed'], dtype=bool)
    W, H = (int(round(v * DBU)) for v in p['canvas'])
    names = [f'h{i}' if i < nh else f's{i}' for i in range(n)] + [f'p{k}' for k in range(len(ports))]
    # Xplace's Bookshelf reader takes the site size as the GCD of movable cell sizes and requires
    # every row to be exactly one site tall, so sizes are snapped up to a QUANTUM grid and the
    # rows/sites are set from those GCDs.
    size = (np.ceil(np.asarray(p['sizes']) * DBU / QUANTUM).astype(np.int64) * QUANTUM).clip(min=QUANTUM)
    movable = size[~fixed]
    site_w = int(np.gcd.reduce(np.concatenate([movable[:, 0], movable[:, 1]]))) if len(movable) else QUANTUM
    site_h = int(np.gcd.reduce(movable[:, 1])) if len(movable) else QUANTUM
    pos = np.asarray(p['initial_positions'], dtype=np.float64) * DBU
    with open(folder / f'{name}.nodes', 'w') as f:
        f.write(f'UCLA nodes 1.0\n\nNumNodes : {n + len(ports)}\nNumTerminals : {int(fixed.sum()) + len(ports)}\n')
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
    (folder / f'{name}.wts').write_text('UCLA wts 1.0\n')
    with open(folder / f'{name}.scl', 'w') as f:
        f.write(f'UCLA scl 1.0\n\nNumRows : {H // site_h}\n')
        for r in range(H // site_h):
            f.write(f'CoreRow Horizontal\n  Coordinate : {r * site_h}\n  Height : {site_h}\n  Sitewidth : {site_w}\n'
                    f'  Sitespacing : {site_w}\n  Siteorient : N\n  Sitesymmetry : Y\n'
                    f'  SubrowOrigin : 0 NumSites : {W // site_w}\nEnd\n')
    (folder / f'{name}.aux').write_text(f'RowBasedPlacement : {name}.nodes {name}.nets {name}.wts {name}.pl {name}.scl\n')
    return folder / f'{name}.aux', names, size


def xplace_gp(p, name, config, work, timeout_s):
    root = Path(os.environ['MOMO_XPLACE_ROOT'])
    py = os.environ.get('MOMO_XPLACE_PYTHON', sys.executable)
    aux, names, size = write_bookshelf(p, work / 'bookshelf', name)
    cfg = XPLACE_CONFIGS[config]
    cmd = [py, 'main.py', '--custom_path', f'aux:{aux},benchmark:momo,design_name:{name}',
           '--load_from_raw', 'True', '--global_placement', 'True', '--legalization', 'False',
           '--detail_placement', 'False', '--write_placement', 'True', '--write_global_placement', 'True',
           '--draw_placement', 'False', '--final_route_eval', 'False', '--seed', '0',
           '--num_threads', str(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 8)), '--result_dir', str(work / config), '--exp_id', 'run',
           '--gpu', '0'] + [x for k, v in cfg.items() for x in (f'--{k}', str(v))]
    with open(work / f'{config}.log', 'w') as log:
        subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout_s)
    pl = sorted((work / config).rglob(f'*{name}_gp.pl'))[-1]
    n = len(p['sizes'])
    index = {nm: i for i, nm in enumerate(names)}
    lower_left = np.full((len(names), 2), np.nan)
    for line in pl.read_text().splitlines()[1:]:
        tok = line.split()
        if len(tok) >= 3 and tok[0] in index:
            lower_left[index[tok[0]]] = float(tok[1]), float(tok[2])
    # Xplace rescales Bookshelf units internally and writes positions in its site units; map
    # them back with a per-axis linear fit on the fixed ports, whose positions are known.
    ports = np.asarray(p['port_positions'], dtype=np.float64).reshape(-1, 2) * DBU
    got = lower_left[n:]
    ok = np.isfinite(got).all(axis=1)
    if ok.sum() < 2:
        raise ValueError('cannot calibrate Xplace output units: too few ports')
    out = np.full((n, 2), np.nan)
    for axis in range(2):
        a, c = np.polyfit(got[ok, axis], ports[ok, axis], 1)
        if np.max(np.abs(a * got[ok, axis] + c - ports[ok, axis])) > 2.0:   # nm
            raise ValueError('Xplace output units are not a linear map of the input')
        out[:, axis] = (a * lower_left[:n, axis] + c + size[:n, axis] / 2) / DBU
    if not np.isfinite(out).all():
        raise ValueError('global placement missing blocks')
    fixed = np.asarray(p['fixed'], dtype=bool)
    out[fixed] = np.asarray(p['initial_positions'])[fixed]
    return out.astype(np.float32)


# ---------- placer ----------

class MomoPlacer:
    def place(self, benchmark):
        t0 = time.monotonic()
        from momo_proxy import Evaluator
        plc = load_plc(benchmark.name)
        p = build_problem(benchmark, plc)
        configs = os.environ.get('MOMO_XPLACE_CONFIGS', 'td075,td090,td060').split(',')
        # Everything (netlist reload, Xplace, legalization, search) fits in one total budget.
        total = float(os.environ.get('MOMO_TOTAL_SECONDS', '3000'))
        left = lambda: total - (time.monotonic() - t0)
        log = dict(benchmark=benchmark.name, total_budget=total, configs={})

        def write_log():
            if os.environ.get('MOMO_LOG'):
                Path(os.environ['MOMO_LOG']).write_text(json.dumps(log, indent=1))
        layouts = []
        work = Path(tempfile.mkdtemp(prefix=f'momo-{benchmark.name}-'))
        try:
            for cfg in configs:
                c0 = time.monotonic()
                room = left() - 400   # keep room for legalization and a minimal search
                if room < 60:
                    log['configs'][cfg] = dict(error='skipped: budget', seconds=0.0)
                    continue
                try:
                    raw = xplace_gp(p, benchmark.name, cfg, work, timeout_s=min(900, room))
                    q = dict(p, initial_positions=raw)
                    legal = legalize(q, min(120, max(10, left() - 300)))
                    with Evaluator(p, legal) as ev:
                        cost = ev.score()['proxy_cost']
                    layouts.append((cost, cfg, legal))
                    log['configs'][cfg] = dict(proxy=cost, seconds=time.monotonic() - c0)
                except Exception as exc:   # one failed setting does not stop the others
                    log['configs'][cfg] = dict(error=f'{type(exc).__name__}: {exc}'[:300], seconds=time.monotonic() - c0)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not layouts:   # fall back to legalizing the given positions (flagged in the log)
            layouts.append((float('inf'), 'given', legalize(p, min(300, max(10, left() - 60)))))
        layouts.sort(key=lambda t: t[0])
        problem = dict(p, initial_positions=layouts[0][2])
        if len(layouts) > 1:
            problem['starting_layouts'] = np.stack([t[2] for t in layouts[1:]])
        # The search program, unchanged, with our evaluator injected as `Evaluator`.
        namespace = {'__name__': 'momo_search', 'Evaluator': Evaluator}
        src = (HERE / 'momo_search.py').read_text()
        exec(compile(src, str(HERE / 'momo_search.py'), 'exec'), namespace)
        spec = os.environ.get('MOMO_SEARCH_SECONDS', '300')
        budget = left() - 30 if spec == 'fill' else min(float(spec), left() - 30)
        log.update(start_seconds=time.monotonic() - t0, search_seconds=budget, best_start=layouts[0][1],
                   fallback=layouts[0][1] == 'given' or any('error' in c for c in log['configs'].values()))
        write_log()
        result = namespace['place'](problem, 42, time_budget_s=max(10.0, budget))
        # Return the search's own float64 positions: it checks legality in float64 with zero
        # tolerance, and a float32 cast could turn a sub-ULP gap into an overlap.
        pos = np.array(result['positions'], dtype=np.float64)
        fixed = np.asarray(p['fixed'], dtype=bool)
        pos[fixed] = benchmark.macro_positions.numpy()[fixed]
        log['total_seconds'] = time.monotonic() - t0
        write_log()
        return torch.from_numpy(pos)
