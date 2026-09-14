"""Numeric candidate interface for the pinned Partcl/TILOS position task.

No candidate code executes here. Net endpoints come directly from the PLC
instead of the challenge loader's unit-weight, macro-deduplicated graph.
"""
import numpy as np

CASES = ('ibm01', 'ibm04', 'ibm08', 'ibm18')


def problem_from_native(benchmark, plc):
    b = benchmark
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
        'schema_version': 'partcl_cpu_positions_v1',
        'canvas': np.array([b.canvas_width, b.canvas_height], dtype=np.float64),
        'initial_positions': b.macro_positions.numpy().copy(),
        'sizes': b.macro_sizes.numpy().copy(),
        'fixed': b.macro_fixed.numpy().copy(),
        'num_hard': b.num_hard_macros,
        'port_positions': b.port_positions.numpy().copy(),
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


def wirelength(problem, positions):
    """Pin HPWL / native normalizer, for adapter parity checks and cheap search."""
    p = problem
    all_centers = np.concatenate([positions, p['port_positions']], axis=0)
    xy = all_centers[p['pin_owner']] + p['pin_offset']
    total = 0.0
    for i, (a, z) in enumerate(zip(p['net_offsets'][:-1], p['net_offsets'][1:])):
        total += p['net_weights'][i] * np.ptp(xy[a:z], axis=0).sum()
    return float(total / p['wirelength_normalizer'])


def reward(costs, *, all_valid):
    costs = np.asarray(costs, dtype=np.float64)
    if not all_valid or not costs.size or not np.isfinite(costs).all() or (costs < 0).any():
        return 0.0
    return max(1e-6, float(1.0 / (1.0 + costs.mean())))

def aggregate(rows):
    from .rewards import invalid, valid
    by_case={r['metrics'].get('case'):r for r in rows}
    if len(rows)!=len(CASES) or set(by_case)!=set(CASES):
        return invalid('incomplete or duplicate placement suite',phase='suite')
    if not all(r['correctness']==1 for r in rows):
        result=invalid('Invalid case: '+'; '.join(f"{c}: {r['msg']}" for c,r in by_case.items() if r['correctness']!=1))
        result['metrics']['cases']=rows;return result
    costs=[by_case[c]['metrics']['proxy_cost'] for c in CASES]
    return valid(reward(costs,all_valid=True),dict(mean_proxy_cost=sum(costs)/len(costs),cases=rows))
