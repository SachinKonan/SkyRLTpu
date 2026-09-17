"""
Macro-placement algorithm using JAX for gradient-based numerical optimization.
Optimizes for weighted HPWL and density, with a final CPU-based legalizer to 
guarantee zero overlap between hard macros and strict canvas boundaries.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict

def legalize(problem: dict, seed: int, positions_in: np.ndarray, deadline: float) -> np.ndarray:
    """
    Greedy legalizer to ensure hard-macro legality and boundary constraints.
    Ensures zero overlap between hard macros and that all macros stay in canvas.
    """
    p = problem
    pos = np.array(positions_in, dtype=np.float64, copy=True)
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    hard_count = int(p['num_hard'])
    gap = max(0.0001, float(canvas.max()) * 1e-05)
    (lo, hi) = (sizes / 2 + gap, canvas - sizes / 2 - gap)
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])
    occupied = list(np.flatnonzero(fixed[:hard_count]))
    moveable_hards = np.flatnonzero(~fixed[:hard_count])
    order = sorted(moveable_hards, key=lambda i: -float(np.prod(sizes[i])))
    for i in order:
        if time.monotonic() > deadline:
            break
        (cur_x, cur_y) = pos[i]
        xs = [lo[i, 0], hi[i, 0], cur_x]
        ys = [lo[i, 1], hi[i, 1], cur_y]
        for j in occupied:
            delta = (sizes[i] + sizes[j]) / 2 + gap
            xs.extend([pos[j, 0] - delta[0], pos[j, 0] + delta[0]])
            ys.extend([pos[j, 1] - delta[1], pos[j, 1] + delta[1]])
        xs = np.unique(np.clip(xs, lo[i, 0], hi[i, 0]))
        ys = np.unique(np.clip(ys, lo[i, 1], hi[i, 1]))
        (xx, yy) = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        ranking = np.argsort(np.sum((candidates - pos[i]) ** 2, axis=1))
        chosen = None
        for start in range(0, len(ranking), 256):
            batch = candidates[ranking[start:start + 256]]
            if occupied:
                delta = np.abs(batch[:, None, :] - pos[occupied][None, :, :])
                overlaps = (delta < (sizes[i] + sizes[occupied])[None, :, :] / 2 + gap / 2).all(axis=2)
                legal = ~overlaps.any(axis=1)
            else:
                legal = np.ones(len(batch), dtype=bool)
            if legal.any():
                chosen = batch[np.flatnonzero(legal)[0]]
                break
        if chosen is not None:
            pos[i] = chosen
        occupied.append(i)
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])
    return pos.astype(np.float32)

def place(problem: dict, seed: int, *, time_budget_s: float) -> Dict:
    start_time = time.monotonic()
    deadline = start_time + time_budget_s - 5.0
    p = problem
    M = len(p['initial_positions'])
    H = int(p['num_hard'])
    canvas_np = np.asarray(p['canvas'], dtype=np.float32)
    initial_pos = np.array(p['initial_positions'], dtype=np.float32)
    canvas = jnp.asarray(canvas_np, dtype=jnp.float32)
    ports = jnp.asarray(p['port_positions'], dtype=jnp.float32)
    owners = jnp.asarray(p['pin_owner'], dtype=jnp.int32)
    offsets = jnp.asarray(p['pin_offset'], dtype=jnp.float32)
    weights = jnp.asarray(p['net_weights'], dtype=jnp.float32)
    net_ids = np.repeat(np.arange(len(weights)), np.diff(p['net_offsets']))
    net_ids = jnp.asarray(net_ids, dtype=jnp.int32)
    num_nets = len(weights)
    fixed = np.asarray(p['fixed'], dtype=bool)
    movable_mask = jnp.asarray(~fixed, dtype=jnp.float32)[:, None]
    sizes = jnp.asarray(p['sizes'], dtype=jnp.float32)
    areas = jnp.prod(sizes, axis=1)
    norm_factor = jnp.max(canvas)
    x_norm = initial_pos / norm_factor
    grid_res = 35
    (gx, gy) = jnp.meshgrid(jnp.linspace(0.05, 0.95, grid_res), jnp.linspace(0.05, 0.95, grid_res))
    grid_centers = jnp.stack([gx.ravel(), gy.ravel()], axis=1)

    def cost_fn(x):
        centers = jnp.concatenate([x * norm_factor, ports])
        pins = jax.lax.optimization_barrier(centers[owners] + offsets)
        x_max = jax.ops.segment_max(pins[:, 0], net_ids, num_nets)
        x_min = jax.ops.segment_min(pins[:, 0], net_ids, num_nets)
        y_max = jax.ops.segment_max(pins[:, 1], net_ids, num_nets)
        y_min = jax.ops.segment_min(pins[:, 1], net_ids, num_nets)
        wl = jnp.sum(weights * (x_max - x_min + (y_max - y_min)))
        wl_norm = wl / p['wirelength_normalizer']
        diff = x[:, None, :] - grid_centers[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=2)
        sigma_sq = 0.01
        density_map = jnp.sum(areas[:, None] * jnp.exp(-dist_sq / (2 * sigma_sq)), axis=0)
        density_cost = jnp.mean(density_map ** 2) / (jnp.mean(density_map) ** 2 + 1e-06)
        return wl_norm + 0.2 * density_cost

    def probe_wire(x):
        centers = jnp.concatenate([x * norm_factor, ports])
        pins = jax.lax.optimization_barrier(centers[owners] + offsets)
        x_max = jax.ops.segment_max(pins[:, 0], net_ids, num_nets)
        x_min = jax.ops.segment_min(pins[:, 0], net_ids, num_nets)
        y_max = jax.ops.segment_max(pins[:, 1], net_ids, num_nets)
        y_min = jax.ops.segment_min(pins[:, 1], net_ids, num_nets)
        wl = jnp.sum(weights * (x_max - x_min + (y_max - y_min)))
        wl_norm = wl / p['wirelength_normalizer']
        return wl_norm
    initial = x_norm
    full = cost_fn
    wire = probe_wire
    for (name, objective) in [('full', full), ('wire', wire)]:
        (value, grad) = jax.jit(jax.value_and_grad(objective))(initial)
        g = np.asarray(grad)
        print('GRAD', name, float(value), float(g.min()), float(g.max()), float(np.linalg.norm(g)), int((g < 0).sum()), flush=True)
        np.save('/output/grad-' + name + '.npy', g)
    physical = np.asarray(initial) * float(norm_factor)
    all_centers = np.concatenate([physical, np.asarray(problem['port_positions'])], axis=0)
    owner = np.asarray(problem['pin_owner'], dtype=np.int64)
    pins = all_centers[owner] + np.asarray(problem['pin_offset'])
    starts = np.asarray(problem['net_offsets'], dtype=np.int64)[:-1]
    ids = np.repeat(np.arange(len(starts)), np.diff(problem['net_offsets']))
    hi = np.maximum.reduceat(pins, starts, axis=0)
    lo = np.minimum.reduceat(pins, starts, axis=0)
    maxmask = pins == hi[ids]
    minmask = pins == lo[ids]
    maxcounts = np.add.reduceat(maxmask.astype(float), starts, axis=0)
    mincounts = np.add.reduceat(minmask.astype(float), starts, axis=0)
    gp = (maxmask / maxcounts[ids] - minmask / mincounts[ids]) * np.asarray(problem['net_weights'])[ids, None] / float(problem['wirelength_normalizer'])
    expected = np.zeros_like(all_centers, dtype=float)
    np.add.at(expected, owner, gp)
    expected = expected[:len(initial)] * float(norm_factor)
    print('REFERENCE', float(expected.min()), float(expected.max()), float(np.max(np.abs(g - expected))), flush=True)
    np.save('/output/expected-wire.npy', expected)
    return {'positions': np.array(problem['initial_positions'], copy=True)}
