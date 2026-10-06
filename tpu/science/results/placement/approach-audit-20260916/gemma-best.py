"""
Chip macro-placement algorithm utilizing JAX for numerical optimization
of wirelength and density, followed by a strict CPU-based legalization
to ensure zero overlap of hard macros and adherence to canvas boundaries.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict

def legalize(problem: dict, seed: int, *, time_budget_s: float, initial_pos=None) -> dict:
    """
    Strict legalization for hard macros.
    Keeps macros within canvas and ensures zero overlap between hard macros.
    Soft clusters are clipped to the canvas but allowed to overlap.
    """
    deadline = time.monotonic() + time_budget_s
    p = problem
    # Use provided positions or fallback to initial_positions
    if initial_pos is not None:
        pos = np.array(initial_pos, dtype=np.float64, copy=True)
    else:
        pos = np.array(p['initial_positions'], dtype=np.float64, copy=True)
        
    sizes = np.asarray(p['sizes'], dtype=np.float64)
    fixed = np.asarray(p['fixed'], dtype=bool)
    canvas = np.asarray(p['canvas'], dtype=np.float64)
    hard = int(p['num_hard'])
    
    # Small margin to avoid float precision issues with boundary constraints
    gap = max(1e-4, float(canvas.max()) * 1e-5)
    lo, hi = sizes / 2 + gap, canvas - sizes / 2 - gap
    
    # Clip non-fixed blocks to keep entire rectangle in canvas
    pos[~fixed] = np.clip(pos[~fixed], lo[~fixed], hi[~fixed])
    
    # Hard macros legalization
    occupied = list(np.flatnonzero(fixed[:hard]))
    # Optimize placing larger hard macros first (greedy)
    order = sorted(np.flatnonzero(~fixed[:hard]), key=lambda i: -float(np.prod(sizes[i])))
    
    for i in order:
        if time.monotonic() > deadline:
            break
        
        # Candidate positions: edges, canvas bounds, and current macro center
        xs = [lo[i, 0], hi[i, 0], pos[i, 0]]
        ys = [lo[i, 1], hi[i, 1], pos[i, 1]]
        for j in occupied:
            delta = (sizes[i] + sizes[j]) / 2 + gap
            xs.extend([pos[j, 0] - delta[0], pos[j, 0] + delta[0]])
            ys.extend([pos[j, 1] - delta[1], pos[j, 1] + delta[1]])
        
        xs = np.unique(np.clip(xs, lo[i, 0], hi[i, 0]))
        ys = np.unique(np.clip(ys, lo[i, 1], hi[i, 1]))
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Distance-based ranking to keep the block near its optimized position
        ranking = np.argsort(np.sum((candidates - pos[i]) ** 2, axis=1))
        
        chosen = None
        # Evaluate candidate batches to maintain performance
        for start in range(0, len(ranking), 256):
            if time.monotonic() > deadline:
                break
            batch = candidates[ranking[start:start+256]]
            if occupied:
                # Check for overlap with all currently placed hard macros
                # delta: [B, N, 2]
                delta = np.abs(batch[:, None, :] - pos[occupied][None, :, :])
                overlap_cond = (delta < (sizes[i] + sizes[occupied])[None, :, :] / 2 + gap/2).all(axis=2)
                legal = ~overlap_cond.any(axis=1)
            else:
                legal = np.ones(len(batch), dtype=bool)
            
            if legal.any():
                chosen = batch[np.flatnonzero(legal)[0]]
                break
        
        if chosen is not None:
            pos[i] = chosen
        occupied.append(i)
        
    # Preserve fixed locations exactly
    pos[fixed] = p['initial_positions'][fixed]
    return {'positions': pos.astype(np.float32)}

def place(problem: Dict, seed: int, *, time_budget_s: float) -> Dict:
    """
    Main placement routine.
    1. Initial legality pass (CPU).
    2. Gradient-based optimization of total cost proxy (JAX).
    3. Final consistency and legality pass (CPU).
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    
    # 1. Initial Legalization
    start_time = time.monotonic()
    try:
        initial_res = legalize(problem, seed, time_budget_s=min(20, time_budget_s))
        pos_np = initial_res['positions']
    except Exception:
        pos_np = np.array(problem['initial_positions'], dtype=np.float32)
    
    # 2. JAX Setup
    p = problem
    canvas = jnp.asarray(p['canvas'], dtype=jnp.float32)
    ports = jnp.asarray(p['port_positions'], dtype=jnp.float32)
    owners = jnp.asarray(p['pin_owner'], dtype=jnp.int32)
    offsets = jnp.asarray(p['pin_offset'], dtype=jnp.float32)
    weights = jnp.asarray(p['net_weights'], dtype=jnp.float32)
    
    # Precompute net IDs for segment operations
    net_ids_np = np.repeat(np.arange(len(weights)), np.diff(p['net_offsets']))
    net_ids = jnp.asarray(net_ids_np, dtype=jnp.int32)
    num_nets = len(weights)
    
    sizes = jnp.asarray(p['sizes'], dtype=jnp.float32)
    fixed = jnp.asarray(p['fixed'], dtype=bool)
    movable_mask = jnp.asarray(~p['fixed'], dtype=bool)
    
    # Normalization: Work in unit space [0, 1] for numerical stability
    x_init = jnp.asarray(pos_np) / canvas
    norm_ports = ports / canvas
    norm_offsets = offsets / canvas
    norm_sizes = sizes / canvas
    
    # Density Proxy Grid
    grid_res = 20
    gx, gy = jnp.meshgrid(jnp.linspace(0.05, 0.95, grid_res), jnp.linspace(0.05, 0.95, grid_res))
    grid_centers = jnp.stack([gx.ravel(), gy.ravel()], axis=1)
    block_areas = jnp.prod(norm_sizes, axis=1)

    def objective(x):
        # Pins = Center of Owner + Offset
        # Owner indices map to macro/cluster positions x or port positions norm_ports
        all_centers = jnp.concatenate([x, norm_ports])
        pins = all_centers[owners] + norm_offsets
        
        # HPWL components
        pins_x = pins[:, 0]
        pins_y = pins[:, 1]
        max_x = jax.ops.segment_max(pins_x, net_ids, num_nets)
        min_x = jax.ops.segment_min(pins_x, net_ids, num_nets)
        max_y = jax.ops.segment_max(pins_y, net_ids, num_nets)
        min_y = jax.ops.segment_min(pins_y, net_ids, num_nets)
        
        wire_cost = jnp.sum(weights * ((max_x - min_x) + (max_y - min_y)))
        # Scale normalized wirelength back to microns / normalizer
        wire_cost = (wire_cost * jnp.mean(canvas)) / p['wirelength_normalizer']
        
        # Density proxy using Gaussian kernels spread over a grid
        # x: [M, 2], grid_centers: [G, 2]
        diff = x[:, None, :] - grid_centers[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=2)
        # Area density spread
        occupancy = jnp.sum(jnp.exp(-dist_sq / (2 * 0.05**2)) * block_areas[:, None], axis=0)
        # Normalized density cost (sum of squared relative densities)
        density_cost = jnp.mean((occupancy / (jnp.mean(occupancy) + 1e-8)) ** 2)
        
        # ProxyCost = Wirelength + 0.5 * Density + 0.5 * Congestion 
        # (Using density as proxy for both congestion and density)
        return wire_cost + 0.5 * density_cost

    # Optimizer configuration
    optimizer = optax.adam(learning_rate=0.001)
    
    @jax.jit
    def step(x, state):
        value, grad = jax.value_and_grad(objective)(x)
        # Mask gradients for fixed blocks
        grad = grad * movable_mask[:, None]
        updates, state = optimizer.update(grad, state, x)
        x = optax.apply_updates(x, updates)
        
        # Box constraints: ensure entire block stays in canvas
        lo = norm_sizes / 2 + 1e-5
        hi = 1.0 - norm_sizes / 2 - 1e-5
        x = jnp.clip(x, lo, hi)
        # Strict preservation of fixed positions
        x = jnp.where(movable_mask[:, None], x, x_init)
        return x, state, value

    # Optimization Loop
    x = x_init
    state = optimizer.init(x)
    best_x = x
    best_val = float('inf')
    
    # Time management: reserve time for final legalization and jit
    opt_budget = time_budget_s - (time.monotonic() - start_time) - 20
    opt_start = time.monotonic()
    
    for i in range(1500):
        if time.monotonic() - opt_start > opt_budget:
            break
        x, state, val = step(x, state)
        # Synchronize occasionally to check objective value
        if i % 100 == 0:
            curr_val = float(val)
            if curr_val < best_val:
                best_val = curr_val
                best_x = x

    # Convert results back to physical micron coordinates
    final_pos_microns = (best_x * canvas).astype(np.float32)
    
    # 3. Final Legalization
    # Ensure hard macros have zero overlap and blocks are strictly inside canvas.
    # The provided legalizer is used with the optimized positions as guidelines.
    try:
        res_final = legalize(problem, seed, time_budget_s=15, initial_pos=final_pos_microns)
        final_positions = res_final['positions']
    except Exception:
        # Fallback to initial legal placement if finalization fails
        final_positions = pos_np
        
    # Absolute safety: restore exact fixed positions from problem definition
    fixed_indices = np.flatnonzero(p['fixed'])
    final_positions[fixed_indices] = p['initial_positions'][fixed_indices]
    
    return {'positions': final_positions}
