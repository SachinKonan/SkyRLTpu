"""
Chip Macro-Placement Algorithm using JAX.
Optimizes soft-cluster centers while enforcing hard-macro legality.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.ops
import optax
from jax import random

def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    """
    Implements a macro-placement algorithm.
    
    1. Legalizes Hard Macros: Ensures no overlap between hard macros using a greedy CPU algorithm.
    2. Optimizes Soft Clusters: Uses JAX + Optax to minimize wirelength and smooth density
       on a grid, while keeping Hard Macros fixed to preserve legality.
    """
    deadline = time.monotonic() + time_budget_s
    
    # 1. Parse Problem Inputs
    p = problem
    M = len(p['initial_positions'])
    H = int(p['num_hard'])
    
    canvas_np = np.array(p['canvas'], dtype=np.float32)
    sizes_np = np.array(p['sizes'], dtype=np.float32)
    fixed_np = np.array(p['fixed'], dtype=bool)
    initial_pos_np = np.array(p['initial_positions'], dtype=np.float32, copy=True)
    
    port_pos_np = np.array(p['port_positions'], dtype=np.float32)
    pin_owner_np = np.array(p['pin_owner'], dtype=np.int32)
    pin_offset_np = np.array(p['pin_offset'], dtype=np.float32)
    net_weights_np = np.array(p['net_weights'], dtype=np.float32)
    net_offsets_np = np.array(p['net_offsets'], dtype=np.int32)
    
    wire_norm = float(p['wirelength_normalizer'])
    grid_shape = p['grid_shape']
    
    # 2. Legalize Hard Macros (Greedy CPU)
    # Strategy:
    # - Ensure Hard Macros (indices 0..H-1) do not overlap with each other.
    # - All blocks (Hard and Soft) must be fully inside the canvas.
    # - Fixed blocks must not move.
    
    gap = max(1e-4, float(np.max(canvas_np)) * 1e-5)
    lo_np = sizes_np / 2 + gap
    hi_np = canvas_np - sizes_np / 2 - gap
    
    # Start from initial positions
    pos = np.array(initial_pos_np, dtype=np.float32)
    
    # Clip all non-fixed blocks to canvas
    pos[~fixed_np] = np.clip(pos[~fixed_np], lo_np[~fixed_np], hi_np[~fixed_np])
    
    # Hard macro packing
    # Sort hard macros by area to place larger ones (harder to place) first.
    hard_indices = np.arange(H)
    moveable_hard_mask = ~fixed_np[:H]
    to_place = hard_indices[moveable_hard_mask]
    
    occupied_indices = list(np.where(fixed_np[:H])[0])
    
    areas = np.prod(sizes_np[:H], axis=1)
    # Sort moveable hard indices by area descending
    to_place_sorted = to_place[np.argsort(-areas[to_place])]
    
    for idx in to_place_sorted:
        if time.monotonic() > deadline:
            break
            
        s_idx = sizes_np[idx]
        # Generate candidate coordinates
        candidates_x = {lo_np[idx, 0], hi_np[idx, 0], pos[idx, 0]}
        candidates_y = {lo_np[idx, 1], hi_np[idx, 1], pos[idx, 1]}
        
        for j in occupied_indices:
            delta = (s_idx + sizes_np[j]) / 2 + gap
            candidates_x.update([pos[j, 0] - delta[0], pos[j, 0] + delta[0]])
            candidates_y.update([pos[j, 1] - delta[1], pos[j, 1] + delta[1]])
        
        cx = sorted(np.clip(list(candidates_x), lo_np[idx, 0], hi_np[idx, 0]))
        cy = sorted(np.clip(list(candidates_y), lo_np[idx, 1], hi_np[idx, 1]))
        
        xx, yy = np.meshgrid(cx, cy)
        cands = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Rank by distance to initial position
        dists = np.sum((cands - pos[idx])**2, axis=1)
        ranking = np.argsort(dists)
        
        chosen = None
        # Batched check for speed/safety
        for start in range(0, len(ranking), 128):
            batch_end = min(start + 128, len(ranking))
            batch_idx = ranking[start:batch_end]
            batch = cands[batch_idx]
            
            # Overlap check
            if len(occupied_indices) > 0:
                delta_mat = np.abs(batch[:, None, :] - pos[occupied_indices][None, :, :])
                dist_thresh = (sizes_np[idx] + sizes_np[occupied_indices]) / 2 + gap/2
                overlaps = (delta_mat < dist_thresh[None, :, :None]).all(axis=2)
                legal_mask = ~overlaps.any(axis=1)
            else:
                legal_mask = np.ones(len(batch), dtype=bool)
            
            if np.any(legal_mask):
                chosen = batch[legal_mask][0]
                break
        
        if chosen is not None:
            pos[idx] = chosen
            occupied_indices.append(idx)
            
    # Restore fixed positions exactly
    pos[fixed_np] = initial_pos_np[fixed_np]
    
    # 3. JAX Optimization
    # We optimize Soft Clusters only. Hard Macros are now legal and fixed.
    
    canvas = jnp.array(canvas_np)
    sizes = jnp.array(sizes_np, dtype=jnp.float32)
    fixed_jax = jnp.array(fixed_np, dtype=jnp.bool_)
    port_pos = jnp.array(port_pos_np, dtype=jnp.float32)
    pin_owner = jnp.array(pin_owner_np, dtype=jnp.int32)
    pin_off = jnp.array(pin_offset_np, dtype=jnp.float32)
    weights = jnp.array(net_weights_np, dtype=jnp.float32)
    wire_n = jnp.array(wire_norm, dtype=jnp.float32)
    
    N_NETS = len(net_weights_np)
    # Map each pin to a net ID
    net_ids = np.repeat(np.arange(N_NETS), np.diff(net_offsets_np)).astype(np.int32)
    net_ids = jnp.array(net_ids, dtype=jnp.int32)
    
    # Initialize JAX positions
    x = jnp.array(pos, dtype=jnp.float32)
    
    # Mask: We only move Soft Clusters (index >= H) AND (~fixed)
    is_hard = jnp.arange(M, dtype=jnp.int32) < H
    # update_mask is True where we want to allow movement
    update_mask = jnp.logical_and(jnp.logical_not(fixed_jax), jnp.logical_not(is_hard))
    update_mask_2d = update_mask[:, None] # (M, 1) or (M,) -> (M, 1) for broadcasting?
    # jnp.where(grads, mask) -> keeps grads.
    
    # Grid for Density Proxy
    # Use grid_shape or a standard resolution
    N_grid_x, N_grid_y = grid_shape
    gx = jnp.linspace(gap, canvas_np[0] - gap, N_grid_x)
    gy = jnp.linspace(gap, canvas_np[1] - gap, N_grid_y)
    X, Y = jnp.meshgrid(gx, gy, indexing='ij')
    centers = jnp.stack([X.ravel(), Y.ravel()], axis=1) # (N_grid, 2)
    
    # Objective Function
    def objective_fn(positions):
        # Positions: (M, 2)
        
        # 1. Wirelength (HPWL)
        # Pin Locations
        # `locals` = [positions (M,2), port_pos (P,2)]
        # `pin_owner` maps to local index:
        # owner i < M -> index i
        # owner M+j -> index i + j (since port_pos starts after M)
        # So we can construct a combined array `locs`
        
        locs = jnp.concatenate([positions, port_pos], axis=0)
        # Global indices for ports are M+j. Since ports are at offset M in `locs`
        # Index `owner` in `locs` is valid directly.
        pin_centers = locs[pin_owner] + pin_off
        
        xmax = jax.ops.segment_max(pin_centers[:, 0], net_ids, N_NETS)
        xmin = jax.ops.segment_min(pin_centers[:, 0], net_ids, N_NETS)
        ymax = jax.ops.segment_max(pin_centers[:, 1], net_ids, N_NETS)
        ymin = jax.ops.segment_min(pin_centers[:, 1], net_ids, N_NETS)
        
        hpwl = (xmax - xmin) + (ymax - ymin)
        wl_cost = jnp.sum(weights * hpwl) / wire_n
        
        # 2. Density Proxy
        # Sum of Squared Gaussian Occupancy on Grid
        sigma_sq = (0.05 * canvas_np.max()) ** 2
        # Use full size for scale?
        # Distance matrix (M, Grid_N)
        dist_mat = jnp.sum((positions[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        
        areas = jnp.prod(sizes, axis=1) # (M,)
        
        # Normalize areas somewhat to avoid dominance of massive hard macros
        # But Hard macros shouldn't move.
        # We compute density from ALL blocks but optimize on Softs.
        
        occ_contrib = areas[:, None] * jnp.exp(-dist_mat / sigma_sq)
        occupancy = jnp.sum(occ_contrib, axis=0)
        density_cost = jnp.mean(occupancy**2)
        
        # We use a simplified cost structure
        return wl_cost + 0.25 * density_cost
        
    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(x)
    
    @jax.jit
    def step_fn(params, state):
        value, grads = jax.value_and_grad(objective_fn)(params)
        
        # Mask gradients: Only update Soft Clusters (Masked by update_mask)
        # update_mask shape (M,)
        grads_masked = grads * update_mask[:, None]
        
        updates, new_state = optimizer.update(grads_masked, state, params)
        params_new = optax.apply_updates(params, updates)
        
        # Clip to bounds
        # Bounds: lo=np.prod(sizes)/2 + gap ? No sizes/2.
        # We recalculate bounds on GPU
        lo_x_gpu = sizes / 2 + gap
        hi_x_gpu = jnp.broadcast_to(canvas, (M, 2)) - sizes / 2 - gap
        
        clipped = jnp.clip(params_new, lo_x_gpu, hi_x_gpu)
        
        # Restore Hard/Fixed positions (Enforce Constraints Strictly)
        # If update_mask is 0, `grads` was 0, so `params_new` == `params` 
        # But `clip` might nudge if `hi_x_gpu` > `params`?
        # params (Hard) are already inside `hi/lo` from legalization.
        # Just to be safe:
        final_params = jnp.where(update_mask[:, None], clipped, params)
        
        return final_params, new_state, value

    # Run Optimization Loop
    start_time = time.monotonic()
    
    for _ in range(400): # Max iterations
        if time.monotonic() > deadline - 15:
            break
        x, opt_state, val = step_fn(x, opt_state)
        
    # Final Cleanup
    final_pos = np.array(x, dtype=np.float32)
    
    # Hard constraints: Fixed positions must be preserved exactly
    final_pos[fixed_np] = initial_pos_np[fixed_np]
    
    # Hard constraints: Canvas bounds
    final_pos = np.clip(final_pos, lo_np, hi_np)
    
    # Return
    return {"positions": final_pos}
