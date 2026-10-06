import numpy as np
import time
import math

# The Evaluator class is assumed to be available in the global environment as specified.

def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    """
    Macro placement algorithm using Simulated Annealing guided by the Evaluator.
    Optimizes the proxy cost (wirelength, density, congestion).
    Ensures hard macro legality and canvas bounds.
    """
    start_time = time.monotonic()
    deadline = start_time + time_budget_s
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Extract and preprocess problem data
    # Use float32 for performance and memory
    canvas = np.asarray(problem['canvas'], dtype=np.float32)
    sizes = np.asarray(problem['sizes'], dtype=np.float32)
    fixed = np.asarray(problem['fixed'], dtype=bool)
    initial_pos = np.asarray(problem['initial_positions'], dtype=np.float32)
    
    num_hard = int(problem['num_hard'])
    M = initial_pos.shape[0]
    
    # Derived geometry
    half_sizes = sizes * 0.5
    
    # Canvas bounds for center coordinates
    # x/y center must be in [w/2, W-w/2]
    min_bounds = half_sizes
    max_bounds = canvas - half_sizes
    
    # Indices for movable blocks
    movable_mask = ~fixed
    movable_indices = np.flatnonzero(movable_mask)
    
    if len(movable_indices) == 0:
        return {"positions": initial_pos}

    # Initialize state
    pos = initial_pos.copy()
    best_pos = initial_pos.copy()
    
    # Initialize Evaluator
    ev = Evaluator(problem, pos)
    current_score = ev.score()['proxy_cost']
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0
    decay = 0.9996 # Slower decay
    batch_size = 16
    rebuild_interval = 5000
    
    last_rebuild = 0
    iterations = 0
    
    while time.monotonic() < deadline - 5.0:
        iterations += 1
        
        # Check budget
        if time.monotonic() >= deadline:
            break
            
        # Periodic rebuild to limit drift
        if iterations - last_rebuild >= rebuild_interval:
            ev.rebuild()
            current_score = ev.score()['proxy_cost']
            last_rebuild = iterations
            
        # Select a block to move
        idx = movable_indices[rng.integers(len(movable_indices))]
        is_hard = (idx < num_hard)
        
        # Step size: smaller for hard macros to maintain legality
        base_scale = min(canvas[0], canvas[1])
        if is_hard:
            step = 0.008 * base_scale
        else:
            step = 0.030 * base_scale
            
        # Generate Candidates
        noise = rng.normal(size=(batch_size, 2)) * step
        cands = pos[idx] + noise
        
        # Enforce Canvas Bounds
        cands = np.clip(cands, min_bounds[idx], max_bounds[idx])
        
        # Filter Hard Macro Overlaps
        valid_mask = np.ones(batch_size, dtype=bool)
        
        if is_hard:
            # Current Hard Geometry
            # We take a slice of current positions
            h_centers = pos[:num_hard]
            h_hs = half_sizes[:num_hard]
            
            # Bounds of current hard macros
            h_min = h_centers - h_hs
            h_max = h_centers + h_hs
            
            # Candidate Geometry
            c_hs = sizes[idx] / 2.0
            c_min = cands - c_hs
            c_max = cands + c_hs
            
            # Check intersection
            # [B, 1] vs [1, H]
            overlap_x = (c_max[:, None, 0] > h_min[None, :, 0]) & (c_min[:, None, 0] < h_max[None, :, 0])
            overlap_y = (c_max[:, None, 1] > h_min[None, :, 1]) & (c_min[:, None, 1] < h_max[None, :, 1])
            overlaps = overlap_x & overlap_y
            
            # Ignore self-overlap
            overlaps[:, idx] = False
            
            # Valid if no overlap with ANY hard macro
            valid_mask = ~np.any(overlaps, axis=1)
            
        if not np.any(valid_mask):
            continue
            
        active_cands = cands[valid_mask]
        
        # Evaluate Moves
        try:
            eval_ids = np.full(len(active_cands), idx, dtype=np.int32)
            scores = ev.evaluate_moves(eval_ids, active_cands)
        except Exception:
            continue
            
        costs = np.array([s['proxy_cost'] for s in scores], dtype=np.float32)
        
        # Select best candidate
        best_local_idx = np.argmin(costs)
        best_cand_cost = costs[best_local_idx]
        best_cand_pos = active_cands[best_local_idx]
        
        # Acceptance Decision (Simulated Annealing)
        delta = best_cand_cost - current_score
        accepted = False
        
        if delta < 0:
            accepted = True
        else:
            if T > 1e-10:
                rate = math.exp(-delta / T)
                if rng.random() < rate:
                    accepted = True
        
        # Annealing Schedule
        T *= decay
        
        if accepted:
            # Apply and Commit Move
            ev.apply(idx, best_cand_pos)
            ev.commit()
            
            # Update internal position state
            pos[idx] = best_cand_pos
            current_score = best_cand_cost
            
            # Update Best Solution
            if current_score < best_score:
                best_score = current_score
                best_pos = pos.copy()
    
    # Final Rebuild
    try:
        ev.rebuild()
    except Exception:
        pass
        
    return {"positions": best_pos}
