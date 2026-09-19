import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple

def is_overlapping(xy, idx, pos, sizes, num_hard, margin):
    """
    Check if moving a hard macro 'idx' to 'xy' causes overlap with other hard macros.
    Hard macros are indices < num_hard.
    """
    if idx >= num_hard:
        return False
    
    # Hard-macro overlap check against all other hard macros
    # pos[:num_hard] contains the current centers of hard macros
    h_pos = pos[:num_hard]
    h_sizes = sizes[:num_hard]
    
    # Vectorized distance check (L-infinity distance for rectangles)
    dx = np.abs(h_pos[:, 0] - xy[0])
    dy = np.abs(h_pos[:, 1] - xy[1])
    
    # Collision if center-to-center distance < sum of half-widths
    min_dx = (h_sizes[:, 0] + sizes[idx, 0]) / 2 + margin
    min_dy = (h_sizes[:, 1] + sizes[idx, 1]) / 2 + margin
    
    overlap = (dx < min_dx) & (dy < min_dy)
    
    # Exclude the block itself from the check
    overlap[idx] = False
    return np.any(overlap)

def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    """
    Optimizes chip macro placement by addressing the identified high congestion and density costs.
    Uses an adaptive Simulated Annealing (SA) strategy with JAX-accelerated centroid 
    calculations and specifically targeted candidates to resolve spatial bottlenecks.
    """
    start_time = time.monotonic()
    # Reserve buffer to ensure we return the best legal solution
    deadline = start_time + max(0.0, time_budget_s - 3.0)
    rng = np.random.default_rng(seed)
    
    # Input extraction
    pos = np.array(problem['initial_positions'], dtype=np.float32, copy=True)
    sizes = np.array(problem['sizes'], dtype=np.float32)
    canvas = np.array(problem['canvas'], dtype=np.float32)
    fixed = np.array(problem['fixed'], dtype=np.bool_)
    num_hard = int(problem['num_hard'])
    movable = np.flatnonzero(~fixed)
    M = len(pos)
    P = len(problem['port_positions'])
    
    if len(movable) == 0:
        return {"positions": pos}

    margin = 1e-3
    best_pos = pos.copy()
    
    # Precompute bounds for each block center
    half_sizes = sizes / 2.0
    min_bounds = half_sizes + margin
    max_bounds = canvas - half_sizes - margin

    # JAX Acceleration for Centroids
    # Pre-build a connectivity matrix including ports as fixed nodes
    # connectivity[i, j] stores the sum of net weights shared by block/port i and j
    conn = np.zeros((M + P, M + P), dtype=np.float32)
    pin_owner = problem['pin_owner']
    net_offsets = problem['net_offsets']
    net_weights = problem['net_weights']
    
    for net_idx in range(len(net_offsets) - 1):
        start, end = net_offsets[net_idx], net_offsets[net_idx + 1]
        owners = pin_owner[start:end]
        weight = net_weights[net_idx]
        for i in range(len(owners)):
            o1 = owners[i]
            for j in range(i + 1, len(owners)):
                o2 = owners[j]
                conn[o1, o2] += weight
                conn[o2, o1] += weight

    # JAX function to update all centroids
    # We include port positions in the current state to calculate centroids correctly
    def compute_all_centroids(current_pos_expanded, conn_matrix):
        # current_pos_expanded shape: [M+P, 2]
        # conn_matrix shape: [M+P, M+P]
        weighted_sum = jnp.dot(conn_matrix, current_pos_expanded)
        total_weights = jnp.sum(conn_matrix, axis=1, keepdims=True)
        # Avoid division by zero for isolated blocks
        centroids = weighted_sum / jnp.where(total_weights > 0, total_weights, 1.0)
        return centroids

    # Initialize JAX constants
    conn_jax = jnp.array(conn)
    jit_centroids = jax.jit(compute_all_centroids)
    
    # Setup augmented positions (Macros + Ports) for JAX
    ports_pos = np.array(problem['port_positions'], dtype=np.float32)
    
    # Initial state tracking
    with Evaluator(problem, pos) as ev:
        current_eval = ev.score()
        current_proxy = current_eval['proxy_cost']
        best_proxy = current_proxy
        
        # Adaptive Temperature: Scale relative to starting cost
        T_start = current_proxy * 0.02
        T_end = current_proxy * 1e-6
        
        iter_count = 0
        
        while time.monotonic() < deadline:
            iter_count += 1
            elapsed = time.monotonic() - start_time
            progress = min(1.0, elapsed / (deadline - start_time))
            
            # Temperature and radius decay
            T = T_start * ((T_end / T_start) ** progress)
            radius_glob = canvas * (0.1 * (1.0 - progress) + 0.001)
            radius_loc = canvas * (0.03 * (1.0 - progress) + 0.0001)
            
            # Periodic update of centroids for all movable blocks
            augmented_pos = np.vstack([pos, ports_pos])
            all_centroids = jit_centroids(jnp.array(augmented_pos), conn_jax)
            all_centroids_np = np.array(all_centroids)
            
            # Process blocks to prevent biased placement
            blocks_to_process = rng.permutation(movable)
            
            for idx in blocks_to_process:
                b_min = min_bounds[idx]
                b_max = max_bounds[idx]
                
                # Proposal generation
                proposals = []
                
                # 1. Centroid move: strong weight on wirelength
                centroid = all_centroids_np[idx]
                proposals.append(np.clip(centroid, b_min, b_max))
                
                # 2. Local sampling for density/congestion repulsion
                # We sample 8 directions to find a local cost gradient descent
                angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                for angle in angles:
                    offset = np.array([np.cos(angle), np.sin(angle)]) * radius_loc
                    proposals.append(np.clip(pos[idx] + offset, b_min, b_max))
                
                # 3. Sparse global jump to escape local minima
                jump = rng.uniform(-1, 1, 2) * radius_glob
                proposals.append(np.clip(pos[idx] + jump, b_min, b_max))
                
                # Legality Filter (Hard Macro Overlap)
                valid_proposals = []
                for p in proposals:
                    if not is_overlapping(p, idx, pos, sizes, num_hard, margin):
                        valid_proposals.append(p)
                
                if not valid_proposals:
                    continue
                
                # Batch evaluate candidates for the current block
                res_list = ev.evaluate_moves(
                    np.array([idx] * len(valid_proposals), dtype=np.int32), 
                    np.array(valid_proposals, dtype=np.float32)
                )
                
                # Find best candidate based on proxy_cost
                best_cand_idx = -1
                min_cand_proxy = float('inf')
                for i, res in enumerate(res_list):
                    p_cost = res['proxy_cost']
                    if p_cost < min_cand_proxy:
                        min_cand_proxy = p_cost
                        best_cand_idx = i
                
                if best_cand_idx != -1:
                    delta = min_cand_proxy - current_proxy
                    # Metropolis criterion for SA
                    if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                        best_xy = valid_proposals[best_cand_idx]
                        
                        ev.apply(idx, best_xy)
                        ev.commit()
                        
                        pos[idx] = best_xy
                        current_proxy = min_cand_proxy
                        
                        if current_proxy < best_proxy:
                            best_proxy = current_proxy
                            best_pos = pos.copy()
            
            # Periodic rebuild to prevent numerical drift inEvaluator
            if iter_count % 100 == 0:
                ev.rebuild()

        # Final state cleanup
        ev.rebuild()

    return {"positions": best_pos}
