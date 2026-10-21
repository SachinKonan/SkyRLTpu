import time
import numpy as np

def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    """
    Optimizes chip macro-placement to minimize proxy cost:
    cost = wirelength + 0.5 * density + 0.5 * congestion.
    
    Strategy:
    1. Select best starting layout
    2. Simulated Annealing with congestion-aware moves
    3. Strict Legalization enforcement for hard macros
    """
    rng = np.random.default_rng(seed)
    start_time = time.monotonic()
    deadline = start_time + time_budget_s - 4.0

    # Problem data
    canvas = np.asarray(problem['canvas'], dtype=np.float64)
    sizes = np.asarray(problem['sizes'], dtype=np.float64)
    half_sizes = sizes / 2.0
    fixed = np.asarray(problem['fixed'], dtype=bool)
    num_hard = int(problem['num_hard'])
    M = len(problem['initial_positions'])
    
    net_offsets = np.asarray(problem['net_offsets'], dtype=np.int32)
    pin_owner = np.asarray(problem['pin_owner'], dtype=np.int32)
    pin_offset = np.asarray(problem['pin_offset'], dtype=np.float32)
    net_weights = np.asarray(problem['net_weights'], dtype=np.float32)
    port_positions = np.asarray(problem['port_positions'], dtype=np.float64)
    
    # Precompute connectivity
    block_to_nets = [[] for _ in range(M)]
    for net_id in range(len(net_weights)):
        start, end = net_offsets[net_id], net_offsets[net_id+1]
        for pin_idx in range(start, end):
            owner = pin_owner[pin_idx]
            if owner < M:
                block_to_nets[owner].append(net_id)

    # Cost function
    def get_cost(s, stage='balanced'):
        if stage == 'congestion':
            return 0.1 * s['wirelength_cost'] + 0.1 * s['density_cost'] + 1.8 * s['congestion_cost']
        else:
            return s['proxy_cost']

    # Explore starting layouts
    start_candidates = [problem['initial_positions']]
    if 'starting_layouts' in problem:
        start_candidates.extend(problem['starting_layouts'])
    
    best_global_pos = np.array(problem['initial_positions'], dtype=np.float64)
    best_global_proxy = float('inf')
    
    explore_budget = max(8.0, 10.0 * len(start_candidates))
    explore_start = time.monotonic()
    
    start_perf = []
    for s_pos_raw in start_candidates:
        if time.monotonic() - explore_start > explore_budget:
            break
        
        pos = np.array(s_pos_raw, dtype=np.float64)
        movable = np.flatnonzero(~fixed)
        if len(movable) == 0:
            # If no movable, layout is fixed. Use initial score.
            with Evaluator(problem, pos) as ev:
                local_best_proxy = ev.score()['proxy_cost']
            start_perf.append((local_best_proxy, pos))
            continue
            
        with Evaluator(problem, pos) as ev:
            scores = ev.score()
            local_best_proxy = scores['proxy_cost']
            local_best_pos = pos.copy()
            
            t_start = get_cost(scores) * 0.03
            s_time = time.monotonic()
            while time.monotonic() - s_time < 6.0:
                sample_ids = rng.choice(movable, size=min(25, len(movable)), replace=False)
                v_ids, v_pos = [], []
                for bid in sample_ids:
                    xy = pos[bid] + rng.normal(0, canvas.max()*0.015, size=2)
                    if np.any(xy - half_sizes[bid] < 0) or np.any(xy + half_sizes[bid] > canvas):
                        continue
                    if bid < num_hard:
                        diff = np.abs(pos[:num_hard] - xy)
                        overlap_mask = np.all(diff < (half_sizes[:num_hard] + half_sizes[bid]), axis=1)
                        if np.any(overlap_mask & (np.arange(num_hard) != bid)):
                            continue
                    v_ids.append(bid)
                    v_pos.append(xy)
                
                if not v_ids:
                    continue
                res = ev.evaluate_moves(np.array(v_ids), np.array(v_pos, dtype=np.float32))
                
                best_idx = -1
                min_delta = float('inf')
                curr_c = get_cost(scores)
                for i, r in enumerate(res):
                    delta = get_cost(r) - curr_c
                    if delta < min_delta:
                        min_delta, best_idx = delta, i
                
                if best_idx != -1 and (min_delta < 0 or rng.random() < np.exp(-min_delta/t_start)):
                    tid, txy = v_ids[best_idx], v_pos[best_idx]
                    ev.apply(tid, txy)
                    ev.commit()
                    pos[tid] = txy
                    scores = ev.score()
                    if scores['proxy_cost'] < local_best_proxy:
                        local_best_proxy = scores['proxy_cost']
                        local_best_pos = pos.copy()
            
            start_perf.append((local_best_proxy, local_best_pos))

    if start_perf:
        best_idx = np.argmin([p[0] for p in start_perf])
        best_global_proxy, best_global_pos = start_perf[best_idx]
    
    # Intensive search on best start
    pos = best_global_pos.copy()
    movable = np.flatnonzero(~fixed)
    if len(movable) == 0:
        return {"positions": pos}

    with Evaluator(problem, pos) as ev:
        scores = ev.score()
        
        t_start = scores['proxy_cost'] * 0.1
        t_end = t_start * 0.001
        
        iter_count = 0
        search_start = time.monotonic()
        while time.monotonic() < deadline:
            iter_count += 1
            elapsed = time.monotonic() - search_start
            progress = min(1.0, elapsed / (deadline - search_start))
            
            # Stage proportions
            if progress < 0.15:
                stage = 'balanced'
            elif progress < 0.60:
                stage = 'congestion'
            else:
                stage = 'refine'
            
            temp = t_start * (1.0 - progress)**1.5 + t_end
            radius = canvas.max() * (0.1 * (1.0 - progress**0.5) + 0.001)
            
            B = 30 
            sample_ids = rng.choice(movable, size=min(B, len(movable)), replace=False)
            v_ids, v_pos = [], []
            
            for bid in sample_ids:
                m_type = rng.random()
                if m_type < 0.5:
                    xy = pos[bid] + rng.normal(0, radius * 0.5, size=2)
                elif m_type < 0.75:
                    if not block_to_nets[bid]:
                        xy = pos[bid] + rng.uniform(-radius, radius, size=2)
                    else:
                        net_id = rng.choice(block_to_nets[bid])
                        start, end = net_offsets[net_id], net_offsets[net_id+1]
                        others = [p for p in range(start, end) if pin_owner[p] != bid]
                        if not others: 
                            xy = pos[bid] + rng.uniform(-radius, radius, size=2)
                        else:
                            q_idx = rng.choice(others)
                            q_owner = pin_owner[q_idx]
                            target_ref = pos[q_owner] if q_owner < M else port_positions[q_owner - M]
                            target_center = target_ref + pin_offset[q_idx]
                            xy = target_center + rng.normal(0, radius * 0.1, size=2)
                elif m_type < 0.95:
                    if not block_to_nets[bid]:
                        xy = pos[bid] + rng.uniform(-radius, radius, size=2)
                    else:
                        net_id = rng.choice(block_to_nets[bid])
                        start, end = net_offsets[net_id], net_offsets[net_id+1]
                        others = [p for p in range(start, end) if pin_owner[p] != bid]
                        if not others: 
                            xy = pos[bid] + rng.uniform(-radius, radius, size=2)
                        else:
                            centroid = np.zeros(2)
                            count = 0
                            for p_idx in others:
                                owner = pin_owner[p_idx]
                                centroid += (pos[owner] if owner < M else port_positions[owner - M])
                                count += 1
                            centroid /= count
                            vec = pos[bid] - centroid
                            dist = np.linalg.norm(vec)
                            if dist < 1e-6: vec = rng.standard_normal(2)
                            xy = pos[bid] + (vec / (dist + 1e-6)) * rng.uniform(radius * 0.1, radius * 0.3)
                else:
                    low = half_sizes[bid] + 1e-4
                    high = canvas - half_sizes[bid] - 1e-4
                    xy = rng.uniform(np.minimum(low, high), np.maximum(low, high))

                if np.any(xy - half_sizes[bid] < 0) or np.any(xy + half_sizes[bid] > canvas):
                    continue
                if bid < num_hard:
                    diff = np.abs(pos[:num_hard] - xy)
                    overlap_mask = np.all(diff < (half_sizes[:num_hard] + half_sizes[bid]), axis=1)
                    if np.any(overlap_mask & (np.arange(num_hard) != bid)):
                        continue
                v_ids.append(bid)
                v_pos.append(xy)

            if not v_ids:
                continue
            
            res = ev.evaluate_moves(np.array(v_ids), np.array(v_pos, dtype=np.float32))
            
            best_idx = -1
            min_delta = float('inf')
            current_search_cost = get_cost(scores, stage)
            
            for i, r in enumerate(res):
                delta = get_cost(r, stage) - current_search_cost
                if delta < min_delta:
                    min_delta, best_idx = delta, i
            
            if best_idx != -1:
                if min_delta < 0 or (temp > 0 and rng.random() < np.exp(-min_delta / temp)):
                    tid, txy = v_ids[best_idx], v_pos[best_idx]
                    ev.apply(tid, txy)
                    ev.commit()
                    pos[tid] = txy
                    scores = ev.score()
                    if scores['proxy_cost'] < best_global_proxy:
                        best_global_proxy = scores['proxy_cost']
                        best_global_pos = pos.copy()

            if iter_count % 1200 == 0:
                ev.rebuild()

    # Final Legalization
    final_pos = best_global_pos.copy()
    for i in range(M):
        if fixed[i]:
            final_pos[i] = problem['initial_positions'][i]
        else:
            low = half_sizes[i] + 1e-6
            high = canvas - half_sizes[i] - 1e-6
            final_pos[i] = np.clip(final_pos[i], low, high)
        
    return {"positions": final_pos}
