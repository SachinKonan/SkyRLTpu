RUST_CODE = r'''
#[derive(Debug, Clone)]
struct LookaheadScores {
    nodes: Vec<(usize, f64)>,
}

impl LookaheadScores {
    fn new(ctx: &SwapSelectionContext<'_>, max_depth: usize, gamma: f64) -> Self {
        let mut weighted_nodes = Vec::new();
        let mut current_layer = ctx.front_layer().node_ids().to_vec();
        let mut visited = std::collections::HashSet::new();
        let mut depth = 0;
        
        let mut pred_counts = ctx.remaining().remaining_predecessor_counts().to_vec();

        while !current_layer.is_empty() && depth < max_depth {
            let mut next_layer = Vec::new();
            let weight = gamma.powi(depth as i32);
            for &node_id in &current_layer {
                if visited.insert(node_id) {
                    if let Some(_) = ctx.circuit().node(node_id).two_qubit_pair() {
                        weighted_nodes.push((node_id, weight));
                    }
                    for &successor in ctx.circuit().node(node_id).successors() {
                        if successor < pred_counts.len() {
                            pred_counts[successor] = pred_counts[successor].saturating_sub(1);
                            if pred_counts[successor] == 0 {
                                next_layer.push(successor);
                            }
                        }
                    }
                }
            }
            current_layer = next_layer;
            depth += 1;
        }
        Self { nodes: weighted_nodes }
    }

    fn score_delta(&self, swap: (usize, usize), ctx: &SwapSelectionContext<'_>) -> f64 {
        let (p1, p2) = swap;
        let topology = ctx.topology();
        let layout = ctx.layout();
        let mut total = 0.0;
        
        for &(node_id, weight) in &self.nodes {
            if let Some((q1, q2)) = ctx.circuit().node(node_id).two_qubit_pair() {
                let phys1 = layout.physical_of_logical(q1);
                let phys2 = layout.physical_of_logical(q2);
                
                let dist_before = topology.distance(phys1, phys2) as f64;
                let mut nphys1 = phys1;
                let mut nphys2 = phys2;
                if nphys1 == p1 { nphys1 = p2; } else if nphys1 == p2 { nphys1 = p1; }
                if nphys2 == p1 { nphys2 = p2; } else if nphys2 == p2 { nphys2 = p1; }
                let dist_after = topology.distance(nphys1, nphys2) as f64;
                total += weight * (dist_before - dist_after);
            }
        }
        total
    }
}

fn choose_improved_layout(ctx: &InitialLayoutContext<'_>) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let num_physical = topology.num_qubits();
    
    if num_logical > num_physical {
        return Err(RouterError::Routing("Too many logical qubits".into()));
    }

    let mut weights = vec![vec![0; num_logical]; num_logical];
    let mut total_interaction = vec![0; num_logical];
    for node_id in circuit.node_ids() {
        if let Some((q1, q2)) = circuit.node(node_id).two_qubit_pair() {
            weights[q1][q2] += 1;
            weights[q2][q1] += 1;
            total_interaction[q1] += 1;
            total_interaction[q2] += 1;
        }
    }

    let mut logical_sorted: Vec<usize> = (0..num_logical).collect();
    logical_sorted.sort_by_key(|&q| std::cmp::Reverse(total_interaction[q]));

    let mut physical_sorted: Vec<usize> = (0..num_physical).collect();
    physical_sorted.sort_by_key(|&p| std::cmp::Reverse(topology.degree(p)));

    let mut mapping = vec![0; num_logical];
    let mut physical_used = vec![false; num_physical];
    let mut placed_logical = Vec::new();

    for &l in &logical_sorted {
        let mut best_p = None;
        let mut min_cost = f64::INFINITY;

        for p in 0..num_physical {
            if physical_used[p] { continue; }
            
            let mut cost = 0.0;
            for &placed_l in &placed_logical {
                let dist = topology.distance(p, mapping[placed_l]) as f64;
                cost += weights[l][placed_l] as f64 * dist;
            }
            
            if cost < min_cost {
                min_cost = cost;
                best_p = Some(p);
            }
        }

        if let Some(p) = best_p {
            mapping[l] = p;
            physical_used[p] = true;
            placed_logical.push(l);
        } else {
            // Fallback
            for p in 0..num_physical {
                if !physical_used[p] {
                    mapping[l] = p;
                    physical_used[p] = true;
                    placed_logical.push(l);
                    break;
                }
            }
        }
    }
    Ok(mapping)
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    lookahead_depth: usize,
    lookahead_gamma: f64,
    progress_bonus: f64,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            lookahead_depth: 8,
            lookahead_gamma: 0.6,
            progress_bonus: 3.0,
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        choose_improved_layout(ctx)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        let front_layer = ctx.front_layer();
        if front_layer.is_empty() { return None; }
        
        let topology = ctx.topology();
        let layout = ctx.layout();
        let node_ids = front_layer.node_ids();
        
        let mut candidate_swaps = std::collections::HashSet::new();
        let num_targets = 8.min(node_ids.len());
        
        for i in 0..num_targets {
            let node_id = if ctx.swaps_since_progress() > 10 {
                node_ids[rng.gen_index(node_ids.len())]
            } else {
                node_ids[i]
            };
            
            if let Some((q1, q2)) = front_layer.logical_pair(node_id) {
                let p1 = layout.physical_of_logical(q1);
                let p2 = layout.physical_of_logical(q2);
                let path = topology.shortest_path(p1, p2);
                for win in path.windows(2) {
                    let edge = if win[0] < win[1] { (win[0], win[1]) } else { (win[1], win[0]) };
                    candidate_swaps.insert(edge);
                }
                for &p in &[p1, p2] {
                    for &neighbor in topology.neighbors(p) {
                        let edge = if p < neighbor { (p, neighbor) } else { (neighbor, p) };
                        candidate_swaps.insert(edge);
                    }
                }
            }
        }

        if candidate_swaps.is_empty() { return None; }

        let lookahead = LookaheadScores::new(ctx, self.lookahead_depth, self.lookahead_gamma);
        let mut best_swaps = Vec::new();
        let mut max_score = f64::NEG_INFINITY;

        for swap in candidate_swaps {
            let (p1, p2) = swap;
            let mut score = 0.0;
            
            // Front layer delta
            for &node_id in node_ids {
                if let Some((q1, q2)) = front_layer.logical_pair(node_id) {
                    let phys1 = layout.physical_of_logical(q1);
                    let phys2 = layout.physical_of_logical(q2);
                    let dist_before = topology.distance(phys1, phys2);
                    
                    let mut nphys1 = phys1;
                    let mut nphys2 = phys2;
                    if nphys1 == p1 { nphys1 = p2; } else if nphys1 == p2 { nphys1 = p1; }
                    if nphys2 == p1 { nphys2 = p2; } else if nphys2 == p2 { nphys2 = p1; }
                    let dist_after = topology.distance(nphys1, nphys2);
                    
                    score += (dist_before as f64 - dist_after as f64) * 2.0;
                    if dist_after == 1 && dist_before > 1 {
                        score += self.progress_bonus;
                    }
                }
            }
            
            score += lookahead.score_delta(swap, ctx);
            
            if let Some(last_swap) = ctx.last_applied_swap() {
                if last_swap == swap {
                    score -= 20.0;
                }
            }

            if score > max_score + 1e-7 {
                max_score = score;
                best_swaps.clear();
                best_swaps.push(swap);
            } else if (score - max_score).abs() <= 1e-7 {
                best_swaps.push(swap);
            }
        }
        
        if best_swaps.is_empty() { None } else { Some(best_swaps[rng.gen_index(best_swaps.len())]) }
    }
}
'''
