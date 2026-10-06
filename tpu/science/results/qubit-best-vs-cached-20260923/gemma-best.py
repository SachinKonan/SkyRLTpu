RUST_CODE = r'''
#[derive(Debug, Clone)]
struct InteractionWindow {
    weights: Vec<Vec<(usize, f64)>>,
}

impl InteractionWindow {
    fn new(num_qubits: usize) -> Self {
        Self {
            weights: vec![Vec::new(); num_qubits],
        }
    }

    fn add_interaction(&mut self, p1: usize, p2: usize, weight: f64) {
        self.weights[p1].push((p2, weight));
        self.weights[p2].push((p1, weight));
    }

    fn score_swap(&self, a: usize, b: usize, topology: TopologyView<'_>) -> f64 {
        let mut delta = 0.0;
        if let Some(w_a) = self.weights.get(a) {
            for &(other, w) in w_a {
                if other == b { continue; }
                delta += w * (topology.distance(b, other) as f64 - topology.distance(a, other) as f64);
            }
        }
        if let Some(w_b) = self.weights.get(b) {
            for &(other, w) in w_b {
                if other == a { continue; }
                delta += w * (topology.distance(a, other) as f64 - topology.distance(b, other) as f64);
            }
        }
        delta
    }
}

fn build_interaction_window(ctx: &SwapSelectionContext<'_>, depth: usize) -> InteractionWindow {
    let mut window = InteractionWindow::new(ctx.topology().num_qubits());
    let mut pred_counts = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut current_layer = ctx.front_layer().node_ids().to_vec();
    
    let weights = [10.0, 5.0, 2.5, 1.25, 0.6, 0.3, 0.15, 0.07, 0.03, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002];
    for d in 0..depth {
        if current_layer.is_empty() { break; }
        let base_w = if d < weights.len() { weights[d] } else { 0.0001 };
        
        let mut next_layer = Vec::new();
        for &node_id in &current_layer {
            let node = ctx.circuit().node(node_id);
            if let Some((la, lb)) = node.two_qubit_pair() {
                let p_la = ctx.layout().physical_of_logical(la);
                let p_lb = ctx.layout().physical_of_logical(lb);
                let criticality = 1.0 + (node.block_count() as f64 * 0.1);
                window.add_interaction(p_la, p_lb, base_w * criticality);
            }
            for &successor in node.successors() {
                if pred_counts[successor] > 0 {
                    pred_counts[successor] -= 1;
                    if pred_counts[successor] == 0 {
                        next_layer.push(successor);
                    }
                }
            }
        }
        current_layer = next_layer;
    }
    window
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    decay_state: Vec<f64>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self { decay_state: Vec::new() }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        let circuit = ctx.circuit();
        let topology = ctx.topology();
        let num_logical = circuit.num_logical_qubits();
        let logical_components = circuit.logical_interaction_components();
        let physical_components = topology.connected_components();

        let mut interactions = vec![vec![0usize; num_logical]; num_logical];
        let mut logical_degree = vec![0usize; num_logical];
        for node_id in circuit.node_ids() {
            if let Some((la, lb)) = circuit.node(node_id).two_qubit_pair() {
                interactions[la][lb] += 1;
                interactions[lb][la] += 1;
                logical_degree[la] += 1;
                logical_degree[lb] += 1;
            }
        }

        let mut phys_centrality = vec![0u32; topology.num_qubits()];
        let distances = topology.distances();
        for p in 0..topology.num_qubits() {
            phys_centrality[p] = distances[p].iter().sum();
        }

        let mut mapping = vec![usize::MAX; num_logical];
        let mut used_physical = vec![false; topology.num_qubits()];
        let mut p_comp_used_count = vec![0usize; physical_components.len()];

        for l_comp in logical_components {
            if l_comp.is_empty() { continue; }

            let l_seed = *l_comp.iter().max_by_key(|&&l| logical_degree[l]).unwrap();
            let mut candidates: Vec<usize> = (0..physical_components.len())
                .filter(|&pc_idx| physical_components[pc_idx].len() >= l_comp.len() + p_comp_used_count[pc_idx])
                .collect();
            
            if candidates.is_empty() {
                candidates = (0..physical_components.len())
                    .filter(|&pc_idx| physical_components[pc_idx].len() > p_comp_used_count[pc_idx])
                    .collect();
            }
            
            if candidates.is_empty() {
                return Err(RouterError::Routing("No physical capacity".to_string()));
            }
            
            let pc_idx = candidates[rng.gen_index(candidates.len())];
            let p_comp = &physical_components[pc_idx];

            let mut p_seeds: Vec<usize> = p_comp.iter()
                .filter(|&&p| !used_physical[p])
                .cloned()
                .collect();
            p_seeds.sort_by_key(|&p| phys_centrality[p]);
            let p_seed = p_seeds[rng.gen_index(std::cmp::min(p_seeds.len(), 2))];
            
            mapping[l_seed] = p_seed;
            used_physical[p_seed] = true;
            p_comp_used_count[pc_idx] += 1;

            let mut mapped_in_comp = std::collections::HashSet::new();
            mapped_in_comp.insert(l_seed);
            let mut queue = VecDeque::new();
            queue.push_back(l_seed);

            while let Some(l_u) = queue.pop_front() {
                for &l_v in l_comp {
                    if mapped_in_comp.contains(&l_v) { continue; }
                    if interactions[l_u][l_v] > 0 {
                        let mut best_p = usize::MAX;
                        let mut min_cost = u32::MAX;

                        for &p_w in p_comp {
                            if used_physical[p_w] { continue; }
                            let mut cost = 0u32;
                            for &l_mapped in l_comp {
                                if mapping[l_mapped] != usize::MAX && interactions[l_v][l_mapped] > 0 {
                                    cost += (interactions[l_v][l_mapped] as u32) * topology.distance(p_w, mapping[l_mapped]);
                                }
                            }
                            if cost < min_cost {
                                min_cost = cost;
                                best_p = p_w;
                            }
                        }

                        if best_p != usize::MAX {
                            mapping[l_v] = best_p;
                            used_physical[best_p] = true;
                            p_comp_used_count[pc_idx] += 1;
                            mapped_in_comp.insert(l_v);
                            queue.push_back(l_v);
                        }
                    }
                }
            }
        }

        let mut free_physical = (0..topology.num_qubits()).filter(|&p| !used_physical[p]);
        for slot in &mut mapping {
            if *slot == usize::MAX {
                *slot = free_physical.next().ok_or_else(|| RouterError::Routing("Out of space".to_string()))?;
            }
        }
        Ok(mapping)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        let num_qubits = ctx.topology().num_qubits();
        if self.decay_state.len() != num_qubits {
            self.decay_state = vec![0.0; num_qubits];
        }

        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(0.0);
        } else if let Some((a, b)) = ctx.last_applied_swap() {
            self.decay_state[a] += 1.0;
            self.decay_state[b] += 1.0;
        }
        
        for val in &mut self.decay_state {
            *val *= 0.95;
        }

        let window = build_interaction_window(ctx, 15);
        let mut candidates = std::collections::HashSet::new();
        for p in 0..num_qubits {
            if !window.weights[p].is_empty() {
                for &neighbor in ctx.topology().neighbors(p) {
                    let edge = if p < neighbor { (p, neighbor) } else { (neighbor, p) };
                    candidates.insert(edge);
                }
            }
        }

        if candidates.is_empty() { return None; }

        let mut best_swaps = Vec::new();
        let mut min_score = f64::INFINITY;
        let epsilon = 1e-7;

        for swap in candidates {
            let delta = window.score_swap(swap.0, swap.1, ctx.topology());
            
            let mut bonus = 0.0;
            for &node_id in ctx.front_layer().node_ids() {
                if let Some((la, lb)) = ctx.front_layer().logical_pair(node_id) {
                    let p_la = ctx.layout().physical_of_logical(la);
                    let p_lb = ctx.layout().physical_of_logical(lb);
                    let old_dist = ctx.topology().distance(p_la, p_lb);
                    let mut new_dist = old_dist;
                    if p_la == swap.0 { new_dist = ctx.topology().distance(swap.1, p_lb); }
                    else if p_la == swap.1 { new_dist = ctx.topology().distance(swap.0, p_lb); }
                    else if p_lb == swap.0 { new_dist = ctx.topology().distance(p_la, swap.1); }
                    else if p_lb == swap.1 { new_dist = ctx.topology().distance(p_la, swap.0); }
                    
                    if new_dist < old_dist {
                        bonus -= 6.0 * (old_dist - new_dist) as f64;
                        if new_dist == 1 { bonus -= 15.0; }
                    } else if new_dist > old_dist {
                        bonus += 10.0 * (new_dist - old_dist) as f64;
                    }
                }
            }

            let penalty = (self.decay_state[swap.0] + self.decay_state[swap.1]) * 0.3;
            let score = delta + penalty + bonus;

            if score < min_score - epsilon {
                min_score = score;
                best_swaps.clear();
                best_swaps.push(swap);
            } else if (score - min_score).abs() <= epsilon {
                best_swaps.push(swap);
            }
        }

        if best_swaps.is_empty() { return None; }
        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
'''
