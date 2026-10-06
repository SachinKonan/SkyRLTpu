RUST_CODE = r'''
#[derive(Debug, Clone)]
struct WeightedRequest {
    pair: (usize, usize),
    weight: f64,
}

#[derive(Debug, Clone)]
struct ScoringState {
    requests: Vec<WeightedRequest>,
    involved_qubits: Vec<usize>,
}

impl ScoringState {
    fn from_ctx(ctx: &SwapSelectionContext<'_>, lookahead_size: usize) -> Self {
        let mut requests = Vec::new();
        let mut involved_qubits = HashSet::new();

        // Front layer: Maximum priority.
        for &node_id in ctx.front_layer().node_ids() {
            if let Some(pair) = ctx.front_layer().logical_pair(node_id) {
                requests.push(WeightedRequest {
                    pair,
                    weight: 200.0,
                });
                involved_qubits.insert(pair.0);
                involved_qubits.insert(pair.1);
            }
        }

        // Lookahead: Use BFS to prioritize gates that will become ready soon.
        let mut pred_counts = ctx.remaining().remaining_predecessor_counts().to_vec();
        let mut queue = VecDeque::new();
        
        for &node_id in ctx.front_layer().node_ids() {
            for &succ in ctx.circuit().node(node_id).successors() {
                pred_counts[succ] -= 1;
                if pred_counts[succ] == 0 {
                    queue.push_back((succ, 1));
                }
            }
        }

        let mut count = 0;
        while !queue.is_empty() && count < lookahead_size {
            let (node_id, dist) = queue.pop_front().unwrap();
            if let Some(pair) = ctx.circuit().node(node_id).two_qubit_pair() {
                let urgency = ctx.circuit().node(node_id).successors().len() as f64;
                // Decay weight exponentially by distance from front layer.
                let weight = (100.0 * 0.4f64.powi(dist as i32)) * (1.0 + 0.2 * urgency);
                
                requests.push(WeightedRequest {
                    pair,
                    weight,
                });
                involved_qubits.insert(pair.0);
                involved_qubits.insert(pair.1);
                count += 1;
            }
            for &succ in ctx.circuit().node(node_id).successors() {
                pred_counts[succ] -= 1;
                if pred_counts[succ] == 0 {
                    queue.push_back((succ, dist + 1));
                }
            }
        }

        Self {
            requests,
            involved_qubits: involved_qubits.into_iter().collect(),
        }
    }

    fn evaluate_swap(&self, swap: (usize, usize), ctx: &SwapSelectionContext<'_>) -> f64 {
        let (p1, p2) = swap;
        let layout = ctx.layout();
        let topology = ctx.topology();
        let mut delta = 0.0;

        for req in &self.requests {
            let phys1 = layout.physical_of_logical(req.pair.0);
            let phys2 = layout.physical_of_logical(req.pair.1);

            let dist_old = topology.distance(phys1, phys2);
            
            let phys1_new = if phys1 == p1 { p2 } else if phys1 == p2 { p1 } else { phys1 };
            let phys2_new = if phys2 == p1 { p2 } else if phys2 == p2 { p1 } else { phys2 };
            let dist_new = topology.distance(phys1_new, phys2_new);
            
            // Use distance squared to prioritize getting qubits close.
            let change = (dist_new * dist_new) as f64 - (dist_old * dist_old) as f64;
            delta += req.weight * change;

            // Extra bonus for achieving adjacency (dist=1).
            if dist_new == 1 && dist_old > 1 {
                delta -= req.weight * 0.5;
            }
        }
        delta
    }
}

fn enumerate_candidates(topology: TopologyView<'_>, layout: LayoutView<'_>, involved_logical: &[usize]) -> Vec<(usize, usize)> {
    let mut candidates = HashSet::new();
    for &logical in involved_logical {
        let phys = layout.physical_of_logical(logical);
        for &neighbor in topology.neighbors(phys) {
            let edge = if phys < neighbor { (phys, neighbor) } else { (neighbor, phys) };
            candidates.insert(edge);
        }
    }
    candidates.into_iter().collect()
}

fn choose_compact_subset(
    topology: TopologyView<'_>, 
    size: usize, 
    component: &[usize]
) -> Vec<usize> {
    if size >= component.len() {
        return component.to_vec();
    }

    let local_index = component.iter().enumerate().map(|(i, &g)| (g, i)).collect::<HashMap<usize, usize>>();
    let mut adj = Array2::<f64>::zeros((component.len(), component.len()));
    for &global_a in component {
        let a = local_index[&global_a];
        for &global_b in topology.neighbors(global_a) {
            if let Some(&b) = local_index.get(&global_b) {
                adj[[a, b]] = 1.0;
            }
        }
    }
    let err_matrix = Array2::<f64>::zeros((component.len(), component.len()));
    let [_, _, best_map] = dense_layout::best_subset(
        size,
        adj.view(),
        0,
        0,
        false,
        true,
        err_matrix.view(),
    );
    
    let chosen: Vec<usize> = best_map.into_iter().take(size).map(|local| component[local]).collect();
    chosen
}

fn perform_layout(ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    
    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    
    let mut logical_sorted = logical_components.to_vec();
    // Randomize sort order occasionally to exploit the 20 trials provided by the engine.
    if rng.gen_f64() < 0.3 {
        rng.shuffle_slice(&mut logical_sorted);
    } else {
        logical_sorted.sort_by_key(|c| std::cmp::Reverse(c.len()));
    }
    
    let mut target_sorted = target_components.iter().enumerate().map(|(i, c)| (i, c.len())).collect::<Vec<_>>();
    target_sorted.sort_by_key(|(_, s)| std::cmp::Reverse(*s));

    let mut free_capacity = target_sorted.iter().map(|&(i, s)| (i, s)).collect::<HashMap<usize, usize>>();
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];

    for logical in logical_sorted {
        let size = logical.len();
        let mut chosen_target = None;
        for (t_idx, _) in &target_sorted {
            if *free_capacity.get(t_idx).unwrap_or(&0) >= size {
                chosen_target = Some(*t_idx);
                break;
            }
        }
        
        if let Some(t_idx) = chosen_target {
            let target_comp = &target_components[t_idx];
            let phys_subset = choose_compact_subset(topology, size, target_comp);
            for (l, p) in logical.iter().zip(phys_subset) {
                mapping[*l] = p;
                used_physical[p] = true;
            }
            *free_capacity.get_mut(&t_idx).unwrap() -= size;
        }
    }

    let mut free_phys = (0..topology.num_qubits()).filter(|&p| !used_physical[p]);
    for slot in &mut mapping {
        if *slot == usize::MAX {
            *slot = free_phys.next().ok_or_else(|| RouterError::Routing("Insufficient physical qubits".into()))?;
        }
    }
    Ok(mapping)
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    lookahead_size: usize,
    penalty_state: Vec<f64>,
    penalty_inc: f64,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            lookahead_size: 60, 
            penalty_state: Vec::new(),
            penalty_inc: 0.025,
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(&mut self, ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
        self.penalty_state.clear();
        perform_layout(ctx, rng)
    }

    fn choose_best_swap(&mut self, ctx: &SwapSelectionContext<'_>, rng: &mut RngState) -> Option<(usize, usize)> {
        let num_qubits = ctx.topology().num_qubits();
        if self.penalty_state.len() != num_qubits {
            self.penalty_state = vec![0.0; num_qubits];
        }

        if ctx.swaps_since_progress() == 0 {
            self.penalty_state.fill(0.0);
        } else if let Some((a, b)) = ctx.last_applied_swap() {
            if ctx.swaps_since_progress() % 24 == 0 {
                self.penalty_state.fill(0.0);
            } else {
                self.penalty_state[a] += self.penalty_inc;
                self.penalty_state[b] += self.penalty_inc;
            }
        }

        let scoring = ScoringState::from_ctx(ctx, self.lookahead_size);
        let candidates = enumerate_candidates(ctx.topology(), ctx.layout(), &scoring.involved_qubits);
        if candidates.is_empty() { return None; }

        let mut best_swaps = Vec::new();
        let mut min_score = f64::INFINITY;

        for swap in candidates {
            let delta = scoring.evaluate_swap(swap, ctx);
            let score = delta + self.penalty_state[swap.0] + self.penalty_state[swap.1];

            if score < min_score - 1e-7 {
                min_score = score;
                best_swaps.clear();
                best_swaps.push(swap);
            } else if (score - min_score).abs() < 1e-7 {
                best_swaps.push(swap);
            }
        }

        if best_swaps.is_empty() { return None; }
        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
'''
