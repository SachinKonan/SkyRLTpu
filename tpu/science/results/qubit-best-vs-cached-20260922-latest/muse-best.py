RUST_CODE = r'''
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

#[derive(Debug, Clone, Default)]
struct FrontLayerScores {
    nodes: Vec<[usize; 2]>,
    qubits: Vec<Option<(usize, usize)>>,
}

impl FrontLayerScores {
    fn from_ctx(ctx: &SwapSelectionContext<'_>) -> Self {
        let mut out = Self {
            nodes: Vec::new(),
            qubits: vec![None; ctx.topology().num_qubits()],
        };
        for pair in ctx.front_layer().physical_pairs() {
            let [a, b] = *pair;
            let index = out.nodes.len();
            out.nodes.push([a, b]);
            out.qubits[a] = Some((index, b));
            out.qubits[b] = Some((index, a));
        }
        out
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn is_active(&self, qubit: usize) -> bool {
        self.qubits[qubit].is_some()
    }

    fn iter_active(&self) -> impl Iterator<Item = &usize> {
        self.nodes.iter().flatten()
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut delta = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            delta += (topology.distance(b, c) as f64) - (topology.distance(a, c) as f64);
        }
        if let Some((_, c)) = self.qubits[b] {
            delta += (topology.distance(a, c) as f64) - (topology.distance(b, c) as f64);
        }
        delta
    }
}

#[derive(Debug, Clone)]
struct ExtendedSetScores {
    qubits: Vec<Vec<usize>>,
    len: usize,
}

impl ExtendedSetScores {
    fn new(num_qubits: usize) -> Self {
        Self {
            qubits: vec![Vec::new(); num_qubits],
            len: 0,
        }
    }

    fn push(&mut self, a: usize, b: usize) {
        self.qubits[a].push(b);
        self.qubits[b].push(a);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for other in &self.qubits[a] {
            if *other == b { continue; }
            total += (topology.distance(b, *other) as f64) - (topology.distance(a, *other) as f64);
        }
        for other in &self.qubits[b] {
            if *other == a { continue; }
            total += (topology.distance(a, *other) as f64) - (topology.distance(b, *other) as f64);
        }
        total
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new(ctx.topology().num_qubits());
    if max_size == 0 { return out; }

    let precomputed = ctx.precomputed_extended_set_logical_pairs();
    if !precomputed.is_empty() {
        for pair in precomputed.iter().take(max_size) {
            out.push(ctx.layout().physical_of_logical(pair[0]), ctx.layout().physical_of_logical(pair[1]));
        }
        return out;
    }

    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = ctx.front_layer().node_ids().to_vec();
    let mut decremented = Vec::<(usize, usize)>::new();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let node_id = to_visit[i];
        for &successor in ctx.circuit().node(node_id).successors() {
            if let Some((_, amount)) = decremented.iter_mut().find(|(idx, _)| *idx == successor) {
                *amount += 1;
            } else {
                decremented.push((successor, 1));
            }
            required_predecessors[successor] -= 1;
            if required_predecessors[successor] == 0 {
                if let Some((a, b)) = ctx.circuit().node(successor).two_qubit_pair() {
                    out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b));
                }
                to_visit.push(successor);
            }
        }
        i += 1;
    }
    out
}

fn enumerate_candidate_swaps(topology: TopologyView<'_>, front_layer: &FrontLayerScores) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();
    for &phys in front_layer.iter_active() {
        for &neighbor in topology.neighbors(phys) {
            if neighbor > phys || !front_layer.is_active(neighbor) {
                out.push((phys, neighbor));
            }
        }
    }
    out
}

fn choose_dense_layout_subset(topology: TopologyView<'_>, logical_component_size: usize, target_component: &[usize]) -> Result<Vec<usize>, RouterError> {
    if logical_component_size > target_component.len() {
        return Err(RouterError::Routing(format!(
            "logical component size {logical_component_size} exceeds target component size {}",
            target_component.len()
        )));
    }
    if logical_component_size == target_component.len() {
        return Ok(target_component.to_vec());
    }

    let local_index = target_component.iter().enumerate().map(|(local, global)| (*global, local)).collect::<std::collections::HashMap<usize, usize>>();
    let mut local_adj = Array2::<f64>::zeros((target_component.len(), target_component.len()));
    for &global_a in target_component {
        let a = local_index[&global_a];
        for &global_b in topology.neighbors(global_a) {
            if let Some(&b) = local_index.get(&global_b) {
                local_adj[[a, b]] = 1.0;
            }
        }
    }
    let error_matrix = Array2::<f64>::zeros((target_component.len(), target_component.len()));
    let [_, _, best_map] = dense_layout::best_subset(
        logical_component_size,
        local_adj.view(),
        0,
        0,
        false,
        true,
        error_matrix.view(),
    );
    let chosen = best_map.into_iter().take(logical_component_size).map(|local| target_component[local]).collect::<Vec<_>>();
    ensure_connected_subset(topology, &chosen)?;
    Ok(chosen)
}

fn ensure_connected_subset(topology: TopologyView<'_>, subset: &[usize]) -> Result<(), RouterError> {
    if subset.is_empty() { return Ok(()); }
    let set = subset.iter().copied().collect::<std::collections::HashSet<_>>();
    let mut seen = std::collections::HashSet::<usize>::new();
    let mut queue = VecDeque::<usize>::new();
    queue.push_back(subset[0]);
    seen.insert(subset[0]);
    while let Some(node) = queue.pop_front() {
        for &next in topology.neighbors(node) {
            if set.contains(&next) && seen.insert(next) {
                queue.push_back(next);
            }
        }
    }
    if seen.len() != set.len() {
        return Err(RouterError::Routing("selected layout subset is not connected".to_string()));
    }
    Ok(())
}

fn assign_components_to_target(logical_components: &[Vec<usize>], target_components: &[Vec<usize>]) -> Result<Vec<(Vec<usize>, usize)>, RouterError> {
    if logical_components.is_empty() { return Ok(Vec::new()); }
    let mut logical_sorted = logical_components.to_vec();
    logical_sorted.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let mut caps: Vec<(usize, usize)> = target_components.iter().enumerate().map(|(i, c)| (i, c.len())).collect();
    let mut assignments = Vec::new();
    for logical in logical_sorted {
        let size = logical.len();
        let mut best_pos = None;
        let mut best_waste = usize::MAX;
        for pos in 0..caps.len() {
            let cap = caps[pos].1;
            if cap >= size {
                let waste = cap - size;
                if waste < best_waste {
                    best_waste = waste;
                    best_pos = Some(pos);
                }
            }
        }
        let pos = best_pos.ok_or_else(|| RouterError::Routing(format!("logical component of size {size} cannot fit any target component")))?;
        let target_idx = caps[pos].0;
        caps[pos].1 -= size;
        assignments.push((logical, target_idx));
    }
    Ok(assignments)
}

fn mapping_heuristic_score(ctx: &InitialLayoutContext<'_>, mapping: &[usize]) -> usize {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let mut sum = 0usize;
    let mut visited = std::collections::HashSet::new();
    let mut queue = VecDeque::new();
    for &node_id in circuit.first_layer_node_ids() {
        if visited.insert(node_id) {
            queue.push_back((node_id, 0usize));
        }
    }
    while let Some((node_id, d)) = queue.pop_front() {
        if circuit.node(node_id).is_two_qubit() {
            if let Some((a, b)) = circuit.node(node_id).two_qubit_pair() {
                if a < mapping.len() && b < mapping.len() {
                    let pa = mapping[a];
                    let pb = mapping[b];
                    sum += topology.distance(pa, pb) as usize;
                }
            }
        }
        if d >= 8 {
            continue;
        }
        for &succ in circuit.node(node_id).successors() {
            if visited.insert(succ) {
                queue.push_back((succ, d + 1));
            }
        }
    }
    sum
}

fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let logical_components = circuit.logical_interaction_components().to_vec();

    let mut logical_adj = vec![Vec::new(); num_logical];
    for node_id in circuit.node_ids() {
        let node = circuit.node(node_id);
        if node.is_two_qubit() {
            if let Some((a, b)) = node.two_qubit_pair() {
                if a < num_logical && b < num_logical {
                    logical_adj[a].push(b);
                    logical_adj[b].push(a);
                }
            }
        }
    }

    const ATTEMPTS: usize = 128;
    let mut best_mapping: Option<Vec<usize>> = None;
    let mut best_score = usize::MAX;

    for _ in 0..ATTEMPTS {
        let mut target_components = topology.connected_components().to_vec();
        rng.shuffle_slice(&mut target_components);
        let assignments = match assign_components_to_target(&logical_components, &target_components) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let mut mapping = vec![usize::MAX; num_logical];
        let mut used_physical = vec![false; topology.num_qubits()];
        let mut ok = true;
        for (logical_component, target_component_idx) in assignments {
            if logical_component.is_empty() { continue; }
            let target_component = &target_components[target_component_idx];
            let local = match choose_dense_layout_subset(topology, logical_component.len(), target_component) {
                Ok(v) => v,
                Err(_) => { ok = false; break; }
            };
            for (logical, physical) in logical_component.iter().zip(local) {
                mapping[*logical] = physical;
                used_physical[physical] = true;
            }
        }
        if !ok { continue; }

        let mut unmapped_logicals: Vec<usize> = mapping.iter().enumerate()
            .filter_map(|(i, &p)| if p == usize::MAX { Some(i) } else { None })
            .collect();
        if !unmapped_logicals.is_empty() {
            unmapped_logicals.sort_by_key(|&l| std::cmp::Reverse(logical_adj[l].len()));
            let mut free_physicals: Vec<usize> = (0..topology.num_qubits()).filter(|q| !used_physical[*q]).collect();
            for &logical in &unmapped_logicals {
                if free_physicals.is_empty() { ok = false; break; }
                let mut best_phys = None;
                let mut best_score_local = usize::MAX;
                let mut best_idx = 0usize;
                for (idx, &phys) in free_physicals.iter().enumerate() {
                    let mut score = 0usize;
                    let mut mapped_neighbors = 0usize;
                    for &nbr in &logical_adj[logical] {
                        let nbr_phys = mapping[nbr];
                        if nbr_phys != usize::MAX {
                            score += topology.distance(phys, nbr_phys) as usize;
                            mapped_neighbors += 1;
                        }
                    }
                    if mapped_neighbors == 0 {
                        score = topology.max_degree().saturating_sub(topology.degree(phys));
                    }
                    if score < best_score_local {
                        best_score_local = score;
                        best_phys = Some(phys);
                        best_idx = idx;
                    }
                }
                if let Some(phys) = best_phys {
                    mapping[logical] = phys;
                    used_physical[phys] = true;
                    free_physicals.swap_remove(best_idx);
                }
            }
            if !ok { continue; }
        }

        let score = mapping_heuristic_score(ctx, &mapping);
        if score < best_score {
            best_score = score;
            best_mapping = Some(mapping);
        }
    }
    best_mapping.ok_or_else(|| RouterError::Routing("failed to find valid initial layout".to_string()))
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    pub basic_weight: f64,
    pub lookahead_weight: f64,
    pub set_scaling: SetScaling,
    pub use_decay: bool,
    decay_state: Vec<f64>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            basic_weight: 9.0,
            lookahead_weight: 3.0,
            set_scaling: SetScaling::Constant,
            use_decay: false,
            decay_state: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    fn refresh_decay_state(&mut self, _ctx: &SwapSelectionContext<'_>) {}
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(&mut self, ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        choose_disjoint_aware_layout(ctx, rng)
    }

    fn choose_best_swap(&mut self, ctx: &SwapSelectionContext<'_>, rng: &mut RngState) -> Option<(usize, usize)> {
        self.refresh_decay_state(ctx);
        let topology = ctx.topology();
        let front_layer = FrontLayerScores::from_ctx(ctx);
        if front_layer.len() == 0 { return None; }

        let mut candidates = enumerate_candidate_swaps(topology, &front_layer);
        if candidates.is_empty() { return None; }

        if let Some((a,b)) = ctx.last_applied_swap() {
            candidates.retain(|&(x,y)| !(x==a && y==b) && !(x==b && y==a));
            if candidates.is_empty() { return None; }
        }

        let swaps_sp = ctx.swaps_since_progress() as f64;
        let basic_mult = 1.0 + swaps_sp * 0.01;
        let lookahead_mult = 1.0 / (1.0 + swaps_sp * 0.01);

        let num_logical = ctx.circuit().num_logical_qubits();
        let dynamic_lookahead_size = num_logical.min(120).max(20);
        let extended_set = build_extended_set(ctx, dynamic_lookahead_size);

        let scale = |weight: f64, size: usize, scaling: SetScaling| -> f64 {
            match scaling {
                SetScaling::Constant => weight,
                SetScaling::Size => if size == 0 { 0.0 } else { weight / (size as f64) },
            }
        };

        let basic_weight = scale(self.basic_weight * basic_mult, front_layer.len(), self.set_scaling);
        let lookahead_weight = scale(self.lookahead_weight * lookahead_mult, extended_set.len(), self.set_scaling);

        let mut improving_candidates = Vec::new();
        for s in &candidates {
            if front_layer.score_delta(*s, topology) < 0.0 {
                improving_candidates.push(*s);
            }
        }
        let eval_candidates = if !improving_candidates.is_empty() { &improving_candidates } else { &candidates };

        let mut best_score = f64::INFINITY;
        let mut best_swaps = Vec::new();
        for swap in eval_candidates {
            let delta_front = front_layer.score_delta(*swap, topology);
            let delta_ext = extended_set.score_delta(*swap, topology);
            let mut score = basic_weight * delta_front + lookahead_weight * delta_ext;

            let mut adj_bonus = 0.0;
            for &node_id in ctx.front_layer().node_ids() {
                if let (Some((la, lb)), Some((pa, pb))) = (ctx.front_layer().logical_pair(node_id), ctx.front_layer().physical_pair(node_id)) {
                    let mut pa2 = pa;
                    let mut pb2 = pb;
                    let (a, b) = *swap;
                    if pa2 == a { pa2 = b; } else if pa2 == b { pa2 = a; }
                    if pb2 == a { pb2 = b; } else if pb2 == b { pb2 = a; }
                    if topology.distance(pa2, pb2) == 1 {
                        adj_bonus += 1.0;
                    }
                }
            }
            score -= adj_bonus * 2000.0;
            score += rng.gen_f64() * 1e-12;
            if score + 1e-12 < best_score {
                best_score = score;
                best_swaps.clear();
                best_swaps.push(*swap);
            } else if (score - best_score).abs() <= 1e-12 {
                best_swaps.push(*swap);
            }
        }
        if best_swaps.is_empty() { None } else { Some(best_swaps[rng.gen_index(best_swaps.len())]) }
    }
}
'''
