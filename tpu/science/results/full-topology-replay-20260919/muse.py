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
}

#[derive(Debug, Clone)]
struct ExtendedSetScores {
    qubits: Vec<Vec<usize>>,
    len: usize,
}

impl ExtendedSetScores {
    fn new(num_qubits: usize) -> Self {
        Self { qubits: vec![Vec::new(); num_qubits], len: 0 }
    }
    fn push(&mut self, a: usize, b: usize) {
        self.qubits[a].push(b);
        self.qubits[b].push(a);
        self.len += 1;
    }
    fn len(&self) -> usize { self.len }
    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for other in &self.qubits[a] {
            if *other == b { continue; }
            total += topology.distance(b, *other) as f64 - topology.distance(a, *other) as f64;
        }
        for other in &self.qubits[b] {
            if *other == a { continue; }
            total += topology.distance(a, *other) as f64 - topology.distance(b, *other) as f64;
        }
        total
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new(ctx.topology().num_qubits());
    if max_size == 0 { return out; }
    for &node_id in ctx.front_layer().node_ids() {
        if out.len() >= max_size { break; }
        if let Some((a, b)) = ctx.circuit().node(node_id).two_qubit_pair() {
            out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b));
        }
    }
    if out.len() < max_size {
        for &node_id in ctx.remaining().ready_node_ids() {
            if out.len() >= max_size { break; }
            if let Some((a, b)) = ctx.circuit().node(node_id).two_qubit_pair() {
                out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b));
            }
        }
    }
    if out.len() >= max_size { return out; }
    let precomputed = ctx.precomputed_extended_set_logical_pairs();
    if !precomputed.is_empty() {
        for pair in precomputed.iter().take(max_size.saturating_sub(out.len())) {
            out.push(ctx.layout().physical_of_logical(pair[0]), ctx.layout().physical_of_logical(pair[1]));
        }
        return out;
    }
    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = ctx.front_layer().node_ids().to_vec();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let node_id = to_visit[i];
        for &successor in ctx.circuit().node(node_id).successors() {
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

fn collect_candidate_swaps(topology: TopologyView<'_>, front_scores: &FrontLayerScores, extended_set: &ExtendedSetScores) -> Vec<(usize, usize)> {
    let n = topology.num_qubits();
    let mut involved = vec![false; n];
    for &pair in &front_scores.nodes {
        involved[pair[0]] = true;
        involved[pair[1]] = true;
    }
    for i in 0..n {
        if !extended_set.qubits[i].is_empty() { involved[i] = true; }
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for i in 0..n {
        if !involved[i] { continue; }
        for &nbr in topology.neighbors(i) {
            let a = i.min(nbr);
            let b = i.max(nbr);
            if seen.insert((a, b)) {
                out.push((a, b));
            }
        }
    }
    out
}

fn choose_dense_layout_subset(topology: TopologyView<'_>, logical_component_size: usize, target_component: &[usize]) -> Result<Vec<usize>, RouterError> {
    if logical_component_size > target_component.len() {
        return Err(RouterError::Routing(format!("logical component size {logical_component_size} exceeds target component size {}", target_component.len())));
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
    let [_, _, best_map] = dense_layout::best_subset(logical_component_size, local_adj.view(), 0, 0, false, true, error_matrix.view());
    let chosen = best_map.into_iter().take(logical_component_size).map(|local| target_component[local]).collect::<Vec<_>>();
    ensure_connected_subset(topology, &chosen)?;
    Ok(chosen)
}

fn ensure_connected_subset(topology: TopologyView<'_>, subset: &[usize]) -> Result<(), RouterError> {
    if subset.is_empty() { return Ok(()); }
    let set = subset.iter().copied().collect::<std::collections::HashSet<_>>();
    let mut seen = std::collections::HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(subset[0]);
    seen.insert(subset[0]);
    while let Some(node) = queue.pop_front() {
        for &next in topology.neighbors(node) {
            if set.contains(&next) && seen.insert(next) { queue.push_back(next); }
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
    let mut target_sorted = target_components.iter().enumerate().map(|(i,c)| (i, c.len())).collect::<Vec<_>>();
    target_sorted.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
    let mut free_capacity = target_sorted.iter().map(|(i,s)| (*i,*s)).collect::<std::collections::HashMap<usize, usize>>();
    let mut assignments = Vec::new();
    for logical in logical_sorted {
        let size = logical.len();
        let mut chosen = None;
        for (idx, _) in &target_sorted {
            if free_capacity.get(idx).copied().unwrap_or(0) >= size { chosen = Some(*idx); break; }
        }
        let target_idx = chosen.ok_or_else(|| RouterError::Routing(format!("logical component of size {size} cannot fit any target component")))?;
        *free_capacity.get_mut(&target_idx).unwrap() -= size;
        assignments.push((logical, target_idx));
    }
    Ok(assignments)
}

fn evaluate_initial_layout(ctx: &InitialLayoutContext<'_>, mapping: &[usize]) -> u64 {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let mut score = 0u64;
    for &node_id in circuit.first_layer_node_ids() {
        if let Some((a,b)) = circuit.node(node_id).two_qubit_pair() {
            let pa = mapping[a];
            let pb = mapping[b];
            score += topology.distance(pa, pb) as u64;
        }
    }
    score
}

fn choose_disjoint_aware_layout_randomized(ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    if target_components.is_empty() { return Err(RouterError::Routing("topology has no connected components".to_string())); }
    let assignments = assign_components_to_target(logical_components, target_components)?;
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];
    for (logical_component, target_component_idx) in assignments {
        if logical_component.is_empty() { continue; }
        let target_component = &target_components[target_component_idx];
        let local = choose_dense_layout_subset(topology, logical_component.len(), target_component)?;
        let mut logicals = logical_component.clone();
        rng.shuffle_slice(&mut logicals);
        for (&logical, &physical) in logicals.iter().zip(local.iter()) {
            mapping[logical] = physical;
            used_physical[physical] = true;
        }
    }
    let mut free_physical: Vec<usize> = (0..topology.num_qubits()).filter(|q| !used_physical[*q]).collect();
    rng.shuffle_slice(&mut free_physical);
    let mut free_iter = free_physical.into_iter();
    for slot in mapping.iter_mut() {
        if *slot == usize::MAX {
            *slot = free_iter.next().ok_or_else(|| RouterError::Routing("not enough physical qubits to complete layout".to_string()))?;
        }
    }
    Ok(mapping)
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    pub basic_weight: f64,
    pub lookahead_weight: f64,
    pub lookahead_size: usize,
    pub set_scaling: SetScaling,
    pub use_decay: bool,
    pub decay_increment: f64,
    pub decay_reset: usize,
    pub best_epsilon: f64,
    decay_state: Vec<f64>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            basic_weight: 2.0,
            lookahead_weight: 4.0,
            lookahead_size: 256,
            set_scaling: SetScaling::Constant,
            use_decay: true,
            decay_increment: 0.03,
            decay_reset: 3,
            best_epsilon: 1e-9,
            decay_state: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    fn refresh_decay_state(&mut self, ctx: &SwapSelectionContext<'_>) {
        if !self.use_decay { return; }
        let n = ctx.topology().num_qubits();
        if self.decay_state.len() != n { self.decay_state = vec![1.0; n]; }
        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(1.0);
            return;
        }
        if let Some((a,b)) = ctx.last_applied_swap() {
            if ctx.swaps_since_progress() % self.decay_reset == 0 {
                self.decay_state.fill(1.0);
            } else {
                self.decay_state[a] += self.decay_increment;
                self.decay_state[b] += self.decay_increment;
            }
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(&mut self, ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        let attempts = 120;
        let mut best_map: Option<Vec<usize>> = None;
        let mut best_score = u64::MAX;
        for _ in 0..attempts {
            if let Ok(map) = choose_disjoint_aware_layout_randomized(ctx, rng) {
                let score = evaluate_initial_layout(ctx, &map);
                if score < best_score {
                    best_score = score;
                    best_map = Some(map);
                }
            }
        }
        if let Some(m) = best_map { Ok(m) } else { choose_disjoint_aware_layout_randomized(ctx, rng) }
    }

    fn choose_best_swap(&mut self, ctx: &SwapSelectionContext<'_>, rng: &mut RngState) -> Option<(usize, usize)> {
        self.refresh_decay_state(ctx);
        let topology = ctx.topology();
        let front_layer = ctx.front_layer();
        let front_scores = FrontLayerScores::from_ctx(ctx);
        let num_logical = ctx.circuit().num_logical_qubits();
        let effective_size = std::cmp::min(self.lookahead_size, num_logical.saturating_mul(2).max(30));
        let extended_set = build_extended_set(ctx, effective_size);
        let candidates = collect_candidate_swaps(topology, &front_scores, &extended_set);
        if candidates.is_empty() { return None; }
        let num_front = front_layer.len();
        let phys_pairs = front_layer.physical_pairs();

        struct Stat {
            swap: (usize, usize),
            front_gain: i64,
            dist_sum: u64,
            adj_inc: u32,
        }
        let mut stats = Vec::with_capacity(candidates.len());
        for &swap in &candidates {
            let (a,b) = swap;
            if let Some((la,lb)) = ctx.last_applied_swap() { if (a == lb && b == la) { continue; } }
            let mut front_gain: i64 = 0;
            let mut dist_sum = 0u64;
            let mut adj_inc = 0u32;
            for i in 0..num_front {
                let [p0,p1] = phys_pairs[i];
                let mut np0 = p0; let mut np1 = p1;
                if np0 == a { np0 = b; } else if np0 == b { np0 = a; }
                if np1 == a { np1 = b; } else if np1 == b { np1 = a; }
                let cur_dist = topology.distance(p0, p1) as i64;
                let new_dist = topology.distance(np0, np1) as i64;
                front_gain += cur_dist - new_dist;
                dist_sum += new_dist as u64;
                if cur_dist > 1 && new_dist == 1 { adj_inc += 1; }
            }
            stats.push(Stat { swap, front_gain, dist_sum, adj_inc });
        }
        if stats.is_empty() { return None; }

        let max_front_gain = stats.iter().map(|s| s.front_gain).max().unwrap();
        let mut filtered: Vec<Stat> = stats.into_iter().filter(|s| s.front_gain == max_front_gain).collect();
        if filtered.is_empty() { return None; }

        let max_adj = filtered.iter().map(|s| s.adj_inc).max().unwrap();
        filtered.retain(|s| s.adj_inc == max_adj);

        let min_dist = filtered.iter().map(|s| s.dist_sum).min().unwrap();
        filtered.retain(|s| s.dist_sum == min_dist);

        let mut best_score = f64::INFINITY;
        let mut final_candidates = Vec::new();
        for s in filtered {
            let mut delta = extended_set.score_delta(s.swap, topology);
            if self.set_scaling == SetScaling::Size && extended_set.len() > 0 {
                delta /= (extended_set.len() as f64 + 1.0);
            }
            let decay_penalty = if self.use_decay { self.decay_state[s.swap.0] + self.decay_state[s.swap.1] } else { 0.0 };
            let score = delta + decay_penalty * 0.1;
            if score < best_score - 1e-12 {
                best_score = score;
                final_candidates.clear();
                final_candidates.push(s.swap);
            } else if (score - best_score).abs() <= 1e-12 {
                final_candidates.push(s.swap);
            }
        }
        if final_candidates.is_empty() { return None; }
        if final_candidates.len() == 1 { return Some(final_candidates[0]); }
        rng.shuffle_slice(&mut final_candidates);
        Some(final_candidates[0])
    }
}
'''
