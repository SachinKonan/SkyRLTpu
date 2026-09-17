RUST_CODE = r'''
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

#[derive(Debug, Clone, Default)]
struct FrontLayerScores {
    pairs: Vec<[usize; 2]>,
    qubits: Vec<Option<(usize, usize)>>,
}

impl FrontLayerScores {
    fn from_ctx(ctx: &SwapSelectionContext<'_>) -> Self {
        let mut out = Self {
            pairs: Vec::new(),
            qubits: vec![None; ctx.topology().num_qubits()],
        };
        for pair in ctx.front_layer().physical_pairs() {
            let [a, b] = *pair;
            let index = out.pairs.len();
            out.pairs.push([a, b]);
            out.qubits[a] = Some((index, b));
            out.qubits[b] = Some((index, a));
        }
        out
    }

    fn len(&self) -> usize {
        self.pairs.len()
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.pairs
            .iter()
            .map(|pair| {
                let d = topology.distance(pair[0], pair[1]) as f64;
                d * d
            })
            .sum()
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut delta = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            let d_old = topology.distance(a, c) as f64;
            let d_new = topology.distance(b, c) as f64;
            delta += d_new * d_new - d_old * d_old;
        }
        if let Some((_, c)) = self.qubits[b] {
            let d_old = topology.distance(b, c) as f64;
            let d_new = topology.distance(a, c) as f64;
            delta += d_new * d_new - d_old * d_old;
        }
        delta
    }
}

#[derive(Debug, Clone)]
struct ScorePair {
    a: usize,
    b: usize,
    weight: f64,
}

#[derive(Debug, Clone)]
struct ExtendedSetScores {
    pairs: Vec<ScorePair>,
}

impl ExtendedSetScores {
    fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    fn len(&self) -> usize {
        self.pairs.len()
    }

    fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.pairs.iter().map(|p| {
            let d = topology.distance(p.a, p.b) as f64;
            p.weight * d * d
        }).sum()
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (s_a, s_b) = swap;
        let mut delta = 0.0;
        for p in &self.pairs {
            if p.a == s_a || p.a == s_b || p.b == s_a || p.b == s_b {
                let d_old = topology.distance(p.a, p.b) as f64;
                let mut a = p.a;
                let mut b = p.b;
                if a == s_a { a = s_b; } else if a == s_b { a = s_a; }
                if b == s_a { b = s_b; } else if b == s_b { b = s_a; }
                let d_new = topology.distance(a, b) as f64;
                delta += p.weight * (d_new * d_new - d_old * d_old);
            }
        }
        delta
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut pairs = Vec::new();
    if max_size == 0 { return ExtendedSetScores { pairs }; }

    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut queue = VecDeque::new();
    
    for &node_id in ctx.front_layer().node_ids() {
        queue.push_back((node_id, 0));
    }

    let mut visited = HashSet::new();
    while let Some((node_id, layer)) = queue.pop_front() {
        if pairs.len() >= max_size { break; }
        if !visited.insert(node_id) { continue; }

        for &successor in ctx.circuit().node(node_id).successors() {
            required_predecessors[successor] -= 1;
            if required_predecessors[successor] == 0 {
                if let Some((a, b)) = ctx.circuit().node(successor).two_qubit_pair() {
                    let weight = 0.5f64.powi(layer as i32 + 1);
                    pairs.push(ScorePair {
                        a: ctx.layout().physical_of_logical(a),
                        b: ctx.layout().physical_of_logical(b),
                        weight,
                    });
                }
                queue.push_back((successor, layer + 1));
            }
        }
    }
    ExtendedSetScores { pairs }
}

fn enumerate_candidate_swaps(
    topology: TopologyView<'_>,
    front_layer: &FrontLayerScores,
) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();
    let mut seen = HashSet::<(usize, usize)>::new();

    for pair in &front_layer.pairs {
        for &phys in pair {
            for &neighbor in topology.neighbors(phys) {
                let edge = if phys < neighbor { (phys, neighbor) } else { (neighbor, phys) };
                if seen.insert(edge) {
                    out.push(edge);
                }
            }
        }
    }
    out
}

fn choose_dense_layout_subset(
    topology: TopologyView<'_>,
    logical_component_size: usize,
    target_component: &[usize],
) -> Result<Vec<usize>, RouterError> {
    if logical_component_size > target_component.len() {
        return Err(RouterError::Routing(format!("size {logical_component_size} > target {}", target_component.len())));
    }
    if logical_component_size == target_component.len() {
        return Ok(target_component.to_vec());
    }

    let local_index = target_component.iter().enumerate().map(|(local, global)| (*global, local)).collect::<HashMap<usize, usize>>();
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
    let set = subset.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::<usize>::new();
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
        return Err(RouterError::Routing("disconnected layout subset".to_string()));
    }
    Ok(())
}

fn assign_components_to_target(
    logical_components: &[Vec<usize>],
    target_components: &[Vec<usize>],
) -> Result<Vec<(Vec<usize>, usize)>, RouterError> {
    if logical_components.is_empty() { return Ok(Vec::new()); }
    let mut logical_sorted = logical_components.to_vec();
    logical_sorted.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let mut target_sorted = target_components.iter().enumerate().map(|(i, c)| (i, c.len())).collect::<Vec<_>>();
    target_sorted.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
    let mut free_capacity = target_sorted.iter().map(|(i, s)| (*i, *s)).collect::<HashMap<usize, usize>>();
    let mut assignments = Vec::new();
    for logical in logical_sorted {
        let size = logical.len();
        let mut chosen = None;
        for (target_idx, _) in &target_sorted {
            if *free_capacity.get(target_idx).unwrap_or(&0) >= size {
                chosen = Some(*target_idx);
                break;
            }
        }
        let target_idx = chosen.ok_or_else(|| RouterError::Routing("insufficient capacity".to_string()))?;
        *free_capacity.get_mut(&target_idx).unwrap() -= size;
        assignments.push((logical, target_idx));
    }
    Ok(assignments)
}

fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let used = circuit.used_logical_qubits();
    if used.is_empty() { return Ok((0..num_logical).collect()); }
    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    let assignments = assign_components_to_target(logical_components, target_components)?;
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];
    for (logical_component, target_component_idx) in assignments {
        let target_component = &target_components[target_component_idx];
        let local = choose_dense_layout_subset(topology, logical_component.len(), target_component)?;
        for (logical, physical) in logical_component.iter().zip(local) {
            mapping[*logical] = physical;
            used_physical[physical] = true;
        }
    }
    let mut free_physical = (0..topology.num_qubits()).filter(|q| !used_physical[*q]);
    for slot in &mut mapping {
        if *slot == usize::MAX {
            *slot = free_physical.next().ok_or_else(|| RouterError::Routing("out of qubits".to_string()))?;
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
            basic_weight: 1.0,
            lookahead_weight: 0.5,
            lookahead_size: 30,
            set_scaling: SetScaling::Size,
            use_decay: true,
            decay_increment: 0.002,
            decay_reset: 4,
            best_epsilon: 1e-9,
            decay_state: Vec::new(),
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(&mut self, ctx: &InitialLayoutContext<'_>, _rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        choose_disjoint_aware_layout(ctx)
    }

    fn choose_best_swap(&mut self, ctx: &SwapSelectionContext<'_>, rng: &mut RngState) -> Option<(usize, usize)> {
        if self.use_decay {
            let num_qubits = ctx.topology().num_qubits();
            if self.decay_state.len() != num_qubits { self.decay_state = vec![1.0; num_qubits]; }
            if ctx.swaps_since_progress() == 0 {
                self.decay_state.fill(1.0);
            } else if let Some((a, b)) = ctx.last_applied_swap() {
                if ctx.swaps_since_progress() % self.decay_reset.max(1) == 0 {
                    self.decay_state.fill(1.0);
                } else {
                    self.decay_state[a] += self.decay_increment;
                    self.decay_state[b] += self.decay_increment;
                }
            }
        }

        let front_layer = FrontLayerScores::from_ctx(ctx);
        let candidates = enumerate_candidate_swaps(ctx.topology(), &front_layer);
        if candidates.is_empty() { return None; }
        let extended_set = build_extended_set(ctx, self.lookahead_size);

        let scale = |weight: f64, size: usize, scaling: SetScaling| -> f64 {
            match scaling {
                SetScaling::Constant => weight,
                SetScaling::Size => if size == 0 { 0.0 } else { weight / (size as f64) }
            }
        };

        let basic_w = scale(self.basic_weight, front_layer.len(), self.set_scaling);
        let look_w = scale(self.lookahead_weight, extended_set.len(), self.set_scaling);

        let abs_score = basic_w * front_layer.total_score(ctx.topology()) + look_w * extended_set.total_score(ctx.topology());

        let mut best_swaps = Vec::new();
        let mut min_score = f64::INFINITY;

        for swap in candidates {
            let delta = basic_w * front_layer.score_delta(swap, ctx.topology()) + look_w * extended_set.score_delta(swap, ctx.topology());
            let mut score = abs_score + delta;
            if self.use_decay {
                score *= self.decay_state[swap.0].max(self.decay_state[swap.1]);
            }
            if score + self.best_epsilon < min_score {
                min_score = score;
                best_swaps.clear();
                best_swaps.push(swap);
            } else if (score - min_score).abs() <= self.best_epsilon {
                best_swaps.push(swap);
            }
        }

        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
'''
