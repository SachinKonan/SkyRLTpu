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

    fn is_empty(self) -> bool {
        self.nodes.is_empty()
    }

    fn is_active(&self, qubit: usize) -> bool {
        self.qubits[qubit].is_some()
    }

    fn iter_active(&self) -> impl Iterator<Item = &usize> {
        self.nodes.iter().flatten()
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.nodes
            .iter()
            .map(|pair| topology.distance(pair[0], pair[1]) as f64)
            .sum()
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

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.qubits
            .iter()
            .enumerate()
            .flat_map(|(a, others)| {
                others
                    .iter()
                    .map(move |b| topology.distance(a, *b) as f64)
            })
            .sum::<f64>()
            * 0.5
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for other in &self.qubits[a] {
            if *other == b {
                continue;
            }
            total += (topology.distance(b, *other) as f64) - (topology.distance(a, *other) as f64);
        }
        for other in &self.qubits[b] {
            if *other == a {
                continue;
            }
            total += (topology.distance(a, *other) as f64) - (topology.distance(b, *other) as f64);
        }
        total
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new(ctx.topology().num_qubits());
    if max_size == 0 {
        return out;
    }

    let precomputed = ctx.precomputed_extended_set_logical_pairs();
    if !precomputed.is_empty() {
        for pair in precomputed.iter().take(max_size) {
            out.push(
                ctx.layout().physical_of_logical(pair[0]),
                ctx.layout().physical_of_logical(pair[1]),
            );
        }
        return out;
    }

    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = Vec::new();
    for &node_id in ctx.front_layer().node_ids() {
        to_visit.push(node_id);
    }
    let mut decremented = Vec::<(usize, usize)>::new();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let node_id = to_visit[i];
        for &successor in ctx.circuit().node(node_id).successors() {
            let mut found = false;
            for (idx, amt) in &mut decremented {
                if *idx == successor {
                    *amt += 1;
                    found = true;
                    break;
                }
            }
            if !found {
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

fn enumerate_candidate_swaps(
    topology: TopologyView<'_>,
    front_layer: &FrontLayerScores,
) -> Vec<(usize, usize)> {
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
            lookahead_size: 20,
            set_scaling: SetScaling::Size,
            use_decay: true,
            decay_increment: 0.001,
            decay_reset: 5,
            best_epsilon: 1e-10,
            decay_state: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    fn refresh_decay_state(&mut self, ctx: &SwapSelectionContext<'_>) {
        if !self.use_decay {
            return;
        }

        let num_qubits = ctx.topology().num_qubits();
        if self.decay_state.len() != num_qubits {
            self.decay_state = vec![1.0; num_qubits];
        }

        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(1.0);
            return;
        }

        let Some((a, b)) = ctx.last_applied_swap() else {
            return;
        };
        let reset = self.decay_reset.max(1);
        if ctx.swaps_since_progress() % reset == 0 {
            self.decay_state.fill(1.0);
        } else {
            self.decay_state[a] += self.decay_increment;
            self.decay_state[b] += self.decay_increment;
        }
    }

    fn greedy_layout(&self, ctx: &InitialLayoutContext<'_>) -> Vec<usize> {
        let topology = ctx.topology();
        let circuit = ctx.circuit();
        let num_logical = circuit.num_logical_qubits();
        let mut mapping = vec![usize::MAX; num_logical];
        let mut used_physical = vec![false; topology.num_qubits()];
        
        let mut logical_degrees = vec![0; num_logical];
        for node_id in circuit.node_ids() {
            let node_view = circuit.node(node_id);
            if node_view.is_two_qubit() {
                if let Some((a, b)) = node_view.two_qubit_pair() {
                    logical_degrees[a] += 1;
                    logical_degrees[b] += 1;
                }
            }
        }
        
        let mut sorted_logical: Vec<usize> = (0..num_logical).collect();
        sorted_logical.sort_by_key(|&l| std::cmp::Reverse(logical_degrees[l]));
        
        let mut sorted_physical: Vec<(usize, usize)> = (0..topology.num_qubits())
            .map(|p| (topology.degree(p), p))
            .collect();
        sorted_physical.sort_by_key(|(d, _)| std::cmp::Reverse(*d));
        
        let mut physical_idx = 0;
        for logical in &sorted_logical {
            while physical_idx < sorted_physical.len() && used_physical[sorted_physical[physical_idx].1] {
                physical_idx += 1;
            }
            if physical_idx < sorted_physical.len() {
                mapping[*logical] = sorted_physical[physical_idx].1;
                used_physical[mapping[*logical]] = true;
            }
        }
        
        for slot in &mut mapping {
            if *slot == usize::MAX {
                for p in 0..topology.num_qubits() {
                    if !used_physical[p] {
                        *slot = p;
                        used_physical[p] = true;
                        break;
                    }
                }
            }
        }
        
        mapping
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        
        let numerator = ctx.circuit().node_count();
        let mapping = self.greedy_layout(ctx);
        
        let num_logical = ctx.circuit().num_logical_qubits();
        let num_physical = ctx.topology().num_qubits();
        let used_physical: std::collections::HashSet<usize> = mapping.iter().copied().collect();
        if used_physical.len() != num_logical {
            return Err(RouterError::Routing(
                "layout is not injective".to_string(),
            ));
        }
        if used_physical.iter().any(|&p| p >= num_physical) {
            return Err(RouterError::Routing(
                "logical to physical mapping out of bounds".to_string(),
            ));
        }
        
        Ok(mapping)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        self.refresh_decay_state(ctx);

        let front_layer = FrontLayerScores::from_ctx(ctx);
        let candidates = enumerate_candidate_swaps(ctx.topology(), &front_layer);
        if candidates.is_empty() {
            return None;
        }

        let extended_set = build_extended_set(ctx, self.lookahead_size);

        let scale = |weight: f64, size: usize, scaling: SetScaling| -> f64 {
            match scaling {
                SetScaling::Constant => weight,
                SetScaling::Size => {
                    if size == 0 {
                        0.0
                    } else {
                        weight / (size as f64)
                    }
                }
            }
        };

        let basic_weight = scale(self.basic_weight, front_layer.len(), self.set_scaling);
        let lookahead_weight = scale(
            self.lookahead_weight,
            extended_set.len(),
            self.set_scaling,
        );

        let mut swap_scores = candidates
            .iter()
            .copied()
            .map(|swap| (swap, 0.0))
            .collect::<Vec<_>>();

        let mut absolute_score = 0.0;
        absolute_score += basic_weight * front_layer.total_score(ctx.topology());
        for (swap, score) in &mut swap_scores {
            *score += basic_weight * front_layer.score_delta(*swap, ctx.topology());
        }

        if !extended_set.is_empty() && self.lookahead_weight != 0.0 {
            absolute_score += lookahead_weight * extended_set.total_score(ctx.topology());
            for (swap, score) in &mut swap_scores {
                *score += lookahead_weight * extended_set.score_delta(*swap, ctx.topology());
            }
        }

        if self.use_decay {
            for (swap, score) in &mut swap_scores {
                *score =
                    (absolute_score + *score) * (self.decay_state[swap.0].min(self.decay_state[swap.1]).clamp(0.1, 2.0));
            }
        } else {
            for (_, score) in &mut swap_scores {
                *score = absolute_score + *score;
            }
        }

        let mut min_score = f64::INFINITY;
        for (_, score) in &swap_scores {
            if *score < min_score {
                min_score = *score;
            }
        }
        
        let mut best_swaps = Vec::<(usize, usize)>::new();
        for (swap, score) in &swap_scores {
            if (*score - min_score).abs() <= self.best_epsilon {
                best_swaps.push(*swap);
            }
        }
        
        if best_swaps.is_empty() {
            return None;
        }

        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
'''
