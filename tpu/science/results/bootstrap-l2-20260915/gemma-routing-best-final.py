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

    fn is_empty(&self) -> bool {
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
        if let Some((_, other)) = self.qubits[a] {
            delta += (topology.distance(b, other) as f64) - (topology.distance(a, other) as f64);
        }
        if let Some((_, other)) = self.qubits[b] {
            delta += (topology.distance(a, other) as f64) - (topology.distance(b, other) as f64);
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
        let mut sum = 0.0;
        for (a, others) in self.qubits.iter().enumerate() {
            for &b in others {
                if a < b {
                    sum += topology.distance(a, b) as f64;
                }
            }
        }
        sum
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for &other in &self.qubits[a] {
            if other == b { continue; }
            total += (topology.distance(b, other) as f64) - (topology.distance(a, other) as f64);
        }
        for &other in &self.qubits[b] {
            if other == a { continue; }
            total += (topology.distance(a, other) as f64) - (topology.distance(b, other) as f64);
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
            out.push(
                ctx.layout().physical_of_logical(pair[0]),
                ctx.layout().physical_of_logical(pair[1]),
            );
        }
        return out;
    }

    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = ctx.front_layer().node_ids().to_vec();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let node_id = to_visit[i];
        for &successor in ctx.circuit().node(node_id).successors() {
            if required_predecessors[successor] > 0 {
                required_predecessors[successor] -= 1;
                if required_predecessors[successor] == 0 {
                    if let Some((a, b)) = ctx.circuit().node(successor).two_qubit_pair() {
                        out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b));
                    }
                    to_visit.push(successor);
                }
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

fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];

    let mut log_comps = logical_components.to_vec();
    log_comps.sort_by_key(|c| std::cmp::Reverse(c.len()));
    
    let mut phys_comps = target_components.to_vec();
    phys_comps.sort_by_key(|c| std::cmp::Reverse(c.len()));

    for log_comp in log_comps {
        if log_comp.is_empty() { continue; }
        
        let mut found = false;
        for phys_comp in &phys_comps {
            let unused: Vec<_> = phys_comp.iter().filter(|&&p| !used_physical[p]).copied().collect();
            if unused.len() >= log_comp.len() {
                let mut cluster = Vec::new();
                let mut available = unused.clone();
                rng.shuffle_slice(&mut available);
                
                let start_node = available[0];
                let mut queue = VecDeque::new();
                let mut seen = HashSet::new();
                queue.push_back(start_node);
                seen.insert(start_node);

                while let Some(curr) = queue.pop_front() {
                    cluster.push(curr);
                    if cluster.len() == log_comp.len() { break; }
                    for &next in topology.neighbors(curr) {
                        if !used_physical[next] && seen.insert(next) {
                            queue.push_back(next);
                        }
                    }
                }

                if cluster.len() < log_comp.len() {
                    for &p in &available {
                        if !cluster.contains(&p) {
                            cluster.push(p);
                            if cluster.len() == log_comp.len() { break; }
                        }
                    }
                }

                for (log_q, phys_q) in log_comp.iter().zip(cluster) {
                    mapping[*log_q] = phys_q;
                    used_physical[phys_q] = true;
                }
                found = true;
                break;
            }
        }
        if !found {
            return Err(RouterError::Routing("Insufficient physical qubits in connected components".into()));
        }
    }

    let mut free_physical = (0..topology.num_qubits()).filter(|&p| !used_physical[p]);
    for slot in &mut mapping {
        if *slot == usize::MAX {
            *slot = free_physical.next().ok_or_else(|| RouterError::Routing("No free physical qubits left".into()))?;
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
            lookahead_weight: 0.8,
            lookahead_size: 32,
            set_scaling: SetScaling::Size,
            use_decay: true,
            decay_increment: 0.001,
            decay_reset: 10,
            best_epsilon: 1e-9,
            decay_state: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    fn refresh_decay_state(&mut self, ctx: &SwapSelectionContext<'_>) {
        if !self.use_decay { return; }
        let num_qubits = ctx.topology().num_qubits();
        if self.decay_state.len() != num_qubits {
            self.decay_state = vec![1.0; num_qubits];
        }
        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(1.0);
            return;
        }
        let Some((a, b)) = ctx.last_applied_swap() else { return; };
        let reset = self.decay_reset.max(1);
        if ctx.swaps_since_progress() % reset == 0 {
            self.decay_state.fill(1.0);
        } else {
            self.decay_state[a] += self.decay_increment;
            self.decay_state[b] += self.decay_increment;
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        choose_disjoint_aware_layout(ctx, rng)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        self.refresh_decay_state(ctx);
        let front_layer = FrontLayerScores::from_ctx(ctx);
        let candidates = enumerate_candidate_swaps(ctx.topology(), &front_layer);
        if candidates.is_empty() { return None; }
        
        let extended_set = build_extended_set(ctx, self.lookahead_size);
        let scale = |weight: f64, size: usize, scaling: SetScaling| -> f64 {
            match scaling {
                SetScaling::Constant => weight,
                SetScaling::Size => if size == 0 { 0.0 } else { weight / (size as f64) },
            }
        };
        
        let basic_weight = scale(self.basic_weight, front_layer.len(), self.set_scaling);
        let lookahead_weight = scale(self.lookahead_weight, extended_set.len(), self.set_scaling);
        
        let mut best_swaps = Vec::<(usize, usize)>::new();
        let mut min_score = f64::INFINITY;
        
        let abs_base = basic_weight * front_layer.total_score(ctx.topology()) 
                     + lookahead_weight * extended_set.total_score(ctx.topology());

        for swap in candidates {
            let mut score = abs_base;
            score += basic_weight * front_layer.score_delta(swap, ctx.topology());
            if !extended_set.is_empty() {
                score += lookahead_weight * extended_set.score_delta(swap, ctx.topology());
            }
            
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
