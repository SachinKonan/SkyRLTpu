RUST_CODE = r'''
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

#[derive(Debug, Clone, Default)]
struct FrontLayerScores {
    qubit_partners: Vec<Vec<(usize, f64)>>,
}

impl FrontLayerScores {
    fn from_ctx(ctx: &SwapSelectionContext<'_>) -> Self {
        let num_qubits = ctx.topology().num_qubits();
        let mut out = Self {
            qubit_partners: vec![Vec::new(); num_qubits],
        };

        for &node_id in ctx.front_layer().node_ids() {
            if let Some((l_a, l_b)) = ctx.circuit().node(node_id).two_qubit_pair() {
                let p_a = ctx.layout().physical_of_logical(l_a);
                let p_b = ctx.layout().physical_of_logical(l_b);
                
                if p_a != p_b {
                   out.qubit_partners[p_a].push((p_b, 1.0));
                   out.qubit_partners[p_b].push((p_a, 1.0));
                }
            }
        }
        out
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        let mut sum = 0.0;
        for (a, partners) in self.qubit_partners.iter().enumerate() {
            for &(b, w) in partners {
                sum += topology.distance(a, b) as f64 * w;
            }
        }
        sum * 0.5
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut delta = 0.0;

        for &(p, w) in &self.qubit_partners[a] {
            if p == b {
                continue;
            }
            delta += (topology.distance(b, p) as f64 - topology.distance(a, p) as f64) * w;
        }

        for &(p, w) in &self.qubit_partners[b] {
            if p == a {
                continue;
            }
            delta += (topology.distance(a, p) as f64 - topology.distance(b, p) as f64) * w;
        }
        delta
    }
}

#[derive(Debug, Clone, Default)]
struct ExtendedSetScores {
    qubits: Vec<Vec<(usize, f64)>>,
    len: usize,
}

impl ExtendedSetScores {
    fn new(num_qubits: usize) -> Self {
        Self {
            qubits: vec![Vec::new(); num_qubits],
            len: 0,
        }
    }

    fn push(&mut self, a: usize, b: usize, weight: f64) {
        self.qubits[a].push((b, weight));
        self.qubits[b].push((a, weight));
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
                    .map(move |(b, w)| topology.distance(a, *b) as f64 * *w)
            })
            .sum::<f64>()
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for (other, weight) in &self.qubits[a] {
            if *other == b {
                continue;
            }
            total += (topology.distance(b, *other) as f64 - topology.distance(a, *other) as f64) * weight;
        }
        for (other, weight) in &self.qubits[b] {
            if *other == a {
                continue;
            }
            total += (topology.distance(a, *other) as f64 - topology.distance(b, *other) as f64) * weight;
        }
        total
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize, lookahead_weight_decay: f64) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new(ctx.topology().num_qubits());
    if max_size == 0 {
        return out;
    }

    let precomputed = ctx.precomputed_extended_set_logical_pairs();
    if !precomputed.is_empty() {
        for (i, pair) in precomputed.iter().enumerate().take(max_size) {
            let weight = 1.0 / (i as f64 + 1.0).powf(lookahead_weight_decay);
            out.push(
                ctx.layout().physical_of_logical(pair[0]),
                ctx.layout().physical_of_logical(pair[1]),
                weight,
            );
        }
        return out;
    }

    for &node_id in ctx.remaining().ready_node_ids() {
        if !ctx.remaining().front_layer_node_ids().contains(&node_id) && out.len() < max_size {
             if let Some((l_a, l_b)) = ctx.circuit().node(node_id).two_qubit_pair() {
                 let a_phys = ctx.layout().physical_of_logical(l_a);
                 let b_phys = ctx.layout().physical_of_logical(l_b);
                 let weight = 1.0;
                 out.push(a_phys, b_phys, weight);
             }
        }
    }

    out
}

fn enumerate_candidate_swaps(
    topology: TopologyView<'_>,
    front_layer: &FrontLayerScores,
) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();
    let mut active_qubits = Vec::new();
    
    for i in 0..front_layer.qubit_partners.len() {
        if !front_layer.qubit_partners[i].is_empty() {
            active_qubits.push(i);
        }
    }

    for &phys in &active_qubits {
        for &neighbor in topology.neighbors(phys) {
            if neighbor > phys {
                out.push((phys, neighbor));
            } else {
                out.push((neighbor, phys));
            }
        }
    }
    out
}

fn ensure_connected_subset(topology: TopologyView<'_>, subset: &[usize]) -> Result<(), RouterError> {
    if subset.is_empty() {
        return Ok(());
    }
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
        return Err(RouterError::Routing(
            "selected layout subset is not connected".to_string(),
        ));
    }
    Ok(())
}

fn assign_components_to_target(
    logical_components: &[Vec<usize>],
    target_components: &[Vec<usize>],
) -> Result<Vec<(Vec<usize>, usize)>, RouterError> {
    if logical_components.is_empty() {
        return Ok(Vec::new());
    }

    let mut logical_sorted = logical_components.to_vec();
    logical_sorted.sort_by_key(|component| std::cmp::Reverse(component.len()));

    let mut target_sorted = target_components
        .iter()
        .enumerate()
        .map(|(idx, component)| (idx, component.len()))
        .collect::<Vec<_>>();
    target_sorted.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    let mut free_capacity = target_sorted
        .iter()
        .map(|(idx, size)| (*idx, *size))
        .collect::<HashMap<usize, usize>>();
    let mut assignments = Vec::<(Vec<usize>, usize)>::new();

    for logical in logical_sorted {
        let size = logical.len();
        let mut chosen = None;
        for (target_idx, _) in &target_sorted {
            let cap = free_capacity.get(target_idx).copied().unwrap_or(0);
            if cap >= size {
                chosen = Some(*target_idx);
                break;
            }
        }
        let Some(target_idx) = chosen else {
            return Err(RouterError::Routing(format!(
                "logical component of size {size} cannot fit any target component"
            )));
        };
        *free_capacity
            .get_mut(&target_idx)
            .expect("selected target component must exist") -= size;
        assignments.push((logical, target_idx));
    }
    Ok(assignments)
}

fn choose_dense_layout_subset(
    topology: TopologyView<'_>,
    logical_component_size: usize,
    target_component: &[usize],
) -> Result<Vec<usize>, RouterError> {
    if logical_component_size > target_component.len() {
        return Err(RouterError::Routing(format!(
            "logical component size {logical_component_size} exceeds target component size {}",
            target_component.len()
        )));
    }
    if logical_component_size == target_component.len() {
        return Ok(target_component.to_vec());
    }

    let local_index = target_component
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect::<HashMap<usize, usize>>();
        
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
    let chosen = best_map
        .into_iter()
        .take(logical_component_size)
        .map(|local| target_component[local])
        .collect::<Vec<_>>();
    ensure_connected_subset(topology, &chosen)?;
    Ok(chosen)
}

fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>, rng: &mut RngState) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let used = circuit.used_logical_qubits();
    if used.is_empty() {
        return Ok((0..num_logical).collect());
    }

    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    if target_components.is_empty() {
        return Err(RouterError::Routing(
            "topology has no connected components".to_string(),
        ));
    }

    let assignments = assign_components_to_target(logical_components, target_components)?;
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];

    for (logical_component, target_component_idx) in assignments {
        let target_component = &target_components[target_component_idx];
        
        let local = choose_dense_layout_subset(topology, logical_component.len(), target_component)?;
        
        let mut phys_candidates = local.clone();
        rng.shuffle_slice(&mut phys_candidates);

        for (logical, &physical) in logical_component.iter().zip(phys_candidates.iter()) {
            mapping[*logical] = physical;
            used_physical[physical] = true;
        }
    }

    let mut free_physical = (0..topology.num_qubits()).filter(|q| !used_physical[*q]).collect::<Vec<_>>();
    rng.shuffle_slice(&mut free_physical);
    let mut free_idx = 0;

    for slot in &mut mapping {
        if *slot == usize::MAX {
            if free_idx < free_physical.len() {
                *slot = free_physical[free_idx];
                free_idx += 1;
            } else {
                return Err(RouterError::Routing("not enough physical qubits to complete layout".to_string()));
            }
        }
    }
    Ok(mapping)
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    pub basic_weight: f64,
    pub lookahead_weight: f64,
    pub lookahead_size: usize,
    pub lookahead_decay: f64,
    pub set_scaling: SetScaling,
    pub use_decay: bool,
    decay_state: Vec<f64>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            basic_weight: 1.8,
            lookahead_weight: 0.6,
            lookahead_size: 30,
            lookahead_decay: 1.5,
            set_scaling: SetScaling::Constant,
            use_decay: false,
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
        if candidates.is_empty() {
            return None;
        }

        let circuit_size = ctx.circuit().node_count();
        let lookahead_size = if circuit_size < 50 { 15.min(self.lookahead_size) } else { self.lookahead_size };
        let extended_set = build_extended_set(ctx, lookahead_size, self.lookahead_decay);

        let circuit_two_qubit_count = ctx.remaining().remaining_two_qubit_node_ids().len();
        let lookahead_scale = if circuit_two_qubit_count > 20 { 0.8 } else { 1.0 };
        let lookahead_weight = if extended_set.is_empty() { 0.0 } else { self.lookahead_weight * lookahead_scale };
        let basic_weight = self.basic_weight;

        let mut swap_scores = candidates
            .iter()
            .copied()
            .map(|swap| (swap, 0.0))
            .collect::<Vec<_>>();

        let mut absolute_score = basic_weight * front_layer.total_score(ctx.topology());
        
        if lookahead_weight != 0.0 {
            absolute_score += lookahead_weight * extended_set.total_score(ctx.topology()) * 0.5;
        }

        for (swap, score) in &mut swap_scores {
            *score += basic_weight * front_layer.score_delta(*swap, ctx.topology());
            if lookahead_weight != 0.0 {
                *score += lookahead_weight * extended_set.score_delta(*swap, ctx.topology()) * 0.5;
            }
        }

        let mut min_score = f64::INFINITY;
        let mut best_swaps = Vec::<(usize, usize)>::new();
        
        let improvement_threshold = if circuit_size < 100 {
            0.0005
        } else {
            0.005
        };

        for (swap, score) in &swap_scores {
            let total = absolute_score + *score;
            
            if total < min_score - improvement_threshold {
                min_score = total;
                best_swaps.clear();
                best_swaps.push(*swap);
            } else if (total - min_score).abs() < improvement_threshold {
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
