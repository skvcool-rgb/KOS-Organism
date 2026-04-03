// KOS-Organism v10.0 — Living Neural Engine
//
// 60Hz Thermodynamic Spreading Activation + Neuroscience Learning
//
// TIER 1: STDP, Eligibility Traces, Neuromodulation, Growing Neural Gas,
//         ACT-R Activation, Fan Effect, BCM Metaplasticity, Synaptic Scaling
// TIER 2: Dual Memory (Hippocampal/Neocortical), Sleep Consolidation,
//         Novelty Search
// TIER 3: Predictive Coding, Global Workspace Broadcast,
//         Hyperdimensional Computing (VSA)

use pyo3::prelude::*;
use std::collections::HashMap;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ═══════════════════════════════════════════════════════════════
// EDGE TYPES — 13 typed relationships with trust multipliers
// ═══════════════════════════════════════════════════════════════

const ET_GENERIC: u8 = 0;
const ET_IS_A: u8 = 1;
const ET_CAUSES: u8 = 2;
const ET_PART_OF: u8 = 3;
const ET_SUPPORTS: u8 = 4;
const ET_CONTRADICTS: u8 = 5;
const ET_DERIVED_FROM: u8 = 6;
const ET_PROCEDURE_STEP: u8 = 7;
const ET_TEMPORAL_BEFORE: u8 = 8;
const ET_TEMPORAL_AFTER: u8 = 9;
const ET_LOCATED_IN: u8 = 10;
const ET_HAS_PROPERTY: u8 = 11;
const ET_ACTIVATES: u8 = 12;

fn edge_trust(et: u8) -> f32 {
    match et {
        1 => 0.9, 2 => 0.85, 3 => 0.8, 4 => 0.8, 5 => 0.3,
        6 => 0.7, 7 => 0.9, 8 => 0.6, 9 => 0.6, 10 => 0.8,
        11 => 0.7, 12 => 0.85, _ => 0.5,
    }
}

// ═══════════════════════════════════════════════════════════════
// SYNAPSE — weighted connection with learning state
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Synapse {
    target_idx: usize,
    weight: f32,
    myelin: f32,         // Myelination level (locks plasticity when high)
    edge_type: u8,
    // TIER 1: STDP + Eligibility Traces
    eligibility: f32,    // Decaying trace for three-factor learning
    usage_count: u64,    // How many times this edge propagated signal
    last_used_tick: u64, // When this edge last carried activation
    // TIER 3: Predictive Coding
    prediction: f32,     // Top-down prediction for this connection
}

impl Synapse {
    fn new(target_idx: usize, weight: f32, edge_type: u8) -> Self {
        Synapse {
            target_idx, weight, myelin: 0.0, edge_type,
            eligibility: 0.0, usage_count: 0, last_used_tick: 0,
            prediction: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// KASM NODE — neuron with full learning state
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
struct KASMNode {
    name: String,
    is_action: bool,
    activation: f32,
    fuel: f32,
    connections: Vec<Synapse>,
    fire_count: u64,
    last_fired_tick: u64,
    // TIER 1: ACT-R base-level activation
    access_times: Vec<u64>,  // Recent access timestamps (capped at 50)
    // TIER 1: BCM Metaplasticity
    bcm_theta: f32,          // Sliding modification threshold
    avg_activation: f32,     // Exponential moving average of activation
    // TIER 1: Growing Neural Gas
    gng_error: f32,          // Accumulated representation error
    // TIER 3: Predictive Coding
    prediction_error: f32,   // Mismatch between predicted and actual input
    predicted_activation: f32, // What this node expects to receive
    // TIER 3: Global Workspace
    workspace_bid: f32,      // Current bid for workspace access
    in_workspace: bool,      // Currently broadcasting?
    // TIER 3: Hyperdimensional Computing
    hd_vector: Vec<f32>,     // Semantic hypervector (D dimensions)
}

impl KASMNode {
    fn new(name: String, is_action: bool, hd_dim: usize, rng: &mut ChaCha8Rng) -> Self {
        // Initialize HD vector with random {-1, +1} components
        let hd_vector: Vec<f32> = (0..hd_dim)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect();

        KASMNode {
            name, is_action,
            activation: 0.0, fuel: 0.0,
            connections: Vec::new(),
            fire_count: 0, last_fired_tick: 0,
            access_times: Vec::new(),
            bcm_theta: 0.5,
            avg_activation: 0.0,
            gng_error: 0.0,
            prediction_error: 0.0,
            predicted_activation: 0.0,
            workspace_bid: 0.0,
            in_workspace: false,
            hd_vector,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// NEUROMODULATORS — four global channels that modulate learning
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Neuromodulators {
    dopamine: f32,       // Reward prediction error → scales reward learning
    acetylcholine: f32,  // Attention/novelty → encoding vs retrieval
    norepinephrine: f32, // Arousal → exploration vs exploitation (gain)
    serotonin: f32,      // Patience → temporal discount / time horizon
}

impl Neuromodulators {
    fn new() -> Self {
        Neuromodulators {
            dopamine: 0.0,
            acetylcholine: 0.5,
            norepinephrine: 0.5,
            serotonin: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// HIPPOCAMPAL MEMORY — fast episodic store (TIER 2)
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
struct Episode {
    node_activations: Vec<(usize, f32)>, // Which nodes were active + how much
    reward: f32,                          // Outcome signal
    tick_stored: u64,                     // When this episode was stored
    replay_count: u32,                    // How many times replayed
}

// ═══════════════════════════════════════════════════════════════
// NOVELTY ARCHIVE — behavioral diversity tracking (TIER 2)
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
struct NoveltyEntry {
    behavior_hash: u64,     // Hash of activation pattern
    activation_snapshot: Vec<f32>, // Top-N node activations as fingerprint
    tick_added: u64,
}

// ═══════════════════════════════════════════════════════════════
// RUST KERNEL — the organism's living neural engine
// ═══════════════════════════════════════════════════════════════

#[pyclass]
pub struct RustKernel {
    arena: Vec<KASMNode>,
    name_to_idx: HashMap<String, usize>,
    max_energy: f32,
    temporal_decay: f32,
    tick_count: u64,
    rng: ChaCha8Rng,

    // TIER 1: Neuromodulation
    modulators: Neuromodulators,

    // TIER 1: STDP parameters
    stdp_a_plus: f32,      // LTP amplitude
    stdp_a_minus: f32,     // LTD amplitude
    stdp_tau: f32,         // Time constant (in ticks)
    eligibility_decay: f32, // Per-tick decay for eligibility traces

    // TIER 1: BCM parameters
    bcm_tau: f32,          // Sliding threshold time constant
    bcm_learning_rate: f32,

    // TIER 1: Synaptic scaling parameters
    scaling_target: f32,   // Target average activation
    scaling_tau: f32,      // Homeostatic time constant

    // TIER 1: Growing Neural Gas parameters
    gng_lambda: u64,       // Insert new node every lambda ticks
    gng_max_age: u64,      // Remove edges older than this
    gng_error_decay: f32,  // Error counter decay

    // TIER 2: Hippocampal episodic memory
    episodes: Vec<Episode>,
    max_episodes: usize,

    // TIER 2: Novelty archive
    novelty_archive: Vec<NoveltyEntry>,
    novelty_k: usize,      // K-nearest neighbors for novelty score

    // TIER 3: Global Workspace
    workspace_threshold: f32,  // Ignition threshold for broadcast
    workspace_slots: usize,    // How many nodes can broadcast at once

    // TIER 3: HDC dimension
    hd_dim: usize,
}

#[pymethods]
impl RustKernel {
    #[new]
    pub fn new() -> Self {
        RustKernel {
            arena: Vec::with_capacity(8192),
            name_to_idx: HashMap::with_capacity(8192),
            max_energy: 3.0,
            temporal_decay: 0.99,
            tick_count: 0,
            rng: ChaCha8Rng::seed_from_u64(42),

            modulators: Neuromodulators::new(),

            // STDP: Bi & Poo 1998 parameters (scaled for 60Hz ticks)
            stdp_a_plus: 0.008,
            stdp_a_minus: 0.009,  // Slightly larger for stability
            stdp_tau: 30.0,       // ~500ms at 60Hz
            eligibility_decay: 0.995, // ~200 ticks half-life

            // BCM: Bienenstock-Cooper-Munro
            bcm_tau: 3000.0,      // Slow timescale (~50 seconds)
            bcm_learning_rate: 0.001,

            // Synaptic scaling: Turrigiano 1998
            scaling_target: 0.3,
            scaling_tau: 6000.0,  // Very slow (~100 seconds)

            // Growing Neural Gas: Fritzke 1994
            gng_lambda: 600,      // Insert node every ~10 seconds
            gng_max_age: 3600,    // Prune unused edges after ~1 minute
            gng_error_decay: 0.995,

            // Hippocampal memory
            episodes: Vec::with_capacity(1024),
            max_episodes: 1000,

            // Novelty search
            novelty_archive: Vec::with_capacity(512),
            novelty_k: 5,

            // Global Workspace
            workspace_threshold: 1.5,
            workspace_slots: 3,

            // Hyperdimensional Computing
            hd_dim: 1024,
        }
    }

    // ═══════════════════════════════════════════════════════════
    // NODE & EDGE MANAGEMENT
    // ═══════════════════════════════════════════════════════════

    pub fn get_or_create_node(&mut self, name: String, is_action: bool) -> usize {
        if let Some(&idx) = self.name_to_idx.get(&name) {
            return idx;
        }
        let idx = self.arena.len();
        self.arena.push(KASMNode::new(name.clone(), is_action, self.hd_dim, &mut self.rng));
        self.name_to_idx.insert(name, idx);
        idx
    }

    pub fn add_connection(&mut self, src: String, tgt: String, weight: f32, edge_type: u8) {
        let src_idx = self.get_or_create_node(src, false);
        let tgt_idx = self.get_or_create_node(tgt, false);
        for syn in &self.arena[src_idx].connections {
            if syn.target_idx == tgt_idx && syn.edge_type == edge_type {
                return;
            }
        }
        let mut syn = Synapse::new(tgt_idx, weight, edge_type);
        syn.last_used_tick = self.tick_count; // Prevent immediate pruning
        self.arena[src_idx].connections.push(syn);
    }

    pub fn add_connection_simple(&mut self, src: String, tgt: String, weight: f32) {
        self.add_connection(src, tgt, weight, ET_GENERIC);
    }

    pub fn inject_energy(&mut self, name: String, amount: f32) {
        if let Some(&idx) = self.name_to_idx.get(&name) {
            let node = &mut self.arena[idx];
            node.fuel = (node.fuel + amount).clamp(0.0, self.max_energy);
            node.activation = (node.activation + amount).clamp(-self.max_energy, self.max_energy);
            // ACT-R: record access time
            let tick = self.tick_count;
            node.access_times.push(tick);
            if node.access_times.len() > 50 {
                node.access_times.remove(0);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // NEUROMODULATION — set from Python (brain.py emotions)
    // ═══════════════════════════════════════════════════════════

    pub fn set_neuromodulators(&mut self, dopamine: f32, acetylcholine: f32,
                                norepinephrine: f32, serotonin: f32) {
        self.modulators.dopamine = dopamine.clamp(-1.0, 1.0);
        self.modulators.acetylcholine = acetylcholine.clamp(0.0, 1.0);
        self.modulators.norepinephrine = norepinephrine.clamp(0.0, 1.0);
        self.modulators.serotonin = serotonin.clamp(0.0, 1.0);
    }

    pub fn get_neuromodulators(&self) -> (f32, f32, f32, f32) {
        (self.modulators.dopamine, self.modulators.acetylcholine,
         self.modulators.norepinephrine, self.modulators.serotonin)
    }

    // ═══════════════════════════════════════════════════════════
    // REWARD SIGNAL — three-factor learning gate (TIER 1)
    // ═══════════════════════════════════════════════════════════

    /// Broadcast a reward signal. Only edges with nonzero eligibility traces update.
    /// This is the key to credit assignment: edges that were recently active in
    /// causal order get strengthened/weakened by delayed reward.
    pub fn broadcast_reward(&mut self, reward: f32) {
        let da = self.modulators.dopamine;
        let reward_signal = reward * (1.0 + da); // Dopamine amplifies reward

        for node in &mut self.arena {
            for syn in &mut node.connections {
                if syn.eligibility.abs() > 0.001 {
                    // Three-factor rule: delta_w = eligibility * reward * (1 - myelin_lock)
                    let myelin_lock = 1.0 / (1.0 + syn.myelin * 0.1);
                    let delta_w = syn.eligibility * reward_signal * 0.01 * myelin_lock;
                    syn.weight = (syn.weight + delta_w).clamp(-1.0, 1.0);
                }
            }
        }

        // Update dopamine as RPE (reward prediction error)
        self.modulators.dopamine = (reward - self.modulators.dopamine * 0.5).clamp(-1.0, 1.0);
    }

    // ═══════════════════════════════════════════════════════════
    // TICK — the 60Hz heartbeat with all learning mechanisms
    // ═══════════════════════════════════════════════════════════

    pub fn tick(&mut self, spatial_decay: f32, threshold: f32) -> Vec<(String, f32)> {
        self.tick_count += 1;
        let tick = self.tick_count;
        let max_e = self.max_energy;
        let t_decay = self.temporal_decay;
        let ne = self.modulators.norepinephrine;

        // ── Phase 1: Temporal decay + ACT-R base-level + collect firing nodes ──
        let mut pending: Vec<(usize, usize, f32, usize)> = Vec::new(); // (src, tgt, energy, syn_idx)

        for (idx, node) in self.arena.iter_mut().enumerate() {
            // Temporal decay
            node.activation *= t_decay;
            node.fuel *= t_decay;

            // ACT-R base-level activation: log of weighted recency sum
            if !node.access_times.is_empty() {
                let base_level: f32 = node.access_times.iter()
                    .map(|&t| {
                        let age = (tick.saturating_sub(t)) as f32 + 1.0;
                        1.0 / age.powf(0.5) // Power-law decay (d=0.5)
                    })
                    .sum::<f32>()
                    .ln()
                    .max(-2.0);
                node.activation += base_level * 0.01; // Subtle base-level boost
            }

            // Norepinephrine modulates activation gain
            let gain = 1.0 + (ne - 0.5) * 0.5; // NE=0: gain=0.75, NE=1: gain=1.25
            node.activation *= gain;
            node.activation = node.activation.clamp(-max_e, max_e);
            node.fuel = node.fuel.clamp(0.0, max_e);

            // Decay eligibility traces on all outgoing edges
            for syn in &mut node.connections {
                syn.eligibility *= self.eligibility_decay;
            }

            // BCM: update sliding threshold (slow timescale)
            let act_sq = node.activation * node.activation;
            node.bcm_theta += (1.0 / self.bcm_tau) * (act_sq - node.bcm_theta);
            node.bcm_theta = node.bcm_theta.clamp(0.01, 5.0);

            // Synaptic scaling: update average activation (very slow)
            node.avg_activation += (1.0 / self.scaling_tau)
                * (node.activation.abs() - node.avg_activation);

            // Fire check
            if node.fuel >= threshold {
                let fire_fuel = node.fuel;
                if !node.is_action {
                    node.fuel = 0.0;
                }
                node.fire_count += 1;
                node.last_fired_tick = tick;

                // Fan effect: divide activation by log(degree + 1)
                let degree = node.connections.len() as f32;
                let fan_penalty = 1.0 / (1.0 + degree.ln().max(0.0));

                for (si, syn) in node.connections.iter_mut().enumerate() {
                    let trust = edge_trust(syn.edge_type);
                    let myelin_boost = 1.0 + syn.myelin * 0.01;
                    let effective_w = syn.weight * myelin_boost * trust;
                    let passed = fire_fuel * effective_w * spatial_decay * fan_penalty;

                    if passed.abs() >= threshold * 0.3 {
                        pending.push((idx, syn.target_idx, passed, si));
                        syn.usage_count += 1;
                        syn.last_used_tick = tick;
                    }
                }
            }
        }

        // ── Phase 2: Apply signals + STDP + Predictive coding ──
        let mut triggered: Vec<(String, f32)> = Vec::new();

        for &(src_idx, tgt_idx, energy, syn_idx) in &pending {
            // STDP: compute timing-based plasticity
            let src_fire = self.arena[src_idx].last_fired_tick;
            let tgt_fire = self.arena[tgt_idx].last_fired_tick;
            if src_fire > 0 && tgt_fire > 0 {
                let delta_t = (tgt_fire as f64 - src_fire as f64) as f32;
                let stdp_signal = if delta_t > 0.0 {
                    // Pre before post → LTP (causal)
                    self.stdp_a_plus * (-delta_t / self.stdp_tau).exp()
                } else if delta_t < 0.0 {
                    // Post before pre → LTD (anti-causal)
                    -self.stdp_a_minus * (delta_t / self.stdp_tau).exp()
                } else {
                    0.0
                };

                // Accumulate into eligibility trace (gated by acetylcholine)
                let ach = self.modulators.acetylcholine;
                self.arena[src_idx].connections[syn_idx].eligibility +=
                    stdp_signal * (0.5 + ach); // High ACh = more encoding
            }

            // BCM-gated weight update (immediate Hebbian component)
            let bcm_theta = self.arena[tgt_idx].bcm_theta;
            let tgt_act = self.arena[tgt_idx].activation;
            let bcm_phi = tgt_act * (tgt_act - bcm_theta);
            let myelin_lock = 1.0 / (1.0 + self.arena[src_idx].connections[syn_idx].myelin * 0.1);
            let bcm_delta = self.bcm_learning_rate * bcm_phi * energy.abs() * myelin_lock;
            self.arena[src_idx].connections[syn_idx].weight =
                (self.arena[src_idx].connections[syn_idx].weight + bcm_delta).clamp(-1.0, 1.0);

            // Micro-myelination on active edges
            self.arena[src_idx].connections[syn_idx].myelin += 0.001;

            // TIER 3: Predictive coding — compute prediction error
            let predicted = self.arena[src_idx].connections[syn_idx].prediction;
            let actual = energy;
            let pred_error = actual - predicted;
            self.arena[tgt_idx].prediction_error += pred_error;
            // Update prediction toward actual (slow learning)
            self.arena[src_idx].connections[syn_idx].prediction += pred_error * 0.01;

            // Apply energy to target node
            let node = &mut self.arena[tgt_idx];
            node.activation = (node.activation + energy).clamp(-max_e, max_e);

            if energy > 0.0 && node.activation > 0.0 {
                node.fuel = (node.fuel + energy).clamp(0.0, max_e);
            }

            // GNG: accumulate error on receiving node
            node.gng_error = (node.gng_error + pred_error.abs() * 0.1).min(1000.0);
            node.prediction_error = node.prediction_error.clamp(-max_e, max_e);

            // Action node trigger
            if node.is_action && node.fuel > 2.0 {
                triggered.push((node.name.clone(), node.fuel));
                node.fuel = 0.0;
            }
        }

        // ── Phase 3: Synaptic scaling (every 60 ticks = ~1 second) ──
        if tick % 60 == 0 {
            let target = self.scaling_target;
            for node in &mut self.arena {
                let avg = node.avg_activation;
                if avg > 0.001 {
                    let scale = 1.0 + 0.001 * (target - avg);
                    for syn in &mut node.connections {
                        syn.weight *= scale;
                        syn.weight = syn.weight.clamp(-1.0, 1.0);
                    }
                }
            }
        }

        // ── Phase 4: Global Workspace — competition + broadcast (TIER 3) ──
        if tick % 10 == 0 {
            // Reset workspace
            for node in &mut self.arena {
                node.in_workspace = false;
                // Bid = activation * prediction_error (surprising & active nodes win)
                node.workspace_bid = node.activation.abs() * (1.0 + node.prediction_error.abs());
            }
            // Find top-N bidders
            let mut bids: Vec<(usize, f32)> = self.arena.iter()
                .enumerate()
                .filter(|(_, n)| !n.is_action && n.workspace_bid > self.workspace_threshold)
                .map(|(i, n)| (i, n.workspace_bid))
                .collect();
            bids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            bids.truncate(self.workspace_slots);

            // Broadcast winners to all nodes (small energy injection)
            let broadcast_indices: Vec<usize> = bids.iter().map(|&(i, _)| i).collect();
            for &winner_idx in &broadcast_indices {
                self.arena[winner_idx].in_workspace = true;
            }
            // Broadcast energy from workspace winners to all connected nodes
            let mut workspace_spread: Vec<(usize, f32)> = Vec::new();
            for &winner_idx in &broadcast_indices {
                let bid = self.arena[winner_idx].workspace_bid;
                for syn in &self.arena[winner_idx].connections {
                    workspace_spread.push((syn.target_idx, bid * 0.05));
                }
            }
            for (idx, energy) in workspace_spread {
                if idx < self.arena.len() {
                    self.arena[idx].activation =
                        (self.arena[idx].activation + energy).clamp(-max_e, max_e);
                }
            }
        }

        // ── Phase 5: GNG — insert node at highest error (TIER 1) ──
        if tick % self.gng_lambda == 0 && self.arena.len() > 2 {
            // Find node with highest accumulated error
            let mut max_err_idx = 0;
            let mut max_err = 0.0_f32;
            for (i, node) in self.arena.iter().enumerate() {
                if node.gng_error > max_err && !node.is_action {
                    max_err = node.gng_error;
                    max_err_idx = i;
                }
            }

            // Find its neighbor with highest error
            if max_err > 0.1 {
                let mut neighbor_idx = None;
                let mut neighbor_err = 0.0_f32;
                for syn in &self.arena[max_err_idx].connections {
                    let n = &self.arena[syn.target_idx];
                    if n.gng_error > neighbor_err && !n.is_action {
                        neighbor_err = n.gng_error;
                        neighbor_idx = Some(syn.target_idx);
                    }
                }

                if let Some(nb_idx) = neighbor_idx {
                    // Insert new node between max_err and neighbor
                    let new_name = format!("gng_{}_{}", tick, self.arena.len());
                    let new_idx = self.arena.len();
                    self.arena.push(KASMNode::new(new_name.clone(), false, self.hd_dim, &mut self.rng));
                    self.name_to_idx.insert(new_name, new_idx);

                    // Average HD vectors of parents
                    for d in 0..self.hd_dim {
                        self.arena[new_idx].hd_vector[d] =
                            (self.arena[max_err_idx].hd_vector[d]
                             + self.arena[nb_idx].hd_vector[d]).signum();
                    }

                    // Connect new node to both parents
                    let mid_w = 0.5 * (self.arena[max_err_idx].connections.iter()
                        .find(|s| s.target_idx == nb_idx)
                        .map(|s| s.weight)
                        .unwrap_or(0.3));
                    self.arena[new_idx].connections.push(Synapse::new(max_err_idx, mid_w, ET_DERIVED_FROM));
                    self.arena[new_idx].connections.push(Synapse::new(nb_idx, mid_w, ET_DERIVED_FROM));
                    self.arena[max_err_idx].connections.push(Synapse::new(new_idx, mid_w, ET_DERIVED_FROM));
                    self.arena[nb_idx].connections.push(Synapse::new(new_idx, mid_w, ET_DERIVED_FROM));

                    // Reduce error on parents
                    self.arena[max_err_idx].gng_error *= 0.5;
                    self.arena[nb_idx].gng_error *= 0.5;
                }
            }

            // Decay all GNG errors
            for node in &mut self.arena {
                node.gng_error *= self.gng_error_decay;
            }
        }

        // ── Phase 6: GNG edge aging — prune old unused edges ──
        if tick % 600 == 0 {
            for node in &mut self.arena {
                node.connections.retain(|syn| {
                    let age = tick.saturating_sub(syn.last_used_tick);
                    age < self.gng_max_age || syn.myelin > 1.0 // Myelinated edges survive
                });
            }
        }

        // Reset prediction errors
        for node in &mut self.arena {
            node.prediction_error = 0.0;
        }

        triggered
    }

    // ═══════════════════════════════════════════════════════════
    // TIER 2: HIPPOCAMPAL MEMORY — fast episodic store
    // ═══════════════════════════════════════════════════════════

    /// Store current activation pattern as an episode
    pub fn store_episode(&mut self, reward: f32) {
        let tick = self.tick_count;
        let activations: Vec<(usize, f32)> = self.arena.iter()
            .enumerate()
            .filter(|(_, n)| n.activation.abs() > 0.1)
            .map(|(i, n)| (i, n.activation))
            .collect();

        if activations.is_empty() { return; }

        self.episodes.push(Episode {
            node_activations: activations,
            reward,
            tick_stored: tick,
            replay_count: 0,
        });

        if self.episodes.len() > self.max_episodes {
            // Remove oldest low-reward episodes first
            self.episodes.sort_by(|a, b| {
                a.reward.abs().partial_cmp(&b.reward.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.episodes.remove(0);
        }
    }

    /// Sleep consolidation: replay episodes + synaptic downscaling
    /// Call this during the dream cycle
    pub fn sleep_consolidation(&mut self, n_replays: usize, downscale_factor: f32) -> (usize, usize) {
        let n = self.episodes.len();
        if n == 0 { return (0, 0); }

        let mut replayed = 0;
        let replay_count = n_replays.min(n);

        // Replay random episodes (interleaved, like biological sleep)
        for _ in 0..replay_count {
            let idx = self.rng.random_range(0..n);
            let episode = self.episodes[idx].clone();

            // Reactivate episode pattern with low learning rate
            for &(node_idx, act) in &episode.node_activations {
                if node_idx < self.arena.len() {
                    self.arena[node_idx].activation += act * 0.1; // Weak reactivation
                    self.arena[node_idx].fuel += act.abs() * 0.05;
                }
            }

            // Run one tick to let activation spread (consolidation)
            self.tick(0.7, 0.3);

            // Apply reward signal from episode
            if episode.reward.abs() > 0.01 {
                self.broadcast_reward(episode.reward * 0.3); // Weaker than live reward
            }

            self.episodes[idx].replay_count += 1;
            replayed += 1;
        }

        // Synaptic homeostasis downscaling (Tononi & Cirelli)
        // Multiply ALL weights by downscale_factor (e.g., 0.95)
        // Preserves relative differences, reduces absolute levels
        let mut pruned = 0;
        for node in &mut self.arena {
            for syn in &mut node.connections {
                syn.weight *= downscale_factor;
            }
            // Prune edges that fell below threshold after downscaling
            let before = node.connections.len();
            node.connections.retain(|s| s.weight.abs() >= 0.02);
            pruned += before - node.connections.len();
        }

        (replayed, pruned)
    }

    /// Get episode count
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    // ═══════════════════════════════════════════════════════════
    // TIER 2: NOVELTY SEARCH
    // ═══════════════════════════════════════════════════════════

    /// Compute novelty score of current activation pattern
    pub fn novelty_score(&self) -> f32 {
        // Create behavioral fingerprint: top-20 activated nodes
        let mut snapshot: Vec<f32> = self.arena.iter()
            .map(|n| n.activation)
            .collect();
        snapshot.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
        snapshot.truncate(20);
        while snapshot.len() < 20 { snapshot.push(0.0); }

        if self.novelty_archive.is_empty() {
            return 1.0; // Everything is novel when archive is empty
        }

        // Compute distances to all archive entries
        let mut distances: Vec<f32> = self.novelty_archive.iter()
            .map(|entry| {
                entry.activation_snapshot.iter()
                    .zip(snapshot.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Average distance to K nearest neighbors
        let k = self.novelty_k.min(distances.len());
        if k == 0 { return 1.0; }
        distances[..k].iter().sum::<f32>() / k as f32
    }

    /// Add current activation pattern to novelty archive
    pub fn archive_novelty(&mut self) {
        let tick = self.tick_count;
        let mut snapshot: Vec<f32> = self.arena.iter()
            .map(|n| n.activation)
            .collect();
        snapshot.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));
        snapshot.truncate(20);
        while snapshot.len() < 20 { snapshot.push(0.0); }

        // Simple hash of activation pattern
        let hash: u64 = snapshot.iter()
            .enumerate()
            .map(|(i, &v)| ((v * 1000.0) as u64).wrapping_mul(i as u64 + 1))
            .fold(0u64, |acc, x| acc.wrapping_add(x));

        self.novelty_archive.push(NoveltyEntry {
            behavior_hash: hash,
            activation_snapshot: snapshot,
            tick_added: tick,
        });

        // Cap archive size
        if self.novelty_archive.len() > 500 {
            self.novelty_archive.remove(0);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // TIER 3: HYPERDIMENSIONAL COMPUTING
    // ═══════════════════════════════════════════════════════════

    /// Bind two node vectors (element-wise multiply) — creates association
    pub fn hd_bind(&self, name_a: &str, name_b: &str) -> Vec<f32> {
        let empty = vec![0.0_f32; self.hd_dim];
        let idx_a = self.name_to_idx.get(name_a);
        let idx_b = self.name_to_idx.get(name_b);
        match (idx_a, idx_b) {
            (Some(&a), Some(&b)) => {
                self.arena[a].hd_vector.iter()
                    .zip(self.arena[b].hd_vector.iter())
                    .map(|(&x, &y)| x * y)
                    .collect()
            }
            _ => empty,
        }
    }

    /// Bundle multiple node vectors (element-wise sum + sign) — creates superposition
    pub fn hd_bundle(&self, names: Vec<String>) -> Vec<f32> {
        let mut result = vec![0.0_f32; self.hd_dim];
        for name in &names {
            if let Some(&idx) = self.name_to_idx.get(name) {
                for (d, v) in self.arena[idx].hd_vector.iter().enumerate() {
                    result[d] += v;
                }
            }
        }
        // Threshold to {-1, +1}
        for v in &mut result {
            *v = if *v >= 0.0 { 1.0 } else { -1.0 };
        }
        result
    }

    /// Cosine similarity between a query vector and a node's HD vector
    pub fn hd_similarity(&self, query: Vec<f32>, name: &str) -> f32 {
        if let Some(&idx) = self.name_to_idx.get(name) {
            let node_vec = &self.arena[idx].hd_vector;
            let dot: f32 = query.iter().zip(node_vec.iter()).map(|(a, b)| a * b).sum();
            let mag_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mag_n: f32 = node_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag_q > 0.0 && mag_n > 0.0 { dot / (mag_q * mag_n) } else { 0.0 }
        } else {
            0.0
        }
    }

    /// Find top-N most similar nodes to a query vector
    pub fn hd_search(&self, query: Vec<f32>, top_n: usize) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self.arena.iter()
            .map(|n| {
                let dot: f32 = query.iter().zip(n.hd_vector.iter()).map(|(a, b)| a * b).sum();
                let mag_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let mag_n: f32 = n.hd_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if mag_q > 0.0 && mag_n > 0.0 { dot / (mag_q * mag_n) } else { 0.0 };
                (n.name.clone(), sim)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_n);
        results
    }

    /// Set a node's HD vector (for ingesting pre-computed embeddings)
    pub fn hd_set_vector(&mut self, name: &str, vector: Vec<f32>) {
        if let Some(&idx) = self.name_to_idx.get(name) {
            let dim = self.hd_dim;
            for (d, v) in vector.iter().enumerate().take(dim) {
                self.arena[idx].hd_vector[d] = *v;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // EXISTING API — backward compatible
    // ═══════════════════════════════════════════════════════════

    pub fn strengthen_edge(&mut self, src: String, tgt: String, delta: f32) {
        if let Some(&src_idx) = self.name_to_idx.get(&src) {
            if let Some(&tgt_idx) = self.name_to_idx.get(&tgt) {
                for syn in &mut self.arena[src_idx].connections {
                    if syn.target_idx == tgt_idx {
                        syn.weight = (syn.weight + delta).clamp(-1.0, 1.0);
                        syn.myelin += delta.abs() * 10.0;
                        return;
                    }
                }
            }
        }
    }

    pub fn weaken_edge(&mut self, src: String, tgt: String, delta: f32) {
        if let Some(&src_idx) = self.name_to_idx.get(&src) {
            if let Some(&tgt_idx) = self.name_to_idx.get(&tgt) {
                for syn in &mut self.arena[src_idx].connections {
                    if syn.target_idx == tgt_idx {
                        syn.weight = (syn.weight - delta).clamp(-1.0, 1.0);
                        return;
                    }
                }
            }
        }
    }

    pub fn triadic_closure(&mut self, max_new: usize) -> usize {
        let mut new_edges: Vec<(usize, usize, f32)> = Vec::new();
        for a_idx in 0..self.arena.len() {
            if new_edges.len() >= max_new { break; }
            let a_conns: Vec<(usize, f32)> = self.arena[a_idx].connections.iter()
                .filter(|s| s.weight > 0.3)
                .map(|s| (s.target_idx, s.weight))
                .collect();
            for &(b_idx, w_ab) in &a_conns {
                if new_edges.len() >= max_new { break; }
                let b_conns: Vec<(usize, f32)> = self.arena[b_idx].connections.iter()
                    .filter(|s| s.weight > 0.3)
                    .map(|s| (s.target_idx, s.weight))
                    .collect();
                for &(c_idx, w_bc) in &b_conns {
                    if c_idx == a_idx { continue; }
                    let exists = self.arena[a_idx].connections.iter().any(|s| s.target_idx == c_idx);
                    if !exists {
                        let inferred_w = (w_ab * w_bc).min(0.5);
                        new_edges.push((a_idx, c_idx, inferred_w));
                        if new_edges.len() >= max_new { break; }
                    }
                }
            }
        }
        let count = new_edges.len();
        for (a, c, w) in new_edges {
            self.arena[a].connections.push(Synapse::new(c, w, ET_DERIVED_FROM));
        }
        count
    }

    pub fn prune_weak_edges(&mut self, threshold: f32) -> usize {
        let mut pruned = 0;
        for node in &mut self.arena {
            let before = node.connections.len();
            node.connections.retain(|s| s.weight.abs() >= threshold);
            pruned += before - node.connections.len();
        }
        pruned
    }

    // ── Dashboard & Introspection ────────────────────────────

    pub fn node_count(&self) -> usize { self.arena.len() }
    pub fn edge_count(&self) -> usize {
        self.arena.iter().map(|n| n.connections.len()).sum()
    }
    pub fn tick_number(&self) -> u64 { self.tick_count }

    pub fn get_activations(&self, top_n: usize) -> Vec<(String, f32, f32, u64)> {
        let mut items: Vec<(usize, f32)> = self.arena.iter()
            .enumerate()
            .map(|(i, n)| (i, n.activation.abs() + n.fuel))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items.truncate(top_n);
        items.iter().map(|(i, _)| {
            let n = &self.arena[*i];
            (n.name.clone(), n.activation, n.fuel, n.fire_count)
        }).collect()
    }

    pub fn get_action_nodes(&self) -> Vec<(String, f32, f32)> {
        self.arena.iter()
            .filter(|n| n.is_action)
            .map(|n| (n.name.clone(), n.activation, n.fuel))
            .collect()
    }

    pub fn stats(&self) -> HashMap<String, f64> {
        let mut s = HashMap::new();
        let n = self.arena.len() as f64;
        let e: f64 = self.arena.iter().map(|n| n.connections.len() as f64).sum();
        let total_act: f64 = self.arena.iter().map(|n| n.activation.abs() as f64).sum();
        let total_fuel: f64 = self.arena.iter().map(|n| n.fuel as f64).sum();
        let total_myelin: f64 = self.arena.iter()
            .flat_map(|n| n.connections.iter().map(|s| s.myelin as f64))
            .sum();
        let total_eligibility: f64 = self.arena.iter()
            .flat_map(|n| n.connections.iter().map(|s| s.eligibility.abs() as f64))
            .sum();
        let total_gng_error: f64 = self.arena.iter().map(|n| n.gng_error as f64).sum();
        let workspace_count = self.arena.iter().filter(|n| n.in_workspace).count() as f64;
        let action_count = self.arena.iter().filter(|n| n.is_action).count() as f64;

        s.insert("nodes".into(), n);
        s.insert("edges".into(), e);
        s.insert("total_activation".into(), total_act);
        s.insert("total_fuel".into(), total_fuel);
        s.insert("total_myelin".into(), total_myelin);
        s.insert("total_eligibility".into(), total_eligibility);
        s.insert("total_gng_error".into(), total_gng_error);
        s.insert("episodes".into(), self.episodes.len() as f64);
        s.insert("novelty_archive".into(), self.novelty_archive.len() as f64);
        s.insert("workspace_active".into(), workspace_count);
        s.insert("action_nodes".into(), action_count);
        s.insert("ticks".into(), self.tick_count as f64);
        s.insert("avg_degree".into(), if n > 0.0 { e / n } else { 0.0 });
        s.insert("dopamine".into(), self.modulators.dopamine as f64);
        s.insert("acetylcholine".into(), self.modulators.acetylcholine as f64);
        s.insert("norepinephrine".into(), self.modulators.norepinephrine as f64);
        s.insert("serotonin".into(), self.modulators.serotonin as f64);
        // Sanitize: replace inf/nan with 0.0 for JSON safety
        for v in s.values_mut() {
            if v.is_infinite() || v.is_nan() {
                *v = 0.0;
            }
        }
        s
    }

    /// Get workspace (globally broadcast) node names
    pub fn get_workspace_nodes(&self) -> Vec<String> {
        self.arena.iter()
            .filter(|n| n.in_workspace)
            .map(|n| n.name.clone())
            .collect()
    }

    pub fn has_node(&self, name: &str) -> bool {
        self.name_to_idx.contains_key(name)
    }

    pub fn node_names(&self) -> Vec<String> {
        self.arena.iter().map(|n| n.name.clone()).collect()
    }

    pub fn reset_activations(&mut self) {
        for node in &mut self.arena {
            node.activation = 0.0;
            node.fuel = 0.0;
            node.prediction_error = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PYMODULE REGISTRATION
// ═══════════════════════════════════════════════════════════════

#[pymodule]
fn kos_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustKernel>()
}
