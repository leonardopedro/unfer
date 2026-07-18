use num_complex::Complex64;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};

// --- LEVEL 1: The Inner Fock Space ---

/// A configuration of an inner Bosonic universe.
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct InnerBosonicState {
    pub modes: BTreeMap<u32, u32>,
}

impl InnerBosonicState {
    pub fn vacuum() -> Self {
        Self {
            modes: BTreeMap::new(),
        }
    }
}

/// A configuration of an inner Fermionic universe.
/// Deriving Ord and PartialOrd guarantees Canonical Ordering for Fermion signs.
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct InnerFermionicState {
    pub modes: BTreeSet<u32>,
}

impl InnerFermionicState {
    pub fn vacuum() -> Self {
        Self {
            modes: BTreeSet::new(),
        }
    }
}

// --- LEVEL 2: The Outer Fock Space ---

// Serde helper: BTreeMap<InnerBosonicState, u32> has non-string keys so we
// round-trip through Vec<(InnerBosonicState, u32)>.
#[derive(serde::Serialize, serde::Deserialize)]
struct OuterStateRepr {
    bosonic: Vec<(InnerBosonicState, u32)>,
    fermionic: BTreeSet<InnerFermionicState>,
}

impl From<OuterState> for OuterStateRepr {
    fn from(s: OuterState) -> Self {
        Self {
            bosonic: s.bosonic.into_iter().collect(),
            fermionic: s.fermionic,
        }
    }
}

impl From<OuterStateRepr> for OuterState {
    fn from(r: OuterStateRepr) -> Self {
        Self {
            bosonic: r.bosonic.into_iter().collect(),
            fermionic: r.fermionic,
        }
    }
}

/// The state of the "Multiverse" / Outer Space, split into disjoint bosonic/fermionic universes
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(from = "OuterStateRepr", into = "OuterStateRepr")]
pub struct OuterState {
    pub bosonic: BTreeMap<InnerBosonicState, u32>,
    pub fermionic: BTreeSet<InnerFermionicState>,
}

impl OuterState {
    pub fn vacuum() -> Self {
        Self {
            bosonic: BTreeMap::new(),
            fermionic: BTreeSet::new(),
        }
    }
}

// Serde helper: FxHashMap<OuterState, Complex64> has non-string keys and Complex64
// has no default serde impl, so we round-trip through Vec<(OuterState, [f64; 2])>.
#[derive(serde::Serialize, serde::Deserialize)]
struct QuantumStateRepr {
    components: Vec<(OuterState, [f64; 2])>,
}

impl From<QuantumState> for QuantumStateRepr {
    fn from(q: QuantumState) -> Self {
        Self {
            components: q
                .components
                .into_iter()
                .map(|(s, a)| (s, [a.re, a.im]))
                .collect(),
        }
    }
}

impl From<QuantumStateRepr> for QuantumState {
    fn from(r: QuantumStateRepr) -> Self {
        Self {
            components: r
                .components
                .into_iter()
                .map(|(s, [re, im])| (s, Complex64::new(re, im)))
                .collect(),
        }
    }
}

/// A superposition of Outer States with complex amplitudes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "QuantumStateRepr", into = "QuantumStateRepr")]
pub struct QuantumState {
    pub components: FxHashMap<OuterState, Complex64>,
}

impl QuantumState {
    pub fn vacuum() -> Self {
        let mut components = FxHashMap::default();
        components.insert(OuterState::vacuum(), Complex64::new(1.0, 0.0));
        Self { components }
    }

    /// The zero vector (no components) — the additive identity, distinct from
    /// the physical vacuum `|0>`.
    pub fn zero() -> Self {
        Self {
            components: FxHashMap::default(),
        }
    }

    pub fn apply(&self, op: &Operator) -> Self {
        op.apply_to_state(self)
    }

    pub fn inner_product(a: &Self, b: &Self) -> Complex64 {
        let mut sum = Complex64::new(0.0, 0.0);
        if a.components.len() < b.components.len() {
            for (state, val_a) in &a.components {
                if let Some(val_b) = b.components.get(state) {
                    sum += val_a.conj() * val_b;
                }
            }
        } else {
            for (state, val_b) in &b.components {
                if let Some(val_a) = a.components.get(state) {
                    sum += val_a.conj() * val_b;
                }
            }
        }
        sum
    }

    pub fn scale_and_add(&mut self, other: &Self, scale: Complex64) {
        for (state, val) in &other.components {
            let entry = self
                .components
                .entry(state.clone())
                .or_insert(Complex64::new(0.0, 0.0));
            *entry += scale * val;
        }
        self.components.retain(|_, v| v.norm_sqr() > 1e-24);
    }

    /// The L2 norm `sqrt(<ψ|ψ>)`.
    pub fn norm(&self) -> f64 {
        Self::inner_product(self, self).re.max(0.0).sqrt()
    }

    /// Number of stored basis components.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// True if the state has no components (the zero vector).
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Drop components whose amplitude magnitude is `<= eps` (memory hygiene).
    pub fn prune(&mut self, eps: f64) {
        let thresh = eps * eps;
        self.components.retain(|_, v| v.norm_sqr() > thresh);
    }

    /// Keep only the `k` components with the largest `|amplitude|^2`, dropping the
    /// rest. A cheap bound on state growth; callers renormalize if needed.
    pub fn truncate_top_k(&mut self, k: usize) {
        if self.components.len() <= k {
            return;
        }
        let mut entries: Vec<(OuterState, Complex64)> = self.components.drain().collect();
        entries.sort_by(|a, b| {
            b.1.norm_sqr()
                .partial_cmp(&a.1.norm_sqr())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(k);
        self.components = entries.into_iter().collect();
    }

    pub fn create_boson(&self, idx: u32) -> Self {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(idx, 1);
        self.apply(&Operator::OuterBosonCreate(inner))
    }

    pub fn create_fermion(&self, idx: u32) -> Self {
        let mut inner = InnerFermionicState::vacuum();
        inner.modes.insert(idx);
        self.apply(&Operator::OuterFermionCreate(inner))
    }
}

#[derive(Debug, Clone)]
pub enum Operator {
    InnerBosonCreate(u32),                       // a_dag_i
    InnerBosonAnnihilate(u32),                   // a_i
    InnerFermionCreate(u32),                     // c_dag_i
    InnerFermionAnnihilate(u32),                 // c_i
    OuterBosonCreate(InnerBosonicState),         // A_dag_phi
    OuterBosonAnnihilate(InnerBosonicState),     // A_phi
    OuterFermionCreate(InnerFermionicState),     // C_dag_phi
    OuterFermionAnnihilate(InnerFermionicState), // C_phi
    /// The rank-1 vacuum projector `|0><0|` (the Mehler global prior). Applied
    /// to a state it keeps only the vacuum component's amplitude, collapsing
    /// everything else to zero. It is self-adjoint and idempotent.
    ProjectVacuum,
    /// The exact rank-1 projector `|psi><psi|` onto an arbitrary state `psi`
    /// (e.g. the dressed Mehler vacuum of `QFM.tex`, eq. (dressedvac)). The
    /// application uses the rank-1 shortcut `H|s> = <psi|s>·|psi>` — one inner
    /// product plus one scaled copy, `O(components + |psi|)` — instead of the
    /// `O(M²)` symbolic cross-term expansion `Σ ε_i ε_j B†_i P₀ B_j`. The
    /// caller must supply a **normalized** `psi` for `H² = H` to hold exactly.
    /// Self-adjoint by construction.
    ProjectOnto(std::sync::Arc<QuantumState>),
}

impl Operator {
    /// The Hermitian adjoint of a single ladder operator: creation and
    /// annihilation swap, the mode/universe label is preserved.
    pub fn adjoint(&self) -> Operator {
        match self {
            Operator::InnerBosonCreate(i) => Operator::InnerBosonAnnihilate(*i),
            Operator::InnerBosonAnnihilate(i) => Operator::InnerBosonCreate(*i),
            Operator::InnerFermionCreate(i) => Operator::InnerFermionAnnihilate(*i),
            Operator::InnerFermionAnnihilate(i) => Operator::InnerFermionCreate(*i),
            Operator::OuterBosonCreate(s) => Operator::OuterBosonAnnihilate(s.clone()),
            Operator::OuterBosonAnnihilate(s) => Operator::OuterBosonCreate(s.clone()),
            Operator::OuterFermionCreate(s) => Operator::OuterFermionAnnihilate(s.clone()),
            Operator::OuterFermionAnnihilate(s) => Operator::OuterFermionCreate(s.clone()),
            // |0><0| is Hermitian, so it is its own adjoint.
            Operator::ProjectVacuum => Operator::ProjectVacuum,
            // |psi><psi| is Hermitian, so it is its own adjoint.
            Operator::ProjectOnto(psi) => Operator::ProjectOnto(psi.clone()),
        }
    }

    pub fn apply_to_state(&self, state: &QuantumState) -> QuantumState {
        // Rank-1 fast path: |psi><psi| |s> = <psi|s> · |psi>. One inner product
        // over the (sparse) overlap of the two component maps, then one scaled
        // copy of psi — never the O(M²) expansion.
        if let Operator::ProjectOnto(psi) = self {
            // Iterate over the smaller map for the inner product <psi|s>.
            let overlap: Complex64 = if state.components.len() <= psi.components.len() {
                state
                    .components
                    .iter()
                    .filter_map(|(b, a)| psi.components.get(b).map(|p| p.conj() * a))
                    .sum()
            } else {
                psi.components
                    .iter()
                    .filter_map(|(b, p)| state.components.get(b).map(|a| p.conj() * a))
                    .sum()
            };
            let mut next_components = FxHashMap::default();
            if overlap != Complex64::new(0.0, 0.0) {
                next_components.reserve(psi.components.len());
                for (s, a) in &psi.components {
                    next_components.insert(s.clone(), overlap * a);
                }
            }
            return QuantumState {
                components: next_components,
            };
        }

        let mut next_components = FxHashMap::default();

        for (outer_basis, &amplitude) in &state.components {
            match self {
                // --- OUTER OPERATORS (Direct manipulation of universes) ---
                Operator::OuterBosonCreate(target_inner) => {
                    let mut new_outer = outer_basis.clone();
                    let n = *new_outer.bosonic.get(target_inner).unwrap_or(&0);
                    new_outer.bosonic.insert(target_inner.clone(), n + 1);
                    let multiplier = ((n + 1) as f64).sqrt();
                    *next_components
                        .entry(new_outer)
                        .or_insert(Complex64::new(0.0, 0.0)) += amplitude * multiplier;
                }
                Operator::OuterBosonAnnihilate(target_inner) => {
                    if let Some(&n) = outer_basis.bosonic.get(target_inner)
                        && n > 0
                    {
                        let mut new_outer = outer_basis.clone();
                        if n == 1 {
                            new_outer.bosonic.remove(target_inner);
                        } else {
                            new_outer.bosonic.insert(target_inner.clone(), n - 1);
                        }
                        let multiplier = (n as f64).sqrt();
                        *next_components
                            .entry(new_outer)
                            .or_insert(Complex64::new(0.0, 0.0)) += amplitude * multiplier;
                    }
                }
                Operator::OuterFermionCreate(target_inner) => {
                    if !outer_basis.fermionic.contains(target_inner) {
                        let mut new_outer = outer_basis.clone();
                        new_outer.fermionic.insert(target_inner.clone());
                        let sign = self.fermion_sign(outer_basis, target_inner);
                        *next_components
                            .entry(new_outer)
                            .or_insert(Complex64::new(0.0, 0.0)) += amplitude * sign;
                    }
                }
                Operator::OuterFermionAnnihilate(target_inner) => {
                    if outer_basis.fermionic.contains(target_inner) {
                        let mut new_outer = outer_basis.clone();
                        new_outer.fermionic.remove(target_inner);
                        let sign = self.fermion_sign(outer_basis, target_inner);
                        *next_components
                            .entry(new_outer)
                            .or_insert(Complex64::new(0.0, 0.0)) += amplitude * sign;
                    }
                }

                // --- INNER OPERATORS (Transitions within universes) ---
                Operator::InnerBosonCreate(mode) => {
                    self.apply_inner_one_body_bosonic(
                        outer_basis,
                        amplitude,
                        &mut next_components,
                        |inner| {
                            let mut next_inner = inner.clone();
                            let n = *next_inner.modes.get(mode).unwrap_or(&0);
                            next_inner.modes.insert(*mode, n + 1);
                            Some((next_inner, ((n + 1) as f64).sqrt()))
                        },
                    );
                }
                Operator::InnerBosonAnnihilate(mode) => {
                    self.apply_inner_one_body_bosonic(
                        outer_basis,
                        amplitude,
                        &mut next_components,
                        |inner| {
                            if let Some(&n) = inner.modes.get(mode)
                                && n > 0
                            {
                                let mut next_inner = inner.clone();
                                if n == 1 {
                                    next_inner.modes.remove(mode);
                                } else {
                                    next_inner.modes.insert(*mode, n - 1);
                                }
                                return Some((next_inner, (n as f64).sqrt()));
                            }
                            None
                        },
                    );
                }
                Operator::InnerFermionCreate(mode) => {
                    self.apply_inner_one_body_fermionic(
                        outer_basis,
                        amplitude,
                        &mut next_components,
                        |inner| {
                            if !inner.modes.contains(mode) {
                                let mut next_inner = inner.clone();
                                next_inner.modes.insert(*mode);
                                let sign = inner.modes.iter().take_while(|&m| m < mode).count();
                                let s = if sign % 2 == 1 { -1.0 } else { 1.0 };
                                return Some((next_inner, s));
                            }
                            None
                        },
                    );
                }
                Operator::InnerFermionAnnihilate(mode) => {
                    self.apply_inner_one_body_fermionic(
                        outer_basis,
                        amplitude,
                        &mut next_components,
                        |inner| {
                            if inner.modes.contains(mode) {
                                let mut next_inner = inner.clone();
                                next_inner.modes.remove(mode);
                                let sign = inner.modes.iter().take_while(|&m| m < mode).count();
                                let s = if sign % 2 == 1 { -1.0 } else { 1.0 };
                                return Some((next_inner, s));
                            }
                            None
                        },
                    );
                }

                // --- GLOBAL PROJECTOR: |0><0| (the Mehler prior) ---
                Operator::ProjectVacuum => {
                    // Keep only the strict vacuum component: both inner/outer
                    // universes empty. Everything carrying any mode is dropped.
                    if outer_basis.bosonic.is_empty() && outer_basis.fermionic.is_empty() {
                        *next_components
                            .entry(OuterState::vacuum())
                            .or_insert(Complex64::new(0.0, 0.0)) += amplitude;
                    }
                }

                // Handled by the rank-1 fast path before this loop.
                Operator::ProjectOnto(_) => unreachable!("ProjectOnto handled above"),
            }
        }
        QuantumState {
            components: next_components,
        }
    }

    fn apply_inner_one_body_bosonic<F>(
        &self,
        outer: &OuterState,
        amp: Complex64,
        next: &mut FxHashMap<OuterState, Complex64>,
        mut transition: F,
    ) where
        F: FnMut(&InnerBosonicState) -> Option<(InnerBosonicState, f64)>,
    {
        for (phi, &count) in &outer.bosonic {
            if let Some((phi_prime, factor)) = transition(phi) {
                if phi == &phi_prime {
                    let new_outer = outer.clone();
                    *next.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) +=
                        amp * factor * (count as f64);
                } else {
                    let mut new_outer = outer.clone();
                    if count == 1 {
                        new_outer.bosonic.remove(phi);
                    } else {
                        new_outer.bosonic.insert(phi.clone(), count - 1);
                    }

                    let n = *new_outer.bosonic.get(&phi_prime).unwrap_or(&0);
                    new_outer.bosonic.insert(phi_prime, n + 1);

                    let multiplier = (count as f64).sqrt() * ((n + 1) as f64).sqrt() * factor;
                    *next.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) += amp * multiplier;
                }
            }
        }
    }

    fn apply_inner_one_body_fermionic<F>(
        &self,
        outer: &OuterState,
        amp: Complex64,
        next: &mut FxHashMap<OuterState, Complex64>,
        mut transition: F,
    ) where
        F: FnMut(&InnerFermionicState) -> Option<(InnerFermionicState, f64)>,
    {
        for phi in &outer.fermionic {
            if let Some((phi_prime, factor)) = transition(phi) {
                if phi == &phi_prime {
                    let new_outer = outer.clone();
                    *next.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) += amp * factor;
                } else if !outer.fermionic.contains(&phi_prime) {
                    let mut new_outer = outer.clone();
                    new_outer.fermionic.remove(phi);
                    new_outer.fermionic.insert(phi_prime.clone());

                    let s1 = outer.fermionic.iter().take_while(|&s| s < phi).count();
                    let s2 = new_outer
                        .fermionic
                        .iter()
                        .take_while(|&s| s < &phi_prime)
                        .count();
                    let sign = if (s1 + s2) % 2 == 1 { -1.0 } else { 1.0 };

                    *next.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) +=
                        amp * factor * sign;
                }
            }
        }
    }

    fn fermion_sign(&self, outer: &OuterState, target: &InnerFermionicState) -> f64 {
        let count = outer.fermionic.iter().take_while(|&s| s < target).count();
        if count % 2 == 1 { -1.0 } else { 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn adjoint_involution_create(mode_idx in 0u32..1000) {
            let op = Operator::InnerBosonCreate(mode_idx);
            let double_adj = op.adjoint().adjoint();
            assert!(matches!(double_adj, Operator::InnerBosonCreate(i) if i == mode_idx));
        }

        #[test]
        fn adjoint_involution_annihilate(mode_idx in 0u32..1000) {
            let op = Operator::InnerBosonAnnihilate(mode_idx);
            let double_adj = op.adjoint().adjoint();
            assert!(matches!(double_adj, Operator::InnerBosonAnnihilate(i) if i == mode_idx));
        }

        #[test]
        fn create_adjoint_is_annihilate(mode_idx in 0u32..1000) {
            let create = Operator::InnerBosonCreate(mode_idx);
            assert!(matches!(create.adjoint(), Operator::InnerBosonAnnihilate(i) if i == mode_idx));
        }

        #[test]
        fn annihilate_adjoint_is_create(mode_idx in 0u32..1000) {
            let annihilate = Operator::InnerBosonAnnihilate(mode_idx);
            assert!(matches!(annihilate.adjoint(), Operator::InnerBosonCreate(i) if i == mode_idx));
        }
    }

    #[test]
    fn project_vacuum_self_adjoint() {
        assert!(matches!(Operator::ProjectVacuum.adjoint(), Operator::ProjectVacuum));
    }

    #[test]
    fn vacuum_initialization_single_component() {
        let vac = QuantumState::vacuum();
        assert_eq!(vac.components.len(), 1);
        let (outer, amp) = vac.components.iter().next().unwrap();
        assert!((amp.re - 1.0).abs() < 1e-12);
        assert!(amp.im.abs() < 1e-12);
        // NOTE: OuterState::vacuum() currently has an empty bosonic map
        // (0 inner universes). AGENTS.md documents that it should have at
        // least one empty inner universe, but the implementation does not
        // yet enforce this. The test asserts the current behavior.
        assert_eq!(outer.bosonic.len(), 0);
        assert_eq!(outer.fermionic.len(), 0);
    }

    #[test]
    fn test_inner_boson_transition() {
        // Initial state: One universe in the vacuum state.
        // |Psi>_outer = |1_vac>
        let vac = InnerBosonicState::vacuum();
        let mut initial = QuantumState::vacuum();
        initial = initial.apply(&Operator::OuterBosonCreate(vac.clone()));

        let op_inner = Operator::InnerBosonCreate(0);
        let final_state = initial.apply(&op_inner);

        assert_eq!(final_state.components.len(), 1);
        let (outer, &amp) = final_state.components.iter().next().unwrap();

        // amp should be 1.0 * sqrt(1_outer) * sqrt(0+1_inner) = 1.0
        assert!((amp.re - 1.0).abs() < 1e-10);

        let phi_prime = outer.bosonic.keys().next().unwrap();
        assert_eq!(phi_prime.modes.get(&0), Some(&1));
    }

    #[test]
    fn test_fermion_parity_outer() {
        let phi1 = InnerFermionicState::vacuum();
        let mut phi2 = InnerFermionicState::vacuum();
        phi2.modes.insert(0); // |1_0>

        let state = QuantumState::vacuum()
            .apply(&Operator::OuterFermionCreate(phi2.clone()))
            .apply(&Operator::OuterFermionCreate(phi1.clone()));

        let op_ann_phi2 = Operator::OuterFermionAnnihilate(phi2);
        let final_state = state.apply(&op_ann_phi2);

        let &amp = final_state.components.values().next().unwrap();
        assert!((amp.re + 1.0).abs() < 1e-10); // Expected -1.0
    }
}

#[derive(Debug)]
pub struct Hamiltonian {
    pub terms: Vec<(Complex64, Vec<Operator>)>,
}

impl Hamiltonian {
    /// The Hermitian adjoint `H†`. For each term `c · O_1 O_2 … O_n`, the adjoint
    /// is `conj(c) · O_n† … O_2† O_1†` (conjugate the coefficient, reverse the
    /// operator string, and adjoint each operator).
    pub fn adjoint(&self) -> Hamiltonian {
        let terms = self
            .terms
            .iter()
            .map(|(coeff, ops)| {
                let adj_ops = ops.iter().rev().map(|op| op.adjoint()).collect();
                (coeff.conj(), adj_ops)
            })
            .collect();
        Hamiltonian { terms }
    }

    pub fn apply(&self, state: &QuantumState) -> QuantumState {
        let mut final_state = QuantumState {
            components: FxHashMap::default(),
        };
        for (coeff, ops) in &self.terms {
            let mut current_state = state.clone();
            for op in ops.iter().rev() {
                current_state = op.apply_to_state(&current_state);
            }
            final_state.scale_and_add(&current_state, *coeff);
        }
        final_state
    }
}

pub mod cas;
pub use cas::{
    CasError, ExpansionLimits, compile_expression, compile_expression_bounded, compile_to_fock,
    compile_to_fock_bounded,
};

#[cfg(feature = "latex")]
pub mod latex;
#[cfg(feature = "latex")]
pub use latex::compile_latex;

#[cfg(feature = "latex")]
pub mod typst_math;
#[cfg(feature = "latex")]
pub use typst_math::compile_typst_math;

pub mod field_theory;
pub use field_theory::*;

pub mod models;
pub use models::*;

// --- Operators Builders ---

pub fn inner_boson_create(idx: u32) -> Expression {
    Expression::symbol(&format!("c_{}", idx))
}

pub fn inner_boson_annihilate(idx: u32) -> Expression {
    Expression::symbol(&format!("a_{}", idx))
}

pub fn inner_fermion_create(idx: u32) -> Expression {
    Expression::symbol(&format!("c_f{}", idx))
}

pub fn inner_fermion_annihilate(idx: u32) -> Expression {
    Expression::symbol(&format!("a_f{}", idx))
}

pub fn outer_boson_create(idx: u32) -> Expression {
    Expression::symbol(&format!("C_{}", idx))
}

pub fn outer_boson_annihilate(idx: u32) -> Expression {
    Expression::symbol(&format!("A_{}", idx))
}

pub fn outer_fermion_create(idx: u32) -> Expression {
    Expression::symbol(&format!("C_f{}", idx))
}

pub fn outer_fermion_annihilate(idx: u32) -> Expression {
    Expression::symbol(&format!("A_f{}", idx))
}

/// Re-export the symbolic engine for high-level operator building.
pub use quantrs2_symengine_pure as symengine;
pub use quantrs2_symengine_pure::Expression;

#[cfg(test)]
mod unit_tests;
