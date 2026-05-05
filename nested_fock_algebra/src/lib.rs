use num_complex::Complex64;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};

// --- LEVEL 1: The Inner Fock Space ---

/// A configuration of an inner Bosonic universe.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InnerBosonicState {
    pub modes: BTreeMap<u32, u32>,
}

impl InnerBosonicState {
    pub fn vacuum() -> Self {
        Self { modes: BTreeMap::new() }
    }
}

/// A configuration of an inner Fermionic universe.
/// Deriving Ord and PartialOrd guarantees Canonical Ordering for Fermion signs.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InnerFermionicState {
    pub modes: BTreeSet<u32>, 
}

impl InnerFermionicState {
    pub fn vacuum() -> Self {
        Self { modes: BTreeSet::new() }
    }
}

// --- LEVEL 2: The Outer Fock Space ---

/// The state of the "Multiverse" / Outer Space, split into disjoint bosonic/fermionic universes
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
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

/// A superposition of Outer States with complex amplitudes
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub components: FxHashMap<OuterState, Complex64>,
}

impl QuantumState {
    pub fn vacuum() -> Self {
        let mut components = FxHashMap::default();
        components.insert(OuterState::vacuum(), Complex64::new(1.0, 0.0));
        Self { components }
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
            let entry = self.components.entry(state.clone()).or_insert(Complex64::new(0.0, 0.0));
            *entry += scale * val;
        }
        self.components.retain(|_, v| v.norm_sqr() > 1e-24);
    }
}

#[derive(Debug, Clone)]
pub enum Operator {
    InnerBosonCreate(u32),       // a_dag_i
    InnerBosonAnnihilate(u32),   // a_i
    InnerFermionCreate(u32),     // c_dag_i
    InnerFermionAnnihilate(u32), // c_i
    OuterBosonCreate(InnerBosonicState), // A_dag_phi
    OuterBosonAnnihilate(InnerBosonicState), // A_phi
    OuterFermionCreate(InnerFermionicState), // C_dag_phi
    OuterFermionAnnihilate(InnerFermionicState), // C_phi
}

impl Operator {
    pub fn apply_to_state(&self, state: &QuantumState) -> QuantumState {
        let mut next_components = FxHashMap::default();

        for (outer_basis, &amplitude) in &state.components {
            match self {
                // --- OUTER OPERATORS (Direct manipulation of universes) ---
                
                Operator::OuterBosonCreate(target_inner) => {
                    let mut new_outer = outer_basis.clone();
                    let n = *new_outer.bosonic.get(target_inner).unwrap_or(&0);
                    new_outer.bosonic.insert(target_inner.clone(), n + 1);
                    let multiplier = ((n + 1) as f64).sqrt();
                    *next_components.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) 
                        += amplitude * multiplier;
                }
                Operator::OuterBosonAnnihilate(target_inner) => {
                    if let Some(&n) = outer_basis.bosonic.get(target_inner) {
                        if n > 0 {
                            let mut new_outer = outer_basis.clone();
                            if n == 1 {
                                new_outer.bosonic.remove(target_inner);
                            } else {
                                new_outer.bosonic.insert(target_inner.clone(), n - 1);
                            }
                            let multiplier = (n as f64).sqrt();
                            *next_components.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) 
                                += amplitude * multiplier;
                        }
                    }
                }
                Operator::OuterFermionCreate(target_inner) => {
                    if !outer_basis.fermionic.contains(target_inner) {
                        let mut new_outer = outer_basis.clone();
                        new_outer.fermionic.insert(target_inner.clone());
                        let sign = self.fermion_sign(outer_basis, target_inner);
                        *next_components.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) 
                            += amplitude * sign;
                    }
                }
                Operator::OuterFermionAnnihilate(target_inner) => {
                    if outer_basis.fermionic.contains(target_inner) {
                        let mut new_outer = outer_basis.clone();
                        new_outer.fermionic.remove(target_inner);
                        let sign = self.fermion_sign(outer_basis, target_inner);
                        *next_components.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) 
                            += amplitude * sign;
                    }
                }

                // --- INNER OPERATORS (Transitions within universes) ---
                
                Operator::InnerBosonCreate(mode) => {
                    self.apply_inner_one_body_bosonic(outer_basis, amplitude, &mut next_components, |inner| {
                        let mut next_inner = inner.clone();
                        let n = *next_inner.modes.get(mode).unwrap_or(&0);
                        next_inner.modes.insert(*mode, n + 1);
                        Some((next_inner, ((n + 1) as f64).sqrt()))
                    });
                }
                Operator::InnerBosonAnnihilate(mode) => {
                    self.apply_inner_one_body_bosonic(outer_basis, amplitude, &mut next_components, |inner| {
                        if let Some(&n) = inner.modes.get(mode) {
                            if n > 0 {
                                let mut next_inner = inner.clone();
                                if n == 1 { next_inner.modes.remove(mode); }
                                else { next_inner.modes.insert(*mode, n - 1); }
                                return Some((next_inner, (n as f64).sqrt()));
                            }
                        }
                        None
                    });
                }
                Operator::InnerFermionCreate(mode) => {
                    self.apply_inner_one_body_fermionic(outer_basis, amplitude, &mut next_components, |inner| {
                        if !inner.modes.contains(mode) {
                            let mut next_inner = inner.clone();
                            next_inner.modes.insert(*mode);
                            let sign = inner.modes.iter().take_while(|&m| m < mode).count();
                            let s = if sign % 2 == 1 { -1.0 } else { 1.0 };
                            return Some((next_inner, s));
                        }
                        None
                    });
                }
                Operator::InnerFermionAnnihilate(mode) => {
                    self.apply_inner_one_body_fermionic(outer_basis, amplitude, &mut next_components, |inner| {
                        if inner.modes.contains(mode) {
                            let mut next_inner = inner.clone();
                            next_inner.modes.remove(mode);
                            let sign = inner.modes.iter().take_while(|&m| m < mode).count();
                            let s = if sign % 2 == 1 { -1.0 } else { 1.0 };
                            return Some((next_inner, s));
                        }
                        None
                    });
                }
            }
        }
        QuantumState { components: next_components }
    }

    fn apply_inner_one_body_bosonic<F>(
        &self,
        outer: &OuterState,
        amp: Complex64,
        next: &mut FxHashMap<OuterState, Complex64>,
        mut transition: F
    ) where F: FnMut(&InnerBosonicState) -> Option<(InnerBosonicState, f64)> {
        for (phi, &count) in &outer.bosonic {
            if let Some((phi_prime, factor)) = transition(phi) {
                if phi == &phi_prime {
                    let new_outer = outer.clone();
                    *next.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) 
                        += amp * factor * (count as f64);
                } else {
                    let mut new_outer = outer.clone();
                    if count == 1 { new_outer.bosonic.remove(phi); }
                    else { new_outer.bosonic.insert(phi.clone(), count - 1); }
                    
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
        mut transition: F
    ) where F: FnMut(&InnerFermionicState) -> Option<(InnerFermionicState, f64)> {
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
                    let s2 = new_outer.fermionic.iter().take_while(|&s| s < &phi_prime).count();
                    let sign = if (s1 + s2) % 2 == 1 { -1.0 } else { 1.0 };
                    
                    *next.entry(new_outer).or_insert(Complex64::new(0.0, 0.0)) += amp * factor * sign;
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
    pub fn apply(&self, state: &QuantumState) -> QuantumState {
        let mut final_state = QuantumState { components: FxHashMap::default() };
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
pub use cas::{compile_expression, compile_to_fock};

/// Re-export the symbolic engine for high-level operator building.
pub use quantrs2_symengine_pure as symengine;
pub use quantrs2_symengine_pure::Expression;
