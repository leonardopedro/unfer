use nested_fock_algebra::{OuterState, QuantumState};
use rustc_hash::FxHashMap;

use crate::linalg::SirkError;

/// StateDictionary maps OuterState topologies to unique indices.
/// This allows flattening the sparse FxHashMap into a dense vector for GPU processing.
///
/// The map is a **linear layout**: the bijection `OuterState ↔ index` is what
/// makes the dense GPU tensor a faithful image of the sparse state. The GPU.md
/// layout-verification idea is realized here as an on-the-fly invariant check
/// ([`StateDictionary::check_bijective`]) plus a `debug_assert!` at every
/// CPU→GPU upload point, so a broken layout fails loudly instead of silently
/// aliasing two basis states in the Gram matrix.
#[derive(Default)]
pub struct StateDictionary {
    pub state_to_index: FxHashMap<OuterState, usize>,
    pub index_to_state: Vec<OuterState>,
}

impl StateDictionary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_insert(&mut self, state: OuterState) -> usize {
        if let Some(&index) = self.state_to_index.get(&state) {
            index
        } else {
            let index = self.index_to_state.len();
            self.state_to_index.insert(state.clone(), index);
            self.index_to_state.push(state);
            index
        }
    }

    pub fn register(&mut self, state: &QuantumState) {
        for outer in state.components.keys() {
            self.get_or_insert(outer.clone());
        }
    }

    pub fn len(&self) -> usize {
        self.index_to_state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index_to_state.is_empty()
    }

    /// True iff the layout is a bijection: every index `i` maps back to `i`
    /// through `state_to_index`, and the two tables have the same size.
    ///
    /// The forward direction (each state maps to a single index) is guaranteed
    /// by `FxHashMap` semantics; this checks the backward direction and the
    /// size agreement, which nothing else enforces.
    pub fn is_bijective(&self) -> bool {
        self.state_to_index.len() == self.index_to_state.len()
            && self
                .index_to_state
                .iter()
                .enumerate()
                .all(|(i, s)| self.state_to_index.get(s).copied() == Some(i))
    }

    /// Returns a machine-readable description of the first bijectivity
    /// violation, if any: the index whose state does not map back to itself.
    pub fn bijective_violation(&self) -> Option<(usize, String)> {
        for (i, s) in self.index_to_state.iter().enumerate() {
            match self.state_to_index.get(s) {
                Some(&j) if j == i => {}
                other => {
                    return Some((
                        i,
                        format!(
                            "index {i} holds a state that re-inserts to {other:?} \
                                (expected Some({i})) — two distinct basis states alias \
                                one dense slot",
                        ),
                    ));
                }
            }
        }
        if self.state_to_index.len() != self.index_to_state.len() {
            return Some((
                self.index_to_state.len(),
                format!(
                    "forward table has {} entries but the backward table has {} — \
                        an index has no state or a state has no index",
                    self.state_to_index.len(),
                    self.index_to_state.len(),
                ),
            ));
        }
        None
    }

    /// Check the layout bijection, returning the machine-readable violation
    /// as a [`SirkError::LayoutNotBijective`] instead of panicking.
    pub fn check_bijective(&self) -> Result<(), SirkError> {
        match self.bijective_violation() {
            None => Ok(()),
            Some((index, message)) => Err(SirkError::LayoutNotBijective { index, message }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::InnerBosonicState;

    fn state_with(mode: u32, occ: u32) -> OuterState {
        let mut inner = InnerBosonicState::vacuum();
        inner.modes.insert(mode, occ);
        let mut s = OuterState::vacuum();
        s.bosonic.insert(inner, 1);
        s
    }

    #[test]
    fn empty_dictionary_is_bijective() {
        let reg = StateDictionary::new();
        assert!(reg.is_bijective());
        assert!(reg.check_bijective().is_ok());
    }

    #[test]
    fn grown_dictionary_stays_bijective() {
        let mut reg = StateDictionary::new();
        for mode in 0..8 {
            let idx = reg.get_or_insert(state_with(mode, 1));
            assert_eq!(idx, mode as usize);
        }
        assert_eq!(reg.len(), 8);
        assert!(reg.is_bijective());
        assert!(reg.check_bijective().is_ok());
        // Re-inserting existing states returns the same index and keeps the
        // bijection (no duplicate entries, no index drift).
        for mode in 0..8 {
            let idx = reg.get_or_insert(state_with(mode, 1));
            assert_eq!(idx, mode as usize);
        }
        assert!(reg.is_bijective());
    }

    #[test]
    fn duplicate_registration_does_not_break_bijection() {
        let mut reg = StateDictionary::new();
        let s = state_with(3, 1);
        let a = reg.get_or_insert(s.clone());
        let b = reg.get_or_insert(s.clone());
        assert_eq!(a, b);
        assert_eq!(reg.len(), 1);
        assert!(reg.is_bijective());
        assert!(reg.check_bijective().is_ok());
    }

    #[test]
    fn corrupted_forward_table_is_detected() {
        let mut reg = StateDictionary::new();
        reg.get_or_insert(state_with(0, 1));
        reg.get_or_insert(state_with(1, 1));
        // Corrupt the forward table: make both states point at index 0.
        let s1 = state_with(1, 1);
        reg.state_to_index.insert(s1, 0);
        assert!(!reg.is_bijective());
        let err = reg.check_bijective().unwrap_err();
        match err {
            SirkError::LayoutNotBijective { index, message } => {
                assert_eq!(index, 1);
                assert!(message.contains("alias"));
            }
            other => panic!("expected LayoutNotBijective, got {other:?}"),
        }
    }

    #[test]
    fn size_mismatch_between_tables_is_detected() {
        let mut reg = StateDictionary::new();
        reg.get_or_insert(state_with(0, 1));
        // Drop a backward entry so the tables disagree in size.
        reg.index_to_state.pop();
        assert!(!reg.is_bijective());
        let err = reg.check_bijective().unwrap_err();
        match err {
            SirkError::LayoutNotBijective { index, message } => {
                assert_eq!(index, 0);
                assert!(message.contains("forward table"));
            }
            other => panic!("expected LayoutNotBijective, got {other:?}"),
        }
    }
}
