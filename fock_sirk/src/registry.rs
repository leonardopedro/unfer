use nested_fock_algebra::{OuterState, QuantumState};
use rustc_hash::FxHashMap;

/// StateDictionary maps OuterState topologies to unique indices.
/// This allows flattening the sparse FxHashMap into a dense vector for GPU processing.
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
}
