use crate::ccg::DerivationTree;
use std::collections::HashMap;

pub type WorldId = usize;

#[derive(Debug, Clone)]
pub struct World {
    pub id: WorldId,
    pub probability: f64,
    pub tree: DerivationTree,
}

#[derive(Debug, Clone)]
pub struct TriggerEntry {
    pub word: String,
    pub category: String,
    pub splits: Vec<(f64, String)>,
}

pub struct TriggerTable {
    entries: Vec<TriggerEntry>,
}

impl Default for TriggerTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TriggerTable {
    pub fn new() -> Self {
        let entries = vec![
            TriggerEntry {
                word: "probably".to_string(),
                category: "S/S".to_string(),
                splits: vec![(0.8, "identity".to_string()), (0.2, "negate".to_string())],
            },
            TriggerEntry {
                word: "might".to_string(),
                category: "(S\\NP)/(S\\NP)".to_string(),
                splits: vec![(0.5, "identity".to_string()), (0.5, "null".to_string())],
            },
            TriggerEntry {
                word: "usually".to_string(),
                category: "S/S".to_string(),
                splits: vec![(0.9, "identity".to_string()), (0.1, "negate".to_string())],
            },
        ];
        Self { entries }
    }

    pub fn lookup(&self, word: &str) -> Option<&TriggerEntry> {
        self.entries.iter().find(|e| e.word == word)
    }

    pub fn is_trigger(&self, word: &str) -> bool {
        self.entries.iter().any(|e| e.word == word)
    }

    pub fn count_triggers(&self, tree: &DerivationTree) -> usize {
        match tree {
            DerivationTree::Leaf { word, .. } => {
                if self.is_trigger(word) {
                    1
                } else {
                    0
                }
            }
            DerivationTree::Application { left, right, .. }
            | DerivationTree::Composition { left, right, .. } => {
                self.count_triggers(left) + self.count_triggers(right)
            }
        }
    }
}

pub const MAX_TRIGGERS: usize = 4;

pub fn split_l1(tree: &DerivationTree, triggers: &TriggerTable) -> Vec<(f64, DerivationTree)> {
    let trigger_count = triggers.count_triggers(tree);
    if trigger_count > MAX_TRIGGERS {
        eprintln!(
            "warning: sentence has {} L1 triggers, exceeding cap of {}",
            trigger_count, MAX_TRIGGERS
        );
        return vec![(1.0, tree.clone())];
    }

    if trigger_count == 0 {
        return vec![(1.0, tree.clone())];
    }

    split_tree(tree, triggers)
}

fn split_tree(tree: &DerivationTree, triggers: &TriggerTable) -> Vec<(f64, DerivationTree)> {
    match tree {
        DerivationTree::Leaf { word, category } => {
            if let Some(trigger) = triggers.lookup(word) {
                trigger
                    .splits
                    .iter()
                    .map(|(prob, action)| {
                        let new_tree = match action.as_str() {
                            "negate" => DerivationTree::Leaf {
                                word: format!("NOT_{}", word),
                                category: category.clone(),
                            },
                            "null" => DerivationTree::Leaf {
                                word: "NULL".to_string(),
                                category: category.clone(),
                            },
                            _ => tree.clone(),
                        };
                        (*prob, new_tree)
                    })
                    .collect()
            } else {
                vec![(1.0, tree.clone())]
            }
        }
        DerivationTree::Application {
            direction,
            result_category,
            left,
            right,
        } => {
            let left_worlds = split_tree(left, triggers);
            let right_worlds = split_tree(right, triggers);

            let mut results = Vec::new();
            for (lp, lt) in &left_worlds {
                for (rp, rt) in &right_worlds {
                    results.push((
                        lp * rp,
                        DerivationTree::Application {
                            direction: direction.clone(),
                            result_category: result_category.clone(),
                            left: Box::new(lt.clone()),
                            right: Box::new(rt.clone()),
                        },
                    ));
                }
            }
            results
        }
        DerivationTree::Composition {
            direction,
            result_category,
            left,
            right,
        } => {
            let left_worlds = split_tree(left, triggers);
            let right_worlds = split_tree(right, triggers);

            let mut results = Vec::new();
            for (lp, lt) in &left_worlds {
                for (rp, rt) in &right_worlds {
                    results.push((
                        lp * rp,
                        DerivationTree::Composition {
                            direction: direction.clone(),
                            result_category: result_category.clone(),
                            left: Box::new(lt.clone()),
                            right: Box::new(rt.clone()),
                        },
                    ));
                }
            }
            results
        }
    }
}

pub fn aggregate_results(worlds: &[(f64, String)]) -> Vec<(String, f64)> {
    let mut map: HashMap<String, f64> = HashMap::new();
    for (prob, result) in worlds {
        *map.entry(result.clone()).or_insert(0.0) += *prob;
    }
    let mut results: Vec<(String, f64)> = map.into_iter().collect();
    // Descending weight, then by key. The tiebreaker is load-bearing: the input
    // is a `HashMap`, so equal-weight keys come out in a different order every
    // process, and `engram::l1keys::weighted_key_set` inherits that ordering
    // while documenting its result as "sorted by descending weight". Anything
    // that hashes or compares the key set downstream then stops being
    // reproducible — the crate elsewhere insists on byte-stable serialization, so
    // a nondeterministic order here is a contradiction rather than a detail.
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    results
}

pub fn verify_world_probabilities(worlds: &[(f64, DerivationTree)], tolerance: f64) -> bool {
    let sum: f64 = worlds.iter().map(|(p, _)| p).sum();
    (sum - 1.0).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_table_lookup() {
        let table = TriggerTable::new();
        assert!(table.is_trigger("probably"));
        assert!(table.is_trigger("might"));
        assert!(!table.is_trigger("John"));
    }

    #[test]
    fn test_split_no_triggers() {
        let table = TriggerTable::new();
        let tree = DerivationTree::Leaf {
            word: "John".to_string(),
            category: crate::ccg::CCGCategory::NP,
        };
        let worlds = split_l1(&tree, &table);
        assert_eq!(worlds.len(), 1);
        assert!((worlds[0].0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_verify_world_probabilities() {
        let table = TriggerTable::new();
        let tree = DerivationTree::Leaf {
            word: "probably".to_string(),
            category: crate::ccg::CCGCategory::S,
        };
        let worlds = split_l1(&tree, &table);
        assert!(verify_world_probabilities(&worlds, 1e-9));
    }
    /// Equal weights must not come out in `HashMap` iteration order.
    ///
    /// The tiebreaker is what makes the output reproducible: without it, two keys
    /// with the same weight swap places between processes, and every caller that
    /// hashes or serializes the result sees a different value each run.
    #[test]
    fn equal_weights_come_back_in_a_stable_order() {
        let worlds: Vec<(f64, String)> = vec![
            (0.25, "charlie".into()),
            (0.25, "alpha".into()),
            (0.25, "delta".into()),
            (0.25, "bravo".into()),
        ];
        let first = aggregate_results(&worlds);
        assert_eq!(
            first,
            vec![
                ("alpha".to_string(), 0.25),
                ("bravo".to_string(), 0.25),
                ("charlie".to_string(), 0.25),
                ("delta".to_string(), 0.25),
            ],
            "ties break by key, ascending"
        );
        // Re-shuffling the input must not change the output.
        let mut shuffled = worlds.clone();
        shuffled.reverse();
        assert_eq!(aggregate_results(&shuffled), first);
    }

    /// Weight still dominates the tiebreaker.
    #[test]
    fn weight_outranks_the_tiebreaker() {
        let worlds: Vec<(f64, String)> = vec![
            (0.1, "zebra".into()),
            (0.9, "apple".into()),
            (0.5, "mango".into()),
        ];
        let out = aggregate_results(&worlds);
        assert_eq!(
            out.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            vec!["apple", "mango", "zebra"]
        );
    }
}
