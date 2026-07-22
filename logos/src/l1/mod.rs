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
                if self.is_trigger(word) { 1 } else { 0 }
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
        eprintln!("warning: sentence has {} L1 triggers, exceeding cap of {}", trigger_count, MAX_TRIGGERS);
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
                trigger.splits.iter()
                    .map(|(prob, action)| {
                        let new_tree = match action.as_str() {
                            "negate" => {
                                DerivationTree::Leaf {
                                    word: format!("NOT_{}", word),
                                    category: category.clone(),
                                }
                            }
                            "null" => {
                                DerivationTree::Leaf {
                                    word: "NULL".to_string(),
                                    category: category.clone(),
                                }
                            }
                            _ => tree.clone(),
                        };
                        (*prob, new_tree)
                    })
                    .collect()
            } else {
                vec![(1.0, tree.clone())]
            }
        }
        DerivationTree::Application { direction, result_category, left, right } => {
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
        DerivationTree::Composition { direction, result_category, left, right } => {
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
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
}
