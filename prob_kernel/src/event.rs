use nested_fock_algebra::OuterState;
use unfer_protocol::{Cmp, EventPredicate};

/// Evaluate whether an `OuterState` satisfies the given `EventPredicate`.
///
/// This is the pure, exhaustive Born-rule event matcher. `probability` sums
/// `|⟨s|ψ⟩|²` over states where `matches` returns true; `condition` retains
/// only those states.
pub fn matches(outer: &OuterState, pred: &EventPredicate) -> bool {
    match pred {
        EventPredicate::BosonModeTotal { mode, cmp, value } => {
            let total: u32 = outer
                .bosonic
                .iter()
                .map(|(inner, &count)| inner.modes.get(mode).copied().unwrap_or(0) * count)
                .sum();
            cmp_eval(*cmp, total, *value)
        }

        EventPredicate::FermionModePresent { mode } => {
            outer.fermionic.iter().any(|f| f.modes.contains(mode))
        }

        EventPredicate::BosonUniverseCount { cmp, value } => {
            let count: u32 = outer.bosonic.values().copied().sum();
            cmp_eval(*cmp, count, *value)
        }

        EventPredicate::FermionUniverseCount { cmp, value } => {
            let count = outer.fermionic.len() as u32;
            cmp_eval(*cmp, count, *value)
        }

        EventPredicate::Vacuum => outer.bosonic.is_empty() && outer.fermionic.is_empty(),

        EventPredicate::And { parts } => parts.iter().all(|p| matches(outer, p)),

        EventPredicate::Or { parts } => parts.iter().any(|p| matches(outer, p)),

        EventPredicate::Not { inner } => !matches(outer, inner),
    }
}

fn cmp_eval(cmp: Cmp, lhs: u32, rhs: u32) -> bool {
    match cmp {
        Cmp::Eq => lhs == rhs,
        Cmp::Ge => lhs >= rhs,
        Cmp::Le => lhs <= rhs,
        Cmp::Gt => lhs > rhs,
        Cmp::Lt => lhs < rhs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nested_fock_algebra::{InnerBosonicState, InnerFermionicState};
    use std::collections::{BTreeMap, BTreeSet};
    use unfer_protocol::Cmp;

    // H11: the Born-rule matcher is the pure, exhaustive predicate evaluator.
    // These tests exercise every predicate arm directly (they previously only
    // ran through the FFI probability/condition path, leaving the matcher's
    // line coverage near-zero in the kernel's own unit suite).

    fn bosonic(mode: u32, count: u32) -> OuterState {
        let mut modes = BTreeMap::new();
        modes.insert(mode, count);
        OuterState {
            bosonic: BTreeMap::from([(InnerBosonicState { modes }, 1)]),
            fermionic: BTreeSet::new(),
        }
    }

    fn fermionic(mode: u32) -> OuterState {
        OuterState {
            bosonic: BTreeMap::new(),
            fermionic: BTreeSet::from([InnerFermionicState {
                modes: BTreeSet::from([mode]),
            }]),
        }
    }

    #[test]
    fn boson_mode_total_evaluates_cmp() {
        let s = bosonic(0, 2);
        assert!(matches(
            &s,
            &EventPredicate::BosonModeTotal {
                mode: 0,
                cmp: Cmp::Ge,
                value: 1
            }
        ));
        assert!(matches(
            &s,
            &EventPredicate::BosonModeTotal {
                mode: 0,
                cmp: Cmp::Eq,
                value: 2
            }
        ));
        assert!(!matches(
            &s,
            &EventPredicate::BosonModeTotal {
                mode: 0,
                cmp: Cmp::Lt,
                value: 2
            }
        ));
        // An absent mode counts as 0.
        assert!(matches(
            &s,
            &EventPredicate::BosonModeTotal {
                mode: 1,
                cmp: Cmp::Eq,
                value: 0
            }
        ));
        assert!(matches(
            &s,
            &EventPredicate::BosonModeTotal {
                mode: 1,
                cmp: Cmp::Le,
                value: 1
            }
        ));
        assert!(!matches(
            &s,
            &EventPredicate::BosonModeTotal {
                mode: 1,
                cmp: Cmp::Gt,
                value: 0
            }
        ));
    }

    #[test]
    fn fermion_mode_present() {
        assert!(matches(
            &fermionic(1),
            &EventPredicate::FermionModePresent { mode: 1 }
        ));
        assert!(!matches(
            &fermionic(1),
            &EventPredicate::FermionModePresent { mode: 2 }
        ));
    }

    #[test]
    fn universe_count_predicates() {
        let two_bosons = bosonic(0, 2);
        assert!(matches(
            &two_bosons,
            &EventPredicate::BosonUniverseCount {
                cmp: Cmp::Eq,
                value: 1
            }
        ));
        assert!(matches(
            &two_bosons,
            &EventPredicate::BosonUniverseCount {
                cmp: Cmp::Ge,
                value: 1
            }
        ));
        assert!(matches(
            &two_bosons,
            &EventPredicate::BosonUniverseCount {
                cmp: Cmp::Le,
                value: 2
            }
        ));
        assert!(!matches(
            &two_bosons,
            &EventPredicate::BosonUniverseCount {
                cmp: Cmp::Lt,
                value: 1
            }
        ));
        assert!(matches(
            &fermionic(3),
            &EventPredicate::FermionUniverseCount {
                cmp: Cmp::Eq,
                value: 1
            }
        ));
        assert!(matches(
            &fermionic(3),
            &EventPredicate::FermionUniverseCount {
                cmp: Cmp::Gt,
                value: 0
            }
        ));
        assert!(!matches(
            &fermionic(3),
            &EventPredicate::FermionUniverseCount {
                cmp: Cmp::Lt,
                value: 1
            }
        ));
    }

    #[test]
    fn vacuum_predicate() {
        assert!(matches(&OuterState::vacuum(), &EventPredicate::Vacuum));
        assert!(!matches(&bosonic(0, 1), &EventPredicate::Vacuum));
        assert!(!matches(&fermionic(0), &EventPredicate::Vacuum));
    }

    #[test]
    fn boolean_combinators() {
        let s = bosonic(0, 2);
        let one = EventPredicate::BosonModeTotal {
            mode: 0,
            cmp: Cmp::Ge,
            value: 1,
        };
        let two = EventPredicate::BosonModeTotal {
            mode: 0,
            cmp: Cmp::Eq,
            value: 2,
        };
        let three = EventPredicate::BosonModeTotal {
            mode: 0,
            cmp: Cmp::Eq,
            value: 3,
        };
        assert!(matches(
            &s,
            &EventPredicate::And {
                parts: vec![one.clone(), two.clone()]
            }
        ));
        assert!(!matches(
            &s,
            &EventPredicate::And {
                parts: vec![one.clone(), three.clone()]
            }
        ));
        assert!(matches(
            &s,
            &EventPredicate::Or {
                parts: vec![three.clone(), one.clone()]
            }
        ));
        assert!(!matches(
            &s,
            &EventPredicate::Or {
                parts: vec![three.clone()]
            }
        ));
        assert!(matches(
            &s,
            &EventPredicate::Not {
                inner: Box::new(three)
            }
        ));
        assert!(!matches(
            &s,
            &EventPredicate::Not {
                inner: Box::new(one)
            }
        ));
    }

    #[test]
    fn all_cmp_ops_covered() {
        let s = bosonic(0, 2);
        for cmp in [Cmp::Eq, Cmp::Ge, Cmp::Le, Cmp::Gt, Cmp::Lt] {
            let _ = matches(
                &s,
                &EventPredicate::BosonModeTotal {
                    mode: 0,
                    cmp,
                    value: 2,
                },
            );
        }
    }
}
