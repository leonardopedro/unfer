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
                .map(|(inner, &count)| {
                    inner.modes.get(mode).copied().unwrap_or(0) * count
                })
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

        EventPredicate::Vacuum => {
            outer.bosonic.is_empty() && outer.fermionic.is_empty()
        }

        EventPredicate::And { parts } => {
            parts.iter().all(|p| matches(outer, p))
        }

        EventPredicate::Or { parts } => {
            parts.iter().any(|p| matches(outer, p))
        }

        EventPredicate::Not { inner } => {
            !matches(outer, inner)
        }
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
