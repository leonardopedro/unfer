//! H7: leader lease for `ConsensusNode::sync` and auction settlement.
//!
//! In a distributed setting exactly one node should fire scheduled consensus /
//! auction work per tick; a lost lease must stop firing without corrupting
//! state. The lease is deterministic given the participant set and the tick
//! (same inputs → same leader), so it never depends on wall-clock or a random
//! source. A node that lost its lease (the epoch/tick moved on, or a different
//! participant was elected) simply stops firing — the ledgers are untouched
//! because the guard returns before any mutation.

/// Deterministic single-leader lease.
///
/// Election rule: the leader for `tick` among `participants` is the member at
/// index `tick % participants.len()`. [`LeaderLease::is_leader`] answers
/// "may I fire this tick?" — exactly one participant answers yes for a given
/// (tick, participant set).
#[derive(Debug, Clone, Copy)]
pub struct LeaderLease;

impl LeaderLease {
    pub fn new() -> Self {
        Self
    }

    /// The participant index that is the single leader for `tick`. With
    /// `participants == 0` no one is leader (fail closed).
    pub fn leader_index(tick: u64, participants: usize) -> Option<usize> {
        if participants == 0 {
            return None;
        }
        Some((tick % participants as u64) as usize)
    }

    /// Whether `node_index` is the single leader allowed to fire `tick`.
    pub fn is_leader(tick: u64, participants: usize, node_index: usize) -> bool {
        Self::leader_index(tick, participants) == Some(node_index)
    }

    /// Lease lifecycle over a moving tick: a node fires `tick` only while it
    /// remains the leader. Returns `false` (lease lost) the moment the tick
    /// elects a different participant — the caller stops firing and the state
    /// is left untouched.
    pub fn tick_held(
        previous_tick: u64,
        tick: u64,
        participants: usize,
        node_index: usize,
    ) -> bool {
        if tick < previous_tick {
            return false;
        }
        (previous_tick..=tick)
            .all(|t| Self::is_leader(t, participants, node_index))
    }
}

impl Default for LeaderLease {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_one_leader_per_tick() {
        // Two nodes, three ticks: each tick elects exactly one leader.
        for tick in 0..6 {
            let a = LeaderLease::is_leader(tick, 2, 0);
            let b = LeaderLease::is_leader(tick, 2, 1);
            assert_eq!(a ^ b, true, "exactly one leader per tick (tick {tick})");
        }
    }

    #[test]
    fn leader_rotates_round_robin() {
        assert!(LeaderLease::is_leader(0, 2, 0));
        assert!(LeaderLease::is_leader(1, 2, 1));
        assert!(LeaderLease::is_leader(2, 2, 0));
        assert!(LeaderLease::is_leader(3, 2, 1));
    }

    #[test]
    fn zero_participants_fail_closed() {
        assert_eq!(LeaderLease::leader_index(0, 0), None);
        assert!(!LeaderLease::is_leader(0, 0, 0));
    }

    #[test]
    fn lost_lease_stops_firing() {
        // Node 0 fires tick 0 (it is leader), but the next tick elects node 1 —
        // node 0 lost the lease and must stop firing.
        assert!(LeaderLease::tick_held(0, 0, 2, 0));
        assert!(!LeaderLease::tick_held(0, 1, 2, 0), "lease lost on tick 1");
        // Node 1 held the whole range [0,1]? No — node 1 is not leader for tick
        // 0, so it never fires tick 0.
        assert!(!LeaderLease::tick_held(0, 0, 2, 1));
    }

    #[test]
    fn clock_regression_never_fires() {
        // A tick moving backwards (clock regression) must not fire anything.
        assert!(!LeaderLease::tick_held(5, 3, 2, 0));
    }
}