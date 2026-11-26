//! D19: what a re-delivered decision does at the recorder.
//!
//! The catch-up loop re-sends a batch when an acknowledgement is lost, so a
//! peer's recorder can see the same slot twice. Three answers are pinned here,
//! because the sender's at-least-once framing relies on all three: an identical
//! re-delivery is an idempotent no-op, a conflicting re-delivery is refused
//! with `ConflictingDecision`, and a batch that carries the conflict applies
//! none of its decisions — not even the ones before it — in memory or on disk.
#![cfg(feature = "network")]

use std::sync::{Arc, Mutex};

use rust_quepaxa::network::PostcardRecorderCodec;
use rust_quepaxa::recorder::RecorderStateStore;
use rust_quepaxa::{
    AllowAllAvailability, Decision, DurableRecorderCore, FileRecorderStore, InMemoryRecorderStore,
    QuePaxaError, RecorderClient, RecorderConfig, ReplicaId, SlotIndex, Step,
};

fn config() -> RecorderConfig {
    RecorderConfig::new(ReplicaId::new(1), (1..=3).map(ReplicaId::new), 1)
        .expect("a 3-member, 1-fault recorder config")
}

fn decision(slot: u64, values: Vec<u64>) -> Decision<u64> {
    Decision::new(
        SlotIndex::new(slot),
        values,
        ReplicaId::new(1),
        Step::ROUND_ONE_PHASE_ZERO,
    )
    .expect("a well-formed decision")
}

fn recorded_slots<V, S>(recorder: &DurableRecorderCore<V, S>) -> Vec<u64>
where
    V: Clone + Eq,
    S: RecorderStateStore<V>,
{
    recorder
        .core()
        .snapshot()
        .decisions
        .keys()
        .map(|slot| slot.get())
        .collect()
}

/// A duplicate that carries the same `value_ids` is a no-op: the slot is
/// recorded once, and the durable snapshot still holds exactly that one
/// decision.
#[test]
fn an_identical_redelivery_is_an_idempotent_no_op() {
    let store = Arc::new(Mutex::new(InMemoryRecorderStore::<u64>::default()));
    let mut recorder =
        DurableRecorderCore::new(config(), Arc::new(AllowAllAvailability), Arc::clone(&store))
            .expect("recorder");

    recorder.inform_decision(decision(1, vec![10])).unwrap();
    recorder.inform_decision(decision(1, vec![10])).unwrap();

    assert_eq!(recorded_slots(&recorder), vec![1]);

    let snapshot = store
        .lock()
        .unwrap()
        .snapshot()
        .cloned()
        .expect("the store was written");
    assert_eq!(snapshot.decisions.len(), 1);
    assert_eq!(
        snapshot.decisions[&SlotIndex::new(1)].value_ids,
        vec![10],
        "the first decision is the one that stands"
    );
}

/// A duplicate that carries different `value_ids` for an already-decided slot
/// is refused by name, and the refusal changes nothing.
#[test]
fn a_conflicting_redelivery_is_refused_and_keeps_the_first_decision() {
    let store = Arc::new(Mutex::new(InMemoryRecorderStore::<u64>::default()));
    let mut recorder =
        DurableRecorderCore::new(config(), Arc::new(AllowAllAvailability), Arc::clone(&store))
            .expect("recorder");

    recorder.inform_decision(decision(1, vec![10])).unwrap();
    let error = recorder
        .inform_decision(decision(1, vec![11]))
        .expect_err("a conflicting decision for slot 1 must be refused");

    assert_eq!(
        error,
        QuePaxaError::ConflictingDecision {
            slot: SlotIndex::new(1)
        }
    );
    let snapshot = store
        .lock()
        .unwrap()
        .snapshot()
        .cloned()
        .expect("the store was written");
    assert_eq!(
        snapshot.decisions[&SlotIndex::new(1)].value_ids,
        vec![10],
        "a refused duplicate must not overwrite the recorded decision"
    );
}

/// The durable wrapper persists a batch atomically: when one decision in it
/// conflicts, the decisions around it are not applied either. This is the
/// property the catch-up sender needs, because it re-sends whole batches.
#[test]
fn a_batch_with_a_conflict_applies_none_of_its_decisions() {
    let store = Arc::new(Mutex::new(InMemoryRecorderStore::<u64>::default()));
    let mut recorder =
        DurableRecorderCore::new(config(), Arc::new(AllowAllAvailability), Arc::clone(&store))
            .expect("recorder");

    recorder.inform_decision(decision(1, vec![10])).unwrap();

    // A re-sent batch: slot 1 now conflicts, slot 2 is new and valid.
    let batch = [decision(1, vec![11]), decision(2, vec![20])];
    let error = recorder
        .inform_decisions(&batch)
        .expect_err("the batch carries a conflicting duplicate");

    assert_eq!(
        error,
        QuePaxaError::ConflictingDecision {
            slot: SlotIndex::new(1)
        }
    );
    assert_eq!(
        recorded_slots(&recorder),
        vec![1],
        "slot 2 must not be applied when the batch is refused"
    );
    let snapshot = store
        .lock()
        .unwrap()
        .snapshot()
        .cloned()
        .expect("the store was written");
    assert_eq!(snapshot.decisions[&SlotIndex::new(1)].value_ids, vec![10]);
    assert!(!snapshot.decisions.contains_key(&SlotIndex::new(2)));
}

/// The answer survives a restart through the file store: the duplicate is
/// still one decision after reopening, and the conflict is still refused.
#[test]
fn duplicates_are_answered_the_same_after_a_restart() {
    let directory =
        std::env::temp_dir().join(format!("unfer-recorder-duplicates-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("recorder.snapshot");

    {
        let mut recorder = DurableRecorderCore::new(
            config(),
            Arc::new(AllowAllAvailability),
            FileRecorderStore::new(&path, PostcardRecorderCodec::default()),
        )
        .expect("recorder");
        recorder.inform_decision(decision(1, vec![10])).unwrap();
        recorder.inform_decision(decision(1, vec![10])).unwrap();
    }

    let mut reopened = DurableRecorderCore::new(
        config(),
        Arc::new(AllowAllAvailability),
        FileRecorderStore::new(&path, PostcardRecorderCodec::default()),
    )
    .expect("reopened recorder");
    assert_eq!(recorded_slots(&reopened), vec![1]);
    assert_eq!(
        reopened
            .inform_decision(decision(1, vec![10]))
            .map_err(|error| error.to_string()),
        Ok(()),
        "an identical duplicate is still a no-op after a restart"
    );
    assert_eq!(
        reopened.inform_decision(decision(1, vec![11])).unwrap_err(),
        QuePaxaError::ConflictingDecision {
            slot: SlotIndex::new(1)
        }
    );

    let _ = std::fs::remove_dir_all(&directory);
}
