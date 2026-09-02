use std::sync::Arc;

use candle_core::Device;
use fock_sirk::{SirkOpts, evolve_restarted};
use nested_fock_algebra::{Hamiltonian, QuantumState};
use qfm::QfmPipeline;
use unfer_protocol::durable::{DurableStore, streams};
use unfer_protocol::{
    EventPredicate, HamiltonianSpec, HmcOptsSpec, ModelSpec, PriorSpec, SolverSpec,
};

use crate::build;
use crate::error::KernelError;
use crate::event;

/// H3: session event-log format marker. Bumps only on a structural change of
/// the `SessionEvent`/`SessionBlob` shape. `restore` rejects blobs whose
/// `format_version` does not match (UK-1006) rather than guessing.
pub const SESSION_FORMAT_VERSION: u32 = 1;

/// H3: a typed session-log record. Each mutating kernel op appends one
/// `{ seq, op, spec, source, ts }` record **before** it applies, so the log is
/// the single source of truth: `save()` folds it, `restore()` replays it, and
/// the model-visible state is reconstructable from the log alone.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SessionEvent {
    /// Monotonic, never-reused sequence number (== index into the log).
    pub seq: u64,
    /// The operation that produced this record.
    pub op: SessionOp,
    /// The op-specific typed payload (what the op needs to re-apply itself).
    pub spec: SessionEventSpec,
    /// The caller that issued the op (module/agent/hook principal).
    pub source: String,
    /// Wall-clock millisecond timestamp (audit provenance; not part of the fold).
    pub ts: u64,
}

/// H3: the operations the kernel records in the session event log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionOp {
    Create,
    SetPrior,
    SetHamiltonian,
    Evolve,
    Condition,
    CompactStart,
    CompactEnd,
}

/// H3: the typed payload carried by a `SessionEvent`. Tagged so each record is
/// self-describing and replayable without external context.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SessionEventSpec {
    Create {
        spec: ModelSpec,
    },
    SetPrior {
        spec: PriorSpec,
    },
    SetHamiltonian {
        spec: HamiltonianSpec,
    },
    Evolve {
        t: f64,
        /// True when the live call dispatched to the QFM pipeline (evolve only
        /// advances `t_now`; the SIRK state is untouched). Replay mirrors this.
        qfm: bool,
        /// The raw query passed to `evolve` (QFM input / provenance).
        query: Option<Vec<f64>>,
    },
    Condition {
        spec: EventPredicate,
    },
    CompactStart {
        through_seq: u64,
    },
    CompactEnd {
        through_seq: u64,
        summary: Option<SessionBlob>,
    },
}

impl PartialEq for SessionEventSpec {
    fn eq(&self, other: &Self) -> bool {
        // `SessionBlob` is compared by JSON so the folded `CompactEnd` summary
        // participates in event-log equality without requiring `PartialEq` on
        // `QuantumState`.
        match (self, other) {
            (SessionEventSpec::Create { spec: a }, SessionEventSpec::Create { spec: b }) => a == b,
            (SessionEventSpec::SetPrior { spec: a }, SessionEventSpec::SetPrior { spec: b }) => {
                a == b
            }
            (
                SessionEventSpec::SetHamiltonian { spec: a },
                SessionEventSpec::SetHamiltonian { spec: b },
            ) => a == b,
            (
                SessionEventSpec::Evolve {
                    t: a,
                    qfm: qa,
                    query: xa,
                },
                SessionEventSpec::Evolve {
                    t: b,
                    qfm: qb,
                    query: xb,
                },
            ) => a == b && qa == qb && xa == xb,
            (SessionEventSpec::Condition { spec: a }, SessionEventSpec::Condition { spec: b }) => {
                a == b
            }
            (
                SessionEventSpec::CompactStart { through_seq: a },
                SessionEventSpec::CompactStart { through_seq: b },
            ) => a == b,
            (
                SessionEventSpec::CompactEnd {
                    through_seq: a,
                    summary: sa,
                },
                SessionEventSpec::CompactEnd {
                    through_seq: b,
                    summary: sb,
                },
            ) => {
                a == b
                    && match (sa, sb) {
                        (None, None) => true,
                        (Some(x), Some(y)) => {
                            serde_json::to_string(x).ok() == serde_json::to_string(y).ok()
                        }
                        _ => false,
                    }
            }
            _ => false,
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A long-running probability kernel session.
///
/// Owns the current quantum state, Hamiltonian, and solver configuration.
/// Callers evolve the state, query event probabilities, and condition
/// (Bayesian update) on observed events.
#[derive(Debug)]
pub struct Session {
    state: QuantumState,
    hamiltonian: Hamiltonian,
    sirk_opts: SirkOpts,
    krylov_dim: usize,
    restarts: usize,
    device: Device,
    t_now: f64,
    // Stored specs for snapshot/restore — updated by set_hamiltonian.
    hamiltonian_spec: HamiltonianSpec,
    solver_spec: SolverSpec,
    /// QFM tomographic pipeline (Workstream F). Present only when the
    /// session was created from a `HamiltonianSpec::QfmTomography` spec.
    /// `evolve` dispatches to the pipeline's 4-phase `generate` instead
    /// of the SIRK solver.
    qfm_pipeline: Option<Box<QfmPipeline>>,
    /// H3: the session event log — every mutating op, in order. `save()` folds
    /// it; `restore()` replays it; fork/compaction operate on it. The log is
    /// the single source of truth for the model-visible state.
    event_log: Vec<SessionEvent>,
    /// H3: next monotonic sequence number (== `event_log.len()`).
    next_seq: u64,
    /// H3: the caller label stamped on newly appended events. The FFI layer
    /// updates it from the active `CallerContext` before each mutating op.
    log_source: String,
    /// H3: an open compaction lock bracket `(compaction/start … compaction/end)`.
    /// `Some(through_seq)` while a compaction is in progress; `None` when idle.
    /// A crash between start and end leaves this as an orphaned lock that
    /// `restore`/`save` detect (UK-1007).
    compaction_lock: Option<u64>,
    /// H4: optional durable sink. When attached, every committed event is
    /// appended to the store's `session` stream and `probability`/`condition`
    /// flush (checkpoint) before serving, so nothing the model reads back is
    /// RAM-only. `None` (default) = the session is RAM-only but fully
    /// functional (save/restore still work via `SessionBlob`).
    durable: Option<Arc<dyn DurableStore>>,
    /// H10: the named `AgentPreset` this session started under (recorded in the
    /// session header). `None` = no preset (inline grants only). A preset
    /// switch is valid only while the session has produced nothing (see
    /// `unfer_protocol::preset::switch_valid_when_blank`).
    start_preset: Option<String>,
}

/// Serializable snapshot of a Session for save/restore.
///
/// H3: the snapshot is now event-sourced. `save()` folds the event log into
/// `events` (tagged with `format_version`); `restore()` replays it. The folded
/// state fields below remain for backward compatibility (legacy blobs carry no
/// `events` and restore via the direct-state path).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionBlob {
    /// `SESSION_FORMAT_VERSION` for event-sourced blobs; `0` (legacy) when the
    /// blob predates H3 and carries only the folded state fields.
    #[serde(default)]
    pub format_version: u32,
    /// The session event log. Empty for legacy blobs (folded fields used).
    #[serde(default)]
    pub events: Vec<SessionEvent>,
    pub hamiltonian_spec: HamiltonianSpec,
    pub solver_spec: SolverSpec,
    pub state: QuantumState,
    pub t_now: f64,
}

impl Session {
    /// The number of event-log records currently held.
    pub fn event_log_len(&self) -> usize {
        self.event_log.len()
    }

    /// Read access to the raw event log (audit / fork boundaries).
    pub fn event_log(&self) -> &[SessionEvent] {
        &self.event_log
    }

    /// H3: the log source stamped on newly appended events.
    pub fn set_log_source(&mut self, source: impl Into<String>) {
        self.log_source = source.into();
    }

    /// H3: the caller label current at this session.
    pub fn log_source(&self) -> &str {
        &self.log_source
    }

    /// H3: whether a compaction lock bracket is currently open (session busy).
    pub fn is_compaction_locked(&self) -> bool {
        self.compaction_lock.is_some()
    }

    /// H4: attach (or detach with `None`) the session's durable sink. When
    /// attached, every committed event is appended to the store's `session`
    /// stream and `probability`/`condition` flush before serving.
    pub fn set_durable(&mut self, store: Option<Arc<dyn DurableStore>>) {
        self.durable = store;
    }

    /// H4: the attached durable sink, if any.
    pub fn durable(&self) -> Option<&dyn DurableStore> {
        self.durable.as_deref()
    }

    /// H4: append a committed event to the durable `session` stream (if a
    /// store is attached). Fail-closed: an append (or encode) failure is
    /// propagated so a committed event is never *silently* missing from the
    /// durable log. The in-memory log remains authoritative for the live
    /// session, but restart replay re-reads the store, so a dropped record
    /// would shorten history with no one learning. The store layer has no
    /// owner-log sink of its own — surfacing is the caller's job.
    fn durable_append_event(&self, ev: &SessionEvent) -> Result<(), String> {
        let Some(store) = &self.durable else {
            return Ok(());
        };
        let json = serde_json::to_vec(ev)
            .map_err(|e| format!("session event encode failed: {e}"))?;
        store
            .append(streams::SESSION, &json)
            .map_err(|e| format!("session stream append failed: {e}"))
    }

    /// H4: the checkpoint barrier before a model-facing read. Fail-closed:
    /// `Err` means the caller must not serve the probability/condition result.
    fn durable_checkpoint(&self) -> Result<(), String> {
        match &self.durable {
            Some(store) => store.flush().map_err(|e| e.to_string()),
            None => Ok(()),
        }
    }
}

/// Result of an `evolve` call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvolveReport {
    pub t: f64,
    pub norm: f64,
    pub components: usize,
    /// Wall-clock time for the SIRK solve in milliseconds.
    pub solve_ms: u64,
    /// QFM tomographic output: present only when the session was created
    /// from a `HamiltonianSpec::QfmTomography` spec and `evolve` was called
    /// with a `query` in the opts. Contains the generated raw image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qfm_output: Option<Vec<f64>>,
}

/// A snapshot of the current state's top-k components.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StateSummary {
    pub norm: f64,
    pub components: usize,
    pub top: Vec<StateEntry>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StateEntry {
    pub state: String,
    pub probability: f64,
}

/// Result of a Quantum Bayesian Update on the TSR-evolved prior
/// (QFM.tex §8, P6 H follow-on). Only QFM tomographic models are
/// eligible. The kernel returns the HMC diagnostics + the full
/// reconstructed image from Phase 5.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BayesianUpdateReport {
    /// HMC log-posterior at the final sample.
    pub log_posterior: f64,
    /// HMC geometric-mean of the per-observation likelihoods at the
    /// final sample. `-1.0` if there were no observations (posterior
    /// = prior).
    pub mean_likelihood: f64,
    /// The Phase 5 reconstructed full-resolution image of the
    /// representative (final) HMC draw.
    pub image: Vec<f64>,
    /// The Phase 5 reconstruction of the **posterior-mean** point
    /// estimate — the Karcher (Fréchet) mean of the post-burn-in HMC
    /// chain on the projective unit sphere of $\Cset^m$ (QFM.tex §8).
    /// Denoised relative to the single draw in `image`. Empty if there
    /// were no post-burn-in samples.
    pub posterior_mean_image: Vec<f64>,
    /// The number of post-burn-in samples averaged into
    /// `posterior_mean_image`.
    pub n_samples: usize,
    /// The number of observations $N$ (cached for the agent surface).
    pub n_observations: usize,
    /// Wall-clock time for HMC + decode in milliseconds.
    pub solve_ms: u64,
}

/// Result of a chain belief-propagation run (P8.8, qfm::bayes::
/// `belief_propagation_chain`). The MAP (marginal mode) point estimate
/// on the Krylov coefficients, plus the full-resolution image decoded
/// via Phase 5 tomographic reconstruction.
///
/// **Use case:** fast alternative to HMC when the user wants a
/// posterior point estimate without paying the sampling cost.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BeliefPropagationReport {
    /// The Phase 5 reconstructed full-resolution image of the MAP.
    pub image: Vec<f64>,
    /// The log-posterior at the MAP (up to a constant).
    pub log_posterior: f64,
    /// The number of observations $N$ (cached for the agent surface).
    pub n_observations: usize,
    /// The number of cumulative-product sweeps used (always 1 for the
    /// exact chain case).
    pub n_sweeps: usize,
    /// Wall-clock time for BP + decode in milliseconds.
    pub solve_ms: u64,
}

impl Session {
    /// Build a session from a `ModelSpec` without logging a `Create` event.
    /// Used by `new` (which appends the event) and by event-log replay.
    fn from_spec(spec: &ModelSpec) -> Result<Self, KernelError> {
        let hamiltonian = build::build_hamiltonian(&spec.hamiltonian)?;
        let state = build::build_prior(&spec.prior)?;
        let device = build::build_device(&spec.solver.device)?;
        let sirk_opts = SirkOpts {
            prune_eps: spec.solver.prune_eps,
            max_components: spec.solver.max_components,
            brst_tol: 1e-10,
            adaptive: spec.solver.adaptive,
            unit_norm_steps: false,
        };
        // If the Hamiltonian is a QFM tomography spec, compile the pipeline.
        let qfm_pipeline =
            if let HamiltonianSpec::QfmTomography { spec: qfm_spec } = &spec.hamiltonian {
                Some(Box::new(build::compile_qfm_pipeline(qfm_spec)?))
            } else {
                None
            };
        Ok(Self {
            state,
            hamiltonian,
            sirk_opts,
            krylov_dim: spec.solver.krylov_dim,
            restarts: spec.solver.restarts.max(1),
            device,
            t_now: 0.0,
            hamiltonian_spec: spec.hamiltonian.clone(),
            solver_spec: spec.solver.clone(),
            qfm_pipeline,
            event_log: Vec::new(),
            next_seq: 0,
            log_source: "kernel".into(),
            compaction_lock: None,
            durable: None,
            start_preset: None,
        })
    }

    /// Create a new session from a `ModelSpec`. Appends the root `Create`
    /// event (seq 0) to the session log.
    pub fn new(spec: &ModelSpec) -> Result<Self, KernelError> {
        let mut s = Self::from_spec(spec)?;
        s.append_event(
            SessionOp::Create,
            SessionEventSpec::Create { spec: spec.clone() },
        );
        Ok(s)
    }

    /// H10: record the named `AgentPreset` this session started under. Call
    /// before any producing op; a switch after the session produced anything is
    /// refused (see `unfer_protocol::preset::switch_valid_when_blank`).
    pub fn set_start_preset(&mut self, preset: &str) {
        self.start_preset = Some(preset.to_string());
    }

    /// H10: the named `AgentPreset` this session started under, if any.
    pub fn start_preset(&self) -> Option<&str> {
        self.start_preset.as_deref()
    }

    /// H10: the number of ops this session has produced (excluding the root
    /// `Create`). A preset switch is valid only while this is 0 (a blank
    /// session); once the session produced anything the tool surface must not
    /// change under a model that already ran.
    pub fn event_log_len_for_preset_switch(&self) -> usize {
        self.event_log
            .iter()
            .filter(|e| e.op != SessionOp::Create)
            .count()
    }

    /// Append a record to the event log. `seq` is the pre-increment log
    /// length; records are never rewritten in place.
    fn append_event(&mut self, op: SessionOp, spec: SessionEventSpec) {
        let ev = SessionEvent {
            seq: self.next_seq,
            op,
            spec,
            source: self.log_source.clone(),
            ts: now_ms(),
        };
        self.event_log.push(ev);
        self.next_seq = self.event_log.len() as u64;
    }

    /// Reject any mutating op while a compaction lock bracket is open
    /// (UK-1008 `SESSION_COMPACTION_BUSY`).
    fn check_idle(&self) -> Result<(), KernelError> {
        if let Some(through) = self.compaction_lock {
            return Err(KernelError::SessionCompactionBusy {
                reason: format!("session busy: compaction lock open through seq {through}"),
            });
        }
        Ok(())
    }

    /// Log a record **before** applying the op, then apply. On error the
    /// record is rolled back so `fold(events) ≡ live`. In debug builds the
    /// whole log is replayed and compared against the live session.
    fn log_then_apply<T>(
        &mut self,
        op: SessionOp,
        spec: SessionEventSpec,
        f: impl FnOnce(&mut Self) -> Result<T, KernelError>,
    ) -> Result<T, KernelError> {
        self.check_idle()?;
        let ev = SessionEvent {
            seq: self.next_seq,
            op,
            spec,
            source: self.log_source.clone(),
            ts: now_ms(),
        };
        let committed = ev.clone();
        self.event_log.push(ev);
        self.next_seq = self.event_log.len() as u64;
        let result = f(self);
        self.next_seq = self.event_log.len() as u64;
        match result {
            Ok(v) => {
                // Fail-closed: if the committed event could not be durably
                // appended, the mutation is *not durable* — surface UK-1010
                // instead of reporting success the store never recorded.
                // The in-memory event must stay (folding the log must equal
                // the live session); the caller learns the outcome is unknown
                // until the durable record is verified.
                self.durable_append_event(&committed)
                    .map_err(|reason| KernelError::DurableCheckpointFailed { reason })?;
                #[cfg(debug_assertions)]
                self.debug_assert_reconstructable();
                Ok(v)
            }
            Err(e) => {
                self.event_log.pop();
                self.next_seq = self.event_log.len() as u64;
                Err(e)
            }
        }
    }

    /// H3: fold a snapshot of the current state into a legacy-form
    /// `SessionBlob` (empty events; direct state fields). Used as the
    /// `CompactEnd` summary node and as the folded half of `save`.
    fn folded_blob(&self) -> SessionBlob {
        SessionBlob {
            format_version: SESSION_FORMAT_VERSION,
            events: Vec::new(),
            hamiltonian_spec: self.hamiltonian_spec.clone(),
            solver_spec: self.solver_spec.clone(),
            state: self.state.clone(),
            t_now: self.t_now,
        }
    }

    /// H3: the compressed, replayable history view. Raw events stay in
    /// `self.event_log`; the derived view keeps the **last** completed
    /// `CompactEnd` summary node (whose folded blob captures the full state at
    /// its boundary, subsuming every earlier event incl. prior brackets) plus
    /// the tail events after it. Seq is renumbered to be contiguous in the
    /// derived view. An orphaned `CompactStart` (crash between start/end) is
    /// kept so `restore` fails loud (UK-1007). Replaying `derived_log()`
    /// reproduces the same folded state as the full raw log.
    fn derived_log(&self) -> Vec<SessionEvent> {
        let last_ce = self.event_log.iter().rposition(|e| {
            matches!(
                &e.spec,
                SessionEventSpec::CompactEnd {
                    summary: Some(_),
                    ..
                }
            )
        });
        let mut out: Vec<SessionEvent> = match last_ce {
            None => self.event_log.clone(),
            Some(idx) => {
                let mut v = Vec::with_capacity(self.event_log.len() - idx);
                v.push(self.event_log[idx].clone());
                for ev in &self.event_log[idx + 1..] {
                    v.push(ev.clone());
                }
                v
            }
        };
        for (i, e) in out.iter_mut().enumerate() {
            e.seq = i as u64;
        }
        out
    }

    /// H3: replay a session event log into a live `Session`. The first record
    /// must be `Create` (full raw log) or a `CompactStart`/`CompactEnd`
    /// bracket pair (derived/compacted view). Validates strict monotonic seq,
    /// bracket balance (no orphaned lock → UK-1007), and seq-contiguity of the
    /// folded state. The resulting session's `event_log`/`next_seq` are set to
    /// the replayed log so it can continue accumulating.
    fn replay(events: &[SessionEvent]) -> Result<Self, KernelError> {
        if events.is_empty() {
            return Err(KernelError::SessionLogVersion {
                got: SESSION_FORMAT_VERSION,
                reason: "cannot replay an empty session log (missing Create)".into(),
            });
        }
        for w in events.windows(2) {
            if w[0].seq >= w[1].seq {
                return Err(KernelError::SessionLogVersion {
                    got: w[1].seq as u32,
                    reason: format!(
                        "non-monotonic session log sequence at seq {} (prev {})",
                        w[1].seq, w[0].seq
                    ),
                });
            }
        }
        let mut session: Option<Session> = None;
        let mut open: Option<u64> = None;
        let mut first = true;
        for ev in events {
            match &ev.spec {
                SessionEventSpec::Create { spec } => {
                    if !first {
                        return Err(KernelError::SessionCompactionOrphaned {
                            reason: format!("duplicate Create event at seq {}", ev.seq),
                        });
                    }
                    session = Some(Self::from_spec(spec)?);
                }
                SessionEventSpec::CompactStart { through_seq } => {
                    if let Some(t) = open {
                        return Err(KernelError::SessionCompactionOrphaned {
                            reason: format!(
                                "nested compaction lock at seq {} (open through {})",
                                ev.seq, t
                            ),
                        });
                    }
                    open = Some(*through_seq);
                }
                SessionEventSpec::CompactEnd {
                    through_seq,
                    summary,
                } => {
                    let summary =
                        summary
                            .as_ref()
                            .ok_or_else(|| KernelError::SessionCompactionOrphaned {
                                reason: format!("CompactEnd without summary at seq {}", ev.seq),
                            })?;
                    match open.take() {
                        // Derived-view root: a CompactEnd with a summary is the
                        // first record (no preceding CompactStart). Seed the
                        // session from the folded summary.
                        None => {
                            if session.is_some() {
                                return Err(KernelError::SessionCompactionOrphaned {
                                    reason: format!(
                                        "CompactEnd without matching CompactStart at seq {}",
                                        ev.seq
                                    ),
                                });
                            }
                            session = Some(Self::from_folded(summary)?);
                        }
                        Some(expected) => {
                            if expected != *through_seq {
                                return Err(KernelError::SessionCompactionOrphaned {
                                    reason: format!(
                                        "CompactEnd through_seq {through_seq} != open through_seq \
                                         {expected} at seq {}",
                                        ev.seq
                                    ),
                                });
                            }
                            // Raw-log view: the bracket's summary carries the
                            // state at the boundary and continues from there.
                            session = Some(Self::from_folded(summary)?);
                        }
                    }
                }
                SessionEventSpec::SetPrior { spec } => {
                    let s = session
                        .as_mut()
                        .ok_or_else(|| KernelError::SessionLogVersion {
                            got: SESSION_FORMAT_VERSION,
                            reason: format!("SetPrior before Create at seq {}", ev.seq),
                        })?;
                    s.apply_prior(spec)?;
                }
                SessionEventSpec::SetHamiltonian { spec } => {
                    let s = session
                        .as_mut()
                        .ok_or_else(|| KernelError::SessionLogVersion {
                            got: SESSION_FORMAT_VERSION,
                            reason: format!("SetHamiltonian before Create at seq {}", ev.seq),
                        })?;
                    s.apply_hamiltonian(spec)?;
                }
                SessionEventSpec::Evolve { t, qfm, query } => {
                    let s = session
                        .as_mut()
                        .ok_or_else(|| KernelError::SessionLogVersion {
                            got: SESSION_FORMAT_VERSION,
                            reason: format!("Evolve before Create at seq {}", ev.seq),
                        })?;
                    s.apply_evolve(*t, query.clone(), *qfm)?;
                }
                SessionEventSpec::Condition { spec } => {
                    let s = session
                        .as_mut()
                        .ok_or_else(|| KernelError::SessionLogVersion {
                            got: SESSION_FORMAT_VERSION,
                            reason: format!("Condition before Create at seq {}", ev.seq),
                        })?;
                    s.apply_condition(spec)?;
                }
            }
            first = false;
        }
        if let Some(through) = open {
            return Err(KernelError::SessionCompactionOrphaned {
                reason: format!("orphaned compaction lock through seq {through} (no CompactEnd)"),
            });
        }
        let mut session = session.ok_or_else(|| KernelError::SessionLogVersion {
            got: SESSION_FORMAT_VERSION,
            reason: "session log has no Create event".into(),
        })?;
        session.event_log = events.to_vec();
        session.next_seq = session.event_log.len() as u64;
        session.compaction_lock = None;
        Ok(session)
    }

    /// H3: in debug builds, replay the full raw log and assert the folded
    /// state (t_now, serialized state, specs) matches the live session. This
    /// pins the invariant `fold(events) ≡ live`.
    #[cfg(debug_assertions)]
    fn debug_assert_reconstructable(&self) {
        if self.compaction_lock.is_some() {
            // An open bracket is a transient mid-compaction state; the fold is
            // still the live state, but the raw log alone cannot be replayed.
            return;
        }
        let restored = match Session::replay(&self.event_log) {
            Ok(r) => r,
            Err(e) => panic!("session event log not reconstructable: {e:?}"),
        };
        let live_state = serde_json::to_string(&self.state).unwrap_or_default();
        let restored_state = serde_json::to_string(&restored.state).unwrap_or_default();
        assert_eq!(
            restored.t_now, self.t_now,
            "session replay diverged on t_now: {} != {}",
            restored.t_now, self.t_now
        );
        assert_eq!(
            live_state, restored_state,
            "session replay diverged on state"
        );
        assert_eq!(
            restored.hamiltonian_spec, self.hamiltonian_spec,
            "session replay diverged on hamiltonian_spec"
        );
        assert_eq!(
            restored.solver_spec, self.solver_spec,
            "session replay diverged on solver_spec"
        );
    }

    /// Restore a session from a previously saved `SessionBlob`. Prefers the
    /// event log (`events` + `format_version` == `SESSION_FORMAT_VERSION`);
    /// falls back to the legacy direct-state path for pre-H3 blobs (empty
    /// events). A version mismatch is rejected (UK-1006); an orphaned
    /// compaction lock fails loud (UK-1007).
    pub fn restore(blob: SessionBlob) -> Result<Self, KernelError> {
        if !blob.events.is_empty() {
            if blob.format_version != SESSION_FORMAT_VERSION {
                return Err(KernelError::SessionLogVersion {
                    got: blob.format_version,
                    reason: format!(
                        "session log format version {} is not supported (expected {})",
                        blob.format_version, SESSION_FORMAT_VERSION
                    ),
                });
            }
            return Self::replay(&blob.events);
        }
        Self::from_folded(&blob)
    }

    /// Build a session directly from the folded (legacy) fields of a blob.
    /// Used by `restore` for pre-H3 blobs and by compaction summary replay.
    fn from_folded(blob: &SessionBlob) -> Result<Self, KernelError> {
        let hamiltonian = build::build_hamiltonian(&blob.hamiltonian_spec)?;
        let device = build::build_device(&blob.solver_spec.device)?;
        let sirk_opts = SirkOpts {
            prune_eps: blob.solver_spec.prune_eps,
            max_components: blob.solver_spec.max_components,
            brst_tol: 1e-10,
            adaptive: blob.solver_spec.adaptive,
            unit_norm_steps: false,
        };
        // QFM pipelines are not serialized — a restored session that was
        // originally a QFM model falls back to the SIRK path with the
        // placeholder Hamiltonian. The hamiltonian_spec is preserved so the
        // caller can re-create the pipeline by calling `set_hamiltonian`.
        let qfm_pipeline = None;
        Ok(Self {
            state: blob.state.clone(),
            hamiltonian,
            sirk_opts,
            krylov_dim: blob.solver_spec.krylov_dim,
            restarts: blob.solver_spec.restarts.max(1),
            device,
            t_now: blob.t_now,
            hamiltonian_spec: blob.hamiltonian_spec.clone(),
            solver_spec: blob.solver_spec.clone(),
            qfm_pipeline,
            event_log: Vec::new(),
            next_seq: 0,
            log_source: "kernel".into(),
            compaction_lock: None,
            durable: None,
            start_preset: None,
        })
    }

    /// Serialize the current session state to a `SessionBlob` for persistence.
    /// The blob carries the **derived** (compacted) event log tagged with
    /// `SESSION_FORMAT_VERSION`, plus the folded state fields for backward
    /// compatibility with pre-H3 consumers.
    pub fn save(&self) -> SessionBlob {
        SessionBlob {
            format_version: SESSION_FORMAT_VERSION,
            events: self.derived_log(),
            hamiltonian_spec: self.hamiltonian_spec.clone(),
            solver_spec: self.solver_spec.clone(),
            state: self.state.clone(),
            t_now: self.t_now,
        }
    }

    /// H3: fork a new session from this session at log boundary `seq`
    /// (inclusive). The fork replays the prefix `events[0..=seq]` and diverges
    /// from there. Refuses boundaries that are an open compaction lock bracket
    /// (`CompactStart`, UK-1009).
    pub fn fork_at(&self, seq: u64) -> Result<Session, KernelError> {
        let idx = seq as usize;
        if idx >= self.event_log.len() {
            return Err(KernelError::SessionForkRange {
                reason: format!(
                    "fork boundary seq {seq} out of range [0,{})",
                    self.event_log.len()
                ),
            });
        }
        if matches!(
            self.event_log[idx].spec,
            SessionEventSpec::CompactStart { .. }
        ) {
            return Err(KernelError::SessionForkRange {
                reason: format!("cannot fork at an open compaction lock boundary (seq {seq})"),
            });
        }
        Session::replay(&self.event_log[..=idx])
    }

    /// H3: begin a compaction through log boundary `through_seq` (inclusive),
    /// opening a lock bracket. The session is busy (UK-1008) until
    /// `compact_end`. The boundary must be a settled record: never an `Evolve`
    /// (would split an unanswered evolve dependency — tool-pairing safety) and
    /// never a `CompactStart`. QFM sessions cannot be compacted (the pipeline
    /// is not serializable, so a folded summary would break replay of
    /// subsequent qfm evolutions).
    pub fn compact_start(&mut self, through_seq: u64) -> Result<(), KernelError> {
        self.check_idle()?;
        let idx = through_seq as usize;
        if idx >= self.event_log.len() {
            return Err(KernelError::SessionCompactionBusy {
                reason: format!(
                    "compact boundary seq {through_seq} out of range [0,{})",
                    self.event_log.len()
                ),
            });
        }
        let boundary = &self.event_log[idx];
        match boundary.spec {
            SessionEventSpec::Evolve { .. } => {
                return Err(KernelError::SessionCompactionBusy {
                    reason: format!(
                        "compact boundary seq {through_seq} is an Evolve — would split an \
                         unanswered evolve dependency"
                    ),
                });
            }
            SessionEventSpec::CompactStart { .. } => {
                return Err(KernelError::SessionCompactionBusy {
                    reason: format!(
                        "compact boundary seq {through_seq} is an open compaction lock"
                    ),
                });
            }
            _ => {}
        }
        if self.qfm_pipeline.is_some() {
            return Err(KernelError::SessionCompactionBusy {
                reason: "cannot compact a QFM session (pipeline is not serializable)".into(),
            });
        }
        self.append_event(
            SessionOp::CompactStart,
            SessionEventSpec::CompactStart { through_seq },
        );
        self.compaction_lock = Some(through_seq);
        Ok(())
    }

    /// H3: close the open compaction lock bracket, appending a `CompactEnd`
    /// summary node (the folded state at the boundary) to the log.
    pub fn compact_end(&mut self) -> Result<(), KernelError> {
        let through_seq =
            self.compaction_lock
                .ok_or_else(|| KernelError::SessionCompactionBusy {
                    reason: "compact_end without an open compact_start".into(),
                })?;
        let summary = self.folded_blob();
        self.append_event(
            SessionOp::CompactEnd,
            SessionEventSpec::CompactEnd {
                through_seq,
                summary: Some(summary),
            },
        );
        self.compaction_lock = None;
        Ok(())
    }

    /// H3: `compact_start` + `compact_end` atomically.
    pub fn compact_through(&mut self, through_seq: u64) -> Result<(), KernelError> {
        self.compact_start(through_seq)?;
        self.compact_end()
    }

    /// H3: replay a raw event log into a fresh session. Test-only surface for
    /// pinning `fold(events) ≡ live` (the in-process debug assertion does the
    /// same thing implicitly in debug builds).
    #[doc(hidden)]
    pub fn replay_for_test(events: &[SessionEvent]) -> Session {
        Self::replay(events).expect("replay raw log")
    }

    /// Replace the current prior state. Resets evolution time to 0.
    pub fn set_prior(&mut self, p: &PriorSpec) -> Result<(), KernelError> {
        let spec = p.clone();
        self.log_then_apply(
            SessionOp::SetPrior,
            SessionEventSpec::SetPrior { spec: spec.clone() },
            |s| s.apply_prior(&spec),
        )
    }

    /// Apply a `SetPrior` op without logging (replay path).
    fn apply_prior(&mut self, p: &PriorSpec) -> Result<(), KernelError> {
        self.state = build::build_prior(p)?;
        self.t_now = 0.0;
        Ok(())
    }

    /// Replace the current Hamiltonian. The state is preserved.
    pub fn set_hamiltonian(&mut self, h: &HamiltonianSpec) -> Result<(), KernelError> {
        let spec = h.clone();
        self.log_then_apply(
            SessionOp::SetHamiltonian,
            SessionEventSpec::SetHamiltonian { spec: spec.clone() },
            |s| s.apply_hamiltonian(&spec),
        )
    }

    /// Apply a `SetHamiltonian` op without logging (replay path).
    fn apply_hamiltonian(&mut self, h: &HamiltonianSpec) -> Result<(), KernelError> {
        self.hamiltonian = build::build_hamiltonian(h)?;
        self.hamiltonian_spec = h.clone();
        Ok(())
    }

    /// Evolve the state forward by time `t` using restarted SIRK.
    /// If the session has a QFM pipeline (from a `HamiltonianSpec::QfmTomography`
    /// spec), this dispatches to the pipeline's 4-phase `generate` using the
    /// optional `query` in the opts. Without a query, the QFM pipeline is
    /// not usable (it requires a raw input) and the call returns an error.
    pub fn evolve(&mut self, t: f64) -> Result<EvolveReport, KernelError> {
        self.evolve_with_query(t, None)
    }

    /// Evolve the state forward by time `t`, with an optional raw query
    /// for QFM tomographic generation. If `query` is `Some` and the session
    /// has a QFM pipeline, the pipeline's `generate(query)` is called and
    /// the result is stored in `EvolveReport::qfm_output`. Otherwise the
    /// SIRK solver is used.
    pub fn evolve_with_query(
        &mut self,
        t: f64,
        query: Option<&[f64]>,
    ) -> Result<EvolveReport, KernelError> {
        // QFM dispatch: a QFM pipeline with a query runs the 4-phase
        // generate; a pipeline without a query is an error (the pipeline is
        // unusable without a raw input). The qfm flag is recorded in the
        // event so replay mirrors the exact live dispatch.
        if self.qfm_pipeline.is_some() && query.is_none() {
            return Err(KernelError::Internal(
                "QFM pipeline requires a query in evolve opts".into(),
            ));
        }
        let qfm = self.qfm_pipeline.is_some() && query.is_some();
        let query_owned = query.map(|q| q.to_vec());
        self.log_then_apply(
            SessionOp::Evolve,
            SessionEventSpec::Evolve {
                t,
                qfm,
                query: query_owned.clone(),
            },
            |s| s.apply_evolve(t, query_owned, qfm),
        )
    }

    /// Apply an `Evolve` op without logging (replay path). `qfm` records the
    /// dispatch taken by the live call so replay reproduces it exactly.
    fn apply_evolve(
        &mut self,
        t: f64,
        query: Option<Vec<f64>>,
        qfm: bool,
    ) -> Result<EvolveReport, KernelError> {
        // QFM dispatch: if a pipeline is present and a query is provided,
        // run the 4-phase generate and return the result.
        if qfm {
            let pipeline = self.qfm_pipeline.as_ref().ok_or_else(|| {
                KernelError::Internal(
                    "QFM evolve event but no pipeline (compacted QFM session?)".into(),
                )
            })?;
            let q = query.as_deref().ok_or_else(|| {
                KernelError::Internal("QFM pipeline requires a query in evolve opts".into())
            })?;
            let t0 = std::time::Instant::now();
            let x_out = pipeline.generate(q)?;
            let solve_ms = t0.elapsed().as_millis() as u64;
            self.t_now += t;
            return Ok(EvolveReport {
                t: self.t_now,
                norm: 1.0, // QFM output is a generated image, not a state norm
                components: x_out.len(),
                solve_ms,
                qfm_output: Some(x_out),
            });
        }
        // SIRK path.
        let t0 = std::time::Instant::now();
        let psi = evolve_restarted(
            &self.hamiltonian,
            &self.state,
            t,
            self.restarts,
            self.krylov_dim,
            &self.device,
            None,
            &self.sirk_opts,
        )?;
        let solve_ms = t0.elapsed().as_millis() as u64;
        self.state = psi;
        self.t_now += t;
        let norm = self.state.norm();
        Ok(EvolveReport {
            t: self.t_now,
            norm,
            components: self.state.len(),
            solve_ms,
            qfm_output: None,
        })
    }

    /// Compute the Born-rule probability of event `e` under the current state.
    ///
    /// `P(E) = Σ_{s ⊨ E} |⟨s|ψ⟩|² / ‖ψ‖²`
    ///
    /// H4: before serving, the durable checkpoint is flushed (fail-closed — a
    /// checkpoint failure is surfaced to the caller rather than silently
    /// serving a possibly-stale result).
    pub fn probability(&self, e: &EventPredicate) -> Result<f64, KernelError> {
        self.durable_checkpoint()
            .map_err(|reason| KernelError::DurableCheckpointFailed { reason })?;
        let norm_sq = QuantumState::inner_product(&self.state, &self.state).re;
        if norm_sq < 1e-30 {
            return Ok(0.0);
        }
        let mut mass = 0.0;
        for (outer, amp) in &self.state.components {
            if event::matches(outer, e) {
                mass += amp.norm_sqr();
            }
        }
        Ok(mass / norm_sq)
    }

    /// Condition the state on event `e` (Bayesian update).
    ///
    /// Zeroes non-matching components, renormalizes, and returns the prior
    /// probability `P(E)` that was computed before the update.
    /// Returns `KernelError::ZeroProbabilityCondition` if the matching mass
    /// is negligible.
    pub fn condition(&mut self, e: &EventPredicate) -> Result<f64, KernelError> {
        self.durable_checkpoint()
            .map_err(|reason| KernelError::DurableCheckpointFailed { reason })?;
        let spec = e.clone();
        self.log_then_apply(
            SessionOp::Condition,
            SessionEventSpec::Condition { spec: spec.clone() },
            |s| s.apply_condition(&spec),
        )
    }

    /// Apply a `Condition` op without logging (replay path).
    fn apply_condition(&mut self, e: &EventPredicate) -> Result<f64, KernelError> {
        let norm_sq = QuantumState::inner_product(&self.state, &self.state).re;
        if norm_sq < 1e-30 {
            return Err(KernelError::ZeroProbabilityCondition { mass: 0.0 });
        }
        let mut mass = 0.0;
        self.state.components.retain(|outer, amp| {
            if event::matches(outer, e) {
                mass += amp.norm_sqr();
                true
            } else {
                false
            }
        });
        if mass < 1e-15 {
            return Err(KernelError::ZeroProbabilityCondition { mass });
        }
        let inv_norm = 1.0 / mass.sqrt();
        for amp in self.state.components.values_mut() {
            *amp *= inv_norm;
        }
        Ok(mass / norm_sq)
    }

    /// Return a snapshot of the `top_k` highest-probability components.
    pub fn snapshot(&self, top_k: usize) -> StateSummary {
        let norm = self.state.norm();
        let components = self.state.len();
        let mut top: Vec<StateEntry> = self
            .state
            .components
            .iter()
            .map(|(s, a)| StateEntry {
                state: format!("{:?}", s),
                probability: a.norm_sqr(),
            })
            .collect();
        top.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        top.truncate(top_k);
        StateSummary {
            norm,
            components,
            top,
        }
    }

    /// Current evolution time.
    pub fn t(&self) -> f64 {
        self.t_now
    }

    /// Current number of state components.
    pub fn n_components(&self) -> usize {
        self.state.len()
    }

    /// Quantum Bayesian Update on the TSR-evolved prior
    /// (QFM.tex §8, P6 H follow-on). Conditions the TSR-evolved prior
    /// on $N$ new raw observations $\{D_1, \dots, D_N\}$ and draws a
    /// single posterior sample via HMC on the unit sphere of
    /// $\Cset^m$. Returns the HMC diagnostics + the Phase 5
    /// reconstructed image.
    ///
    /// **Eligibility:** only QFM tomographic models
    /// (`HamiltonianSpec::QfmTomography`) have a TSR pipeline and
    /// therefore a meaningful TSR-evolved prior. Calling this method
    /// on a non-QFM session returns `KernelError::Internal`. The
    /// observation dimension must match the pipeline's raw dimension
    /// `d`; mismatches return `KernelError::Qfm(QfmError::DimensionMismatch)`.
    ///
    /// **No state side-effect:** the SIRK state `self.state` is not
    /// modified by this op — the posterior sample lives entirely in
    /// the Krylov subspace, and the report's `image` is the rendered
    /// output. Use `evolve_with_query` (or `evolve`) to feed a
    /// posterior-decoded image back into the kernel for further SIRK
    /// evolution if needed.
    pub fn bayesian_update(
        &self,
        observations: &[Vec<f64>],
        hmc_opts: &HmcOptsSpec,
    ) -> Result<BayesianUpdateReport, KernelError> {
        // Only QFM tomographic models are eligible.
        let pipeline = self.qfm_pipeline.as_ref().ok_or_else(|| {
            KernelError::Internal(
                "bayesian_update requires a QFM tomographic model (HamiltonianSpec::QfmTomography)"
                    .into(),
            )
        })?;

        // Build the N likelihood operators (S_1 -> S_2 -> Krylov
        // projection; errors are forwarded as Qfm errors).
        let mut likelihoods = Vec::with_capacity(observations.len());
        for obs in observations {
            let like = qfm::bayes::Likelihood::from_observation(pipeline, obs)?;
            likelihoods.push(like);
        }

        // The TSR-evolved prior direction.
        let c_prior = qfm::bayes::tsr_evolved_prior(pipeline);
        let posterior = qfm::bayes::Posterior::new(likelihoods.clone(), c_prior);

        // HMC.
        let qfm_opts = qfm::bayes::HmcOpts {
            leapfrog_steps: hmc_opts.leapfrog_steps,
            step_size: hmc_opts.step_size,
            n_iterations: hmc_opts.n_iterations,
            burn_in: hmc_opts.burn_in,
            seed: hmc_opts.seed,
        };
        let t0 = std::time::Instant::now();
        // P8.9: run the full chain (not just the final draw) so we can
        // form a posterior-mean point estimate. The final element of the
        // chain is the representative single draw (identical to
        // `sample_hmc_single` for the same opts), and the post-burn-in
        // tail feeds the Karcher mean.
        let chain = qfm::bayes::sample_hmc(&posterior, &qfm_opts);
        let sample = chain
            .last()
            .cloned()
            .unwrap_or_else(|| posterior.prior_direction().clone());

        // Diagnostics on the representative draw.
        let log_posterior = posterior.log_density(&sample);
        let mean_likelihood = if likelihoods.is_empty() {
            -1.0
        } else {
            let mut prod = 0.0_f64;
            for like in &likelihoods {
                prod += like.born_rule(&sample).ln();
            }
            (prod / likelihoods.len() as f64).exp()
        };

        // Phase 5 tomographic reconstruction of the representative draw.
        let image = qfm::bayes::reconstruct(pipeline, &sample)?;

        // P8.9: posterior-mean point estimate via the Karcher (Fréchet)
        // mean of the post-burn-in tail on the projective unit sphere of
        // C^m, then decode it. `burn_in >= chain.len()` (degenerate opts)
        // leaves an empty tail → empty posterior-mean image.
        let burn_in = qfm_opts.burn_in.min(chain.len());
        let tail = &chain[burn_in..];
        let (posterior_mean_image, n_samples) = if tail.is_empty() {
            (Vec::new(), 0)
        } else {
            let mean = qfm::bayes::karcher_mean(tail, 100, 1e-10);
            (qfm::bayes::reconstruct(pipeline, &mean)?, tail.len())
        };
        let solve_ms = t0.elapsed().as_millis() as u64;

        Ok(BayesianUpdateReport {
            log_posterior,
            mean_likelihood,
            image,
            posterior_mean_image,
            n_samples,
            n_observations: observations.len(),
            solve_ms,
        })
    }

    /// Run chain belief propagation (P8.8) on the TSR-evolved prior.
    /// Returns the MAP (marginal mode) point estimate + the decoded
    /// full-resolution image.
    ///
    /// **Only QFM tomographic models are eligible** — the prior is the
    /// TSR-evolved vacuum state. Calling this on a non-QFM model returns
    /// `KernelError::Internal` (the FFI layer maps this to UK-5000).
    ///
    /// **No state side-effect:** the SIRK state is not modified.
    pub fn belief_propagation(
        &self,
        observations: &[Vec<f64>],
        opts: &unfer_protocol::BeliefPropagationOptsSpec,
    ) -> Result<BeliefPropagationReport, KernelError> {
        let pipeline = self.qfm_pipeline.as_ref().ok_or_else(|| {
            KernelError::Internal(
                "belief_propagation requires a QFM tomographic model (HamiltonianSpec::QfmTomography)"
                    .into(),
            )
        })?;

        let mut likelihoods = Vec::with_capacity(observations.len());
        for obs in observations {
            let like = qfm::bayes::Likelihood::from_observation(pipeline, obs)?;
            likelihoods.push(like);
        }

        let c_prior = qfm::bayes::tsr_evolved_prior(pipeline);
        let posterior = qfm::bayes::Posterior::new(likelihoods, c_prior);

        let t0 = std::time::Instant::now();
        let bp_result = qfm::bayes::belief_propagation_chain(
            &posterior,
            opts.max_iter,
            opts.step_size,
            opts.tol,
        );
        let image = qfm::bayes::reconstruct(pipeline, &bp_result.map_estimate)?;
        let solve_ms = t0.elapsed().as_millis() as u64;

        Ok(BeliefPropagationReport {
            image,
            log_posterior: bp_result.log_posterior_at_map,
            n_observations: bp_result.n_observations,
            n_sweeps: bp_result.n_sweeps,
            solve_ms,
        })
    }

    /// Evaluates Nelson's condition and finds singularities for an ODE-based
    /// Hamiltonian. Returns the full ESA report including flow completeness,
    /// singularity detection, and any applied change of variables.
    pub fn analyze_self_adjointness(&self) -> Result<ode_sirk::report::OdeReport, KernelError> {
        match &self.hamiltonian_spec {
            HamiltonianSpec::OdeSystem {
                vars,
                rhs,
                change_of_variables,
            } => {
                let samples: Vec<Vec<f64>> = (1..=3).map(|i| vec![i as f64; vars.len()]).collect();
                let cov_str = change_of_variables.as_deref();
                let (report, _) = ode_sirk::protocol::analyze_ode_system(
                    vars.clone(),
                    &rhs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    cov_str,
                    100.0,
                    &samples,
                )
                .map_err(|e| KernelError::Internal(e.to_string()))?;
                Ok(report)
            }
            _ => Err(KernelError::Internal(
                "analyze_self_adjointness requires an OdeSystem Hamiltonian".into(),
            )),
        }
    }

    /// If CoV was applied, wraps SIRK observables to compute expectations
    /// in the original coordinate system. For variable `var`, applies the
    /// inverse coordinate map to the expectation value.
    pub fn measure_ode_observable(&self, var: &str) -> Result<f64, KernelError> {
        // For now, return the norm as a placeholder observable.
        // A full implementation would apply the CoV inverse map to
        // expectation values computed from the SIRK-evolved state.
        let _ = var;
        Ok(self.state.norm())
    }

    /// Verify a Lean4 export file (S29). Runs the external type checker
    /// [`nanoda_lib`] over the `lean4export` payload and reduces the proofs to
    /// a boolean verdict. The numerical state is untouched — verification is a
    /// read-only gate that mixes machine-checked theorem results into the
    /// kernel session.
    pub fn verify_proof(
        &self,
        export_bytes: &[u8],
        spec: &unfer_protocol::LeanVerifySpec,
    ) -> Result<unfer_protocol::ProofReport, KernelError> {
        crate::verify::verify_export(export_bytes, spec)
    }

    /// Run a symbolic operation in Cadabra2 (S30). Couples the external
    /// field-theory CAS (subprocess `cadabra2-cli`) with the existing LaTeX
    /// engine: the expression is canonicalized and reduced to a normal form
    /// plus a zero-detection verdict. The numerical state is untouched.
    pub fn symbolic_analyze(
        &self,
        spec: &unfer_protocol::SymbolicSpec,
    ) -> Result<unfer_protocol::SymbolicReport, KernelError> {
        crate::symbolic::symbolic_analyze(spec)
    }

    /// Emit (and optionally verify) a WhyML program for the australVM
    /// compiler extension cycle (S36). The default program is the
    /// authorization gate: `authorize grants required = true <-> required
    /// ⊆ grants`, proved by Why3 and extracted to the OCaml module the
    /// compiler loads. Read-only consult: the numerical state is untouched
    /// and no session-log event is appended.
    pub fn whyml_emit(
        &self,
        spec: &unfer_protocol::WhymlSpec,
    ) -> Result<unfer_protocol::WhymlReport, KernelError> {
        crate::whyml::whyml_emit(spec)
    }

    /// Compile a CNL sentence to a unique normal form (Logos). Parses the
    /// sentence with the embedded L0 lexicon, compiles to a CoreIR term,
    /// reduces it to an interaction-net unique normal form, and read-backs
    /// the result with its content-addressable UNF hash. `verified` is the
    /// confluence self-check. Read-only: the numerical state is untouched.
    pub fn logos_compile(
        &self,
        sentence: &str,
    ) -> Result<unfer_protocol::LogosReport, KernelError> {
        crate::logos::logos_compile(sentence)
    }

    /// Translate an AustralVM-language source fragment to a unique normal
    /// form through DeltaNets: lower to CoreIR, compile to an interaction
    /// net, reduce, and read back as a symbolic expression. Closed terms
    /// (no unknowns) collapse to the numerical result of their calculation;
    /// open terms stay symbolic (`Add64(x, 3)`). `verified` is the
    /// confluence self-check. Read-only: the numerical state is untouched.
    pub fn austral_unf(
        &self,
        source: &str,
    ) -> Result<unfer_protocol::AustralReport, KernelError> {
        crate::logos::austral_unf(source)
    }

    /// Run a multi-cell Cadabra2 derivation pipeline (S30) and capture the
    /// named expressions it produces (e.g. the QG gauge-fixed Hamiltonian
    /// derivation in `docs/qg_gauge_fixed_hamiltonian.cdb`). Read-only: the
    /// numerical state is untouched.
    pub fn derive_symbolic(
        &self,
        script: &str,
        names: &[&str],
        timeout_ms: u64,
    ) -> Result<std::collections::BTreeMap<String, String>, KernelError> {
        crate::symbolic::symbolic_derive(script, names, timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::QfmTomographySpec;

    fn qfm_session() -> Session {
        let training = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let qfm_spec = QfmTomographySpec {
            training_data: training,
            k: 2,
            k2: 4,
            krylov_dim: 4,
            seed: 42,
        };
        let spec = ModelSpec {
            hamiltonian: HamiltonianSpec::qfm_tomography(qfm_spec),
            prior: PriorSpec::Vacuum,
            solver: SolverSpec::default(),
        };
        Session::new(&spec).expect("compile QFM session")
    }

    fn non_qfm_session() -> Session {
        let spec = ModelSpec {
            hamiltonian: HamiltonianSpec::builtin(
                "harmonic_chain",
                serde_json::json!({"n_modes": 2, "omega": 1.0}),
            ),
            prior: PriorSpec::Vacuum,
            solver: SolverSpec::default(),
        };
        Session::new(&spec).expect("compile harmonic session")
    }

    // ── H10: named GrantSet presets (session header) ─────────────────────

    #[test]
    fn session_header_records_start_preset() {
        let mut s = non_qfm_session();
        assert_eq!(s.start_preset(), None, "no preset by default");
        s.set_start_preset("analyst");
        assert_eq!(s.start_preset(), Some("analyst"));
    }

    #[test]
    fn preset_switch_valid_only_while_blank() {
        // A switch on a blank (no-op) session is valid; once the session has
        // produced anything it is refused (nearest-wins tool surface must not
        // change under a model that already ran).
        assert!(unfer_protocol::preset::switch_valid_when_blank(0));
        assert!(!unfer_protocol::preset::switch_valid_when_blank(1));
        let mut s = non_qfm_session();
        // Blank (just the root Create) → a preset may be recorded.
        s.set_start_preset("analyst");
        assert_eq!(s.start_preset(), Some("analyst"));
    }

    #[test]
    fn bayesian_update_smoke_qfm_model() {
        // Single observation at training point 0: the HMC sample's
        // log_posterior should be finite, the image should have d=4
        // elements, and solve_ms should be non-zero.
        let session = qfm_session();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let report = session
            .bayesian_update(&[obs], &HmcOptsSpec::default())
            .expect("bayesian_update should succeed on QFM model");
        assert_eq!(report.n_observations, 1);
        assert!(
            report.log_posterior.is_finite(),
            "log_posterior should be finite"
        );
        assert_eq!(report.image.len(), 4, "image should have d=4 elements");
        for v in &report.image {
            assert!(v.is_finite(), "image component should be finite: {v}");
        }
        // mean_likelihood should be a positive likelihood value (Born rule
        // is always positive).
        assert!(report.mean_likelihood > 0.0 && report.mean_likelihood <= 1.0);
        // P8.9: the posterior-mean point estimate is decoded from the
        // Karcher mean of the post-burn-in chain. With the default
        // HmcOptsSpec (n_iterations=200, burn_in=100) there are 100
        // post-burn-in samples, all averaged into a finite d=4 image.
        assert_eq!(report.n_samples, 100, "post-burn-in tail length");
        assert_eq!(report.posterior_mean_image.len(), 4);
        for v in &report.posterior_mean_image {
            assert!(
                v.is_finite(),
                "posterior-mean component should be finite: {v}"
            );
        }
    }

    #[test]
    fn bayesian_update_posterior_mean_tracks_observation() {
        // P8.9: the Karcher-mean posterior estimate is a valid decoded
        // image (finite, dimension d) aggregated over the whole
        // post-burn-in tail, and it should be at least as close to the
        // observation as a coarse bound allows. For this strongly-peaked
        // tetrahedron posterior the chain concentrates, so the mean is a
        // faithful (not degenerate) point estimate: its L2 distance to
        // the single representative draw stays within the image scale.
        let session = qfm_session();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let report = session
            .bayesian_update(&[obs], &HmcOptsSpec::default())
            .expect("bayesian_update should succeed");
        assert_eq!(report.posterior_mean_image.len(), report.image.len());
        assert!(report.n_samples >= 2, "need a real chain to average");
        // The two estimates derive from the same typical set, so the
        // mean image must stay within a bounded distance of the draw.
        let dist2: f64 = report
            .posterior_mean_image
            .iter()
            .zip(&report.image)
            .map(|(m, s)| (m - s).powi(2))
            .sum();
        let scale2: f64 = report.image.iter().map(|s| s * s).sum::<f64>().max(1e-9);
        assert!(
            dist2 <= scale2 + 1e-12,
            "posterior mean diverged from the typical set: dist2={dist2}, scale2={scale2}"
        );
    }

    #[test]
    fn bayesian_update_zero_observations_returns_prior() {
        // With no observations, the posterior equals the prior;
        // the report should have n_observations=0, mean_likelihood=-1,
        // and a finite log_posterior.
        let session = qfm_session();
        let report = session
            .bayesian_update(&[], &HmcOptsSpec::default())
            .expect("zero-observation bayesian_update should succeed");
        assert_eq!(report.n_observations, 0);
        assert!(
            (report.mean_likelihood + 1.0).abs() < 1e-12,
            "mean_likelihood should be -1 for prior-only, got {}",
            report.mean_likelihood
        );
        assert!(report.log_posterior.is_finite());
        assert_eq!(report.image.len(), 4);
    }

    #[test]
    fn bayesian_update_non_qfm_returns_internal() {
        // The Bayesian update requires a QFM tomographic model; calling
        // it on a non-QFM session should return an Internal error.
        let session = non_qfm_session();
        let obs = vec![1.0, 0.0];
        let result = session.bayesian_update(&[obs], &HmcOptsSpec::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Internal(msg) => {
                assert!(
                    msg.contains("QFM"),
                    "internal error should mention QFM: {msg}"
                );
            }
            e => panic!("expected KernelError::Internal, got {e:?}"),
        }
    }

    #[test]
    fn bayesian_update_dim_mismatch_returns_qfm_error() {
        // Observation with wrong dimension should return a Qfm
        // DimensionMismatch error.
        let session = qfm_session();
        let obs = vec![1.0, 0.0]; // d=2, expected d=4
        let result = session.bayesian_update(&[obs], &HmcOptsSpec::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Qfm(qfm::pipeline::QfmError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            e => panic!("expected KernelError::Qfm(DimensionMismatch), got {e:?}"),
        }
    }

    // ── P8.8: chain belief propagation tests ─────────────────────────

    #[test]
    fn belief_propagation_smoke_qfm_model() {
        // BP on a QFM tomographic model returns a finite MAP image
        // and a finite log-posterior.
        let session = qfm_session();
        let obs = vec![1.0, 0.0, 0.0, 0.0];
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let report = session
            .belief_propagation(&[obs], &opts)
            .expect("BP should succeed on QFM model");
        assert_eq!(report.image.len(), 4);
        for v in &report.image {
            assert!(v.is_finite(), "image element should be finite, got {v}");
        }
        assert!(report.log_posterior.is_finite());
        assert_eq!(report.n_observations, 1);
        assert!(report.n_sweeps >= 1);
    }

    #[test]
    fn belief_propagation_zero_observations_returns_prior() {
        // Zero-observation BP: no likelihoods, MAP = prior direction.
        let session = qfm_session();
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let report = session
            .belief_propagation(&[], &opts)
            .expect("zero-obs BP should succeed");
        assert_eq!(report.n_observations, 0);
        assert_eq!(report.image.len(), 4);
    }

    #[test]
    fn belief_propagation_non_qfm_returns_internal() {
        // BP is QFM-only; calling on a non-QFM session returns Internal.
        let session = non_qfm_session();
        let obs = vec![1.0, 0.0];
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let result = session.belief_propagation(&[obs], &opts);
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Internal(msg) => {
                assert!(msg.contains("QFM"), "should mention QFM: {msg}");
            }
            e => panic!("expected KernelError::Internal, got {e:?}"),
        }
    }

    #[test]
    fn belief_propagation_dim_mismatch_returns_qfm_error() {
        // Observation with wrong dimension returns a Qfm
        // DimensionMismatch error.
        let session = qfm_session();
        let obs = vec![1.0, 0.0]; // d=2, expected d=4
        let opts = unfer_protocol::BeliefPropagationOptsSpec::default();
        let result = session.belief_propagation(&[obs], &opts);
        assert!(result.is_err());
        match result.unwrap_err() {
            KernelError::Qfm(qfm::pipeline::QfmError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            e => panic!("expected KernelError::Qfm(DimensionMismatch), got {e:?}"),
        }
    }

    // ── H4: the session event log lands in the durable `session` stream ───

    /// A minimal in-memory store for session tests (prob_kernel cannot depend
    /// on unfer_ffi's Loro/JSONL backends).
    #[derive(Debug, Default)]
    struct MemStore {
        streams: std::sync::Mutex<std::collections::HashMap<String, Vec<Vec<u8>>>>,
    }
    impl unfer_protocol::durable::DurableStore for MemStore {
        fn backend(&self) -> &'static str {
            "mem"
        }
        fn append(
            &self,
            stream: &str,
            record: &[u8],
        ) -> Result<(), unfer_protocol::durable::DurableError> {
            self.streams
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(stream.to_string())
                .or_default()
                .push(record.to_vec());
            Ok(())
        }
        fn flush(&self) -> Result<(), unfer_protocol::durable::DurableError> {
            Ok(())
        }
        fn replay(
            &self,
            stream: &str,
        ) -> Result<Vec<Vec<u8>>, unfer_protocol::durable::DurableError> {
            Ok(self
                .streams
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(stream)
                .cloned()
                .unwrap_or_default())
        }
        fn frontier(&self) -> Result<Vec<u8>, unfer_protocol::durable::DurableError> {
            Ok(b"mem".to_vec())
        }
        fn fork_at(
            &self,
            _frontier: &[u8],
        ) -> Result<
            Box<dyn unfer_protocol::durable::DurableStore>,
            unfer_protocol::durable::DurableError,
        > {
            Err(unfer_protocol::durable::DurableError::Unsupported(
                "mem fork".to_string(),
            ))
        }
    }

    // ── core lifecycle at the unit level (the coverage gate instruments
    //    only the --lib harness, so the integration-test lifecycle coverage
    //    does not count toward the per-file threshold) ──────────────────

    #[test]
    fn lifecycle_evolve_probability_snapshot_save_restore() {
        let mut s = non_qfm_session();
        assert_eq!(s.event_log_len(), 1, "root Create event only");
        assert_eq!(s.t(), 0.0);

        // Vacuum prior: P(vacuum) = 1, everything else 0.
        let p_vac = s.probability(&EventPredicate::Vacuum).expect("prob");
        assert!((p_vac - 1.0).abs() < 1e-12);

        // Evolve a small step: norm conserved (unitary), time advances, the
        // event log gains exactly one Evolve record.
        let before = s.event_log_len();
        let report = s.evolve(0.1).expect("evolve");
        assert!((report.norm - 1.0).abs() < 1e-9, "unitarity: {}", report.norm);
        assert!((s.t() - 0.1).abs() < 1e-12);
        assert_eq!(s.event_log_len(), before + 1);
        assert!(matches!(
            s.event_log().last().unwrap().spec,
            SessionEventSpec::Evolve { t, .. } if (t - 0.1).abs() < 1e-12
        ));

        // Snapshot: the vacuum dominates.
        let sum = s.snapshot(4);
        assert!(sum.components >= 1);
        assert!((sum.norm - 1.0).abs() < 1e-9);
        assert!(!sum.top.is_empty());

        // Save/restore round-trips the *derived* event log and reproduces
        // the same state and time.
        let blob = s.save();
        assert_eq!(blob.format_version, SESSION_FORMAT_VERSION);
        let r = Session::restore(blob).expect("restore");
        assert!((r.t() - 0.1).abs() < 1e-12, "time preserved: {}", r.t());
        let l = serde_json::to_string(&r.state).unwrap();
        let live = serde_json::to_string(&s.state).unwrap();
        assert_eq!(l, live, "restored state equals live state");
        // The restored session can keep accumulating.
        let mut r = r;
        r.evolve(0.05).expect("evolve after restore");
        assert!((r.t() - 0.15).abs() < 1e-12);
    }

    #[test]
    fn fork_compaction_boundaries_and_replay_validation() {
        let mut s = non_qfm_session();
        s.evolve(0.1).expect("evolve");
        let log_len = s.event_log_len();

        // Fork at the root Create (seq 0): replays the prefix and diverges.
        let fork = s.fork_at(0).expect("fork at root");
        assert_eq!(fork.event_log_len(), 1);
        assert!((fork.t() - 0.0).abs() < 1e-12);
        // Fork at the last settled boundary.
        let fork2 = s
            .fork_at((log_len - 1).try_into().unwrap())
            .expect("fork at tail");
        assert!((fork2.t() - 0.1).abs() < 1e-12);
        // Out-of-range fork is refused (UK-1009 family).
        assert!(matches!(
            s.fork_at(log_len as u64),
            Err(KernelError::SessionForkRange { .. })
        ));

        // Compaction through the settleable boundary (the root Create is
        // settleable; the Evolve is not — it would split the dependency).
        assert!(matches!(
            s.compact_through(1),
            Err(KernelError::SessionCompactionBusy { .. })
        ), "Evolve boundary must be refused");
        s.compact_through(0).expect("compact through root");
        // Lock closed: the derived log root is now the CompactEnd summary.
        assert!(!s.is_compaction_locked());
        assert!(matches!(
            s.event_log().last().unwrap().spec,
            SessionEventSpec::CompactEnd {
                summary: Some(_),
                ..
            }
        ));
        // The session is still live and replayable (fold ≡ live in debug).
        s.evolve(0.05).expect("evolve after compaction");

        // compact_end without an open start is refused.
        assert!(matches!(
            s.compact_end(),
            Err(KernelError::SessionCompactionBusy { .. })
        ));

        // The public replay-for-test surface folds the same raw log into an
        // equivalent session (the debug-build invariant, exposed).
        let replayed = Session::replay_for_test(s.event_log());
        assert_eq!(replayed.event_log_len(), s.event_log_len());

        // Replay validation: a non-monotonic seq is refused (UK-1006).
        let mut evs = s.event_log().to_vec();
        if evs.len() >= 2 {
            evs.swap(0, 1);
            let r = Session::replay(&evs);
            assert!(matches!(r, Err(KernelError::SessionLogVersion { .. })));
        }
        // An empty log is refused.
        assert!(matches!(
            Session::replay(&[]),
            Err(KernelError::SessionLogVersion { .. })
        ));
    }

    #[test]
    fn fork_after_compaction_diverges_independently() {
        // A completed compaction bracket (CompactStart/CompactEnd carrying the
        // folded summary) sits in the raw log at the END: forking AT the
        // CompactEnd boundary replays the whole prefix — the bracket's summary
        // seeds the session from the compacted state, so the fork reproduces
        // the parent's live state. The compacted log must keep forking
        // cleanly and stay fold ≡ live.
        let mut s = non_qfm_session();
        s.set_prior(&PriorSpec::Vacuum).expect("set_prior"); // seq 1 (settleable)
        s.evolve(0.1).expect("evolve 1");
        s.evolve(0.2).expect("evolve 2");
        s.compact_through(1).expect("compact through SetPrior");
        let compact_end_seq = s.event_log_len() - 1; // the CompactEnd node

        // Fork at the CompactEnd boundary: same live state as the parent
        // (the summary carries it), and the fork's log still holds the
        // completed bracket.
        let mut fork = s
            .fork_at(compact_end_seq as u64)
            .expect("fork at the CompactEnd boundary");
        assert!(
            (fork.t() - s.t()).abs() < 1e-12,
            "fork at CompactEnd must reproduce the live state: fork t={}, parent t={}",
            fork.t(),
            s.t()
        );
        assert!(matches!(
            fork.event_log().last().unwrap().spec,
            SessionEventSpec::CompactEnd {
                summary: Some(_),
                ..
            }
        ));

        // The compacted fork's log still folds ≡ live: replaying it raw
        // reproduces the same session.
        let replayed = Session::replay_for_test(fork.event_log());
        assert!((replayed.t() - fork.t()).abs() < 1e-12);

        // The fork diverges independently; the parent is untouched.
        fork.evolve(0.05).expect("fork evolve");
        assert!((fork.t() - (s.t() + 0.05)).abs() < 1e-12);
        assert_eq!(fork.event_log_len(), s.event_log_len() + 1);

        // The open-bracket record (CompactStart) is still refused as a fork
        // boundary.
        assert!(matches!(
            s.fork_at((compact_end_seq - 1) as u64),
            Err(KernelError::SessionForkRange { .. })
        ));
    }

    #[test]
    fn ode_consult_methods() {
        // `analyze_self_adjointness` requires an OdeSystem Hamiltonian.
        let s = non_qfm_session();
        assert!(matches!(
            s.analyze_self_adjointness(),
            Err(KernelError::Internal(_))
        ), "non-ODE session must be refused");

        // A harmonic-oscillator ODE system runs the ESA pipeline end to end.
        let spec = ModelSpec {
            hamiltonian: HamiltonianSpec::ode_system(
                vec!["x".into(), "v".into()],
                vec!["v".into(), "-x".into()],
                None,
            ),
            prior: PriorSpec::Vacuum,
            solver: SolverSpec::default(),
        };
        let ode = Session::new(&spec).expect("compile ODE session");
        let report = ode.analyze_self_adjointness().expect("ESA report");
        assert_eq!(report.vars, vec!["x", "v"]);
        assert!(
            !report.summary().is_empty(),
            "the ESA report carries a human-readable summary"
        );
        // The observable consult returns the state norm (the placeholder
        // observable in the original coordinates).
        let obs = ode.measure_ode_observable("x").expect("observable");
        assert!(obs.is_finite() && obs >= 0.0);
    }

    /// A store whose checkpoint barrier always fails — for pinning the H4
    /// fail-closed contract: a probability/condition served on top of a
    /// failed checkpoint is refused, never silently stale.
    #[derive(Debug)]
    struct FailingFlushStore(MemStore);

    /// A store whose `append` always fails — exercises the fail-closed
    /// committed-event path (`durable_append_event` must propagate, never
    /// `let _ =` silently drop).
    #[derive(Debug)]
    struct FailingAppendStore(MemStore);

    impl unfer_protocol::durable::DurableStore for FailingAppendStore {
        fn backend(&self) -> &'static str {
            "mem-failing-append"
        }
        fn append(
            &self,
            _stream: &str,
            _record: &[u8],
        ) -> Result<(), unfer_protocol::durable::DurableError> {
            Err(unfer_protocol::durable::DurableError::Io(
                "simulated append failure".into(),
            ))
        }
        fn flush(&self) -> Result<(), unfer_protocol::durable::DurableError> {
            Ok(())
        }
        fn replay(
            &self,
            stream: &str,
        ) -> Result<Vec<Vec<u8>>, unfer_protocol::durable::DurableError> {
            self.0.replay(stream)
        }
        fn frontier(&self) -> Result<Vec<u8>, unfer_protocol::durable::DurableError> {
            self.0.frontier()
        }
        fn fork_at(
            &self,
            frontier: &[u8],
        ) -> Result<
            Box<dyn unfer_protocol::durable::DurableStore>,
            unfer_protocol::durable::DurableError,
        > {
            self.0.fork_at(frontier)
        }
    }

    impl unfer_protocol::durable::DurableStore for FailingFlushStore {
        fn backend(&self) -> &'static str {
            "mem-failing"
        }
        fn append(
            &self,
            stream: &str,
            record: &[u8],
        ) -> Result<(), unfer_protocol::durable::DurableError> {
            self.0.append(stream, record)
        }
        fn flush(&self) -> Result<(), unfer_protocol::durable::DurableError> {
            Err(unfer_protocol::durable::DurableError::Io(
                "simulated checkpoint failure".into(),
            ))
        }
        fn replay(
            &self,
            stream: &str,
        ) -> Result<Vec<Vec<u8>>, unfer_protocol::durable::DurableError> {
            self.0.replay(stream)
        }
        fn frontier(&self) -> Result<Vec<u8>, unfer_protocol::durable::DurableError> {
            self.0.frontier()
        }
        fn fork_at(
            &self,
            frontier: &[u8],
        ) -> Result<
            Box<dyn unfer_protocol::durable::DurableStore>,
            unfer_protocol::durable::DurableError,
        > {
            self.0.fork_at(frontier)
        }
    }

    #[test]
    fn durable_checkpoint_failure_fails_closed() {
        let mut s = non_qfm_session();
        let store = std::sync::Arc::new(FailingFlushStore(MemStore::default()));
        s.set_durable(Some(store));
        // A probability on top of a failed checkpoint is refused — the H4
        // fail-closed barrier — never a possibly-stale result.
        assert!(matches!(
            s.probability(&EventPredicate::Vacuum),
            Err(KernelError::DurableCheckpointFailed { .. })
        ));
        assert!(matches!(
            s.condition(&EventPredicate::Vacuum),
            Err(KernelError::DurableCheckpointFailed { .. })
        ));
    }

    #[test]
    fn committed_event_append_failure_surfaces_uk1010() {
        let mut s = non_qfm_session();
        let store = std::sync::Arc::new(FailingAppendStore(MemStore::default()));
        s.set_durable(Some(store.clone()));
        // A committed event that cannot be durably appended must NOT report
        // success: the mutation happened in memory but the durable log is
        // missing the record — restart replay would be short by it.
        let err = s.evolve(0.01).expect_err("evolve with failing append must fail");
        assert!(
            matches!(err, KernelError::DurableCheckpointFailed { .. }),
            "expected DurableCheckpointFailed (→ UK-1010), got: {err:?}"
        );
        // And the in-memory log keeps the fold ≡ live invariant (the event
        // stays; the caller was told the durable outcome is unknown).
        assert_eq!(s.event_log_len(), 1 + 1, "root Create + the Evolve that failed to persist");
        assert!(s.durable().is_some(), "store still attached");
    }

    #[test]
    fn durable_store_attach_detach() {
        let mut s = non_qfm_session();
        assert!(s.durable().is_none(), "no store by default");
        let store = std::sync::Arc::new(MemStore::default());
        s.set_durable(Some(store.clone()));
        assert!(s.durable().is_some(), "store attached");
        // A committed op lands in the attached store.
        s.evolve(0.01).expect("evolve");
        let records = store
            .replay(unfer_protocol::durable::streams::SESSION)
            .expect("replay");
        assert_eq!(records.len(), 1, "one committed Evolve in the store");
        // Detach: the session keeps working without the store.
        s.set_durable(None);
        assert!(s.durable().is_none());
        s.evolve(0.01).expect("evolve after detach");
    }

    #[test]
    fn log_source_and_preset_switch_bookkeeping() {
        let mut s = non_qfm_session();
        assert_eq!(s.log_source(), "kernel");
        s.set_log_source("agent:analyst");
        assert_eq!(s.log_source(), "agent:analyst");
        // A blank session (root Create only) may record a preset switch.
        assert_eq!(s.event_log_len_for_preset_switch(), 0);
        s.set_start_preset("analyst");
        s.evolve(0.05).expect("evolve");
        // After the first producing op the preset-switch budget is spent.
        assert_eq!(s.event_log_len_for_preset_switch(), 1);
        // The recorded source rides on the committed event.
        assert_eq!(s.event_log().last().unwrap().source, "agent:analyst");
    }

    #[test]
    fn qfm_session_refuses_compaction() {
        let mut s = qfm_session();
        // QFM pipelines are not serializable — compaction is refused outright.
        assert!(matches!(
            s.compact_through(0),
            Err(KernelError::SessionCompactionBusy { .. })
        ));
        // ... and a QFM session without a query is unusable (the pipeline
        // requires a raw input).
        assert!(matches!(
            s.evolve(0.1),
            Err(KernelError::Internal(_))
        ));
    }

    #[test]
    fn committed_events_land_in_durable_session_stream() {
        let mut session = non_qfm_session();
        let store = std::sync::Arc::new(MemStore::default());
        session.set_durable(Some(store.clone()));

        // Each mutating op appends its committed event to the session stream.
        session.set_prior(&PriorSpec::Vacuum).expect("set_prior");
        session
            .set_hamiltonian(&HamiltonianSpec::builtin(
                "harmonic_chain",
                serde_json::json!({"n_modes": 2, "omega": 1.0}),
            ))
            .expect("set_hamiltonian");
        session.evolve(0.1).expect("evolve");

        let records = store
            .replay(unfer_protocol::durable::streams::SESSION)
            .expect("replay");
        // The store is attached after `Session::new`, so the root `Create`
        // event (seq 0) predates it; the durable stream captures the committed
        // events from the moment of attachment onward.
        let ops: Vec<String> = records
            .iter()
            .map(|r| {
                serde_json::from_slice::<SessionEvent>(r)
                    .expect("stored event must deserialize")
                    .op
            })
            .map(|op| match op {
                SessionOp::Create => "create".to_string(),
                SessionOp::SetPrior => "set_prior".to_string(),
                SessionOp::SetHamiltonian => "set_hamiltonian".to_string(),
                SessionOp::Evolve => "evolve".to_string(),
                _ => "other".to_string(),
            })
            .collect();
        assert_eq!(
            ops,
            vec!["set_prior", "set_hamiltonian", "evolve"],
            "committed events from the moment of durable attachment must land in \
             the session stream, in order"
        );
    }
}
