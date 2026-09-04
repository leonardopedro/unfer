//! Math catastrophe bond ledger (Plan R, SPV with nanoda trigger).
//!
//! A math bond is a catastrophe bond whose trigger is a purely mathematical
//! proof: the sponsor locks collateral (e-coins), investors buy the bond for a
//! coupon, and if `prob_kernel::verify::verify_export` (nanoda) verifies a
//! Lean4-export proof of the specified theorem, the collateral is paid out as
//! a bounty to the researcher plus a catastrophe payment to the sponsor. If
//! the proof never arrives before maturity, investors recover their principal
//! plus coupon.
//!
//! The trigger engine runs deterministically inside `apply_op` — no human
//! oracle, no external dependency, no zkVM. Bond probability trading uses the
//! unified auction mechanism (Prebid-model) for conditional-token-like shares.
//!
//! Error codes: UK-7401..UK-7407 (see `unfer_protocol::codes`).

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use unfer_protocol::{
    Code, Diagnostic, LeanVerifySpec, MathBondId, MathBondOpKind, MathBondReport, MathBondState,
    MathBondTrigger, ProofReport, Severity,
};

/// Internal record for one investment position.
#[derive(Debug, Clone)]
pub struct Investment {
    pub investor_did: String,
    pub amount: u64,
    pub seq: u64,
}

/// Full internal state for one math bond.
#[derive(Debug, Clone)]
pub struct BondState {
    pub bond_id: MathBondId,
    pub trigger: MathBondTrigger,
    pub state: MathBondState,
    pub principal: u64,
    /// Total invested by all investors (must not exceed principal).
    pub invested: u64,
    pub coupon_rate_bps: u64,
    pub maturity_seq: u64,
    pub researcher_did: String,
    pub sponsor_did: String,
    pub investments: Vec<Investment>,
    /// The proof report if a proof was submitted (None = no submission yet).
    pub proof_report: Option<ProofReport>,
    /// The consensus-log seq at which the trigger fired (for settlement).
    pub trigger_seq: Option<u64>,
}

/// The deterministic math-bond state-transition engine.
#[derive(Debug, Default)]
pub struct MathBondLedger {
    bonds: HashMap<[u8; 32], BondState>,
}

impl MathBondLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a bond by id.
    pub fn bond(&self, id: &MathBondId) -> Option<&BondState> {
        self.bonds.get(&id.0)
    }

    /// Read-only report for a bond.
    pub fn report(&self, id: &MathBondId) -> Option<MathBondReport> {
        let s = self.bonds.get(&id.0)?;
        Some(MathBondReport {
            bond_id: s.bond_id,
            trigger: s.trigger.clone(),
            state: s.state,
            principal: s.principal,
            invested: s.invested,
            coupon_rate_bps: s.coupon_rate_bps,
            maturity_seq: s.maturity_seq,
            researcher_did: s.researcher_did.clone(),
            sponsor_did: s.sponsor_did.clone(),
            proof_report: s.proof_report.clone(),
            trigger_seq: s.trigger_seq,
        })
    }

    /// All open (non-settled) bonds.
    pub fn open_bonds(&self) -> Vec<MathBondReport> {
        self.bonds
            .values()
            .filter(|s| s.state != MathBondState::Settled)
            .filter_map(|s| self.report(&s.bond_id))
            .collect()
    }

    /// Dispatch a signed math bond op against the ledger. `actor` is the
    /// signer's DID (already verified by the caller). `seq` is the
    /// consensus-log sequence.
    pub fn apply_op(
        &mut self,
        actor: &str,
        kind: &MathBondOpKind,
        seq: u64,
    ) -> Result<Option<MathBondTrigger>, Diagnostic> {
        match kind {
            MathBondOpKind::Issue {
                trigger,
                principal,
                coupon_rate_bps,
                maturity_seq,
                researcher_did,
            } => {
                self.apply_issue(
                    actor,
                    trigger,
                    *principal,
                    *coupon_rate_bps,
                    *maturity_seq,
                    researcher_did,
                    seq,
                )?;
                Ok(None)
            }
            MathBondOpKind::Invest { bond_id, amount } => {
                self.apply_invest(actor, bond_id, *amount, seq)?;
                Ok(None)
            }
            MathBondOpKind::SubmitProof {
                bond_id,
                export_bytes,
            } => {
                let triggered = self.apply_submit_proof(actor, bond_id, export_bytes, seq)?;
                Ok(triggered)
            }
            MathBondOpKind::Mature { bond_id } => {
                self.apply_mature(bond_id, seq)?;
                Ok(None)
            }
            MathBondOpKind::Settle { bond_id } => {
                self.apply_settle(actor, bond_id, seq)?;
                Ok(None)
            }
        }
    }

    /// Issue a new math bond. Only the sponsor may issue; the bond_id must be
    /// unique.
    // The parameter list mirrors the `MathBondOp::Issue` transaction fields;
    // grouping them into a struct would churn the op/protocol types.
    #[allow(clippy::too_many_arguments)]
    fn apply_issue(
        &mut self,
        actor: &str,
        trigger: &MathBondTrigger,
        principal: u64,
        coupon_rate_bps: u64,
        maturity_seq: u64,
        researcher_did: &str,
        _seq: u64,
    ) -> Result<MathBondId, Diagnostic> {
        if principal == 0 {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "principal must be positive",
                Severity::Error,
            ));
        }
        if maturity_seq == 0 {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "maturity_seq must be positive",
                Severity::Error,
            ));
        }
        if trigger.theorem.is_empty() {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "trigger theorem label must not be empty",
                Severity::Error,
            ));
        }
        if coupon_rate_bps > 10_000 {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                format!("coupon_rate_bps {coupon_rate_bps} exceeds 10000 (100%)"),
                Severity::Error,
            ));
        }
        if researcher_did.is_empty() {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "researcher_did must not be empty",
                Severity::Error,
            ));
        }

        // Deterministic bond id from the full issue parameters (trigger spec,
        // sponsor, principal AND terms) so two bonds with the same theorem but
        // different coupon/maturity/researcher do not collide.
        let bond_id = compute_bond_id(
            trigger,
            actor,
            principal,
            coupon_rate_bps,
            maturity_seq,
            researcher_did,
        );

        if self.bonds.contains_key(&bond_id.0) {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "a bond with this trigger already exists",
                Severity::Error,
            ));
        }

        self.bonds.insert(
            bond_id.0,
            BondState {
                bond_id,
                trigger: trigger.clone(),
                state: MathBondState::Issued,
                principal,
                invested: 0,
                coupon_rate_bps,
                maturity_seq,
                researcher_did: researcher_did.to_string(),
                sponsor_did: actor.to_string(),
                investments: Vec::new(),
                proof_report: None,
                trigger_seq: None,
            },
        );
        Ok(bond_id)
    }

    /// An investor funds the bond. The bond must be in `Issued` or `Funded`
    /// state; the total invested must not exceed the principal.
    fn apply_invest(
        &mut self,
        actor: &str,
        bond_id: &MathBondId,
        amount: u64,
        seq: u64,
    ) -> Result<(), Diagnostic> {
        let bond = match self.bonds.get_mut(&bond_id.0) {
            Some(b) => b,
            None => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_UNKNOWN,
                    "unknown bond id",
                    Severity::Error,
                ));
            }
        };
        if bond.state != MathBondState::Issued && bond.state != MathBondState::Funded {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                format!("bond is {:?}, expected Issued or Funded", bond.state),
                Severity::Error,
            ));
        }
        if amount == 0 {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                "investment amount must be positive",
                Severity::Error,
            ));
        }
        let new_invested = bond.invested.checked_add(amount).ok_or_else(|| {
            Diagnostic::new(
                Code::MATHBOND_OVERFUNDED,
                "investment amount overflow",
                Severity::Error,
            )
        })?;
        if new_invested > bond.principal {
            return Err(Diagnostic::new(
                Code::MATHBOND_OVERFUNDED,
                format!(
                    "invested {} would exceed principal {}",
                    new_invested, bond.principal
                ),
                Severity::Error,
            ));
        }
        bond.invested = new_invested;
        bond.investments.push(Investment {
            investor_did: actor.to_string(),
            amount,
            seq,
        });
        if new_invested == bond.principal {
            bond.state = MathBondState::Funded;
        }
        Ok(())
    }

    /// Researcher submits a proof attempt. The bond must be `Funded` (or
    /// `Issued` for a self-funded sponsor-researcher); the actor must be the
    /// designated researcher. The proof is verified by nanoda deterministically.
    fn apply_submit_proof(
        &mut self,
        actor: &str,
        bond_id: &MathBondId,
        export_bytes: &[u8],
        seq: u64,
    ) -> Result<Option<MathBondTrigger>, Diagnostic> {
        let bond = match self.bonds.get_mut(&bond_id.0) {
            Some(b) => b,
            None => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_UNKNOWN,
                    "unknown bond id",
                    Severity::Error,
                ));
            }
        };
        if bond.state == MathBondState::Triggered {
            return Err(Diagnostic::new(
                Code::MATHBOND_ALREADY_TRIGGERED,
                "bond already triggered",
                Severity::Error,
            ));
        }
        if bond.state == MathBondState::Settled || bond.state == MathBondState::Matured {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                format!("bond is {:?}", bond.state),
                Severity::Error,
            ));
        }
        if actor != bond.researcher_did {
            return Err(Diagnostic::new(
                Code::MATHBOND_NOT_RESEARCHER,
                format!(
                    "submitter {} is not the designated researcher {}",
                    actor, bond.researcher_did
                ),
                Severity::Error,
            ));
        }
        if export_bytes.len() > bond.trigger.max_export_bytes {
            return Err(Diagnostic::new(
                Code::MATHBOND_PROOF_OVERSIZE,
                format!(
                    "proof {} bytes exceeds limit {}",
                    export_bytes.len(),
                    bond.trigger.max_export_bytes
                ),
                Severity::Error,
            ));
        }

        // Deterministic nanoda verification — the trigger engine.
        let spec = LeanVerifySpec {
            permitted_axioms: bond.trigger.permitted_axioms.clone(),
            strict: bond.trigger.strict,
            nat_extension: bond.trigger.nat_extension,
            string_extension: bond.trigger.string_extension,
            ..LeanVerifySpec::default()
        };
        let report = prob_kernel::verify::verify_export(export_bytes, &spec).map_err(|e| {
            Diagnostic::new(
                Code::MATHBOND_PROOF_REJECTED,
                format!("nanoda verification error: {e}"),
                Severity::Error,
            )
        })?;

        let triggered = report.verified;
        bond.proof_report = Some(report);

        if triggered {
            bond.state = MathBondState::Triggered;
            bond.trigger_seq = Some(seq);
            Ok(Some(bond.trigger.clone()))
        } else {
            // Proof rejected — bond stays in its current state (investors can
            // try again or wait for maturity).
            Err(Diagnostic::new(
                Code::MATHBOND_PROOF_REJECTED,
                format!(
                    "proof rejected: {}",
                    bond.proof_report
                        .as_ref()
                        .and_then(|r| r.error.as_deref())
                        .unwrap_or("unknown reason")
                ),
                Severity::Error,
            ))
        }
    }

    /// Record that the bond reached maturity without a successful trigger.
    /// The seq check (current consensus-log position vs `maturity_seq`) is the
    /// enforcement — a premature Mature is rejected even though the passage of
    /// time itself needs no authority.
    fn apply_mature(&mut self, bond_id: &MathBondId, seq: u64) -> Result<(), Diagnostic> {
        let bond = match self.bonds.get_mut(&bond_id.0) {
            Some(b) => b,
            None => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_UNKNOWN,
                    "unknown bond id",
                    Severity::Error,
                ));
            }
        };
        match bond.state {
            MathBondState::Issued | MathBondState::Funded => {}
            MathBondState::Triggered => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_ALREADY_TRIGGERED,
                    "bond already triggered; settle the trigger payout instead",
                    Severity::Error,
                ));
            }
            MathBondState::Matured => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_WRONG_STATE,
                    "bond already matured",
                    Severity::Error,
                ));
            }
            MathBondState::Settled => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_WRONG_STATE,
                    "bond already settled",
                    Severity::Error,
                ));
            }
        }
        if seq < bond.maturity_seq {
            return Err(Diagnostic::new(
                Code::MATHBOND_WRONG_STATE,
                format!(
                    "not yet mature: log seq {seq} < maturity_seq {}",
                    bond.maturity_seq
                ),
                Severity::Error,
            ));
        }
        bond.state = MathBondState::Matured;
        Ok(())
    }

    /// Finalize the bond: distribute collateral per the trigger/maturity
    /// outcome. Only a `Triggered` bond (trigger payout) or a `Matured` bond
    /// (maturity refund) may settle — a live `Issued`/`Funded` bond has an
    /// open trigger window and cannot be settled early.
    fn apply_settle(
        &mut self,
        _actor: &str,
        bond_id: &MathBondId,
        _seq: u64,
    ) -> Result<(), Diagnostic> {
        let bond = match self.bonds.get_mut(&bond_id.0) {
            Some(b) => b,
            None => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_UNKNOWN,
                    "unknown bond id",
                    Severity::Error,
                ));
            }
        };
        match bond.state {
            MathBondState::Triggered | MathBondState::Matured => {
                // Trigger payout or maturity refund — both final.
            }
            MathBondState::Issued | MathBondState::Funded => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_WRONG_STATE,
                    "bond is not triggered or matured; the trigger window is still open",
                    Severity::Error,
                ));
            }
            MathBondState::Settled => {
                return Err(Diagnostic::new(
                    Code::MATHBOND_WRONG_STATE,
                    "bond already settled",
                    Severity::Error,
                ));
            }
        }

        // Mark as settled. The actual certificate transfers (Taler e-coin
        // payments) are rowed by the settlement service on top of this
        // deterministic state machine — exactly like the auction + escrow
        // pattern.
        bond.state = MathBondState::Settled;
        Ok(())
    }
}

/// Deterministic bond id from the full issue parameters: the trigger spec,
/// sponsor, principal, AND the terms (coupon, maturity, researcher). The
/// version tag distinguishes the id space from the earlier terms-blind hash.
/// Public so clients can reference a bond (Invest/SubmitProof/Mature/Settle)
/// without round-tripping through a report.
pub fn compute_bond_id(
    trigger: &MathBondTrigger,
    sponsor: &str,
    principal: u64,
    coupon_rate_bps: u64,
    maturity_seq: u64,
    researcher_did: &str,
) -> MathBondId {
    let mut ctx = Sha256::new();
    ctx.update(b"unfer:mathbond:v2");
    ctx.update(trigger.theorem.as_bytes());
    ctx.update(trigger.spec_hash.as_bytes());
    ctx.update(sponsor.as_bytes());
    ctx.update(principal.to_le_bytes());
    ctx.update(coupon_rate_bps.to_le_bytes());
    ctx.update(maturity_seq.to_le_bytes());
    ctx.update(researcher_did.as_bytes());
    MathBondId(ctx.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use unfer_protocol::MathBondOpKind;

    fn trigger() -> MathBondTrigger {
        MathBondTrigger {
            theorem: "P_eq_NP".to_string(),
            spec_hash: "deadbeef".to_string(),
            max_export_bytes: 16 * 1024 * 1024,
            permitted_axioms: vec![
                "Quot.sound".to_string(),
                "Classical.choice".to_string(),
                "propext".to_string(),
            ],
            strict: false,
            nat_extension: false,
            string_extension: false,
        }
    }

    fn sponsor() -> &'static str {
        "did:unfer:sponsor"
    }

    fn investor() -> &'static str {
        "did:unfer:investor"
    }

    fn researcher() -> &'static str {
        "did:unfer:researcher"
    }

    fn issue_bond(l: &mut MathBondLedger) -> MathBondId {
        let trig = trigger();
        l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trig,
                principal: 10000,
                coupon_rate_bps: 500,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            1,
        )
        .unwrap();
        // Compute the expected bond id from the full issue parameters.
        compute_bond_id(&trigger(), sponsor(), 10000, 500, 1000, researcher())
    }

    #[test]
    fn issue_and_invest_lifecycle() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Issued);

        // Invest 6000 — not yet fully funded.
        l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 6000,
            },
            2,
        )
        .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Issued);
        assert_eq!(l.bond(&bond_id).unwrap().invested, 6000);

        // Invest 4000 — now fully funded.
        l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 4000,
            },
            3,
        )
        .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Funded);
        assert_eq!(l.bond(&bond_id).unwrap().invested, 10000);
    }

    #[test]
    fn overfunding_rejected() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        let err = l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 10001,
            },
            2,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_OVERFUNDED);
    }

    #[test]
    fn unknown_bond_rejected() {
        let mut l = MathBondLedger::new();
        let err = l.apply_op(
            "did:unfer:x",
            &MathBondOpKind::Invest {
                bond_id: MathBondId([99u8; 32]),
                amount: 100,
            },
            1,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_UNKNOWN);
    }

    #[test]
    fn non_researcher_proof_rejected() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        let err = l.apply_op(
            "did:unfer:impostor",
            &MathBondOpKind::SubmitProof {
                bond_id,
                export_bytes: b"fake proof".to_vec(),
            },
            2,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_NOT_RESEARCHER);
    }

    #[test]
    fn early_settle_rejected() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        // A live Issued bond cannot be settled while the trigger window is open.
        let err = l.apply_op(sponsor(), &MathBondOpKind::Settle { bond_id }, 2);
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Issued);
    }

    #[test]
    fn premature_mature_rejected() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        // Maturity_seq is 1000; maturing at seq 2 is refused.
        let err = l.apply_op("did:unfer:anyone", &MathBondOpKind::Mature { bond_id }, 2);
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Issued);
    }

    #[test]
    fn maturity_refund_lifecycle() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        // Fund the bond.
        l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 10000,
            },
            2,
        )
        .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Funded);

        // Mature at/after maturity_seq.
        l.apply_op(
            "did:unfer:anyone",
            &MathBondOpKind::Mature { bond_id },
            1000,
        )
        .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Matured);

        // A Matured bond settles as a maturity refund; the trigger window is closed.
        let err = l.apply_op(
            researcher(),
            &MathBondOpKind::SubmitProof {
                bond_id,
                export_bytes: b"too late".to_vec(),
            },
            1001,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);

        l.apply_op(sponsor(), &MathBondOpKind::Settle { bond_id }, 1001)
            .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Settled);

        // Double-settle is rejected.
        let err = l.apply_op(sponsor(), &MathBondOpKind::Settle { bond_id }, 1002);
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);
    }

    #[test]
    fn triggered_bond_settles() {
        let mut l = MathBondLedger::new();
        // A self-funded bond where the sponsor is also the researcher.
        let trig = trigger();
        l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trig.clone(),
                principal: 1000,
                coupon_rate_bps: 0,
                maturity_seq: 1000,
                researcher_did: sponsor().to_string(),
            },
            1,
        )
        .unwrap();
        let bond_id = compute_bond_id(&trig, sponsor(), 1000, 0, 1000, sponsor());
        l.apply_op(
            sponsor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 1000,
            },
            2,
        )
        .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Funded);

        // Settle before trigger/maturity is still rejected.
        let err = l.apply_op(sponsor(), &MathBondOpKind::Settle { bond_id }, 3);
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);

        // Force the triggered state by directly transitioning (the nanoda path
        // is covered by valid_proof_triggers_bond) and settle the payout.
        l.bonds.get_mut(&bond_id.0).unwrap().state = MathBondState::Triggered;
        l.bonds.get_mut(&bond_id.0).unwrap().trigger_seq = Some(3);
        l.apply_op(sponsor(), &MathBondOpKind::Settle { bond_id }, 4)
            .unwrap();
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Settled);
        assert_eq!(l.bond(&bond_id).unwrap().trigger_seq, Some(3));
    }

    #[test]
    fn report_matches_state() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        let report = l.report(&bond_id).unwrap();
        assert_eq!(report.state, MathBondState::Issued);
        assert_eq!(report.principal, 10000);
        assert_eq!(report.coupon_rate_bps, 500);
        assert_eq!(report.researcher_did, researcher());
        assert!(report.proof_report.is_none());
    }

    #[test]
    fn replay_is_deterministic() {
        let trigger = trigger();

        let ops: Vec<(&str, MathBondOpKind, u64)> = vec![
            (
                sponsor(),
                MathBondOpKind::Issue {
                    trigger: trigger.clone(),
                    principal: 10000,
                    coupon_rate_bps: 500,
                    maturity_seq: 1000,
                    researcher_did: researcher().to_string(),
                },
                1,
            ),
            (
                investor(),
                MathBondOpKind::Invest {
                    bond_id: compute_bond_id(&trigger, sponsor(), 10000, 500, 1000, researcher()),
                    amount: 10000,
                },
                2,
            ),
        ];

        let mut a = MathBondLedger::new();
        let mut b = MathBondLedger::new();
        for (actor, kind, seq) in &ops {
            let r_a = a.apply_op(actor, kind, *seq);
            let r_b = b.apply_op(actor, kind, *seq);
            assert_eq!(r_a.is_ok(), r_b.is_ok(), "ops must agree at seq {seq}");
        }
        // Both ledgers converge on the same bond state.
        let bond_id = compute_bond_id(&trigger, sponsor(), 10000, 500, 1000, researcher());
        assert_eq!(a.report(&bond_id).unwrap().state, MathBondState::Funded);
        assert_eq!(b.report(&bond_id).unwrap().state, MathBondState::Funded);
        assert_eq!(a.report(&bond_id).unwrap().invested, 10000);
        assert_eq!(b.report(&bond_id).unwrap().invested, 10000);
    }

    #[test]
    fn proof_oversize_rejected() {
        let mut l = MathBondLedger::new();

        let small_trigger = MathBondTrigger {
            max_export_bytes: 100,
            ..trigger()
        };

        l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: small_trigger.clone(),
                principal: 1000,
                coupon_rate_bps: 500,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            1,
        )
        .unwrap();
        let bond_id = compute_bond_id(&small_trigger, sponsor(), 1000, 500, 1000, researcher());

        let err = l.apply_op(
            researcher(),
            &MathBondOpKind::SubmitProof {
                bond_id,
                export_bytes: vec![0u8; 200],
            },
            2,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_PROOF_OVERSIZE);
    }

    #[test]
    fn valid_proof_triggers_bond() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        // Fund it fully.
        l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 10000,
            },
            2,
        )
        .unwrap();

        // Use the real nanoda-valid export from prob_kernel's test fixtures.
        // The confluence proof needs nat_extension.
        let trigger_with_nat = MathBondTrigger {
            theorem: "confluence".to_string(),
            spec_hash: "confluence".to_string(),
            max_export_bytes: 16 * 1024 * 1024,
            permitted_axioms: vec![
                "Quot.sound".to_string(),
                "Classical.choice".to_string(),
                "propext".to_string(),
                "Lean.trustCompiler".to_string(),
            ],
            strict: false,
            nat_extension: true,
            string_extension: true,
        };
        l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trigger_with_nat.clone(),
                principal: 10000,
                coupon_rate_bps: 500,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            1,
        )
        .unwrap();
        let bond_id = compute_bond_id(&trigger_with_nat, sponsor(), 10000, 500, 1000, researcher());

        l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 10000,
            },
            2,
        )
        .unwrap();

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../prob_kernel/tests/fixtures/confluence.ndjson"
        );
        let export_bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => {
                // If the fixture is not available, skip this test gracefully.
                eprintln!(
                    "confluence.ndjson fixture not available, skipping valid_proof_triggers_bond"
                );
                return;
            }
        };

        let result = l.apply_op(
            researcher(),
            &MathBondOpKind::SubmitProof {
                bond_id,
                export_bytes,
            },
            3,
        );
        // The proof should trigger the bond.
        assert!(result.is_ok(), "valid proof should trigger: {result:?}");
        let trigger_opt = result.unwrap();
        assert!(trigger_opt.is_some(), "trigger should fire");
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Triggered);
        assert_eq!(l.bond(&bond_id).unwrap().trigger_seq, Some(3));
        // The report exposes the trigger signal for market resolution.
        assert_eq!(l.report(&bond_id).unwrap().trigger_seq, Some(3));
    }

    #[test]
    fn invalid_proof_rejected_bond_stays_funded() {
        let mut l = MathBondLedger::new();
        let bond_id = issue_bond(&mut l);

        l.apply_op(
            investor(),
            &MathBondOpKind::Invest {
                bond_id,
                amount: 10000,
            },
            2,
        )
        .unwrap();

        // Submit garbage bytes — nanoda will reject it.
        let result = l.apply_op(
            researcher(),
            &MathBondOpKind::SubmitProof {
                bond_id,
                export_bytes: b"this is not a valid lean4 export".to_vec(),
            },
            3,
        );
        assert!(result.is_err(), "garbage proof should be rejected");
        assert_eq!(result.unwrap_err().code, Code::MATHBOND_PROOF_REJECTED);
        // Bond stays in Funded state (not triggered).
        assert_eq!(l.bond(&bond_id).unwrap().state, MathBondState::Funded);
    }

    #[test]
    fn bond_ids_distinguish_terms() {
        // Two bonds with the same theorem/sponsor/principal but different
        // terms get different ids and can coexist.
        let id_a = compute_bond_id(&trigger(), sponsor(), 10000, 500, 1000, researcher());
        let id_b = compute_bond_id(&trigger(), sponsor(), 10000, 600, 1000, researcher());
        let id_c = compute_bond_id(&trigger(), sponsor(), 10000, 500, 2000, researcher());
        assert_ne!(id_a, id_b);
        assert_ne!(id_a, id_c);

        let mut l = MathBondLedger::new();
        l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trigger(),
                principal: 10000,
                coupon_rate_bps: 500,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            1,
        )
        .unwrap();
        // Same theorem, different coupon — must not collide with the first.
        l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trigger(),
                principal: 10000,
                coupon_rate_bps: 600,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            2,
        )
        .unwrap();
        assert!(l.bond(&id_a).is_some());
        assert!(l.bond(&id_b).is_some());
    }

    #[test]
    fn issue_validates_terms() {
        let mut l = MathBondLedger::new();
        // Coupon above 100% is refused.
        let err = l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trigger(),
                principal: 10000,
                coupon_rate_bps: 10001,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            1,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);
        // Empty researcher is refused.
        let err = l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trigger(),
                principal: 10000,
                coupon_rate_bps: 500,
                maturity_seq: 1000,
                researcher_did: String::new(),
            },
            2,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);
        // Zero principal and zero maturity are refused.
        let err = l.apply_op(
            sponsor(),
            &MathBondOpKind::Issue {
                trigger: trigger(),
                principal: 0,
                coupon_rate_bps: 500,
                maturity_seq: 1000,
                researcher_did: researcher().to_string(),
            },
            3,
        );
        assert_eq!(err.unwrap_err().code, Code::MATHBOND_WRONG_STATE);
    }
}
