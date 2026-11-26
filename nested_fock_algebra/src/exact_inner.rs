//! Ordered inner CCR/CAR action and a single outer `dGamma` enclosure.
//!
//! A basis label contains both bosonic occupations and fermionic occupations.
//! Products act right to left on that *same* inner state before the outer lift.
//! No normal-ordering, scalar subtraction, occupation cutoff or amplitude pruning
//! is performed. "Exact" refers to operator algebra; coefficients and square
//! roots still use floating point and carry no interval-rounding certificate.
//!
//! This is an additive construction API. It does not identify the full physical
//! SM term ledger, select its outer statistics, or replace the legacy CAS API.

use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Tensor-product inner occupation label, with canonical zero-free support.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct JointInnerBasis {
    bosons: BTreeMap<u32, u32>,
    fermions: BTreeSet<u32>,
}

impl JointInnerBasis {
    pub fn vacuum() -> Self {
        Self::default()
    }

    /// Duplicate bosonic mode entries are summed; Pauli duplicates are rejected.
    pub fn new(
        bosons: impl IntoIterator<Item = (u32, u32)>,
        fermions: impl IntoIterator<Item = u32>,
    ) -> Result<Self, ActionError> {
        let mut basis = Self::vacuum();
        for (mode, n) in bosons {
            if n != 0 {
                let old = basis.bosons.get(&mode).copied().unwrap_or(0);
                basis.bosons.insert(
                    mode,
                    old.checked_add(n).ok_or(ActionError::OccupationOverflow)?,
                );
            }
        }
        for mode in fermions {
            if !basis.fermions.insert(mode) {
                return Err(ActionError::PauliOccupation);
            }
        }
        Ok(basis)
    }

    pub fn bosons(&self) -> &BTreeMap<u32, u32> {
        &self.bosons
    }

    pub fn fermions(&self) -> &BTreeSet<u32> {
        &self.fermions
    }

    pub fn total_bosons(&self) -> u64 {
        self.bosons.values().map(|&n| u64::from(n)).sum()
    }
    pub fn is_odd(&self) -> bool {
        self.fermions.len() % 2 == 1
    }
}

pub type InnerVector = BTreeMap<JointInnerBasis, Complex64>;
pub type NestedVector = BTreeMap<JointOuterBasis, Complex64>;

/// A numerical resource failure aborts the action; it never truncates the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionLimits {
    pub max_components: usize,
    pub max_work: usize,
}

impl Default for ActionLimits {
    fn default() -> Self {
        Self {
            max_components: 100_000,
            max_work: 10_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionError {
    ComponentLimit { limit: usize },
    WorkLimit { limit: usize },
    OccupationOverflow,
    PauliOccupation,
    NonFiniteAmplitude,
    UnsupportedBasis,
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ordered nested-Fock action failed: {self:?}")
    }
}

impl std::error::Error for ActionError {}

struct Budget {
    limits: ActionLimits,
    work: usize,
}

impl Budget {
    fn tick(&mut self) -> Result<(), ActionError> {
        if self.work >= self.limits.max_work {
            return Err(ActionError::WorkLimit {
                limit: self.limits.max_work,
            });
        }
        self.work += 1;
        Ok(())
    }

    /// Charge a batch of work. An action that asks for more than
    /// `max_work` fails; asking for exactly `max_work` succeeds, matching
    /// [`Self::tick`] (which errors on the attempt to exceed the limit).
    fn charge(&mut self, work: usize) -> Result<(), ActionError> {
        self.work = self.work.saturating_add(work);
        if self.work > self.limits.max_work {
            return Err(ActionError::WorkLimit {
                limit: self.limits.max_work,
            });
        }
        Ok(())
    }

    fn add<B: Ord>(
        &mut self,
        vector: &mut BTreeMap<B, Complex64>,
        basis: B,
        amplitude: Complex64,
    ) -> Result<(), ActionError> {
        self.tick()?;
        if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
            return Err(ActionError::NonFiniteAmplitude);
        }
        if amplitude == Complex64::new(0.0, 0.0) {
            return Ok(());
        }
        let previous = vector.get(&basis).copied();
        if previous.is_none() && vector.len() >= self.limits.max_components {
            return Err(ActionError::ComponentLimit {
                limit: self.limits.max_components,
            });
        }
        let sum = previous.unwrap_or_default() + amplitude;
        if !sum.re.is_finite() || !sum.im.is_finite() {
            return Err(ActionError::NonFiniteAmplitude);
        }
        if sum == Complex64::new(0.0, 0.0) {
            vector.remove(&basis);
        } else {
            vector.insert(basis, sum);
        }
        Ok(())
    }

    fn validate<B: Ord>(&mut self, input: &BTreeMap<B, Complex64>) -> Result<(), ActionError> {
        if input.len() > self.limits.max_components {
            return Err(ActionError::ComponentLimit {
                limit: self.limits.max_components,
            });
        }
        for amplitude in input.values() {
            self.tick()?;
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(ActionError::NonFiniteAmplitude);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerLadder {
    BosonCreate(u32),
    BosonAnnihilate(u32),
    FermionCreate(u32),
    FermionAnnihilate(u32),
}

impl InnerLadder {
    pub fn adjoint(self) -> Self {
        match self {
            Self::BosonCreate(m) => Self::BosonAnnihilate(m),
            Self::BosonAnnihilate(m) => Self::BosonCreate(m),
            Self::FermionCreate(m) => Self::FermionAnnihilate(m),
            Self::FermionAnnihilate(m) => Self::FermionCreate(m),
        }
    }

    fn transition(
        self,
        basis: &JointInnerBasis,
    ) -> Result<Option<(JointInnerBasis, f64)>, ActionError> {
        let mut next = basis.clone();
        let factor = match self {
            Self::BosonCreate(m) => {
                let n = basis.bosons.get(&m).copied().unwrap_or(0);
                let n1 = n.checked_add(1).ok_or(ActionError::OccupationOverflow)?;
                next.bosons.insert(m, n1);
                f64::from(n1).sqrt()
            }
            Self::BosonAnnihilate(m) => {
                let Some(&n) = basis.bosons.get(&m) else {
                    return Ok(None);
                };
                if n == 1 {
                    next.bosons.remove(&m);
                } else {
                    next.bosons.insert(m, n - 1);
                }
                f64::from(n).sqrt()
            }
            Self::FermionCreate(m) | Self::FermionAnnihilate(m) => {
                let create = matches!(self, Self::FermionCreate(_));
                if basis.fermions.contains(&m) == create {
                    return Ok(None);
                }
                let sign = if basis.fermions.range(..m).count().is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                if create {
                    next.fermions.insert(m);
                } else {
                    next.fermions.remove(&m);
                }
                sign
            }
        };
        Ok(Some((next, factor)))
    }
}

/// Factorized polynomial in inner ladders. Product factors are written in
/// mathematical order: `Product([B,C])` applies C first, then B.
/// An empty sum is zero; an empty product and `Scalar(1)` are the inner identity.
#[derive(Clone, Debug)]
pub enum InnerExpr {
    Scalar(Complex64),
    Ladder(InnerLadder),
    Sum(Vec<Self>),
    Product(Vec<Self>),
}

impl InnerExpr {
    /// Variance-one Hermite coordinate `q = a† + a`.
    pub fn coordinate(mode: u32) -> Self {
        Self::Sum(vec![
            Self::Ladder(InnerLadder::BosonCreate(mode)),
            Self::Ladder(InnerLadder::BosonAnnihilate(mode)),
        ])
    }

    /// Same-mode conjugate momentum `p = i(a† - a)/2`, so `[q,p] = i`.
    pub fn momentum(mode: u32) -> Self {
        Self::Sum(vec![
            Self::Ladder(InnerLadder::BosonCreate(mode)).scaled(Complex64::new(0.0, 0.5)),
            Self::Ladder(InnerLadder::BosonAnnihilate(mode)).scaled(Complex64::new(0.0, -0.5)),
        ])
    }

    pub fn scaled(self, coefficient: Complex64) -> Self {
        Self::Product(vec![Self::Scalar(coefficient), self])
    }

    pub fn adjoint(&self) -> Self {
        match self {
            Self::Scalar(c) => Self::Scalar(c.conj()),
            Self::Ladder(l) => Self::Ladder(l.adjoint()),
            Self::Sum(terms) => Self::Sum(terms.iter().map(Self::adjoint).collect()),
            Self::Product(factors) => {
                Self::Product(factors.iter().rev().map(Self::adjoint).collect())
            }
        }
    }

    pub fn apply(
        &self,
        input: &InnerVector,
        limits: ActionLimits,
    ) -> Result<InnerVector, ActionError> {
        let mut budget = Budget { limits, work: 0 };
        budget.validate(input)?;
        self.apply_with_budget(input, &mut budget)
    }

    fn apply_with_budget(
        &self,
        input: &InnerVector,
        budget: &mut Budget,
    ) -> Result<InnerVector, ActionError> {
        budget.tick()?;
        match self {
            Self::Scalar(c) => {
                // Reject invalid coefficients even when the input is zero.
                if !c.re.is_finite() || !c.im.is_finite() {
                    return Err(ActionError::NonFiniteAmplitude);
                }
                let mut out = InnerVector::new();
                let (produced, _) = map_entries(input, |b, a| Ok(Some((b.clone(), c * a))))?;
                for (b, value) in produced {
                    budget.add(&mut out, b, value)?;
                }
                Ok(out)
            }
            Self::Ladder(ladder) => {
                let mut out = InnerVector::new();
                let (produced, work) = map_entries(input, |b, a| {
                    Ok(ladder
                        .transition(b)?
                        .map(|(next, factor)| (next, a * factor)))
                })?;
                budget.charge(work)?;
                for (b, value) in produced {
                    budget.add(&mut out, b, value)?;
                }
                Ok(out)
            }
            Self::Sum(terms) => {
                let mut out = InnerVector::new();
                let threads = sum_threads(terms.len());
                if threads <= 1 || in_worker() {
                    for term in terms {
                        for (b, a) in term.apply_with_budget(input, budget)? {
                            budget.add(&mut out, b, a)?;
                        }
                    }
                    return Ok(out);
                }
                // One worker per term. Each term's image is merged as a whole
                // map in term order, and a term contributes at most one value
                // per output label, so every label accumulates in the same
                // term order as the serial path: bit-identical, not merely close.
                let by_reference = input;
                let limits = budget.limits;
                let joined = std::thread::scope(|scope| {
                    let handles: Vec<_> = terms
                        .iter()
                        .map(|term| {
                            scope.spawn(move || {
                                with_worker_context(|| {
                                    let mut local = Budget { limits, work: 0 };
                                    let image = term.apply_with_budget(by_reference, &mut local)?;
                                    Ok::<_, ActionError>((image, local.work))
                                })
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().expect("action worker panicked"))
                        .collect::<Result<Vec<_>, ActionError>>()
                })?;
                for (image, work) in joined {
                    budget.charge(work)?;
                    for (b, a) in image {
                        budget.add(&mut out, b, a)?;
                    }
                }
                Ok(out)
            }
            Self::Product(factors) => {
                let mut current = input.clone();
                for factor in factors.iter().rev() {
                    current = factor.apply_with_budget(&current, budget)?;
                }
                Ok(current)
            }
        }
    }

    /// Apply `sum_ij h_ij C†_i A_j` after completing the entire inner action.
    /// Scalar inner expressions therefore give `c N_out`, not `c I_out`.
    /// Symmetric, antisymmetric and parity-graded outer statistics are supported.
    /// Physical Hilbert/core identification remains a separate proof obligation.
    pub fn d_gamma(
        &self,
        input: &NestedVector,
        limits: ActionLimits,
    ) -> Result<NestedVector, ActionError> {
        let mut budget = Budget { limits, work: 0 };
        budget.validate(input)?;
        let mut out = NestedVector::new();
        // Apply the factorized inner expression to a whole one-particle vector,
        // rather than separately expanding the image of every occupied label.
        // Bosonic and fermionic outer statistics remain separate summands.
        if input.keys().all(|outer| outer.particle_number() <= 1) {
            for statistics in [0, 1, 2] {
                let mut inner = InnerVector::new();
                for (outer, amplitude) in input {
                    let category = match outer {
                        JointOuterBasis::Bosonic(_) => 0,
                        JointOuterBasis::Fermionic(_) => 1,
                        JointOuterBasis::Graded(_) => 2,
                    };
                    if category == statistics
                        && let Some(source) = outer.occupied().next()
                    {
                        budget.add(&mut inner, source.clone(), *amplitude)?;
                    }
                }
                if inner.is_empty() {
                    continue;
                }
                for (target, amplitude) in self.apply_with_budget(&inner, &mut budget)? {
                    let outer = match statistics {
                        0 => JointOuterBasis::bosonic([(target, 1)])?,
                        1 => JointOuterBasis::fermionic([target])?,
                        _ => JointOuterBasis::graded([(target, 1)])?,
                    };
                    budget.add(&mut out, outer, amplitude)?;
                }
            }
            return Ok(out);
        }
        let entries: Vec<(&JointOuterBasis, &Complex64)> = input.iter().collect();
        let threads = action_threads(entries.len());
        self.apply_outer_labels(&entries, limits, &mut out, threads)?;
        Ok(out)
    }

    /// The multi-particle branch of [`Self::d_gamma`], with the worker count
    /// passed in so a test can compare the parallel and serial paths on the same
    /// input without touching the process environment.
    ///
    /// The expensive part is `apply_with_budget` on a one-particle basis vector,
    /// once per (outer label, occupied inner label). Those units are
    /// independent, so they are farmed out over contiguous chunks of the input
    /// order. The *contributions* come back per entry and are merged on this
    /// thread in the original order, which keeps the floating-point accumulation
    /// sequence identical to the serial path: the parallel result is
    /// bit-identical, not merely close.
    fn apply_outer_labels(
        &self,
        entries: &[(&JointOuterBasis, &Complex64)],
        limits: ActionLimits,
        out: &mut NestedVector,
        threads: usize,
    ) -> Result<(), ActionError> {
        let mut budget = Budget { limits, work: 0 };
        if threads <= 1 {
            for (outer, amplitude) in entries {
                for source in outer.occupied() {
                    budget.tick()?;
                    let basis_vector =
                        InnerVector::from([(source.clone(), Complex64::new(1.0, 0.0))]);
                    let image = self.apply_with_budget(&basis_vector, &mut budget)?;
                    for (target, coefficient) in image {
                        if let Some((next, factor)) = outer.transition(source, &target)? {
                            budget.add(out, next, *amplitude * coefficient * factor)?;
                        }
                    }
                }
            }
            return Ok(());
        }
        let chunk = entries.len().div_ceil(threads);
        let joined = std::thread::scope(|scope| {
            let handles: Vec<_> = entries
                .chunks(chunk)
                .map(|group| {
                    scope.spawn(move || {
                        // A private counter: the work limit is checked against
                        // the aggregate below, so the total tick count is the
                        // same as the serial path.
                        let mut local = Budget { limits, work: 0 };
                        let mut produced = Vec::with_capacity(group.len());
                        for (outer, amplitude) in group {
                            let mut per_entry = Vec::new();
                            for source in outer.occupied() {
                                local.tick()?;
                                let basis_vector =
                                    InnerVector::from([(source.clone(), Complex64::new(1.0, 0.0))]);
                                let image = self.apply_with_budget(&basis_vector, &mut local)?;
                                for (target, coefficient) in image {
                                    if let Some((next, factor)) =
                                        outer.transition(source, &target)?
                                    {
                                        per_entry.push((next, *amplitude * coefficient * factor));
                                    }
                                }
                            }
                            produced.push(per_entry);
                        }
                        Ok::<_, ActionError>((produced, local.work))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("action worker panicked"))
                .collect::<Result<Vec<_>, ActionError>>()
        })?;
        for (produced, work) in joined {
            budget.charge(work)?;
            for per_entry in produced {
                for (next, value) in per_entry {
                    budget.add(out, next, value)?;
                }
            }
        }
        Ok(())
    }
}

/// Map over the entries of an inner vector, optionally across threads.
///
/// Returns the contributions in the *input order* — chunks are contiguous, so
/// concatenating them reproduces the serial sequence exactly — plus the number
/// of entries, which the caller charges as work so the limit accounting is the
/// same serial and parallel. Returning contributions instead of partial sums is
/// what makes the parallel result bit-identical rather than merely close.
fn map_entries<F>(
    input: &InnerVector,
    f: F,
) -> Result<(Vec<(JointInnerBasis, Complex64)>, usize), ActionError>
where
    F: Fn(
        &JointInnerBasis,
        &Complex64,
    ) -> Result<Option<(JointInnerBasis, Complex64)>, ActionError>,
{
    // Deliberately serial: splitting this map across threads measured *slower*
    // (46.0 ms -> 62.5 ms per C = 20 matvec) because one fork/join per factor
    // does not amortize against a per-entry cost of a few map lookups. The
    // parallelism is taken at the coarser sum-over-terms level instead.
    let mut produced = Vec::with_capacity(input.len());
    for (b, a) in input {
        if let Some(item) = f(b, a)? {
            produced.push(item);
        }
    }
    Ok((produced, input.len()))
}

thread_local! {
    /// Set inside a sum worker so a nested sum does not spawn again.
    static IN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn in_worker() -> bool {
    IN_WORKER.with(|flag| flag.get())
}

fn with_worker_context<R>(f: impl FnOnce() -> R) -> R {
    IN_WORKER.with(|flag| {
        let previous = flag.replace(true);
        let out = f();
        flag.set(previous);
        out
    })
}

/// Test-only override so a test can force the serial path without touching the
/// process environment (which would race with the other tests in this binary).
#[cfg(test)]
static FORCED_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The worker count a caller asked for: the test override, else
/// `FOCK_SIRK_ACTION_THREADS`. Forcing a count is how the serial path is
/// measured against the parallel one.
fn requested_threads() -> Option<usize> {
    #[cfg(test)]
    {
        let forced = FORCED_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        if forced > 0 {
            return Some(forced);
        }
    }
    std::env::var("FOCK_SIRK_ACTION_THREADS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|count| *count > 0)
}

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Worker count for the sum-over-terms split: one per term, bounded by the
/// machine.
fn sum_threads(terms: usize) -> usize {
    if terms < 2 {
        return 1;
    }
    if let Some(forced) = requested_threads() {
        return forced.clamp(1, terms);
    }
    available_threads().clamp(1, terms)
}

/// Worker count for the outer-label action. One thread below 64 entries (the
/// fork/join cost dominates there), otherwise bounded by the machine and by the
/// number of chunks worth splitting into.
fn action_threads(entries: usize) -> usize {
    if entries < 64 {
        return 1;
    }
    if let Some(forced) = requested_threads() {
        return forced.clamp(1, entries);
    }
    available_threads().clamp(1, entries / 32).max(1)
}

/// Symmetric, antisymmetric or parity-graded Fock space over the joint inner basis.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum JointOuterBasis {
    Bosonic(BosonicOccupations),
    Fermionic(FermionicOccupations),
    Graded(GradedOccupations),
}

/// Private support prevents explicit zero occupations and overflow aliases.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BosonicOccupations(BTreeMap<JointInnerBasis, u32>);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct FermionicOccupations(BTreeSet<JointInnerBasis>);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GradedOccupations {
    even: BosonicOccupations,
    odd: FermionicOccupations,
}

impl JointOuterBasis {
    /// Even inner configurations obey CCR outer statistics, odd configurations
    /// CAR outer statistics. Inner parity is never confused with outer number.
    pub fn graded(
        input: impl IntoIterator<Item = (JointInnerBasis, u32)>,
    ) -> Result<Self, ActionError> {
        let mut occupations = GradedOccupations::default();
        for (b, n) in input {
            if n == 0 {
                continue;
            }
            if b.is_odd() {
                if n != 1 || !occupations.odd.0.insert(b) {
                    return Err(ActionError::PauliOccupation);
                }
            } else {
                let old = occupations.even.0.get(&b).copied().unwrap_or(0);
                occupations.even.0.insert(
                    b,
                    old.checked_add(n).ok_or(ActionError::OccupationOverflow)?,
                );
            }
        }
        Ok(Self::Graded(occupations))
    }
    pub fn bosonic(
        input: impl IntoIterator<Item = (JointInnerBasis, u32)>,
    ) -> Result<Self, ActionError> {
        let mut occupations = BTreeMap::<JointInnerBasis, u32>::new();
        for (b, n) in input {
            if n != 0 {
                let old = occupations.get(&b).copied().unwrap_or(0);
                occupations.insert(
                    b,
                    old.checked_add(n).ok_or(ActionError::OccupationOverflow)?,
                );
            }
        }
        Ok(Self::Bosonic(BosonicOccupations(occupations)))
    }

    pub fn fermionic(
        input: impl IntoIterator<Item = JointInnerBasis>,
    ) -> Result<Self, ActionError> {
        let mut occupations = BTreeSet::new();
        for b in input {
            if !occupations.insert(b) {
                return Err(ActionError::PauliOccupation);
            }
        }
        Ok(Self::Fermionic(FermionicOccupations(occupations)))
    }

    pub fn particle_number(&self) -> u64 {
        match self {
            Self::Bosonic(ns) => ns.0.values().map(|&n| u64::from(n)).sum(),
            Self::Fermionic(ns) => ns.0.len() as u64,
            Self::Graded(ns) => {
                ns.even.0.values().map(|&n| u64::from(n)).sum::<u64>() + ns.odd.0.len() as u64
            }
        }
    }

    /// Read-only joint labels/counts; this does not flatten the tensor factor.
    pub fn inner_occupations(&self) -> Vec<(&JointInnerBasis, u32)> {
        match self {
            Self::Bosonic(ns) => ns.0.iter().map(|(b, &n)| (b, n)).collect(),
            Self::Fermionic(ns) => ns.0.iter().map(|b| (b, 1)).collect(),
            Self::Graded(ns) => ns
                .even
                .0
                .iter()
                .map(|(b, &n)| (b, n))
                .chain(ns.odd.0.iter().map(|b| (b, 1)))
                .collect(),
        }
    }

    fn occupied(&self) -> Box<dyn Iterator<Item = &JointInnerBasis> + '_> {
        match self {
            Self::Bosonic(ns) => Box::new(ns.0.keys()),
            Self::Fermionic(ns) => Box::new(ns.0.iter()),
            Self::Graded(ns) => Box::new(ns.even.0.keys().chain(ns.odd.0.iter())),
        }
    }

    fn transition(
        &self,
        source: &JointInnerBasis,
        target: &JointInnerBasis,
    ) -> Result<Option<(Self, f64)>, ActionError> {
        match self {
            Self::Graded(ns) => {
                let mut next = ns.clone();
                let mut factor = if source.is_odd() {
                    let sign = if next.odd.0.range(..source).count() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    next.odd.0.remove(source);
                    sign
                } else {
                    let n = next.even.0[source];
                    if n == 1 {
                        next.even.0.remove(source);
                    } else {
                        next.even.0.insert(source.clone(), n - 1);
                    }
                    f64::from(n).sqrt()
                };
                if target.is_odd() {
                    if next.odd.0.contains(target) {
                        return Ok(None);
                    }
                    if next.odd.0.range(..target).count() % 2 == 1 {
                        factor *= -1.0;
                    }
                    next.odd.0.insert(target.clone());
                } else {
                    let n = next.even.0.get(target).copied().unwrap_or(0);
                    let n1 = n.checked_add(1).ok_or(ActionError::OccupationOverflow)?;
                    next.even.0.insert(target.clone(), n1);
                    factor *= f64::from(n1).sqrt();
                }
                Ok(Some((Self::Graded(next), factor)))
            }
            Self::Bosonic(ns) => {
                let n = ns.0[source];
                let mut next = ns.clone();
                if n == 1 {
                    next.0.remove(source);
                } else {
                    next.0.insert(source.clone(), n - 1);
                }
                let m = next.0.get(target).copied().unwrap_or(0);
                let m1 = m.checked_add(1).ok_or(ActionError::OccupationOverflow)?;
                next.0.insert(target.clone(), m1);
                Ok(Some((
                    Self::Bosonic(next),
                    f64::from(n).sqrt() * f64::from(m1).sqrt(),
                )))
            }
            Self::Fermionic(ns) => {
                let n_before = ns.0.range(..source).count();
                let mut next = ns.clone();
                next.0.remove(source);
                if next.0.contains(target) {
                    return Ok(None);
                }
                let m_before = next.0.range(..target).count();
                let sign = if (n_before + m_before) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                next.0.insert(target.clone());
                Ok(Some((Self::Fermionic(next), sign)))
            }
        }
    }
}

#[cfg(test)]
mod parallel_action_tests {
    use super::*;
    use crate::phi4_momentum::ScalarMomCont;

    /// A wide multi-particle state over the continuous-momentum phi4 window.
    fn wide_state(width: usize) -> NestedVector {
        let model = ScalarMomCont::unit();
        let mut state = NestedVector::new();
        for n in 0..width {
            let label = model.inner_basis([n as u32, 0, 0, 0]).unwrap();
            // Two-particle outer labels take the multi-particle branch, and the
            // mixed amplitudes make the accumulation order observable.
            let basis = JointOuterBasis::bosonic([(label, 2)]).unwrap();
            let amplitude = Complex64::new(1.0 / (3.0 + n as f64), 1.0 / (7.0 + n as f64));
            state.insert(basis, amplitude);
        }
        state
    }

    fn bits(v: &NestedVector) -> Vec<(JointOuterBasis, u64, u64)> {
        v.iter()
            .map(|(b, a)| (b.clone(), a.re.to_bits(), a.im.to_bits()))
            .collect()
    }

    #[test]
    fn parallel_outer_labels_reproduce_the_serial_bits() {
        let model = ScalarMomCont::unit();
        let h = model.hamiltonian();
        let limits = ActionLimits {
            max_components: 100_000,
            max_work: 100_000_000,
        };
        let state = wide_state(96);
        assert!(state.len() >= 64, "need enough entries to engage workers");
        let entries: Vec<(&JointOuterBasis, &Complex64)> = state.iter().collect();

        let mut serial = NestedVector::new();
        h.apply_outer_labels(&entries, limits, &mut serial, 1)
            .unwrap();
        assert!(!serial.is_empty());

        // Every non-zero component must match to the bit, not merely closely.
        for threads in [2, 3, 4, 7, 16] {
            let mut parallel = NestedVector::new();
            h.apply_outer_labels(&entries, limits, &mut parallel, threads)
                .unwrap();
            assert_eq!(
                bits(&serial),
                bits(&parallel),
                "threads={threads} changed the accumulation order"
            );
        }

        // And the public entry point, which picks its own worker count, agrees.
        let public = h.d_gamma(&state, limits).unwrap();
        assert_eq!(bits(&serial), bits(&public));
    }

    #[test]
    fn parallel_sum_reproduces_the_serial_bits() {
        let model = ScalarMomCont::unit();
        let h = model.hamiltonian();
        let limits = ActionLimits {
            max_components: 4_000_000,
            max_work: 100_000_000,
        };
        // A one-particle outer state over the widest window: the inner action
        // then maps over ~1000 inner entries, which is the sweep's hot path.
        let mut state = NestedVector::new();
        for (n, basis) in ScalarMomCont::window_labels(20).into_iter().enumerate() {
            let amplitude = Complex64::new(1.0 / (2.5 + n as f64), 1.0 / (11.0 + n as f64));
            state.insert(basis, amplitude);
        }
        assert!(state.len() > 900);

        let serial = {
            FORCED_THREADS.store(1, std::sync::atomic::Ordering::Relaxed);
            let image = h.d_gamma(&state, limits).unwrap();
            FORCED_THREADS.store(0, std::sync::atomic::Ordering::Relaxed);
            image
        };
        let parallel = h.d_gamma(&state, limits).unwrap();
        assert_eq!(
            bits(&serial),
            bits(&parallel),
            "the inner map changed the accumulation order"
        );

        // And the serial result is the same as forcing an odd chunk count.
        let ragged = {
            FORCED_THREADS.store(7, std::sync::atomic::Ordering::Relaxed);
            let image = h.d_gamma(&state, limits).unwrap();
            FORCED_THREADS.store(0, std::sync::atomic::Ordering::Relaxed);
            image
        };
        assert_eq!(bits(&serial), bits(&ragged));
    }

    #[test]
    fn parallel_path_keeps_the_work_limit() {
        let model = ScalarMomCont::unit();
        let h = model.hamiltonian();
        let state = wide_state(72);
        let entries: Vec<(&JointOuterBasis, &Complex64)> = state.iter().collect();

        // Find the serial work count by bisecting the limit, then check that the
        // parallel path reports the same failure at the same total work.
        let fits = |limit: usize, threads: usize| {
            let mut out = NestedVector::new();
            h.apply_outer_labels(
                &entries,
                ActionLimits {
                    max_components: 100_000,
                    max_work: limit,
                },
                &mut out,
                threads,
            )
            .is_ok()
        };
        let mut lo = 1usize;
        let mut hi = 10_000_000usize;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid, 1) { hi = mid } else { lo = mid + 1 }
        }
        assert_eq!(
            lo,
            {
                let mut hi = 10_000_000usize;
                let mut lo2 = 1usize;
                while lo2 < hi {
                    let mid = lo2 + (hi - lo2) / 2;
                    if fits(mid, 4) {
                        hi = mid
                    } else {
                        lo2 = mid + 1
                    }
                }
                lo2
            },
            "parallel path charges a different total work than the serial one"
        );
    }
}
