//! Auto-`m` heuristics for the SIRK solver (Stage S16).
//!
//! The Krylov dimension `m` is user-fixed today; the Gram spectrum has a
//! knowable ceiling (AGENTS.md rank-saturation data: ~6 for single-mode
//! inputs). These helpers turn that observation into a rule: measure the
//! effective Gram rank of one restart, then set `m` just past it so the basis
//! saturates instead of wasting width on numerically-null directions.

use num_complex::Complex64;

/// Number of eigenvalues in a Hermitian spectrum above `rel_tol * max`.
///
/// Mirrors the [`crate::linalg::whiten_gram`] rank rule exactly, so the auto-`m`
/// heuristic and the actual whitening agree about which directions are useful.
pub fn effective_rank(eigenvalues: &[f64], rel_tol: f64) -> usize {
    let max_eig = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_eig.is_nan() || max_eig <= 0.0 {
        return 0;
    }
    let threshold = rel_tol * max_eig;
    eigenvalues.iter().filter(|&&e| e > threshold).count()
}

/// Suggest a Krylov dimension from a measured Gram `rank`, plus a reserve of
/// redundant directions (kept so the projector/whitening keep unitarity /
/// Hermiticity even when the next restart drifts). Clamped to `[min_m, max_m]`.
pub fn auto_krylov_dim(measured_rank: usize, reserve: usize, min_m: usize, max_m: usize) -> usize {
    let suggested = measured_rank.saturating_add(reserve);
    suggested.clamp(min_m, max_m)
}

/// Split the `[0, m)` shift index range into consecutive batches of at most
/// `batch` shifts each — the "budgeted batch of shifts per restart" half of the
/// S16 feature. Returns half-open ranges `(start, end)` covering all `m` shifts.
pub fn budgeted_shift_batches(m: usize, batch: usize) -> Vec<(usize, usize)> {
    if batch == 0 || m == 0 {
        return Vec::new();
    }
    let mut batches = Vec::new();
    let mut start = 0;
    while start < m {
        let end = (start + batch).min(m);
        batches.push((start, end));
        start = end;
    }
    batches
}

/// Build the standard SIRK shifts (the imaginary schedule used by
/// [`crate::evolve::evolve_restarted`]) for a sub-range of the Krylov dimension.
pub fn shifts_for_range(range: (usize, usize)) -> Vec<Complex64> {
    (range.0..range.1)
        .map(|j| Complex64::new(0.0, 1.0 + (j as f64) * 0.2))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_rank_keeps_significant_spectrum() {
        // max = 1.0, so the threshold = rel_tol. Values strictly above it survive.
        let spec = [1.0, 0.5, 1e-8, 1e-13, 0.0];
        assert_eq!(effective_rank(&spec, 1e-12), 3); // 1e-8 > 1e-12 survives
        assert_eq!(effective_rank(&spec, 1e-7), 2); // 1e-8 < 1e-7 -> dropped
    }

    #[test]
    fn effective_rank_all_equal_is_full_rank() {
        assert_eq!(effective_rank(&[2.0; 8], 1e-12), 8);
        assert_eq!(effective_rank(&[0.5; 3], 1e-12), 3);
    }

    #[test]
    fn effective_rank_degenerate_is_zero() {
        assert_eq!(effective_rank(&[0.0; 4], 1e-12), 0);
        assert_eq!(effective_rank(&[f64::NAN; 3], 1e-12), 0);
        assert_eq!(effective_rank(&[], 1e-12), 0);
    }

    #[test]
    fn auto_m_saturates_at_measured_rank() {
        // AGENTS.md: effective gram rank caps ~6 for single-mode inputs.
        assert_eq!(auto_krylov_dim(6, 1, 3, 12), 7);
        assert_eq!(auto_krylov_dim(6, 0, 3, 12), 6);
    }

    #[test]
    fn auto_m_clamps_to_bounds() {
        // Degenerate spectrum -> floor.
        assert_eq!(auto_krylov_dim(0, 2, 3, 12), 3);
        // Huge measured rank -> ceiling.
        assert_eq!(auto_krylov_dim(400, 20, 3, 12), 12);
    }

    #[test]
    fn auto_m_min_must_be_at_least_one() {
        assert_eq!(auto_krylov_dim(0, 0, 0, 12), 0);
        assert_eq!(auto_krylov_dim(1, 0, 0, 12), 1);
    }

    #[test]
    fn batches_are_exhaustive_and_disjoint() {
        let batches = budgeted_shift_batches(9, 3);
        assert_eq!(batches, vec![(0, 3), (3, 6), (6, 9)]);
        let flat: Vec<usize> = batches.iter().flat_map(|&(s, e)| s..e).collect();
        assert_eq!(flat, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn batches_respect_budget() {
        // Batch larger than m => single range.
        assert_eq!(budgeted_shift_batches(5, 6), vec![(0, 5)]);
        // Budget smaller than the dimension.
        let batches = budgeted_shift_batches(10, 4);
        assert_eq!(batches, vec![(0, 4), (4, 8), (8, 10)]);
        // No batch ever exceeds the budget.
        assert!(batches.iter().all(|&(s, e)| e - s <= 4));
        // Zero budget / zero dim degrade to nothing.
        assert_eq!(budgeted_shift_batches(9, 0), Vec::new());
        assert_eq!(budgeted_shift_batches(0, 3), Vec::new());
    }

    #[test]
    fn shift_batches_use_stable_schedule() {
        let first = shifts_for_range((0, 3));
        assert_eq!(first.len(), 3);
        assert_eq!(first[0].re, 0.0);
        assert_eq!(first[0].im, 1.0);
        assert_eq!(first[2].im, 1.4);
        let second = shifts_for_range((3, 6));
        assert_eq!(second[0].im, 1.6);
    }
}
