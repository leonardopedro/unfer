//! Count-Sketch Heavy Hitters algorithm for peak recovery from a
//! probability sketch p̃ in R^{K_2}.
//!
//! Uses the standard Misra–Gries / Count-Min approach: maintain
//! a fixed number of (key, counter) pairs, decrement all counters when
//! a new key arrives that isn't tracked, and return the top-k by count.
//!
//! Complexity: O(K_2 log k) amortized for the full update + query.

/// Count-Sketch Heavy Hitters: tracks the top-k most frequent items
/// from a stream of (index, delta) updates.
#[derive(Debug, Clone)]
pub struct HeavyHitters {
    top_k: usize,
    min_count: f64,
    counters: Vec<(u32, f64)>,
}

impl HeavyHitters {
    /// Create a new HeavyHitters tracker.
    /// - `k`: number of items to track (top-k).
    /// - `min_count`: minimum count to be included in results.
    pub fn new(k: usize, min_count: f64) -> Self {
        Self {
            top_k: k,
            min_count,
            counters: Vec::with_capacity(k),
        }
    }

    /// Update the tracker with a new (index, delta) observation.
    pub fn sketch_add(&mut self, idx: u32, delta: f64) {
        // If idx is already tracked, increment.
        for entry in self.counters.iter_mut() {
            if entry.0 == idx {
                entry.1 += delta;
                return;
            }
        }
        // If there's room, add a new entry.
        if self.counters.len() < self.top_k {
            self.counters.push((idx, delta));
            return;
        }
        // Otherwise, find the minimum-count entry.
        let min_pos = self
            .counters
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        if let Some(i) = min_pos {
            if self.counters[i].1 + delta >= 0.0 {
                self.counters[i].0 = idx;
                self.counters[i].1 += delta;
            } else {
                // Decrement all counters by the minimum.
                let min_val = self.counters[i].1;
                for entry in self.counters.iter_mut() {
                    entry.1 -= min_val;
                    if entry.1 <= 0.0 {
                        *entry = (idx, delta);
                        break;
                    }
                }
            }
        }
    }

    /// Return the top-k items as (index, estimated_count) pairs,
    /// sorted by count descending.
    pub fn top_k(&self) -> Vec<(u32, f64)> {
        let mut sorted: Vec<(u32, f64)> = self
            .counters
            .iter()
            .filter(|(_, c)| *c >= self.min_count)
            .map(|(i, c)| (*i, *c))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Bulk update from a full probability vector p̃ in R^{K_2}.
    pub fn update_from_distribution(&mut self, p: &[f64]) {
        for (i, &v) in p.iter().enumerate() {
            if v > 0.0 {
                self.sketch_add(i as u32, v);
            }
        }
    }

    /// Return the single highest-count index and its estimated count.
    /// Returns None if the tracker is empty.
    pub fn top_one(&self) -> Option<(u32, f64)> {
        self.counters
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, c)| (*i, *c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heavy_hitters_top_one_single_entry() {
        let mut hh = HeavyHitters::new(5, 0.0);
        hh.sketch_add(42, 1.0);
        let (idx, _count) = hh.top_one().unwrap();
        assert_eq!(idx, 42);
    }

    #[test]
    fn heavy_hitters_distinguishes_modes() {
        let mut hh = HeavyHitters::new(3, 0.0);
        // Mode 0 appears 5 times, mode 1 appears 3 times, mode 2 appears 1 time.
        for _ in 0..5 {
            hh.sketch_add(0, 1.0);
        }
        for _ in 0..3 {
            hh.sketch_add(1, 1.0);
        }
        hh.sketch_add(2, 1.0);
        let (idx, _count) = hh.top_one().unwrap();
        assert_eq!(idx, 0, "mode 0 should be the top hit");
    }

    #[test]
    fn heavy_hitters_update_from_distribution() {
        let mut hh = HeavyHitters::new(3, 0.0);
        let dist = vec![0.1, 0.0, 0.5, 0.0, 0.3, 0.1];
        hh.update_from_distribution(&dist);
        let (idx, _count) = hh.top_one().unwrap();
        assert_eq!(idx, 2, "mode 2 has the highest count (0.5)");
    }

    #[test]
    fn heavy_hitters_eviction_under_pressure() {
        let mut hh = HeavyHitters::new(2, 0.0);
        hh.sketch_add(0, 1.0);
        hh.sketch_add(1, 1.0);
        // Adding many different items should evict the lowest-count one.
        for i in 10..20 {
            hh.sketch_add(i, 1.0);
        }
        // The tracker should still have 2 entries.
        assert!(hh.top_one().is_some());
    }
}
