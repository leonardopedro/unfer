//! Temporary verification helper: loads trajectory files and prints shape/stats.

use pcnl::persist::load_trajectory;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dump_trajectory FILE");
    let rows = load_trajectory(Path::new(&path)).expect("load");
    let p = rows[0].len();
    let mean: f64 = rows.iter().flatten().map(|x| *x as f64).sum::<f64>() / (rows.len() * p) as f64;
    let norm0: f64 = rows[0]
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norms: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt())
        .collect();
    let nmin = norms.iter().cloned().fold(f64::INFINITY, f64::min);
    let nmax = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "samples={} P={} mean={:.6} ||theta_0||={:.3} ||theta||_range=[{:.3}, {:.3}]",
        rows.len(),
        p,
        mean,
        norm0,
        nmin,
        nmax
    );
}
