//! The one SVG both dependency-graph pages are drawn with.
//!
//! `--vis dag.html` (`cli::formalize::dag_html`) grew this layout first, as
//! this project's own stand-in for ProofFlow's `vis.py` — a Plotly/matplotlib
//! figure that needs Python to open. `mm0web` wants the same artefact for an
//! mm0 submission, and a second copy of the layout, the edge geometry and the
//! escaping would have been a second place for them to drift. So the canvas
//! lives here and both pages hand it their boxes.
//!
//! The split is deliberate: **this module draws, it does not judge.** Layering,
//! escape and truncation are shared; what a box means — ProofFlow's four node
//! kinds and its score, or mm0's declarations and their structural faults —
//! stays with the caller, as do the verdicts (`formalize::graph::check_DAG`
//! there, `mm0web::graph` here).

/// One box, in layer order.
pub struct BoxView {
    /// `0` for a box nothing depends on; `1 + max(layer(deps))` otherwise.
    /// Both callers compute this against their own graph, because the rule for
    /// what a *dependency* is belongs to them.
    pub layer: usize,
    pub id: String,
    /// Drawn beside the id in grey — the node kind, the declaration kind.
    pub subtitle: String,
    /// Box stroke: green proved, amber declared, red faulty.
    pub stroke: &'static str,
    /// Body lines, left-aligned under the id.
    pub lines: Vec<String>,
    /// Ids of the boxes this one rests on, for the arrows.
    pub deps: Vec<String>,
    /// An extra line in its own colour (an error, a fault), after `lines`.
    pub accent: Option<(String, &'static str)>,
    /// Right-aligned on the id line (a score).
    pub badge: Option<String>,
}

impl BoxView {
    pub fn new(layer: usize, id: impl Into<String>, stroke: &'static str) -> Self {
        Self {
            layer,
            id: id.into(),
            subtitle: String::new(),
            stroke,
            lines: Vec::new(),
            deps: Vec::new(),
            accent: None,
            badge: None,
        }
    }
}

/// Canvas constants. The height is computed from the tallest box, so a caller
/// with two body lines and one with three both fit.
#[derive(Clone, Copy)]
pub struct Canvas {
    pub box_width: f64,
    /// `aria-label` on the `<svg>`.
    pub label: &'static str,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            box_width: 320.0,
            label: "proof dependency graph",
        }
    }
}

const BOX_H_PAD: f64 = 20.0;
const LINE_H: f64 = 20.0;
const LINE0: f64 = 48.0;
const ID_Y: f64 = 26.0;
const GAP_X: f64 = 56.0;
const GAP_Y: f64 = 26.0;
const MARGIN: f64 = 28.0;
const LINE_MAX: usize = 44;
const ID_MAX: usize = 26;

/// Draw the boxes and their edges as one self-contained `<svg>`: no script,
/// no `src`, no network.
pub fn svg(boxes: &[BoxView], canvas: &Canvas) -> String {
    const ROW_H_MIN: f64 = 92.0;
    let box_height = boxes
        .iter()
        .map(|b| {
            let mut h = LINE0 + LINE_H * b.lines.len().max(1) as f64;
            if b.accent.is_some() {
                h += LINE_H - 6.0;
            }
            h + BOX_H_PAD
        })
        .fold(ROW_H_MIN, f64::max);

    // Placement: one row per layer, in the order the caller listed the boxes.
    let mut next_in_layer: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(boxes.len());
    let (mut widest, mut tallest) = (1usize, 1usize);
    for b in boxes {
        let row = next_in_layer.entry(b.layer).or_insert(0);
        positions.push((
            MARGIN + *row as f64 * (canvas.box_width + GAP_X),
            MARGIN + b.layer as f64 * (box_height + GAP_Y),
        ));
        *row += 1;
        widest = widest.max(*row);
        tallest = tallest.max(b.layer + 1);
    }

    let width = MARGIN * 2.0 + widest as f64 * (canvas.box_width + GAP_X);
    let height = MARGIN * 2.0 + tallest as f64 * (box_height + GAP_Y);

    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" \
         width=\"100%\" role=\"img\" aria-label=\"{}\">\n",
        canvas.label
    ));
    out.push_str(
        "<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" \
         markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">\
         <path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#8a8f98\"/></marker></defs>\n",
    );

    // Edges first, so the boxes paint over the arrowheads.
    let by_id: std::collections::HashMap<&str, (f64, f64)> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id.as_str(), positions[i]))
        .collect();
    for (i, b) in boxes.iter().enumerate() {
        let (x, y) = positions[i];
        for dep in &b.deps {
            let Some(&(dx, dy)) = by_id.get(dep.as_str()) else {
                continue;
            };
            let (x1, y1) = (dx + canvas.box_width, dy + box_height / 2.0);
            let (x2, y2) = (x, y + box_height / 2.0);
            out.push_str(&format!(
                "<path d=\"M {x1:.1} {y1:.1} C {x1:.1} {y2:.1} {x2:.1} {y1:.1} {x2:.1} {y2:.1}\" \
                 fill=\"none\" stroke=\"#8a8f98\" stroke-width=\"1.4\" \
                 marker-end=\"url(#arrow)\"/>\n"
            ));
        }
    }

    for (i, b) in boxes.iter().enumerate() {
        let (x, y) = positions[i];
        out.push_str(&format!(
            "<g><rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w}\" height=\"{h}\" rx=\"8\" \
             fill=\"#ffffff\" stroke=\"{}\" stroke-width=\"2\"/>\n",
            b.stroke,
            w = canvas.box_width,
            h = box_height,
            x = x,
            y = y,
        ));
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" \
             font-size=\"15\" font-weight=\"700\" fill=\"#1b1e23\">{} <tspan \
             font-weight=\"400\" fill=\"#6b7076\">{}</tspan></text>\n",
            x + 14.0,
            y + ID_Y,
            escape(&truncate(&b.id, ID_MAX)),
            escape(&b.subtitle),
        ));
        if let Some(badge) = &b.badge {
            out.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" \
                 font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" \
                 font-size=\"12\" fill=\"#1b1e23\">{}</text>\n",
                x + canvas.box_width - 14.0,
                y + ID_Y,
                escape(badge),
            ));
        }
        for (n, line) in b.lines.iter().enumerate() {
            out.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" \
                 font-size=\"12\" fill=\"#3a3f45\">{}</text>\n",
                x + 14.0,
                y + LINE0 + n as f64 * LINE_H,
                escape(&truncate(line, LINE_MAX)),
            ));
        }
        if let Some((text, colour)) = &b.accent {
            let n = b.lines.len();
            out.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" \
                 font-size=\"11\" fill=\"{colour}\">{}</text>\n",
                x + 14.0,
                y + LINE0 + n as f64 * LINE_H + 10.0,
                escape(&truncate(text, LINE_MAX)),
            ));
        }
        out.push_str("</g>\n");
    }
    out.push_str("</svg>\n");
    out
}

/// Truncate to `n` visible characters, keeping the head and marking the cut.
pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let head: String = s.chars().take(n.saturating_sub(1)).collect();
    format!("{head}…")
}

/// Minimal HTML escaping. Box content is model output or submitted source, so
/// this is a correctness requirement, not a nicety.
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> Vec<BoxView> {
        let mut a = BoxView::new(0, "tc_1", "#1a7f37");
        a.subtitle = "TheoremCondition".into();
        a.lines = vec!["st0".into()];
        let mut b = BoxView::new(1, "l1", "#bf8700");
        b.subtitle = "Lemma".into();
        b.lines = vec!["run_cons".into(), "Love(john, mary)".into()];
        b.badge = Some("score 1.00".into());
        b.deps = vec!["tc_1".into()];
        let mut c = BoxView::new(2, "ts_1", "#a4262c");
        c.lines = vec!["q".into()];
        c.accent = Some(("failed: nope".into(), "#a4262c"));
        c.deps = vec!["l1".into()];
        vec![a, b, c]
    }

    #[test]
    fn the_figure_is_self_contained_and_escapes_box_content() {
        let html = svg(&chain(), &Canvas::default());
        assert!(html.starts_with("<svg "));
        assert!(html.contains("viewBox"));
        assert!(!html.contains("<script"));
        assert!(!html.contains(" src="));
        assert!(html.contains("Love(john, mary)"), "body lines are drawn");
        assert!(html.contains("score 1.00"), "badges are drawn");
        assert!(html.contains("#a4262c"), "stroke and accent survive");
        assert!(html.contains("marker-end=\"url(#arrow)\""), "edges are arrows");

        let mut hostile = BoxView::new(0, "<b>&", "#8a8f98");
        hostile.lines = vec!["<script>".to_string()];
        let out = svg(&[hostile], &Canvas::default());
        assert!(!out.contains("<script>"), "content must be escaped: {out}");
        assert!(out.contains("&lt;script&gt;"));
    }

    #[test]
    fn boxes_stack_below_their_layer_and_the_canvas_grows_with_content() {
        let html = svg(&chain(), &Canvas::default());
        // Layer 0 sits at the margin; each further layer is one box height
        // below the last, and every box is as tall as the tallest one.
        assert!(html.contains("y=\"28.0\""), "layer 0 at the margin");
        assert!(html.contains("y=\"162.0\""), "layer 1 one box below");
        assert!(
            html.contains("height=\"108\""),
            "a uniform box height, sized for the tallest box"
        );
        let short = svg(&[BoxView::new(0, "a", "#8a8f98")], &Canvas::default());
        assert!(
            short.contains("height=\"92\""),
            "a single line needs less room"
        );
    }

    #[test]
    fn truncation_marks_the_cut() {
        assert_eq!(truncate("short", 10), "short");
        let long = "x".repeat(60);
        let t = truncate(&long, 10);
        assert_eq!(t.chars().count(), 10);
        assert!(t.ends_with('…'));
        assert_eq!(escape("<script>alert(1)</script>"), "&lt;script&gt;alert(1)&lt;/script&gt;");
    }
}
