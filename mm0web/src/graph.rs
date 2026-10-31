//! The submission's dependency graph — this service's ProofFlow-shaped view of
//! an mm0 development.
//!
//! `ProofFlow` (checkout at `../../../ProofFlow/`, MIT) formalizes a natural-
//! language proof by first building a **dependency graph** of its steps and
//! validating it (`check_DAG`: cycles, orphans, forward refs) before anything
//! is proved. mm0web does not take that pipeline's input — an `.mm0`+`.mm1`
//! pair is what arrives here, and `mm0-rs`/`mm0-c` remain the only authority
//! on whether it is *correct* — but the graph itself is exactly as useful
//! here, and it is what `physlib`'s website shows for its own library:
//! `../../../physlib/scripts/website/informal.lean` generates the library's
//! dependency graph as a DOT file (rendered to `Docs/graph.svg` with
//! graphviz), and `import-graph.py` in the same directory plots the import
//! graph of the whole corpus. A proof is easier to trust when you can *see*
//! what rests on what.
//!
//! So this module derives the same artefact from the submission, textually:
//!
//! ```text
//! split statements (`;` outside `$ … $`) → classify → merge `.mm0` + `.mm1`
//!   → reference edges → faults (unresolved / forward / cycle) → SVG + JSON
//! ```
//!
//! Three properties this deliberately keeps:
//!
//! 1. **It is a view, never a gate.** Every verdict the service issues comes
//!    from the two checkers (`crate::checker`); a fault here is a hint about
//!    the *structure* of the submission, and an empty graph never rejects it.
//! 2. **Derived, never stored.** The graph is recomputed from the sources the
//!    index already holds, so the Zenodo snapshot/delta format
//!    (`crate::store`) does not change by a byte.
//! 3. **Self-contained output.** Like `logos`' `dag.html`
//!    (`../../logos/src/cli/formalize.rs`), the page is one HTML file with an
//!    inline SVG, inline CSS and no script — the same choice `physlib` makes
//!    by rendering its graph to a static SVG rather than asking a reviewer's
//!    browser to fetch a plotting library.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{json, Value};

use crate::http::escape_html;
use logos::dagviz;

/// Above this many declarations the boxes stop being readable and the page
/// stops being fast; the JSON endpoint stays available for the full graph.
pub const MAX_RENDER_DECLS: usize = 500;

/// How many `unused` names are listed before the list is summarised.
const MAX_UNUSED_LISTED: usize = 40;

/// How many faults the page lists before it says how many more there are.
const MAX_FAULTS_LISTED: usize = 40;

/// `1` when a count is not exactly one, for the handful of places that
/// pluralise themselves.
fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Keywords that would otherwise look like references to declarations nobody
/// made. Only what this parser recognises as a statement head or a `pub`/
/// `import` marker — everything else in the submission is a name.
const KEYWORDS: &[&str] = &[
    "sort", "provable", "strict", "term", "axiom", "def", "theorem", "macro", "pub", "import",
    "delimiter", "prefix", "infixl", "infixr", "statement", "notation", "coercion", "local",
    "hypothesis",
];

/// The page's own stylesheet. Inline on purpose: the document has to render
/// from disk, from a `<pre>` dump, or off a host with no outbound network.
/// Public because an embedding view (`web::render_result`) serves the same
/// rules, or the legend's swatches mean nothing there.
pub const STYLE: &str = "\n\
             body{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:24px;\
             color:#1b1e23;background:#fbfbfc;max-width:1180px}\n\
             h1{font-size:20px;margin:0 0 6px}\n\
             p{margin:6px 0}\n\
             .summary{color:#3a3f45}\n\
             .muted{color:#6b7076;font-size:12px}\n\
             .ok{color:#1a7f37}\n\
             .legend{color:#6b7076;font-size:12px;margin:10px 0}\n\
             .sw{display:inline-block;width:11px;height:11px;border-radius:3px;\
             vertical-align:-1px;margin:0 4px 0 14px;border:2px solid}\n\
             .sw.proved{border-color:#1a7f37} .sw.declared{border-color:#bf8700}\n\
             .sw.bad{border-color:#a4262c}\n\
             .faults{font-size:13px;padding-left:18px}\n\
             .faults .bad{color:#a4262c} .faults .info{color:#6b7076}\n\
             code{background:#f0f1f3;padding:1px 5px;border-radius:4px;font-size:12px}\n\
             .nav{font-size:13px;margin-top:18px}\n";

/// A declaration kind, as the relaxed MM0 surface spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Sort,
    Term,
    Axiom,
    Def,
    Theorem,
    Other,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Sort => "sort",
            Kind::Term => "term",
            Kind::Axiom => "axiom",
            Kind::Def => "def",
            Kind::Theorem => "theorem",
            Kind::Other => "other",
        }
    }
}

/// One declaration of the submission, with what it refers to.
#[derive(Clone, Debug)]
pub struct Decl {
    pub name: String,
    pub kind: Kind,
    /// The statement, whitespace-normalised, without its proof tail.
    pub statement: String,
    /// The proof tail (`pub theorem … = '…`), when the submission carries one.
    pub proof: Option<String>,
    /// Declared in the `.mm0` spec.
    pub in_spec: bool,
    /// Declared in the `.mm1` proofs file.
    pub in_proof: bool,
    pub spec_line: Option<u32>,
    pub proof_line: Option<u32>,
    /// Names this declaration refers to, in the order they appear, deduped and
    /// with its own binders removed. Local names never match a declaration, so
    /// they drop out of this set on their own.
    pub deps: Vec<String>,
}

impl Decl {
    /// What the submission actually carries for this name.
    pub fn source(&self) -> &'static str {
        match (self.in_spec, self.in_proof) {
            (true, true) => "spec+proof",
            (true, false) => "spec",
            (false, true) => "proof",
            (false, false) => "declared",
        }
    }

    /// The line to cite: the `.mm1` when the name is in it (that is the file
    /// `mm0-rs compile` reads), otherwise the `.mm0`.
    pub fn line(&self) -> u32 {
        self.proof_line.or(self.spec_line).unwrap_or(0)
    }

    /// A proof term is present — the ProofFlow legend's "has tactics" case.
    pub fn proved(&self) -> bool {
        self.proof.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    /// A reference to a name nothing in the submission declares.
    Unresolved,
    /// The same, but the submission imports modules — the name is theirs.
    External,
    /// A reference to a declaration that appears later in the source.
    Forward,
    /// A reference that closes a cycle.
    Cycle,
}

impl FaultKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FaultKind::Unresolved => "unresolved",
            FaultKind::External => "external",
            FaultKind::Forward => "forward",
            FaultKind::Cycle => "cycle",
        }
    }

    /// Only a real structural fault paints a node red; an `external` name is
    /// information about where the submission ends, not about what is wrong.
    pub fn is_fault(self) -> bool {
        !matches!(self, FaultKind::External)
    }
}

#[derive(Clone, Debug)]
pub struct Fault {
    pub kind: FaultKind,
    pub node: String,
    pub detail: String,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub decls: Vec<Decl>,
    pub faults: Vec<Fault>,
    /// Declared but referenced by nothing — `physlib`'s "unused" line and this
    /// project's own `theory coverage: used=… unused=…` header, computed.
    pub unused: Vec<String>,
    /// Modules the submission imports (`import "…";`).
    pub imports: Vec<String>,
    /// Identifiers that nothing declares but *some* declaration binds — the
    /// variables of the development. The `.mm1` restates a theorem with the
    /// binders the `.mm0` gave it, so its statement reads as free names unless
    /// binders from anywhere in the submission are counted; without this they
    /// all become phantom "undeclared" faults (peano: 8 840 of them).
    pub variables: HashSet<String>,
    /// Names the declaration grammar does not declare but the submission does
    /// introduce: notation tokens (`infixl subset: $C_$`, `prefix ex: $E.$`)
    /// and the proof-script helpers of a `do { … }` block
    /// (`(def (induct a c . xs) …)`). Neither is an mm0 declaration, and
    /// neither is a reference to something missing.
    pub symbols: HashSet<String>,
    /// Whether the two checkers accepted this submission. Every stored proof
    /// was; a rejected result page was not — and that difference decides how
    /// an undeclared name is reported (see [`Graph::severe`]).
    pub accepted: bool,
}

impl Graph {
    /// Build the graph of a submission. `mm1` is parsed first when present:
    /// it is the file `mm0-rs compile` walks, so its order *is* the order a
    /// forward reference has to be judged against.
    pub fn build(mm0: &str, mm1: &str) -> Graph {
        let (spec, spec_imports, spec_symbols) = parse_file(mm0, false);
        let (proof, proof_imports, proof_symbols) = parse_file(mm1, true);

        let mut g = Graph {
            imports: {
                let mut v = proof_imports;
                for i in spec_imports {
                    if !v.contains(&i) {
                        v.push(i);
                    }
                }
                v
            },
            ..Default::default()
        };

        let mut index: HashMap<String, usize> = HashMap::new();
        let mut locals: Vec<HashSet<String>> = Vec::new();

        for p in proof {
            match index.get(&p.name) {
                Some(&i) => {
                    let d = &mut g.decls[i];
                    d.in_proof = true;
                    d.proof_line = Some(p.line);
                    d.kind = p.kind;
                    d.statement = p.statement;
                    d.proof = p.proof;
                    locals[i].extend(p.locals);
                }
                None => {
                    index.insert(p.name.clone(), g.decls.len());
                    locals.push(p.locals);
                    g.decls.push(Decl {
                        name: p.name,
                        kind: p.kind,
                        statement: p.statement,
                        proof: p.proof,
                        in_spec: false,
                        in_proof: true,
                        spec_line: None,
                        proof_line: Some(p.line),
                        deps: Vec::new(),
                    });
                }
            }
        }
        for p in spec {
            match index.get(&p.name) {
                Some(&i) => {
                    let d = &mut g.decls[i];
                    d.in_spec = true;
                    d.spec_line = Some(p.line);
                    locals[i].extend(p.locals);
                }
                None => {
                    index.insert(p.name.clone(), g.decls.len());
                    locals.push(p.locals);
                    g.decls.push(Decl {
                        name: p.name,
                        kind: p.kind,
                        statement: p.statement,
                        proof: p.proof,
                        in_spec: true,
                        in_proof: false,
                        spec_line: Some(p.line),
                        proof_line: None,
                        deps: Vec::new(),
                    });
                }
            }
        }

        let names: HashSet<String> = g.decls.iter().map(|d| d.name.clone()).collect();
        // Bound in one declaration, declared in none: a variable of the
        // development rather than a reference to something missing.
        let mut bound: HashSet<String> = HashSet::new();
        for l in &locals {
            bound.extend(l.iter().cloned());
        }
        g.variables = bound.difference(&names).cloned().collect();
        g.symbols = spec_symbols.union(&proof_symbols).cloned().collect();
        let mut unknown: Vec<Vec<String>> = Vec::with_capacity(g.decls.len());
        for i in 0..g.decls.len() {
            let mut text = g.decls[i].statement.clone();
            if let Some(p) = &g.decls[i].proof {
                text.push(' ');
                text.push_str(p);
            }
            let self_name = g.decls[i].name.clone();
            let mut deps: Vec<String> = Vec::new();
            let mut unres: Vec<String> = Vec::new();
            for id in identifiers(&text) {
                if id == self_name || locals[i].contains(&id) {
                    continue;
                }
                if names.contains(id.as_str()) {
                    if !deps.contains(&id) {
                        deps.push(id);
                    }
                } else if !KEYWORDS.contains(&id.as_str())
                    && !g.variables.contains(&id)
                    && !g.symbols.contains(&id)
                    && !unres.contains(&id)
                {
                    // A name nothing here declares, no declaration binds, and
                    // no notation or script helper introduces: a candidate
                    // fault, decided by `check_references` once imports are
                    // known.
                    unres.push(id);
                }
            }
            g.decls[i].deps = deps;
            unknown.push(unres);
        }

        g.check_references(&index, &unknown);
        g.check_cycles(&index);
        g.find_unused(&index);
        g
    }

    fn check_references(&mut self, index: &HashMap<String, usize>, unknown: &[Vec<String>]) {
        let has_imports = !self.imports.is_empty();
        let mut faults = Vec::new();
        for (i, d) in self.decls.iter().enumerate() {
            for dep in &unknown[i] {
                faults.push(Fault {
                    kind: if has_imports {
                        FaultKind::External
                    } else {
                        FaultKind::Unresolved
                    },
                    node: d.name.clone(),
                    detail: format!(
                        "uses `{dep}`, which neither file declares{}",
                        if has_imports {
                            " — it must come from an imported module"
                        } else {
                            ""
                        }
                    ),
                });
            }
            for dep in &d.deps {
                if let Some(&j) = index.get(dep.as_str())
                    && self.is_forward(i, j)
                {
                    faults.push(Fault {
                        kind: FaultKind::Forward,
                        node: d.name.clone(),
                        detail: format!(
                            "uses `{dep}`, declared later (line {})",
                            self.decls[j].line()
                        ),
                    });
                }
            }
        }
        self.faults.extend(faults);
    }

    /// Is `j` declared *later than* `i` — and therefore a forward reference?
    ///
    /// Only within one file. The merged declaration list is the `.mm1` order
    /// first and the `.mm0`-only names appended, so comparing positions across
    /// the merge would call every spec-only definition a forward reference;
    /// the two files are separate streams and `mm0-rs` reads them separately.
    /// Both in the proof file → proof lines. Both only in the spec → spec
    /// lines. Anything else straddles the two, where "later" has no meaning.
    fn is_forward(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.decls[i], &self.decls[j]);
        match (a.proof_line, b.proof_line) {
            (Some(x), Some(y)) => y > x,
            _ => match (a.spec_line, b.spec_line) {
                (Some(x), Some(y)) if a.proof_line.is_none() && b.proof_line.is_none() => y > x,
                _ => false,
            },
        }
    }

    /// Iterative DFS over the edges, so a deep chain cannot overflow the
    /// stack: mm0 is sequential, and a submitted 4 MiB development may be deep.
    fn check_cycles(&mut self, index: &HashMap<String, usize>) {
        let n = self.decls.len();
        let mut state = vec![0u8; n]; // 0 unvisited · 1 on the path · 2 done
        let mut found: Vec<Fault> = Vec::new();
        for start in 0..n {
            if state[start] != 0 {
                continue;
            }
            state[start] = 1;
            let mut path = vec![start];
            let mut frames: Vec<(usize, usize)> = vec![(start, 0)];
            while let Some((u, k)) = frames.last().copied() {
                if k < self.decls[u].deps.len() {
                    frames.last_mut().expect("frame").1 = k + 1;
                    let dep = self.decls[u].deps[k].clone();
                    let Some(&v) = index.get(dep.as_str()) else {
                        continue;
                    };
                    match state[v] {
                        0 => {
                            state[v] = 1;
                            frames.push((v, 0));
                            path.push(v);
                        }
                        1 => {
                            // A back edge: the cycle is the tail of the path
                            // plus the node it points at.
                            let from = path.iter().position(|&x| x == v).unwrap_or(0);
                            let members: Vec<&str> = path[from..]
                                .iter()
                                .map(|&x| self.decls[x].name.as_str())
                                .collect();
                            found.push(Fault {
                                kind: FaultKind::Cycle,
                                node: self.decls[u].name.clone(),
                                detail: format!("{} → `{dep}`", members.join(" → ")),
                            });
                        }
                        _ => {}
                    }
                } else {
                    state[u] = 2;
                    frames.pop();
                    path.pop();
                }
            }
        }
        // One fault per cycle, not one per back edge into it.
        let mut seen: HashSet<String> = HashSet::new();
        for f in found {
            if seen.insert(f.detail.clone()) {
                self.faults.push(f);
            }
        }
    }

    fn find_unused(&mut self, index: &HashMap<String, usize>) {
        let mut indeg = vec![0usize; self.decls.len()];
        for d in &self.decls {
            for dep in &d.deps {
                if let Some(&j) = index.get(dep.as_str()) {
                    indeg[j] += 1;
                }
            }
        }
        self.unused = self
            .decls
            .iter()
            .enumerate()
            .filter(|(i, _)| indeg[*i] == 0)
            .map(|(_, d)| d.name.clone())
            .collect();
    }

    pub fn len(&self) -> usize {
        self.decls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.decls.iter().any(|d| d.name == name)
    }

    pub fn reference_count(&self) -> usize {
        self.decls.iter().map(|d| d.deps.len()).sum()
    }

    pub fn counts_by_kind(&self) -> BTreeMap<&'static str, usize> {
        let mut m: BTreeMap<&'static str, usize> = BTreeMap::new();
        for d in &self.decls {
            *m.entry(d.kind.as_str()).or_insert(0) += 1;
        }
        m
    }

    /// Mark the submission as one the two checkers accepted. Every proof
    /// stored by this service was; the rejected result page does not.
    pub fn accepted(mut self) -> Graph {
        self.accepted = true;
        self
    }

    /// Is this fault worth painting red?
    ///
    /// In a submission the checkers accepted, an undeclared name *cannot* be
    /// a defect — `mm0-rs` would have rejected it. What is left is by
    /// construction a variable, a notation token or a proof-script helper
    /// this parser does not model, so it is reported as information. The
    /// structural rules still hold either way, and a rejected submission
    /// keeps every undeclared name as the hint it is.
    fn severe(&self, f: &Fault) -> bool {
        match f.kind {
            FaultKind::External => false,
            FaultKind::Unresolved => !self.accepted,
            FaultKind::Forward | FaultKind::Cycle => true,
        }
    }

    /// Structural faults only — `external` names are context, not failures.
    pub fn fault_count(&self) -> usize {
        self.faults.iter().filter(|f| self.severe(f)).count()
    }

    pub fn faulty_names(&self) -> HashSet<String> {
        self.faults
            .iter()
            .filter(|f| self.severe(f))
            .map(|f| f.node.clone())
            .collect()
    }

    pub fn to_json(&self) -> Value {
        let decls: Vec<Value> = self
            .decls
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "kind": d.kind.as_str(),
                    "source": d.source(),
                    "line": d.line(),
                    "statement": d.statement,
                    "proof": d.proof.as_deref(),
                    "proved": d.proved(),
                    "deps": d.deps,
                })
            })
            .collect();
        let faults: Vec<Value> = self
            .faults
            .iter()
            .map(|f| {
                json!({"kind": f.kind.as_str(), "node": f.node, "detail": f.detail})
            })
            .collect();
        let by_kind: serde_json::Map<String, Value> = self
            .counts_by_kind()
            .into_iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect();
        json!({
            "declarations": decls,
            "faults": faults,
            "unused": self.unused,
            "imports": self.imports,
            "counts": {
                "declarations": self.decls.len(),
                "references": self.reference_count(),
                "faults": self.fault_count(),
                "variables": self.variables.len(),
                "symbols": self.symbols.len(),
                "accepted": self.accepted,
                "by_kind": by_kind,
            },
        })
    }

    /// The standalone page: `<h1>`, the caller's `note`, and
    /// [`Graph::body_fragment`].
    ///
    /// `note` is raw HTML (the caller escapes what it interpolates) and
    /// `failed` is the set of node names a rejection log named, which the
    /// legend's red swatch covers.
    pub fn html(&self, title: &str, note: &str, failed: &HashSet<String>) -> String {
        format!(
            "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
             <title>{title}</title><style>{style}</style></head><body>\n\
             <h1>{title}</h1>\n{note}\n{body}\n\
             <p class=\"nav\"><a href=\"/\">home</a> · \
             <a href=\"/search\">search</a> · \
             <a href=\"/stats\">stats</a></p>\n</body></html>\n",
            title = escape_html(title),
            note = note,
            body = self.body_fragment(failed),
            style = STYLE,
        )
    }

    /// The page body, without the document shell — so a result page or any
    /// other server-rendered view can embed the same artefact instead of
    /// linking out to it.
    pub fn body_fragment(&self, failed: &HashSet<String>) -> String {
        let mut faulty = self.faulty_names();
        faulty.extend(failed.iter().cloned());

        let kinds: Vec<String> = self
            .counts_by_kind()
            .into_iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect();
        let summary = format!(
            "{} declarations · {} references · {}{}{}",
            self.decls.len(),
            self.reference_count(),
            kinds.join(" · "),
            if self.variables.is_empty() {
                String::new()
            } else {
                format!(" · {} bound variable{}", self.variables.len(), plural(self.variables.len()))
            },
            if kinds.is_empty() {
                " (nothing declared — the graph is read from the submission's \
                 own statements, so an empty one means no declarations parsed)"
            } else {
                ""
            },
        );

        let faults_html = if self.faults.is_empty() {
            "<p class=\"ok\">no dependency faults</p>".to_string()
        } else {
            // The page is for reading: a development can fault thousands of
            // times, and the list has to end somewhere. The JSON endpoint
            // carries all of them.
            let items: Vec<String> = self
                .faults
                .iter()
                .take(MAX_FAULTS_LISTED)
                .map(|f| {
                    format!(
                        "<li class=\"{}\"><code>{}</code> <b>{}</b> — {}</li>",
                        if self.severe(f) { "bad" } else { "info" },
                        escape_html(f.kind.as_str()),
                        escape_html(&f.node),
                        escape_html(&f.detail),
                    )
                })
                .collect();
            let more = if self.faults.len() > MAX_FAULTS_LISTED {
                format!(
                    "<li class=\"info\">… and {} more — the JSON view of \
                     this graph carries all of them.</li>",
                    self.faults.len() - MAX_FAULTS_LISTED
                )
            } else {
                String::new()
            };
            let verdict = if self.accepted {
                "<p class=\"muted\">Both checkers accepted this submission, so an \
                 undeclared name below cannot be a defect: it is a variable, a \
                 notation token or a proof-script helper this parser does not \
                 model, listed as information.</p>"
            } else {
                ""
            };
            let severe = self.fault_count();
            let info = self.faults.len() - severe;
            let heading = match (severe, info) {
                (0, 0) => String::new(),
                (0, n) => format!(
                    "{n} undeclared name{} — listed as information:",
                    plural(n)
                ),
                (n, 0) => format!("{n} structural fault{}:", plural(n)),
                (f, n) => format!(
                    "{f} structural fault{} and {n} undeclared name{} — the \
                     undeclared ones are listed as information:",
                    plural(f),
                    plural(n)
                ),
            };
            format!(
                "{verdict}<p class=\"muted\">{heading}</p>\
                 <ul class=\"faults\">{}{}</ul>",
                items.join(""),
                more
            )
        };

        let unused_html = if self.unused.is_empty() {
            String::new()
        } else {
            let shown: Vec<&str> = self
                .unused
                .iter()
                .take(MAX_UNUSED_LISTED)
                .map(String::as_str)
                .collect();
            let more = if self.unused.len() > MAX_UNUSED_LISTED {
                format!(
                    " and {} more",
                    self.unused.len() - MAX_UNUSED_LISTED
                )
            } else {
                String::new()
            };
            format!(
                "<p class=\"muted\">{} declared, referenced by nothing: {}{}.</p>",
                self.unused.len(),
                escape_html(&shown.join(", ")),
                escape_html(&more)
            )
        };

        let imports_html = if self.imports.is_empty() {
            String::new()
        } else {
            format!(
                "<p>imports: {}</p>",
                self.imports
                    .iter()
                    .map(|i| format!("<code>{}</code>", escape_html(i)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let graph_html = if self.decls.len() > MAX_RENDER_DECLS {
            format!(
                "<p class=\"muted\">{} declarations is more than this page \
                 draws ({MAX_RENDER_DECLS} is the ceiling on readable boxes); \
                 the full graph is at <a href=\"/api/v1/graph\">the JSON \
                 endpoint</a>.</p>",
                self.decls.len()
            )
        } else {
            self.svg(&faulty, failed)
        };
        format!(
            "<p class=\"summary\">{summary}</p>\n{imports_html}\n\
             <p class=\"legend\"><span class=\"sw proved\"></span>proved by a \
             proof term in this submission <span class=\"sw declared\"></span> \
             declared, no proof obligation (sort/term/axiom/def)\n\
             <span class=\"sw bad\"></span>structural fault, or named in the \
             checker's log</p>\n\
             {faults_html}\n{unused_html}\n{graph_html}"
        )
    }

    /// The figure itself comes from `logos::dagviz` — the canvas this
    /// project's `--vis dag.html` already draws with, shared so the two graph
    /// pages cannot drift apart. What goes *in* a box is mm0's: the statement,
    /// the line it sits on, how the name is sourced, and the fault that paints
    /// it red.
    fn svg(&self, faulty: &HashSet<String>, failed: &HashSet<String>) -> String {
        let layers = self.layers();
        let boxes: Vec<dagviz::BoxView> = self
            .decls
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let bad = faulty.contains(&d.name) || failed.contains(&d.name);
                let stroke = if bad {
                    "#a4262c"
                } else if d.proved() {
                    "#1a7f37"
                } else {
                    "#bf8700"
                };
                let accent = self
                    .faults
                    .iter()
                    .find(|f| f.node == d.name && f.kind.is_fault())
                    .map(|f| (f.detail.clone(), "#a4262c"))
                    .or_else(|| {
                        failed
                            .contains(&d.name)
                            .then(|| ("named in the checker's log".to_string(), "#a4262c"))
                    });
                let mut b = dagviz::BoxView::new(layers[i], d.name.clone(), stroke);
                b.subtitle = d.kind.as_str().to_string();
                b.lines = vec![
                    d.statement.clone(),
                    format!(
                        "line {} · {} ref{} · {}",
                        d.line(),
                        d.deps.len(),
                        if d.deps.len() == 1 { "" } else { "s" },
                        d.source()
                    ),
                ];
                b.accent = accent;
                b.deps = d.deps.clone();
                b
            })
            .collect();
        dagviz::svg(
            &boxes,
            &dagviz::Canvas {
                box_width: 300.0,
                label: "declaration dependency graph",
            },
        )
    }

    /// Longest-path layering, iterative: a cyclic or very deep submission
    /// must still produce a page.
    fn layers(&self) -> Vec<usize> {
        let n = self.decls.len();
        let index: HashMap<&str, usize> = self
            .decls
            .iter()
            .enumerate()
            .map(|(i, d)| (d.name.as_str(), i))
            .collect();
        let mut layer = vec![0usize; n];
        let mut state = vec![0u8; n];
        for start in 0..n {
            if state[start] == 2 {
                continue;
            }
            state[start] = 1;
            let mut frames: Vec<(usize, usize)> = vec![(start, 0)];
            while let Some((u, k)) = frames.last().copied() {
                if k < self.decls[u].deps.len() {
                    frames.last_mut().expect("frame").1 = k + 1;
                    let dep = self.decls[u].deps[k].clone();
                    if let Some(&v) = index.get(dep.as_str())
                        && state[v] == 0
                    {
                        state[v] = 1;
                        frames.push((v, 0));
                    }
                } else {
                    let mut l = 0usize;
                    for dep in &self.decls[u].deps {
                        if let Some(&v) = index.get(dep.as_str()) {
                            l = l.max(layer[v] + 1);
                        }
                    }
                    layer[u] = l;
                    state[u] = 2;
                    frames.pop();
                }
            }
        }
        layer
    }
}

// ── parsing ────────────────────────────────────────────────────────────────

struct Parsed {
    kind: Kind,
    name: String,
    statement: String,
    proof: Option<String>,
    locals: HashSet<String>,
    line: u32,
}

/// Split on `;` outside a `$ … $` math string, dropping `--` line comments.
/// The two constructs that decide the whole scan: a `;` inside math is data,
/// and `--` inside math (or inside a comment's own `--`) is not a comment.
fn split_statements(src: &str) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    let mut buf = String::new();
    let mut line: u32 = 1;
    let mut start: Option<u32> = None;
    let mut in_math = false;
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if !in_math && c == '-' && i + 1 < chars.len() && chars[i + 1] == '-' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            buf.push(' ');
            continue;
        }
        if c == '\n' {
            line += 1;
            i += 1;
            if !buf.trim().is_empty() && start.is_none() {
                start = Some(line.saturating_sub(1));
            }
            buf.push(c);
            continue;
        }
        if c == '$' {
            in_math = !in_math;
        }
        if c == ';' && !in_math {
            if !buf.trim().is_empty() {
                out.push((std::mem::take(&mut buf), start.unwrap_or(line)));
            } else {
                buf.clear();
            }
            start = None;
            i += 1;
            continue;
        }
        if !c.is_whitespace() && start.is_none() {
            start = Some(line);
        }
        buf.push(c);
        i += 1;
    }
    if !buf.trim().is_empty() {
        out.push((buf, start.unwrap_or(line)));
    }
    out
}

/// Read one file: its statements and the modules it imports.
fn parse_file(src: &str, _from_proof: bool) -> (Vec<Parsed>, Vec<String>, HashSet<String>) {
    let mut decls: Vec<Parsed> = Vec::new();
    let mut imports: Vec<String> = Vec::new();
    let mut symbols: HashSet<String> = HashSet::new();

    for (raw, line) in split_statements(src) {
        let norm = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if norm.is_empty() {
            continue;
        }
        let s = norm.strip_prefix("pub ").unwrap_or(&norm);

        if let Some(rest) = s.strip_prefix("import ") {
            // `import "metamath/set.mm0";`
            if let Some(q) = rest.find('"')
                && let Some(e) = rest[q + 1..].find('"')
            {
                imports.push(rest[q + 1..q + 1 + e].to_string());
            }
            continue;
        }

        // Modifiers come before the statement keyword: a real spec spells its
        // sorts `strict provable sort wff;`, and dropping that statement for
        // want of a recognised head would declare nothing — every use of
        // `wff` then reads as unresolved, which is precisely what the first
        // run against `peano.mm0` showed (9 000 phantom faults for two
        // missing sorts).
        let mut head = s;
        let (kw, after) = loop {
            let end = head
                .find(|c: char| c.is_whitespace() || c == '(')
                .unwrap_or(head.len());
            let tok = &head[..end];
            if tok == "strict" || tok == "provable" {
                head = head[end..].trim_start();
                continue;
            }
            break (tok, head[end..].trim_start());
        };

        // From the keyword on: `head` is where the modifier loop stopped, so
        // the statement we record and scan excludes `pub`, `strict`,
        // `provable` — they are syntax, not names, and scanning them would
        // report `strict` as an undeclared reference.
        let stmt_src = head;
        let (kind, rest) = match kw {
            // Not declarations, but they do introduce names the rest of the
            // submission uses: a notation's token lives in its math string,
            // and a `do { … }` block defines the helpers its proof terms call.
            "infixl" | "infixr" | "prefix" | "postfix" | "notation" | "delimiter" => {
                // Including single-character tokens: `prefix ex: $E.$`
                // declares `E.`, whose first word is one letter long.
                for id in identifiers(stmt_src) {
                    if !KEYWORDS.contains(&id.as_str()) {
                        symbols.insert(id);
                    }
                }
                continue;
            }
            "do" => {
                symbols.extend(do_block_defs(stmt_src));
                continue;
            }
            "sort" => (Kind::Sort, after),
            "term" => (Kind::Term, after),
            "axiom" => (Kind::Axiom, after),
            "def" => (Kind::Def, after),
            "theorem" => (Kind::Theorem, after),
            "macro" => (Kind::Other, after),
            // `delimiter`, `prefix`, `infixr`, … — syntax, not declarations.
            _ => continue,
        };

        // Name: up to whitespace, `(`, `:` or `,`.
        let name_end = rest
            .find(|c: char| c.is_whitespace() || c == '(' || c == ':' || c == ',')
            .unwrap_or(rest.len());
        if name_end == 0 {
            continue;
        }

        let names: Vec<String> = if kind == Kind::Sort {
            // `sort wff, set;` declares both.
            rest.split(',')
                .map(|seg| {
                    let seg = seg.trim();
                    let e = seg
                        .find(|c: char| c.is_whitespace() || c == '(' || c == ':')
                        .unwrap_or(seg.len());
                    seg[..e].to_string()
                })
                .filter(|n| !n.is_empty())
                .collect()
        } else {
            vec![rest[..name_end].to_string()]
        };
        if names.is_empty() {
            continue;
        }

        // A theorem carries its proof after `= '…`; a def's `= $ … $` body is
        // part of the statement, so only the `= '` form splits.
        let (statement, proof) = match (kind == Kind::Theorem).then(|| stmt_src.find("= '")) {
            Some(Some(p)) => (
                stmt_src[..p].trim_end().to_string(),
                // `= '` is three characters; the proof starts after them.
                Some(stmt_src[p + 3..].trim().to_string()),
            ),
            _ => (stmt_src.to_string(), None),
        };

        let after_name = rest[name_end..].to_string();
        let locals = binder_names(&after_name);

        for name in names {
            decls.push(Parsed {
                kind,
                name,
                statement: statement.clone(),
                proof: proof.clone(),
                locals: locals.clone(),
                line,
            });
        }
    }
    (decls, imports, symbols)
}

/// The names a `do { … }` block defines: `(def (induct a c . xs) …)` and
/// `(def last (…))` both bind `induct` / `last` for the proof terms that
/// follow. None of it is an mm0 declaration, so none of it reaches the graph
/// as a node — but a proof term that calls one must not read as a reference
/// to something that does not exist.
fn do_block_defs(stmt: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = stmt;
    while let Some(i) = rest.find("(def ") {
        rest = rest[i + 5..].trim_start();
        let t = rest.strip_prefix('(').unwrap_or(rest);
        let end = t
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '\''))
            .unwrap_or(t.len());
        let name = &t[..end];
        if !name.is_empty() && !name.starts_with(|c: char| c.is_ascii_digit()) {
            out.push(name.to_string());
        }
        rest = &rest[rest.len().min(1)..];
    }
    out
}

/// Binder names of a signature: the identifiers left of a `:` inside a paren
/// or brace group. `(s: St)` binds `s`, and so does the implicit `{x: nat}`;
/// the `St` / `nat` stays a reference.
fn binder_names(rest: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut depth = 0usize;
    let mut group = String::new();
    for c in rest.chars() {
        match c {
            '(' | '{' => {
                if depth == 0 {
                    depth = 1;
                    group.clear();
                    continue;
                }
                depth += 1;
            }
            ')' | '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some((before, _)) = group.split_once(':') {
                            for id in identifiers(before) {
                                out.insert(id);
                            }
                        }
                        continue;
                    }
                }
            }
            _ => {}
        }
        if depth > 0 {
            group.push(c);
        }
    }
    out
}

/// Identifier tokens, in order of appearance. A token made only of `'` (the
/// proof marker) is not a name.
fn identifiers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty()
            && !cur.starts_with(|c: char| c.is_ascii_digit())
            && cur.chars().any(|c| c.is_alphanumeric())
        {
            out.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' || c == '\'' {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape every certificate in this project has: a spec stream and a
    /// line-identical proof stream with `pub theorem … = '…` tails.
    const SPEC: &str = r#"
provable sort Prop;
sort St;
sort L;
delimiter $ ( $ $ ) $;
term st0: St;
term tl_nil: L;
term tl_cons (h: St) (t: L): L;
term run (s: St) (l: L): Prop;
axiom run_nil (s: St): $ run s tl_nil $;
axiom run_cons (h: St) (t: L) (s: St):
  $ run s t $ > $ run s (tl_cons h t) $;
def trace: L = $ tl_cons st0 tl_nil $;
theorem t_trace: $ run st0 trace $;
"#;

    const PROOF: &str = r#"
provable sort Prop;
sort St;
sort L;
delimiter $ ( $ $ ) $;
term st0: St;
term tl_nil: L;
term tl_cons (h: St) (t: L): L;
term run (s: St) (l: L): Prop;
axiom run_nil (s: St): $ run s tl_nil $;
axiom run_cons (h: St) (t: L) (s: St):
  $ run s t $ > $ run s (tl_cons h t) $;
def trace: L = $ tl_cons st0 tl_nil $;
pub theorem t_trace: $ run st0 trace $ = '(run_cons run_nil);
"#;

    #[test]
    fn every_declaration_kind_is_named_and_ordered() {
        let g = Graph::build(SPEC, "");
        let names: Vec<&str> = g.decls.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "Prop", "St", "L", "st0", "tl_nil", "tl_cons", "run", "run_nil",
                "run_cons", "trace", "t_trace"
            ]
        );
        let kinds: Vec<Kind> = g.decls.iter().map(|d| d.kind).collect();
        assert_eq!(kinds[0], Kind::Sort);
        assert_eq!(kinds[3], Kind::Term);
        assert_eq!(kinds[7], Kind::Axiom);
        assert_eq!(kinds[9], Kind::Def);
        assert_eq!(kinds[10], Kind::Theorem);
        // `delimiter …` declares nothing.
        assert!(!g.contains("delimiter"));
        assert_eq!(g.counts_by_kind()["sort"], 3);
        assert_eq!(g.counts_by_kind()["theorem"], 1);
    }

    #[test]
    fn references_are_edges_and_binders_are_not() {
        let g = Graph::build(SPEC, PROOF);
        let d = g.decls.iter().find(|d| d.name == "run_cons").unwrap();
        // `(h: St) (t: L) (s: St)` — h/t/s are binders, not edges; St/L and
        // the body's `run`/`tl_cons` are.
        assert_eq!(
            d.deps,
            vec![
                "St".to_string(),
                "L".to_string(),
                "run".to_string(),
                "tl_cons".to_string()
            ]
        );
        assert!(!d.deps.iter().any(|x| x == "s" || x == "h" || x == "t"));

        let t = g.decls.iter().find(|d| d.name == "t_trace").unwrap();
        assert_eq!(
            t.deps,
            vec![
                "run".to_string(),
                "st0".to_string(),
                "trace".to_string(),
                "run_cons".to_string(),
                "run_nil".to_string()
            ],
            "the proof tail contributes edges too: {:?}",
            t.deps
        );
    }

    #[test]
    fn a_clean_submission_has_no_faults_and_the_top_theorem_is_unused() {
        let g = Graph::build(SPEC, PROOF);
        assert_eq!(g.fault_count(), 0, "{:?}", g.faults);
        assert_eq!(g.unused, vec!["t_trace".to_string()]);
        let t = g.decls.iter().find(|d| d.name == "t_trace").unwrap();
        assert!(t.proved());
        assert_eq!(t.source(), "spec+proof");
        assert!(t.line() > 0);
        assert!(!g.decls.iter().find(|d| d.name == "run").unwrap().proved());
    }

    #[test]
    fn a_forward_reference_names_the_line_it_waits_for() {
        let spec = "\
provable sort Prop;\ndef a: Prop = $ b $;\nterm b: Prop;\n";
        let g = Graph::build(spec, "");
        let f: Vec<&Fault> = g.faults.iter().filter(|f| f.kind.is_fault()).collect();
        assert_eq!(f.len(), 1, "{:?}", g.faults);
        assert_eq!(f[0].kind, FaultKind::Forward);
        assert_eq!(f[0].node, "a");
        assert!(f[0].detail.contains("b"), "{}", f[0].detail);
        assert_eq!(g.faulty_names(), ["a".to_string()].into_iter().collect());
    }

    #[test]
    fn an_unknown_name_is_unresolved_alone_and_external_when_imports_exist() {
        let spec = "provable sort Prop;\ndef a: Prop = $ mystery $;\n";
        let g = Graph::build(spec, "");
        assert_eq!(g.faults[0].kind, FaultKind::Unresolved);
        assert!(g.faults[0].detail.contains("mystery"));
        assert!(g.faults[0].detail.contains("neither file declares"));

        let with_import = "import \"metamath/set.mm0\";\nprovable sort Prop;\ndef a: Prop = $ mystery $;\n";
        let g = Graph::build(with_import, "");
        assert_eq!(g.imports, vec!["metamath/set.mm0".to_string()]);
        assert_eq!(g.faults[0].kind, FaultKind::External);
        assert!(!g.faults[0].kind.is_fault());
        assert_eq!(g.fault_count(), 0, "an import is context, not a fault");
    }

    #[test]
    fn a_cycle_is_reported_once_and_still_lays_out() {
        let spec = "\
provable sort Prop;\ndef x: Prop = $ y $;\ndef y: Prop = $ x $;\n";
        let g = Graph::build(spec, "");
        let cycles: Vec<&Fault> = g
            .faults
            .iter()
            .filter(|f| f.kind == FaultKind::Cycle)
            .collect();
        assert_eq!(cycles.len(), 1, "{:?}", g.faults);
        assert!(cycles[0].detail.contains('x') && cycles[0].detail.contains('y'));
        // Layering must terminate: every node gets a layer, no panic.
        assert_eq!(g.layers().len(), 3);
        let html = g.html("cycle", "", &HashSet::new());
        assert!(html.contains("<svg"), "a cycle still renders");
    }

    #[test]
    fn comments_and_math_strings_do_not_leak_into_statements() {
        let spec = "\
-- a comment with a ; and a $ inside\n\
provable sort Prop;  -- trailing, also has ;\n\
term a: Prop; -- outer -- inner\n";
        let g = Graph::build(spec, "");
        let names: Vec<&str> = g.decls.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["Prop", "a"], "{names:?}");
        assert!(!g.decls.iter().any(|d| d.name.contains("comment")));
    }

    #[test]
    fn the_page_is_one_self_contained_file() {
        let g = Graph::build(SPEC, PROOF);
        let html = g.html("t_trace — graph", "<p>accepted</p>", &HashSet::new());
        assert!(html.contains("<svg"));
        assert!(html.contains("viewBox"));
        assert!(html.contains("proved by a proof term"));
        assert!(html.contains("accepted"));
        assert!(html.contains("no dependency faults"));
        // No script, no external fetch — the whole point of §the header.
        assert!(!html.contains("<script"));
        assert!(!html.contains("src="));
        assert!(!html.contains("unpkg"));
        // Nothing unescaped: a hostile name must not become markup.
        let hostile = Graph::build("provable sort Prop;\ndef <b>: Prop = $ a $;", "");
        let hh = hostile.html("x\"<t>", "", &HashSet::new());
        assert!(hostile.contains("<b>"), "the parser takes it literally");
        assert!(
            !hh.contains("<b>:") && !hh.contains("\"<b>"),
            "but the page never emits it raw: {hh}"
        );
        assert!(hh.contains("&lt;b&gt;"), "escaped instead: {hh}");
        assert!(!hh.contains("<t>"), "title escaped too: {hh}");
    }

    #[test]
    fn a_rejection_log_can_mark_the_node_it_named() {
        let g = Graph::build(SPEC, PROOF);
        let mut failed = HashSet::new();
        failed.insert("run_cons".to_string());
        let html = g.html("rejected", "", &failed);
        assert!(html.contains("#a4262c"), "the failed node paints red");
        assert!(html.contains("run_cons"));
    }

    #[test]
    fn past_the_rendering_ceiling_the_page_says_so_and_json_still_answers() {
        let mut src = String::from("provable sort Prop;\n");
        for i in 0..(MAX_RENDER_DECLS + 20) {
            src.push_str(&format!("term t{i}: Prop;\n"));
        }
        let g = Graph::build(&src, "");
        assert!(g.len() > MAX_RENDER_DECLS);
        let html = g.html("big", "", &HashSet::new());
        assert!(!html.contains("<svg"));
        assert!(html.contains("is more than this page draws"));
        assert_eq!(g.to_json()["counts"]["declarations"], g.len());
    }

    #[test]
    fn json_carries_everything_a_client_needs() {
        let g = Graph::build(SPEC, PROOF);
        let v = g.to_json();
        assert_eq!(v["counts"]["declarations"], 11);
        assert_eq!(v["counts"]["references"], g.reference_count());
        assert_eq!(v["counts"]["by_kind"]["axiom"], 2);
        let t = v["declarations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["name"] == "t_trace")
            .unwrap();
        assert_eq!(t["kind"], "theorem");
        assert_eq!(t["proved"], true);
        assert_eq!(t["source"], "spec+proof");
        assert!(t["proof"].as_str().unwrap().contains("run_cons"));
        assert!(t["deps"].as_array().unwrap().len() >= 5);
        assert_eq!(v["faults"].as_array().unwrap().len(), 0);
        assert_eq!(v["unused"][0], "t_trace");
    }

    #[test]
    fn statement_modifiers_do_not_hide_the_declaration() {
        // How a real spec spells its sorts. Skipping the statement because
        // `strict` was not a keyword it knew declared nothing, and `wff` —
        // the sort most of the file is built from — then reads as unresolved
        // in every single statement.
        let src = "strict provable sort wff;\nstrict sort set;\nprovable sort Prop;\
                   \nterm a: wff;\n";
        let g = Graph::build(src, "");
        let names: Vec<&str> = g.decls.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["wff", "set", "Prop", "a"]);
        assert_eq!(g.fault_count(), 0, "{:?}", g.faults);
        assert_eq!(g.counts_by_kind()["sort"], 3);
    }

    #[test]
    fn forward_is_judged_within_one_file_not_across_the_merge() {
        // The merged list is the `.mm1` order with the `.mm0`-only names
        // appended, so a definition the proof stream legitimately uses from
        // the spec must not be called "declared later" — the two files are
        // separate streams and `mm0-rs` reads them separately.
        let mm0 = "provable sort Prop;\nterm base: Prop;\ndef helper: Prop = $ base $;\n";
        let mm1 = "provable sort Prop;\nterm base: Prop;\
                   \npub theorem uses: $ helper $ = '(helper);\n";
        let g = Graph::build(mm0, mm1);
        let helper = g.decls.iter().position(|d| d.name == "helper").unwrap();
        let uses = g.decls.iter().position(|d| d.name == "uses").unwrap();
        assert!(helper > uses, "spec-only names are appended by the merge");
        assert_eq!(
            g.fault_count(),
            0,
            "using a spec definition from a proof is not a fault: {:?}",
            g.faults
        );

        // …while a genuine forward reference inside the proof file still is.
        let fwd_mm1 = "provable sort Prop;\npub theorem a: $ b $ = '(x);\
                       \npub theorem b: $ Prop $ = '(y);\n";
        let g = Graph::build("", fwd_mm1);
        assert!(
            g.faults
                .iter()
                .any(|f| f.kind == FaultKind::Forward && f.node == "a"),
            "{:?}",
            g.faults
        );
    }

    #[test]
    fn a_name_bound_somewhere_is_a_variable_not_a_missing_declaration() {
        // The `.mm1` restates a theorem with the binders the `.mm0` gave it
        // left out, so its statement reads as free names — `a` and `b` here.
        // They are bound in another declaration, so they are variables of the
        // development, not references to something that does not exist.
        let src = "strict provable sort wff;\
                   \naxiom ax (a b: wff): $ a -> b $;\
                   \npub theorem a1i (h: $ b $): $ a -> b $;\n";
        let g = Graph::build("", src);
        assert_eq!(
            g.fault_count(),
            0,
            "a and b are bound somewhere: {:?}",
            g.faults
        );
        assert_eq!(
            g.variables,
            ["a".to_string(), "b".to_string(), "h".to_string()]
                .into_iter()
                .collect(),
            "{:?}",
            g.variables
        );
        assert_eq!(g.to_json()["counts"]["variables"], 3);

        // A name nothing binds *and* nothing declares is still a fault.
        let g = Graph::build("", "strict provable sort wff;\npub theorem t: $ ghost $;\n");
        assert_eq!(g.fault_count(), 1, "{:?}", g.faults);
        assert!(g.faults[0].detail.contains("ghost"));
    }

    #[test]
    fn an_accepted_submission_reports_undeclared_names_without_calling_them_defects() {
        let g = Graph::build("", "strict provable sort wff;\npub theorem t: $ ghost $;\n");
        assert_eq!(g.fault_count(), 1, "unaccepted, it is a hint: {:?}", g.faults);

        let g = g.accepted();
        assert_eq!(
            g.fault_count(),
            0,
            "the checkers already judged it: {:?}",
            g.faults
        );
        assert_eq!(g.faults.len(), 1, "the name itself is still reported");
        assert_eq!(g.to_json()["counts"]["accepted"], true);

        let html = g.html("t", "", &HashSet::new());
        assert!(html.contains("class=\"info\""), "drawn as information");
        assert!(html.contains("ghost"), "and still listed: {html}");
        assert!(
            html.contains("Both checkers accepted"),
            "the page says why: {html}"
        );
    }

    /// The DAG rules are not a second opinion invented here: the same three
    /// checks already exist in this workspace as ProofFlow's `check_DAG`,
    /// ported and parity-locked against the Python oracle
    /// (`logos::formalize::graph`, `docs/PROOFFLOW_STUDY.md`). Where the two
    /// overlap they must say the same thing — this test is what stops the
    /// report here from drifting away from the port.
    ///
    /// Where they differ, the difference is deliberate and one-sided:
    /// ProofFlow raises on the *first* problem and treats an orphan as one;
    /// this service reports every structural fault at once and lets an unused
    /// declaration stand, because an mm0 development may end in a theorem
    /// nothing else needs.
    fn oracle(g: &Graph) -> Result<logos::formalize::graph::Validated, logos::formalize::graph::GraphError>
    {
        use logos::formalize::graph::{GraphNode, check_dag};
        let nodes = g
            .decls
            .iter()
            .map(|d| {
                GraphNode::new(
                    d.name.clone(),
                    String::new(),
                    d.statement.clone(),
                    d.deps.clone(),
                )
            })
            .collect();
        check_dag(nodes)
    }

    #[test]
    fn the_dag_rules_agree_with_the_ported_proof_flow_check_dag() {
        use logos::formalize::graph::GraphError;

        // A graph the oracle accepts: no structural fault is reported here.
        let ok = Graph::build("provable sort Prop;\nterm ts_end: Prop;\n", "");
        assert!(oracle(&ok).is_ok(), "{:?}", oracle(&ok));
        assert_eq!(ok.fault_count(), 0, "{:?}", ok.faults);

        // A cycle: the oracle raises `Cycle`, this module reports a cycle.
        let cyc = Graph::build(
            "provable sort Prop;\ndef x: Prop = $ y $;\ndef y: Prop = $ x $;\n",
            "",
        );
        assert!(
            matches!(oracle(&cyc), Err(GraphError::Cycle)),
            "{:?}",
            oracle(&cyc)
        );
        assert!(
            cyc.faults.iter().any(|f| f.kind == FaultKind::Cycle),
            "{:?}",
            cyc.faults
        );

        // A forward reference: same node, same dependency, both sides.
        let fwd = Graph::build(
            "provable sort Prop;\ndef a: Prop = $ b $;\nterm b: Prop;\
             \ndef ts_end: Prop = $ a $;\n",
            "",
        );
        match oracle(&fwd) {
            Err(GraphError::ForwardReference { id, dependency }) => assert!(
                fwd.faults.iter().any(|f| {
                    f.kind == FaultKind::Forward
                        && f.node == id
                        && f.detail.contains(&format!("`{dependency}`"))
                }),
                "oracle: {id} → {dependency}; here: {:?}",
                fwd.faults
            ),
            other => panic!("expected a forward reference from the oracle, got {other:?}"),
        }

        // The one-sided difference: an unused declaration is ProofFlow's
        // orphan error and this service's information, never a fault.
        let unused = Graph::build(SPEC, PROOF);
        assert!(matches!(
            oracle(&unused),
            Err(GraphError::OrphanNode { .. })
        ));
        assert_eq!(unused.fault_count(), 0, "{:?}", unused.faults);
        assert_eq!(unused.unused, vec!["t_trace".to_string()]);
    }
}
