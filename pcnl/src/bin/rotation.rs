//! Thin CLI for the R12 signed rotation session.
//!
//! ```text
//! cargo run --release -p pcnl --bin rotation -- init --state /tmp/r12.json
//! cargo run --release -p pcnl --bin rotation -- update --state /tmp/r12.json \
//!     --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --per-class 10
//! cargo run --release -p pcnl --bin rotation -- predict --state /tmp/r12.json \
//!     --fixture docs/fixtures/ultrabert_agnews_embeddings.safetensors --row 0
//! ```
//!
//! The encoder is external: `--embedding` accepts an already prepared vector.

use pcnl::rotation_head::{
    HeadSpec, RotationHead, RotationSession, SessionConfig, TextInput,
};
use pcnl::text_head::{EmbeddingTable, Split};
use std::path::{Path, PathBuf};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn number<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    flag(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn default_fixture() -> PathBuf {
    PathBuf::from("docs/fixtures/ultrabert_agnews_embeddings.safetensors")
}

fn load_fixture(args: &[String]) -> Result<EmbeddingTable, String> {
    let path = flag(args, "--fixture").map(PathBuf::from).unwrap_or_else(default_fixture);
    EmbeddingTable::load(&path).map_err(|e| e.to_string())
}

fn init(args: &[String]) -> Result<(), String> {
    let state = flag(args, "--state").ok_or("init needs --state PATH")?;
    let hidden = flag(args, "--hidden")
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![32, 16]);
    let dim = number(args, "--dim", 768usize);
    let classes = number(args, "--classes", 4usize);
    let head = RotationHead::with_name(
        &flag(args, "--name").unwrap_or_else(|| "rotation-head".into()),
        dim,
        classes,
        &hidden,
    )
    .map_err(|e| e.to_string())?;
    let config = SessionConfig {
        sigma: number(args, "--sigma", 0.5),
        lambda: number(args, "--lambda", 1.0),
        beta: number(args, "--beta", 0.2),
        burn: number(args, "--burn", 400usize),
        steps: number(args, "--steps", 1_200usize),
        thin: number(args, "--thin", 10usize),
        chains: number(args, "--chains", 4usize),
        draws: number(args, "--draws", 40usize),
        seed: number(args, "--seed", 20261118u64),
    };
    let session = RotationSession::new(head, config).map_err(|e| e.to_string())?;
    session.save(Path::new(&state)).map_err(|e| e.to_string())?;
    println!("initialized {state} dim={dim} classes={classes} hidden={hidden:?}");
    Ok(())
}

fn update(args: &[String]) -> Result<(), String> {
    let state = flag(args, "--state").ok_or("update needs --state PATH")?;
    let mut session =
        RotationSession::load(Path::new(&state)).map_err(|e| e.to_string())?;
    let table = load_fixture(args)?;
    let per_class = number(args, "--per-class", 8usize);
    let dim = table.dim();
    let classes = table.classes();
    if dim != session.head().dim() || classes != session.head().classes() {
        return Err(format!(
            "fixture has dim {dim}/classes {classes}, session has {}/{}",
            session.head().dim(),
            session.head().classes()
        ));
    }
    let x = table.x(Split::Train);
    let y = table.y(Split::Train);
    let mut taken = vec![0usize; classes];
    let mut added = 0usize;
    for (row, &class) in y.iter().enumerate() {
        if taken[class as usize] >= per_class {
            continue;
        }
        let step = x[row * dim..(row + 1) * dim].to_vec();
        let id = format!("fixture:train:{row}");
        if session
            .append_event(id, vec![step], class)
            .map_err(|e| e.to_string())?
        {
            added += 1;
        }
        taken[class as usize] += 1;
    }
    let report = session.update().map_err(|e| e.to_string())?;
    session.save(Path::new(&state)).map_err(|e| e.to_string())?;
    println!(
        "updated {state} checkpoint={} added={added} proposals={} accepted={} retained={}",
        report.checkpoint, report.proposals, report.accepted, report.retained_draws
    );
    Ok(())
}

fn predict(args: &[String]) -> Result<(), String> {
    let state = flag(args, "--state").ok_or("predict needs --state PATH")?;
    let mut session =
        RotationSession::load(Path::new(&state)).map_err(|e| e.to_string())?;
    let (text, embedding) = if let Some(raw) = flag(args, "--embedding") {
        let values = raw
            .split(',')
            .map(|v| v.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("--embedding: {e}"))?;
        (
            flag(args, "--text").unwrap_or_else(|| "embedding".into()),
            values,
        )
    } else {
        let table = load_fixture(args)?;
        let split = match flag(args, "--split").as_deref() {
            Some("train") => Split::Train,
            _ => Split::Test,
        };
        let row = number(args, "--row", 0usize);
        let x = table.x(split);
        let n = table.y(split).len();
        if row >= n {
            return Err(format!("--row {row} outside 0..{n}"));
        }
        let dim = table.dim();
        (
            format!("fixture:{split:?}:{row}"),
            x[row * dim..(row + 1) * dim].to_vec(),
        )
    };
    let input = TextInput::new(&text, &embedding, session.head().dim())
        .map_err(|e| e.to_string())?;
    let prediction = session.predict(&input).map_err(|e| e.to_string())?;
    println!(
        "checkpoint={} draw={} basis={:?}",
        session.checkpoint(),
        prediction.states.len(),
        prediction.states[0].basis().labels()
    );
    println!("p={:?}", prediction.probabilities);
    for (i, state) in prediction.states.iter().enumerate() {
        println!("psi[{i}]={:?}", state.amplitudes());
    }
    Ok(())
}

fn show(args: &[String]) -> Result<(), String> {
    let state = flag(args, "--state").ok_or("show needs --state PATH")?;
    let session = RotationSession::load(Path::new(&state)).map_err(|e| e.to_string())?;
    let spec = HeadSpec::from_head(session.head());
    println!("head={spec:?}");
    println!("config={:?}", session.config());
    println!(
        "checkpoint={} updates={} retained={}",
        session.checkpoint(),
        session.updates(),
        session.retained_draws().len()
    );
    for event in session.events() {
        println!(
            "event={} steps={} observed={}",
            event.id,
            event.steps.len(),
            event.observed
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("");
    let rest = if args.is_empty() { &args[..] } else { &args[1..] };
    let result = match command {
        "init" => init(rest),
        "update" => update(rest),
        "predict" => predict(rest),
        "show" => show(rest),
        _ => {
            eprintln!(
                "usage: rotation <init|update|predict|show> [flags]\n\
                 state flags: --state PATH\n\
                 init: --dim --classes --hidden --sigma --lambda --beta --burn --steps --thin --chains --draws --seed\n\
                 update: --fixture PATH --per-class N\n\
                 predict: --fixture PATH [--split train|test] [--row N] | --embedding v1,v2,... [--text TEXT]"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("rotation: {e}");
        std::process::exit(1);
    }
}
