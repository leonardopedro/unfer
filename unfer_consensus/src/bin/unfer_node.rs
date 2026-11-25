//! D11: one consensus replica, as its own process.
//!
//! ```text
//! unfer_node serve  --config node.json  [--ready ready.json]
//! unfer_node submit --config client.json --to 127.0.0.1:17001 --id 7 --tx tx.json
//! unfer_node digest --config node.json
//! unfer_node job    --config node.json --job settle-auction
//! ```
//!
//! `serve` owns exactly one state directory (locked for its lifetime), one
//! ledger and one TLS listener; it publishes the address it bound in the ready
//! file and stops on SIGINT/SIGTERM. `submit` is a client: it proposes one
//! complete value to a member and prints the outcome as JSON. `digest` reads a
//! node's durable committed log without taking its lock, which is how a test
//! compares what two processes applied.
//!
//! Configuration is JSON, not TOML: certificates and keys are hex DER inside the
//! file, so a fixture is self-contained and two processes never resolve a shared
//! path. Validation lives in `unfer_consensus::net`, not here — the binary only
//! parses arguments and prints results.

#[cfg(feature = "network")]
fn main() -> std::process::ExitCode {
    use std::process::ExitCode;

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("unfer_node: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "network"))]
fn main() -> std::process::ExitCode {
    eprintln!("unfer_node: built without the `network` feature; a node has no transport");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "network")]
fn run() -> Result<(), String> {
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or_else(|| {
        "usage: unfer_node <serve|submit|digest> --config <path> ...".to_owned()
    })?;
    let mut config: Option<PathBuf> = None;
    let mut ready: Option<PathBuf> = None;
    let mut to: Option<String> = None;
    let mut tx: Option<PathBuf> = None;
    let mut id: Option<u64> = None;
    let mut job: Option<String> = None;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--config" => config = Some(PathBuf::from(value)),
            "--ready" => ready = Some(PathBuf::from(value)),
            "--to" => to = Some(value),
            "--tx" => tx = Some(PathBuf::from(value)),
            "--job" => job = Some(value),
            "--id" => {
                id = Some(
                    value
                        .parse()
                        .map_err(|error| format!("--id `{value}`: {error}"))?,
                )
            }
            other => return Err(format!("unknown flag `{other}`")),
        }
    }
    let config = config.ok_or_else(|| "missing --config".to_owned())?;

    match command.as_str() {
        "serve" => serve(&config, ready.as_deref()),
        "submit" => submit(&config, &to.ok_or_else(|| "missing --to".to_owned())?, &tx
            .ok_or_else(|| "missing --tx".to_owned())?, id
            .ok_or_else(|| "missing --id".to_owned())?),
        "digest" => digest(&config),
        "job" => job_state(&config, &job.ok_or_else(|| "missing --job".to_owned())?),
        other => Err(format!(
            "unknown command `{other}`; expected serve, submit, digest or job"
        )),
    }
}

#[cfg(feature = "network")]
fn serve(config: &std::path::Path, ready: Option<&std::path::Path>) -> Result<(), String> {
    use unfer_consensus::net::{NodeConfig, start_node};

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| format!("could not start the node runtime: {error}"))?;
    runtime.block_on(async {
        let config = NodeConfig::load(config)?;
        let node = start_node(config).await?;
        let published = format!(
            "{{\"replica\":{},\"address\":\"{}\"}}",
            node.replica_id.get(),
            node.address
        );
        if let Some(path) = ready {
            std::fs::write(path, published.as_bytes())
                .map_err(|error| format!("could not write {}: {error}", path.display()))?;
        }
        println!("{published}");
        // A node runs until it is told to stop. The signal handler is the
        // process's only shutdown input, so a supervisor's SIGTERM is a graceful
        // stop and a SIGKILL is the abrupt one D13 injects.
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .map_err(|error| format!("could not install the SIGTERM handler: {error}"))?;
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
        node.shutdown().await;
        Ok(())
    })
}

#[cfg(feature = "network")]
fn submit(
    config: &std::path::Path,
    to: &str,
    tx_path: &std::path::Path,
    id: u64,
) -> Result<(), String> {
    use rust_quepaxa::network::SubmissionOutcome;
    use unfer_consensus::net::{SubmitConfig, encode_ledger_value, submit_client};
    use unfer_protocol::ConsensusTransaction;

    let address = to
        .parse()
        .map_err(|error| format!("--to `{to}` is not a socket address: {error}"))?;
    let config = SubmitConfig::load(config)?;
    let text = std::fs::read_to_string(tx_path)
        .map_err(|error| format!("could not read {}: {error}", tx_path.display()))?;
    let transaction: ConsensusTransaction = serde_json::from_str(&text)
        .map_err(|error| format!("{} is not a consensus transaction: {error}", tx_path.display()))?;
    // The client frames the value itself: the reservation id travels inside the
    // payload, so the replica that applies it never needs a local map (D10).
    let value = encode_ledger_value(id, &transaction)?;
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| format!("could not start the client runtime: {error}"))?;
    runtime.block_on(async {
        let client = submit_client(&config, address)?;
        match client.submit(id, vec![value]).await {
            Ok(SubmissionOutcome::Committed(decision)) => {
                println!("{{\"outcome\":\"committed\",\"slot\":{}}}", decision.slot.get());
            }
            Ok(SubmissionOutcome::Accepted) => println!("{{\"outcome\":\"accepted\"}}"),
            Ok(SubmissionOutcome::Duplicate(decision)) => match decision {
                Some(decision) => println!(
                    "{{\"outcome\":\"duplicate\",\"slot\":{}}}",
                    decision.slot.get()
                ),
                None => println!("{{\"outcome\":\"duplicate\"}}"),
            },
            Err(error) => return Err(format!("submission refused: {error}")),
        }
        Ok(())
    })
}

#[cfg(feature = "network")]
#[cfg(feature = "network")]
fn job_state(config: &std::path::Path, job: &str) -> Result<(), String> {
    use unfer_consensus::net::{NodeConfig, state_dir_job};

    let config = NodeConfig::load(config)?;
    println!("{}", state_dir_job(&config.state_dir, job)?);
    Ok(())
}

#[cfg(feature = "network")]
fn digest(config: &std::path::Path) -> Result<(), String> {
    use unfer_consensus::net::{NodeConfig, state_dir_digest};

    let config = NodeConfig::load(config)?;
    let digest = state_dir_digest(&config.state_dir)?;
    let (committed, _) = digest
        .split_once(':')
        .ok_or_else(|| format!("digest `{digest}` is not `count:hash`"))?;
    println!("{{\"committed\":{committed},\"digest\":\"{digest}\"}}");
    Ok(())
}
