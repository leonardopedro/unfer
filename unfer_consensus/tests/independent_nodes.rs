//! D11: three independent node processes, no shared state.
//!
//! Each process gets its own state directory, its own ledger and its own TLS
//! identity, and is started from a validated JSON config. The test drives them
//! only through their published interface: submit a value over TLS to one
//! member, then compare the digest of the committed log each node applied. A
//! node that read another node's storage could not be distinguished by the
//! digest alone — so the isolation is also asserted directly: each directory
//! holds only its own replica's files, and a second process pointed at a live
//! node's directory is refused.
//!
//! Steps (PLAN D11):
//! 1. startup is per process, from validated config;
//! 2. three processes, disjoint state directories, submissions through
//!    different nodes, identical digests;
//! 3. a node is killed, the survivors commit without it, and it catches up when
//!    restarted.

#![cfg(feature = "network")]

use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rcgen::{CertifiedKey, generate_simple_self_signed};
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_protocol::{ConsensusTransaction, IdentityOp, IdentityOpKind};

const NODE_BIN: &str = env!("CARGO_BIN_EXE_unfer_node");

struct Identity {
    cert_hex: String,
    key_hex: String,
}

fn identity() -> Identity {
    let CertifiedKey { cert, key_pair } = generate_simple_self_signed(["localhost".to_owned()])
        .expect("self-signed certificate");
    Identity {
        cert_hex: hex::encode(cert.der()),
        key_hex: hex::encode(key_pair.serialize_der()),
    }
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(90)
}

fn temp_root(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "unfer-d11-{name}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Three loopback ports reserved and released: the fixture hands each process a
/// concrete address, because members must agree on each other's address before
/// any of them binds.
fn free_ports() -> Vec<u16> {
    let listeners = (0..3)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect::<Vec<_>>();
    let ports = listeners
        .iter()
        .map(|listener| listener.local_addr().unwrap().port())
        .collect();
    drop(listeners);
    ports
}

fn deployment_json(deployment: &str, ports: &[u16], identities: &[Identity], client_id: &str) -> String {
    let members = (0..3)
        .map(|index| {
            format!(
                "{{\"replica\":{},\"address\":\"127.0.0.1:{}\",\"certificate_hex\":\"{}\"}}",
                index + 1,
                ports[index],
                identities[index].cert_hex
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"deployment\":\"{deployment}\",\"lane\":1,\"members\":[{members}],\
         \"clients\":[{{\"client_id_hex\":\"{client_id}\",\"certificate_hex\":\"{}\"}}]}}",
        identities[3].cert_hex
    )
}

#[allow(clippy::too_many_arguments)]
fn node_json(
    deployment: &str,
    replica: usize,
    state_dir: &Path,
    identities: &[Identity],
    ports: &[u16],
    client_id: &str,
) -> String {
    format!(
        "{{\"replica\":{replica},\"state_dir\":\"{}\",\"certificate_hex\":\"{}\",\
         \"key_hex\":\"{}\",\"rpc_timeout_ms\":30000,\"deployment\":{}}}",
        state_dir.display(),
        identities[replica - 1].cert_hex,
        identities[replica - 1].key_hex,
        deployment_json(deployment, ports, identities, client_id)
    )
}

fn client_json(
    deployment: &str,
    client_id: &str,
    identities: &[Identity],
    ports: &[u16],
) -> String {
    format!(
        "{{\"client_id_hex\":\"{client_id}\",\"certificate_hex\":\"{}\",\"key_hex\":\"{}\",\
         \"rpc_timeout_ms\":30000,\"deployment\":{}}}",
        identities[3].cert_hex,
        identities[3].key_hex,
        deployment_json(deployment, ports, identities, client_id)
    )
}

fn transaction(label: u64) -> ConsensusTransaction {
    let did = Keypair::generate();
    let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
        did: did.did(),
        op_kind: IdentityOpKind::Create,
        signing_key: did.public_key(),
        signature: [0u8; 64],
        seq: label,
        service_endpoint: None,
    });
    sign_transaction(&mut tx, &did);
    tx
}

struct Node {
    child: Child,
    ready: PathBuf,
}

impl Node {
    fn spawn(config: &Path, ready: &Path) -> Self {
        let child = Command::new(NODE_BIN)
            .arg("serve")
            .arg("--config")
            .arg(config)
            .arg("--ready")
            .arg(ready)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn unfer_node serve");
        Self {
            child,
            ready: ready.to_path_buf(),
        }
    }

    /// Block until the process publishes the address it bound.
    fn wait_ready(&self) -> SocketAddr {
        let until = deadline();
        while Instant::now() < until {
            if let Ok(text) = std::fs::read_to_string(&self.ready) {
                if let Some(address) = field(&text, "address") {
                    return address.parse().expect("ready address parses");
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("node never published {}", self.ready.display());
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn field(json: &str, name: &str) -> Option<String> {
    let key = format!("\"{name}\":");
    let start = json.find(&key)? + key.len();
    let rest = json[start..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        stripped.find('"').map(|end| stripped[..end].to_owned())
    } else {
        let end = rest
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '-')
            .unwrap_or(rest.len());
        Some(rest[..end].to_owned())
    }
}

fn run(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(NODE_BIN)
        .args(args)
        .output()
        .expect("run unfer_node");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// `count:hash`, the form `unfer_node digest` prints.
fn digest(config: &Path) -> String {
    let (ok, stdout, stderr) = run(&["digest", "--config", config.to_str().unwrap()]);
    assert!(ok, "digest failed: {stderr}");
    field(&stdout, "digest").expect("digest field")
}

fn wait_digest(config: &Path, want_committed: u64) -> String {
    let until = deadline();
    while Instant::now() < until {
        let current = digest(config);
        if current.starts_with(&format!("{want_committed}:")) {
            return current;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!(
        "{} never reached {want_committed} committed; last {}",
        config.display(),
        digest(config)
    );
}

#[test]
fn three_processes_commit_identically_and_one_catches_up_after_a_kill() {
    let root = temp_root("independent");
    let ports = free_ports();
    // One non-zero client id, and one different deployment for the refusal
    // case: a command signed for another deployment must not commit here.
    let client_id = "07".repeat(16);
    let identities = (0..4).map(|_| identity()).collect::<Vec<_>>();

    let dirs = (1..=3)
        .map(|replica| {
            let dir = root.join(format!("node-{replica}"));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        })
        .collect::<Vec<_>>();
    let configs = (1..=3)
        .map(|replica| {
            let path = root.join(format!("node-{replica}.json"));
            std::fs::write(
                &path,
                node_json(
                    "1",
                    replica,
                    &dirs[replica - 1],
                    &identities,
                    &ports,
                    &client_id,
                ),
            )
            .unwrap();
            path
        })
        .collect::<Vec<_>>();
    let client = root.join("client.json");
    std::fs::write(
        &client,
        client_json("1", &client_id, &identities, &ports),
    )
    .unwrap();

    // ---- Step 1: three processes, three disjoint state directories.
    let mut nodes = (0..3)
        .map(|index| {
            let ready = root.join(format!("ready-{}.json", index + 1));
            let node = Node::spawn(&configs[index], &ready);
            let address = node.wait_ready();
            assert_eq!(address.port(), ports[index], "each node binds its own port");
            node
        })
        .collect::<Vec<_>>();

    // ---- Step 2: submit through different nodes; everyone applies.
    let mint = root.join("mint.json");
    std::fs::write(&mint, serde_json::to_vec(&transaction(1)).unwrap()).unwrap();
    let transfer = root.join("transfer.json");
    std::fs::write(&transfer, serde_json::to_vec(&transaction(2)).unwrap()).unwrap();

    for (index, tx, id) in [(0usize, &mint, "1"), (1usize, &transfer, "2")] {
        let (ok, stdout, stderr) = run(&[
            "submit",
            "--config",
            client.to_str().unwrap(),
            "--to",
            &format!("127.0.0.1:{}", ports[index]),
            "--id",
            id,
            "--tx",
            tx.to_str().unwrap(),
        ]);
        assert!(ok, "submit through node {} failed: {stderr}", index + 1);
        assert!(
            stdout.contains("\"outcome\":\"committed\"") || stdout.contains("\"outcome\":\"accepted\""),
            "unexpected submit outcome: {stdout}"
        );
    }

    let first = wait_digest(&configs[0], 2);
    assert_eq!(wait_digest(&configs[1], 2), first, "nodes 1 and 2 agree");
    assert_eq!(wait_digest(&configs[2], 2), first, "node 3 agrees");

    // ---- Isolation: each process wrote only its own directory.
    for (index, dir) in dirs.iter().enumerate() {
        let own = dir.join(format!("node-{}", index + 1));
        assert!(own.is_dir(), "{} owns {}", dir.display(), own.display());
        assert!(dir.join("ledger.json").is_file(), "each node has its own ledger");
        for other in 1..=3 {
            if other != index + 1 {
                assert!(
                    !dir.join(format!("node-{other}")).exists(),
                    "{} must not hold replica {other}'s state",
                    dir.display()
                );
            }
        }
    }
    // A second process pointed at a live node's directory is refused, so no two
    // nodes can share one application store even by accident.
    let (ok, _, stderr) = run(&[
        "serve",
        "--config",
        configs[0].to_str().unwrap(),
        "--ready",
        root.join("second-ready.json").to_str().unwrap(),
    ]);
    assert!(!ok, "a second process on a live state dir must fail");
    assert!(
        stderr.contains("locked by another node process"),
        "unexpected refusal: {stderr}"
    );

    // ---- A command for another deployment is refused by name.
    let other = root.join("other-deployment.json");
    std::fs::write(
        &other,
        client_json("2", &client_id, &identities, &ports),
    )
    .unwrap();
    let (ok, _, stderr) = run(&[
        "submit",
        "--config",
        other.to_str().unwrap(),
        "--to",
        &format!("127.0.0.1:{}", ports[0]),
        "--id",
        "9",
        "--tx",
        mint.to_str().unwrap(),
    ]);
    assert!(!ok, "a command for deployment 2 must not commit into deployment 1");
    assert!(
        !stderr.is_empty(),
        "the refusal must say why it was refused"
    );

    // ---- Step 3a: stop node 3, commit without it, survivors keep the payload.
    nodes[2].kill();
    let third = root.join("third.json");
    std::fs::write(&third, serde_json::to_vec(&transaction(3)).unwrap()).unwrap();
    let (ok, _, stderr) = run(&[
        "submit",
        "--config",
        client.to_str().unwrap(),
        "--to",
        &format!("127.0.0.1:{}", ports[1]),
        "--id",
        "3",
        "--tx",
        third.to_str().unwrap(),
    ]);
    assert!(ok, "submit with node 3 down failed: {stderr}");

    let survivors = wait_digest(&configs[0], 3);
    assert_eq!(wait_digest(&configs[1], 3), survivors);
    assert!(
        digest(&configs[2]).starts_with("2:"),
        "the stopped process must not have applied the third commit"
    );

    // ---- Step 3b: restart node 3 on the same directory, address and identity.
    let ready = root.join("ready-3-again.json");
    let restarted = Node::spawn(&configs[2], &ready);
    let address = restarted.wait_ready();
    assert_eq!(address.port(), ports[2], "a restarted node rebinds its address");
    nodes[2] = restarted;

    // The rejoining replica is behind by one commit. Decide once more and check
    // that it converges on the survivors' full log — not just on the new slot.
    let fourth = root.join("fourth.json");
    std::fs::write(&fourth, serde_json::to_vec(&transaction(4)).unwrap()).unwrap();
    let (ok, _, stderr) = run(&[
        "submit",
        "--config",
        client.to_str().unwrap(),
        "--to",
        &format!("127.0.0.1:{}", ports[0]),
        "--id",
        "4",
        "--tx",
        fourth.to_str().unwrap(),
    ]);
    assert!(ok, "submit after the restart failed: {stderr}");
    let caught_up = wait_digest(&configs[0], 4);
    assert_eq!(wait_digest(&configs[1], 4), caught_up);
    eprintln!("survivors: {caught_up}; rejoined: {}", digest(&configs[2]));
    assert_eq!(
        wait_digest(&configs[2], 4),
        caught_up,
        "the restarted node must converge on the same log, not just the new slot"
    );

    // Leave the fixture's directory intact for inspection on failure only.
    if std::env::var_os("UNFER_D11_KEEP").is_none() {
        for node in nodes.iter_mut() {
            node.kill();
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
