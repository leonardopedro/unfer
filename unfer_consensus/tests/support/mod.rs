//! Fixtures shared by the process-level distributed tests (D11 and D13).
//!
//! Everything here drives the built `unfer_node` binary the way an operator
//! would: JSON configs on disk, a TLS endpoint per replica, `serve`/`submit`/
//! `digest` as the only interface. Nothing reaches into a node's memory, which is
//! what makes a crash test here a statement about the process and its files.

#![allow(dead_code)]

use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rcgen::{CertifiedKey, generate_simple_self_signed};
use unfer_consensus::signing::{Keypair, sign_transaction};
use unfer_protocol::{ConsensusTransaction, IdentityOp, IdentityOpKind};

pub const NODE_BIN: &str = env!("CARGO_BIN_EXE_unfer_node");

pub struct Identity {
    pub cert_hex: String,
    pub key_hex: String,
}

pub fn identity() -> Identity {
    let CertifiedKey { cert, key_pair } =
        generate_simple_self_signed(["localhost".to_owned()]).expect("self-signed certificate");
    Identity {
        cert_hex: hex::encode(cert.der()),
        key_hex: hex::encode(key_pair.serialize_der()),
    }
}

pub fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(90)
}

/// A temporary root that is unique per test process *and* per call, so a test
/// that runs several fixtures in one process cannot collide with itself.
pub fn temp_root(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("unfer-{name}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Three loopback ports reserved and released: the fixture hands each process a
/// concrete address, because members must agree on each other's address before
/// any of them binds.
pub fn free_ports() -> Vec<u16> {
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

pub fn deployment_json(
    deployment: &str,
    ports: &[u16],
    identities: &[Identity],
    client_id: &str,
) -> String {
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
pub fn node_json(
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

pub fn client_json(
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

/// A signed identity create. Deterministic in `label` only up to the keypair,
/// which is fresh each call: two calls are two distinct ops, never a replay.
pub fn transaction(label: u64) -> ConsensusTransaction {
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

/// A *failing* identity op: a create whose DID does not match its key, so every
/// node refuses it in its own slot. Used where the matrix needs an entry the
/// domain rejects, to show a refusal is durable rather than retried.
pub fn refused_transaction(label: u64) -> ConsensusTransaction {
    let signer = Keypair::generate();
    let other = Keypair::generate();
    let mut tx = ConsensusTransaction::IdentityOp(IdentityOp {
        did: other.did(),
        op_kind: IdentityOpKind::Create,
        signing_key: signer.public_key(),
        signature: [0u8; 64],
        seq: label,
        service_endpoint: None,
    });
    sign_transaction(&mut tx, &signer);
    tx
}

pub struct Node {
    pub child: Child,
    pub ready: PathBuf,
    pub config: PathBuf,
}

impl Node {
    pub fn spawn(config: &Path, ready: &Path) -> Self {
        // A ready file left by a previous process would satisfy `wait_ready`
        // before the new process has bound anything, so a restarted node would
        // look ready while it is still starting.
        let _ = std::fs::remove_file(ready);
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
            config: config.to_path_buf(),
        }
    }

    /// Block until the process publishes the address it bound.
    pub fn wait_ready(&self) -> SocketAddr {
        let until = deadline();
        while Instant::now() < until {
            if let Ok(text) = std::fs::read_to_string(&self.ready)
                && let Some(address) = field(&text, "address")
            {
                return address.parse().expect("ready address parses");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("node never published {}", self.ready.display());
    }

    /// SIGKILL and reap: the process gets no chance to flush, finalize or
    /// acknowledge. This is a process-crash measurement, not a power-loss one.
    pub fn kill(&mut self) {
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

pub fn field(json: &str, name: &str) -> Option<String> {
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

pub fn run(args: &[&str]) -> (bool, String, String) {
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

/// Submit `tx` to one address under a stable request id.
pub fn submit(client: &Path, to: &str, id: u64, tx: &Path) -> (bool, String, String) {
    run(&[
        "submit",
        "--config",
        client.to_str().unwrap(),
        "--to",
        to,
        "--id",
        &id.to_string(),
        "--tx",
        tx.to_str().unwrap(),
    ])
}

/// The committed count a `digest` line reports.
pub fn committed(digest: &str) -> u64 {
    digest
        .split(':')
        .next()
        .and_then(|count| count.parse().ok())
        .unwrap_or(0)
}

/// The committed job state, read from a node's durable ledger without taking
/// its lock (`unfer_node job`).
pub fn job_report(config: &Path, job: &str) -> String {
    let (ok, stdout, stderr) = run(&["job", "--config", config.to_str().unwrap(), "--job", job]);
    assert!(ok, "job report failed: {stderr}");
    stdout.trim().to_owned()
}

/// `count:hash`, the form `unfer_node digest` prints.
pub fn digest(config: &Path) -> String {
    let (ok, stdout, stderr) = run(&["digest", "--config", config.to_str().unwrap()]);
    assert!(ok, "digest failed: {stderr}");
    field(&stdout, "digest").expect("digest field")
}

pub fn wait_digest(config: &Path, want_committed: u64) -> String {
    let until = deadline();
    loop {
        let current = digest(config);
        if current.starts_with(&format!("{want_committed}:")) {
            return current;
        }
        if Instant::now() >= until {
            panic!(
                "{} never reached {want_committed} committed; last {current}",
                config.display()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Wait until at least `want_committed` slots are committed. Unlike
/// `wait_digest` this tolerates *more*: a crash-retry test must not fail because
/// a duplicate slipped in, it must observe it.
pub fn wait_at_least(config: &Path, want_committed: u64) -> String {
    let until = deadline();
    loop {
        let current = digest(config);
        if committed(&current) >= want_committed {
            return current;
        }
        if Instant::now() >= until {
            panic!(
                "{} never reached {want_committed} committed; last {current}",
                config.display()
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// A whole three-process fixture: configs, directories and the client file.
pub struct Deployment {
    pub root: PathBuf,
    pub ports: Vec<u16>,
    pub identities: Vec<Identity>,
    pub dirs: Vec<PathBuf>,
    pub configs: Vec<PathBuf>,
    pub client: PathBuf,
}

impl Deployment {
    pub fn new(name: &str) -> Self {
        let root = temp_root(name);
        let ports = free_ports();
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
        std::fs::write(&client, client_json("1", &client_id, &identities, &ports)).unwrap();
        Self {
            root,
            ports,
            identities,
            dirs,
            configs,
            client,
        }
    }

    pub fn address(&self, index: usize) -> String {
        format!("127.0.0.1:{}", self.ports[index])
    }

    pub fn spawn(&self, index: usize) -> Node {
        let ready = self.root.join(format!("ready-{index}.json"));
        let node = Node::spawn(&self.configs[index], &ready);
        let address = node.wait_ready();
        assert_eq!(
            address.port(),
            self.ports[index],
            "each node binds its own port"
        );
        node
    }

    pub fn spawn_all(&self) -> Vec<Node> {
        (0..3).map(|index| self.spawn(index)).collect()
    }

    /// Write one transaction and submit it to `index` under `id`.
    pub fn submit(
        &self,
        index: usize,
        id: u64,
        tx: &ConsensusTransaction,
    ) -> (bool, String, String) {
        let path = self.root.join(format!("tx-{id}.json"));
        std::fs::write(&path, serde_json::to_vec(tx).unwrap()).unwrap();
        submit(&self.client, &self.address(index), id, &path)
    }

    pub fn keep(&self) -> bool {
        std::env::var_os("UNFER_D13_KEEP").is_some()
    }

    pub fn clean(&self, nodes: &mut [Node]) {
        if self.keep() {
            return;
        }
        for node in nodes.iter_mut() {
            node.kill();
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
