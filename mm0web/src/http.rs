//! HTTP: the server side is a small `std::net` reader/writer (no framework),
//! and the client side is `curl` — see `README.md`, "Why curl", for why this
//! crate does not depend on `ureq`/`axum` like the rest of the workspace.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};

/// Upper bound on a submitted proof (8 MiB) — a submitted `.mm0`+`.mm1` pair is
/// kilobytes in this project, and an unbounded read is a trivial DoS.
pub const MAX_BODY: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: HashMap<String, String>,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub peer: IpAddr,
}

impl Request {
    /// URL-encoded form fields from the body (browser form posts).
    pub fn form(&self) -> HashMap<String, String> {
        form_fields(&self.body)
    }

    /// A field from the form body first, then from the query string.
    pub fn field(&self, name: &str) -> Option<String> {
        self.form()
            .get(name)
            .cloned()
            .or_else(|| self.query.get(name).cloned())
    }

    pub fn wants_json(&self) -> bool {
        self.path.starts_with("/api/")
            || self
                .headers
                .get("accept")
                .map(|a| a.contains("application/json"))
                .unwrap_or(false)
    }
}

pub fn parse_request(stream: &mut TcpStream) -> std::io::Result<Request> {
    let peer = stream.peer_addr().map(|a| a.ip())?;
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "empty request",
        ));
    }
    let mut parts = line.trim_end().split(' ');
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or("/").to_string();

    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (target, HashMap::new()),
    };

    let mut headers = HashMap::new();
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().to_string();
            if key == "content-length" {
                content_length = val.parse().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }

    let mut body = Vec::new();
    if content_length > MAX_BODY {
        // Read and discard nothing: refuse loudly, the caller turns this into
        // a 413 rather than allocating.
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("body of {content_length} bytes exceeds {MAX_BODY}"),
        ));
    }
    if content_length > 0 {
        body.resize(content_length, 0);
        reader.read_exact(&mut body)?;
    }

    Ok(Request {
        method,
        path,
        query,
        headers,
        body,
        peer,
    })
}

pub fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        _ => "Status",
    }
}

pub fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter(|p| !p.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(pair), String::new()),
        })
        .collect()
}

pub fn form_fields(body: &[u8]) -> HashMap<String, String> {
    let text = String::from_utf8_lossy(body);
    text.split('&')
        .filter(|p| !p.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(pair), String::new()),
        })
        .collect()
}

/// `%XX` plus `+` → space, the `application/x-www-form-urlencoded` rules.
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── curl transport ───────────────────────────────────────────────────

/// `(status, body)` for one HTTP request, via `curl`. `body` is sent as the
/// request entity when present.
pub fn curl(
    method: &str,
    url: &str,
    headers: &[(String, String)],
    body: Option<&[u8]>,
    upload: Option<&Path>,
) -> Result<(u16, String), String> {
    let mut cmd = Command::new("curl");
    cmd.args([
        "-sS",
        "--max-time",
        "120",
        "-X",
        method,
        "-w",
        "\n%{http_code}",
    ]);
    for (k, v) in headers {
        cmd.arg("-H").arg(format!("{k}: {v}"));
    }
    let mut tmp: Option<std::path::PathBuf> = None;
    if let Some(data) = body {
        let path = std::env::temp_dir().join(format!(
            "mm0web-body-{}-{}",
            std::process::id(),
            tmp_counter()
        ));
        std::fs::write(&path, data).map_err(|e| format!("write body: {e}"))?;
        cmd.arg("--data-binary").arg(format!("@{}", path.display()));
        tmp = Some(path);
    }
    if let Some(path) = upload {
        cmd.arg("-T").arg(path);
    }
    cmd.arg(url).stdout(Stdio::piped()).stderr(Stdio::piped());

    let out = cmd
        .output()
        .map_err(|e| format!("cannot run curl: {e} (is curl installed?)"))?;
    if let Some(p) = tmp {
        let _ = std::fs::remove_file(p);
    }
    if !out.status.success() && out.stdout.is_empty() {
        return Err(format!(
            "curl failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let (body, code) = match stdout.rsplit_once('\n') {
        Some((b, c)) => (b.to_string(), c.trim().parse::<u16>().unwrap_or(0)),
        None => (stdout, 0),
    };
    Ok((code, body))
}

fn tmp_counter() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

/// Download a URL to bytes (Zenodo file download links).
pub fn curl_download(url: &str, headers: &[(String, String)]) -> Result<Vec<u8>, String> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "--max-time", "120", "-L"]);
    for (k, v) in headers {
        cmd.arg("-H").arg(format!("{k}: {v}"));
    }
    cmd.arg(url).stdout(Stdio::piped()).stderr(Stdio::piped());
    let out = cmd.output().map_err(|e| format!("cannot run curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "download {url} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decoding_covers_plus_and_hex() {
        assert_eq!(percent_decode("a+b"), "a b");
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("%2Ftmp%2Fx"), "/tmp/x");
        // a truncated escape is passed through, not panicked on
        assert_eq!(percent_decode("100%"), "100%");
    }

    #[test]
    fn form_fields_parses_a_browser_post() {
        let f = form_fields(b"project=unfer&title=a+b+proof&mm0=sort+Prop%0A");
        assert_eq!(f.get("project").unwrap(), "unfer");
        assert_eq!(f.get("title").unwrap(), "a b proof");
        assert_eq!(f.get("mm0").unwrap(), "sort Prop\n");
    }

    #[test]
    fn html_escaping_blocks_injection() {
        assert_eq!(
            escape_html("<script>x</script>"),
            "&lt;script&gt;x&lt;/script&gt;"
        );
    }

    #[test]
    fn a_real_request_parses_over_a_socket() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(
                b"POST /verify?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 11\r\n\r\nproject=abc",
            )
            .unwrap();
            s
        });
        let (mut stream, _) = listener.accept().unwrap();
        let req = parse_request(&mut stream).unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/verify");
        assert_eq!(req.query.get("x").unwrap(), "1");
        assert_eq!(req.field("project").unwrap(), "abc");
        respond(&mut stream, 200, "text/plain", b"ok").unwrap();
        let _ = client.join().unwrap();
    }

    #[test]
    fn an_oversized_body_is_refused_before_allocating() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            let _ = s.write_all(
                format!(
                    "POST /verify HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n",
                    MAX_BODY + 1
                )
                .as_bytes(),
            );
            std::thread::sleep(std::time::Duration::from_millis(200));
        });
        let (mut stream, _) = listener.accept().unwrap();
        let err = parse_request(&mut stream).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
