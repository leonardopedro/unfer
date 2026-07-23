use sha2::{Digest, Sha256};

pub fn build_magnet_uri(content_cid: &str, display_name: Option<&str>) -> String {
    let mut uri = format!("magnet:?xt=urn:btih:{content_cid}");
    if let Some(name) = display_name {
        let encoded: String = name
            .chars()
            .map(|c| match c {
                ' ' => "+".to_string(),
                c if c.is_alphanumeric() || "-_.~".contains(c) => c.to_string(),
                c => format!("%{:02X}", c as u32),
            })
            .collect();
        uri.push_str(&format!("&dn={encoded}"));
    }
    uri
}

pub fn content_cid_from_chunks(chunk_cids: &[String]) -> String {
    let mut hasher = Sha256::new();
    for cid in chunk_cids {
        hasher.update(cid.as_bytes());
    }
    hex::encode(hasher.finalize())
}

pub fn parse_magnet_cid(magnet_uri: &str) -> Option<String> {
    let params = magnet_uri.strip_prefix("magnet:?")?;
    let xt = params
        .split('&')
        .find(|part| part.starts_with("xt=urn:btih:"))?;
    Some(xt["xt=urn:btih:".len()..].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_magnet_basic() {
        let uri = build_magnet_uri("abc123", None);
        assert_eq!(uri, "magnet:?xt=urn:btih:abc123");
    }

    #[test]
    fn build_magnet_with_name() {
        let uri = build_magnet_uri("abc123", Some("my video.mp4"));
        assert!(uri.starts_with("magnet:?xt=urn:btih:abc123"));
        assert!(uri.contains("&dn=my+video.mp4"));
    }

    #[test]
    fn build_magnet_encodes_special_chars() {
        let uri = build_magnet_uri("abc", Some("a&b=c"));
        assert!(uri.contains("dn=a%26b%3Dc"));
    }

    #[test]
    fn content_cid_is_deterministic() {
        let cids = vec!["aaa".to_string(), "bbb".to_string()];
        let cid1 = content_cid_from_chunks(&cids);
        let cid2 = content_cid_from_chunks(&cids);
        assert_eq!(cid1, cid2);
        assert_eq!(cid1.len(), 64);
    }

    #[test]
    fn content_cid_differs_for_different_chunks() {
        let c1 = content_cid_from_chunks(&["aaa".to_string()]);
        let c2 = content_cid_from_chunks(&["bbb".to_string()]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn content_cid_order_matters() {
        let c1 = content_cid_from_chunks(&["aaa".to_string(), "bbb".to_string()]);
        let c2 = content_cid_from_chunks(&["bbb".to_string(), "aaa".to_string()]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn parse_magnet_roundtrip() {
        let uri = build_magnet_uri("deadbeef1234", Some("test"));
        let cid = parse_magnet_cid(&uri).unwrap();
        assert_eq!(cid, "deadbeef1234");
    }

    #[test]
    fn parse_magnet_no_xt_returns_none() {
        assert!(parse_magnet_cid("https://example.com").is_none());
    }

    #[test]
    fn empty_chunk_list_produces_valid_cid() {
        let cid = content_cid_from_chunks(&[]);
        assert_eq!(cid.len(), 64);
    }
}
