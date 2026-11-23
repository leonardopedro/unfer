//! D10: two independent decoders accept the golden vectors, and the framed
//! wire behaves at every truncation, split and coalescing boundary.

use unfer_protocol::distributed::{
    decode_exact, decode_frame, preimage, signing_preimage, Command, FrameDecoder, GoldenVector,
    ReplicatedValue, SignatureVersion, SigningDomain, WireError, GOLDEN_VECTORS, HEADER_BYTES,
    MAGIC, MAX_BODY_BYTES, MAX_COMMAND_BYTES, MAX_FRAME_BYTES, MAX_DOMAIN_FIELD_BYTES, SCHEMA_BYTES,
    SIGNATURE_DOMAIN_PREFIX, WIRE_VERSION,
};

fn frame_hex(vector: &GoldenVector) -> Vec<u8> {
    hex::decode(vector.hex).expect("golden vector is hex")
}

/// A second decoder, written independently of the module's: it walks the frame
/// with an explicit cursor and compares fields, so a bug shared with
/// `decode_frame` cannot make both wrong in the same way.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.at + n > self.bytes.len() {
            return Err(format!("wanted {n} byte(s) at {}, have {}", self.at, self.bytes.len()));
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        let raw = self.take(2)?;
        Ok(((raw[0] as u16) << 8) | raw[1] as u16)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let raw = self.take(4)?;
        Ok(((raw[0] as u32) << 24) | ((raw[1] as u32) << 16) | ((raw[2] as u32) << 8) | raw[3] as u32)
    }
}

fn independent_decode(bytes: &[u8]) -> Result<(u16, Option<(u16, Vec<u8>)>), String> {
    let mut cursor = Cursor { bytes, at: 0 };
    if cursor.take(4)? != MAGIC {
        return Err("magic".into());
    }
    let version = cursor.u16()?;
    let kind = cursor.u8()?;
    let body_len = cursor.u32()? as usize;
    let body = cursor.take(body_len)?;
    if cursor.at != bytes.len() {
        return Err("trailing".into());
    }
    match kind {
        0 if body.is_empty() => Ok((version, None)),
        1 => {
            if body.len() < SCHEMA_BYTES {
                return Err("short body".into());
            }
            let schema = ((body[0] as u16) << 8) | body[1] as u16;
            Ok((version, Some((schema, body[SCHEMA_BYTES..].to_vec()))))
        }
        other => Err(format!("kind {other}")),
    }
}

#[test]
fn both_decoders_accept_the_golden_vectors_and_re_encode_them() {
    for vector in GOLDEN_VECTORS {
        let bytes = frame_hex(vector);
        // 1. the module's decoder
        assert_eq!(
            decode_exact(&bytes).unwrap_or_else(|error| panic!("{}: {error}", vector.name)),
            vector.value.to_value().unwrap(),
            "{}: module decoder disagrees",
            vector.name
        );
        // 2. the independent one
        let (version, body) = independent_decode(&bytes)
            .unwrap_or_else(|error| panic!("{}: independent decoder: {error}", vector.name));
        assert_eq!(version, WIRE_VERSION, "{}: version", vector.name);
        match (&vector.value.to_value().unwrap(), body) {
            (ReplicatedValue::Noop, None) => {}
            (ReplicatedValue::Command(command), Some((schema, payload))) => {
                assert_eq!(schema, command.schema(), "{}: schema", vector.name);
                assert_eq!(payload, command.payload(), "{}: payload", vector.name);
            }
            (value, body) => panic!("{}: decoders disagree: {value:?} vs {body:?}", vector.name),
        }
        // 3. the bytes are canonical: encoding reproduces them exactly
        assert_eq!(
            vector.value.to_value().unwrap().encode().unwrap(),
            bytes,
            "{}: re-encoding is not the pinned frame",
            vector.name
        );
    }
}

#[test]
fn the_golden_vectors_use_every_declared_width() {
    for vector in GOLDEN_VECTORS {
        let bytes = frame_hex(vector);
        assert_eq!(bytes[..4], MAGIC, "{}", vector.name);
        assert_eq!(u16::from_be_bytes([bytes[4], bytes[5]]), WIRE_VERSION);
        assert!(bytes.len() <= MAX_FRAME_BYTES);
    }
    // The no-op is exactly a header and nothing else.
    assert_eq!(frame_hex(&GOLDEN_VECTORS[0]).len(), HEADER_BYTES);
}

#[test]
fn every_split_point_of_every_vector_streams_to_the_same_value() {
    for vector in GOLDEN_VECTORS {
        let bytes = frame_hex(vector);
        for split in 0..=bytes.len() {
            let mut decoder = FrameDecoder::new();
            let mut values = decoder.push(&bytes[..split]).expect("prefix");
            values.extend(decoder.push(&bytes[split..]).expect("suffix"));
            assert_eq!(values, vec![vector.value.to_value().unwrap()], "{} split at {split}", vector.name);
            assert_eq!(decoder.buffered(), 0, "{} split at {split}", vector.name);
            decoder.finish().expect("nothing pending");
        }
        // Byte at a time.
        let mut decoder = FrameDecoder::new();
        let mut values = Vec::new();
        for byte in &bytes {
            values.extend(decoder.push(std::slice::from_ref(byte)).expect("byte"));
        }
        assert_eq!(values, vec![vector.value.to_value().unwrap()], "{} byte-at-a-time", vector.name);
    }
}

#[test]
fn coalesced_frames_decode_in_order() {
    let mut stream = Vec::new();
    for vector in GOLDEN_VECTORS {
        stream.extend_from_slice(&frame_hex(vector));
    }
    let mut decoder = FrameDecoder::new();
    let mut values = Vec::new();
    // In one push, and in arbitrary chunks.
    values.extend(decoder.push(&stream).expect("one push"));
    assert_eq!(
        values,
        GOLDEN_VECTORS
            .iter()
            .map(|v| v.value.to_value().unwrap())
            .collect::<Vec<_>>()
    );
    let mut decoder = FrameDecoder::new();
    let mut chunked = Vec::new();
    for chunk in stream.chunks(3) {
        chunked.extend(decoder.push(chunk).expect("chunk"));
    }
    assert_eq!(chunked, values);
}

#[test]
fn every_truncation_is_refused_and_names_what_it_needs() {
    for vector in GOLDEN_VECTORS {
        let bytes = frame_hex(vector);
        for cut in 0..bytes.len() {
            match decode_exact(&bytes[..cut]) {
                Err(WireError::Truncated { .. }) => {}
                other => panic!("{} cut at {cut}: {other:?}", vector.name),
            }
            // Streaming keeps the partial frame and reports nothing.
            let mut decoder = FrameDecoder::new();
            assert!(decoder.push(&bytes[..cut]).expect("partial").is_empty());
            assert_eq!(decoder.buffered(), cut);
            if cut > 0 {
                assert!(matches!(decoder.finish(), Err(WireError::Truncated { .. })));
            } else {
                decoder.finish().expect("an empty stream is complete");
            }
        }
        decode_exact(&bytes).expect("the whole frame decodes");
    }
}

#[test]
fn trailing_garbage_is_refused_and_two_frames_are_not_one() {
    let bytes = frame_hex(&GOLDEN_VECTORS[1]);
    let mut long = bytes.clone();
    long.push(0x00);
    match decode_exact(&long) {
        Err(WireError::TrailingGarbage { consumed, total }) => {
            assert_eq!(consumed, bytes.len());
            assert_eq!(total, long.len());
        }
        other => panic!("{other:?}"),
    }
    let mut two = bytes.clone();
    two.extend_from_slice(&frame_hex(&GOLDEN_VECTORS[0]));
    assert!(matches!(
        decode_exact(&two),
        Err(WireError::TrailingGarbage { .. })
    ));
    // …but the framed decoder reads both.
    let (first, used) = decode_frame(&two).expect("first");
    assert_eq!(first, GOLDEN_VECTORS[1].value.to_value().unwrap());
    assert_eq!(used, bytes.len());
    assert_eq!(decode_exact(&two[used..]).expect("second"), ReplicatedValue::Noop);
}

#[test]
fn an_oversized_body_is_refused_from_the_header_alone() {
    let mut header = Vec::new();
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(&WIRE_VERSION.to_be_bytes());
    header.push(1);
    header.extend_from_slice(&((MAX_BODY_BYTES + 1) as u32).to_be_bytes());
    match decode_frame(&header) {
        Err(WireError::BodyTooLong { body_len, max }) => {
            assert_eq!(body_len, MAX_BODY_BYTES + 1);
            assert_eq!(max, MAX_BODY_BYTES);
        }
        other => panic!("{other:?}"),
    }
    // The same holds when no payload bytes follow at all: the length is refused
    // before availability is considered.
    assert_eq!(header.len(), HEADER_BYTES);
    // And a length that would overflow the frame budget is the same refusal.
    let mut huge = header.clone();
    huge[7..11].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(matches!(
        decode_frame(&huge),
        Err(WireError::BodyTooLong { .. })
    ));
}

#[test]
fn malformed_headers_are_refused_by_the_rule_they_break() {
    let good = frame_hex(&GOLDEN_VECTORS[1]);

    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        decode_exact(&bad_magic),
        Err(WireError::BadMagic { .. })
    ));

    let mut bad_version = good.clone();
    bad_version[5] = 2;
    assert_eq!(
        decode_exact(&bad_version),
        Err(WireError::UnsupportedVersion { found: 2 })
    );

    let mut bad_kind = good.clone();
    bad_kind[6] = 7;
    assert_eq!(decode_exact(&bad_kind), Err(WireError::UnknownKind { found: 7 }));

    // A no-op that declares a body.
    let mut noop_with_body = frame_hex(&GOLDEN_VECTORS[0]);
    noop_with_body[10] = 1;
    noop_with_body.push(0x00);
    assert!(matches!(
        decode_exact(&noop_with_body),
        Err(WireError::NoopWithBody { body_len: 1 })
    ));

    // A command with no schema tag.
    let mut short = good.clone();
    short[7..11].copy_from_slice(&1u32.to_be_bytes());
    short.truncate(HEADER_BYTES + 1);
    assert_eq!(
        decode_exact(&short),
        Err(WireError::CommandBodyTooShort { body_len: 1 })
    );

    // Schema 0 is not a schema.
    let mut zero_schema = good.clone();
    zero_schema[HEADER_BYTES] = 0;
    zero_schema[HEADER_BYTES + 1] = 0;
    assert_eq!(decode_exact(&zero_schema), Err(WireError::ZeroSchema));
}

#[test]
fn a_command_is_bounded_at_construction_and_at_decode() {
    assert_eq!(
        Command::new(0, Vec::new()),
        Err(WireError::ZeroSchema),
        "0 is never a schema tag"
    );
    assert!(matches!(
        Command::new(1, vec![0u8; MAX_COMMAND_BYTES + 1]),
        Err(WireError::BodyTooLong { .. })
    ));
    let full = Command::new(1, vec![7u8; MAX_COMMAND_BYTES]).expect("exactly at the bound");
    let frame = ReplicatedValue::Command(full.clone()).encode().unwrap();
    assert_eq!(frame.len(), MAX_FRAME_BYTES);
    assert_eq!(decode_exact(&frame).unwrap(), ReplicatedValue::Command(full));
    // An empty payload is a legal command: the schema decides what it means.
    let empty = Command::new(9, Vec::new()).unwrap();
    assert_eq!(
        decode_exact(&ReplicatedValue::Command(empty.clone()).encode().unwrap()).unwrap(),
        ReplicatedValue::Command(empty)
    );
}

#[test]
fn a_frame_for_another_deployment_audience_schema_or_context_has_another_preimage() {
    let frame = frame_hex(&GOLDEN_VECTORS[1]);
    let base = SigningDomain {
        deployment: "prod-eu",
        audience: "ledger",
        schema: "certificate",
        context: "lane-1",
    };
    let preimages = |domain: SigningDomain<'_>| signing_preimage(&domain, &frame).unwrap();
    let reference = preimages(base);
    assert!(reference.starts_with(SIGNATURE_DOMAIN_PREFIX));
    for other in [
        SigningDomain { deployment: "prod-us", ..base },
        SigningDomain { audience: "auction", ..base },
        SigningDomain { schema: "session", ..base },
        SigningDomain { context: "lane-2", ..base },
    ] {
        assert_ne!(preimages(other), reference, "{other:?} shares a preimage");
    }
    // The same domain over different bytes differs, and the frame is inside.
    assert_ne!(signing_preimage(&base, &frame_hex(&GOLDEN_VECTORS[0])).unwrap(), reference);
    assert!(reference.ends_with(&frame));
}

#[test]
fn domain_separation_is_length_tagged_not_delimited() {
    // Moving a byte across a field boundary must change the preimage: a
    // delimiter-joined layout would collide here.
    let frame = frame_hex(&GOLDEN_VECTORS[0]);
    let left = SigningDomain { deployment: "ab", audience: "c", ..Default::default() };
    let right = SigningDomain { deployment: "a", audience: "bc", ..Default::default() };
    assert_ne!(
        signing_preimage(&left, &frame).unwrap(),
        signing_preimage(&right, &frame).unwrap()
    );
    // An over-long field is refused, not hashed.
    let long = "x".repeat(MAX_DOMAIN_FIELD_BYTES + 1);
    match signing_preimage(
        &SigningDomain { deployment: &long, ..Default::default() },
        &frame,
    ) {
        Err(WireError::DomainFieldTooLong { field, .. }) => assert_eq!(field, "deployment"),
        other => panic!("{other:?}"),
    }
    // A frame larger than the wire's own bound is refused by both versions.
    let too_long = vec![0u8; MAX_FRAME_BYTES + 1];
    assert!(matches!(
        signing_preimage(&left, &too_long),
        Err(WireError::FrameTooLong { .. })
    ));
    assert!(matches!(
        preimage(SignatureVersion::Legacy, &left, &too_long),
        Err(WireError::FrameTooLong { .. })
    ));
}

#[test]
fn the_legacy_version_is_sha256_of_the_frame_alone() {
    use sha2::{Digest, Sha256};
    let frame = frame_hex(&GOLDEN_VECTORS[1]);
    let domain = SigningDomain {
        deployment: "irrelevant",
        audience: "irrelevant",
        schema: "irrelevant",
        context: "irrelevant",
    };
    let legacy = preimage(SignatureVersion::Legacy, &domain, &frame).unwrap();
    assert_eq!(legacy, Sha256::digest(&frame).to_vec());
    // Its whole point: the domain does not enter it.
    let elsewhere = SigningDomain { deployment: "another", ..domain };
    assert_eq!(preimage(SignatureVersion::Legacy, &elsewhere, &frame).unwrap(), legacy);
    // The domain-separated version does not equal it.
    assert_ne!(
        preimage(SignatureVersion::DomainSeparated, &domain, &frame).unwrap(),
        legacy
    );
}
