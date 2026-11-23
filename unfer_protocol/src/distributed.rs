//! D10: the bytes a replicated command travels as, and the domain a signature
//! over it is separated by.
//!
//! QuePaxa agrees on a value; today that value is a `u64` transaction id and the
//! command itself waits in a process-local map (`net.rs`'s pending store), so a
//! committed id can name nothing at all after a restart on another node. This
//! module is the replacement value: [`ReplicatedValue`] carries the whole
//! bounded command, and it carries an explicit [`ReplicatedValue::Noop`] so a
//! gap filler is a *value*, not a reserved integer that a command could never
//! legally take.
//!
//! The frame is canonical and versioned, not `serde_json`: every byte has one
//! encoding, every integer has a stated width and byte order, and there is no
//! map or float whose rendering could differ between two writers.
//!
//! ```text
//! magic(4) = "UFDC" | version(u16 BE) | kind(u8) | body_len(u32 BE) | body
//! kind 0 (noop):    body_len = 0
//! kind 1 (command): body = schema(u16 BE) | payload(body_len - 2)
//! ```
//!
//! `body_len` is bounded by [`MAX_BODY_BYTES`] (the payload bound plus the
//! schema tag) and checked **before** any allocation, so a hostile header
//! cannot ask a decoder for a gigabyte.
//!
//! ```text
//! use unfer_protocol::distributed::{Command, ReplicatedValue};
//! let value = ReplicatedValue::Command(Command::new(1, b"hi".to_vec()).unwrap());
//! let frame = value.encode().unwrap();
//! assert_eq!(unfer_protocol::distributed::decode_exact(&frame).unwrap(), value);
//! ```
//!
//! Nothing here identifies a command by a process-local id: [`ReplicatedValue`]
//! has no numeric identity to resolve, so a committed slot always names the
//! bytes it agreed on.

use std::fmt;

/// The only frame version this module writes. A decoder refuses anything else
/// instead of guessing: an unknown version is a compatibility step its owner
/// must take deliberately.
pub const WIRE_VERSION: u16 = 1;

/// `"UFDC"` — the frame's first four bytes.
pub const MAGIC: [u8; 4] = *b"UFDC";

/// magic + version + kind + body length.
pub const HEADER_BYTES: usize = 11;

/// `schema(u16)` at the head of a command body.
pub const SCHEMA_BYTES: usize = 2;

/// The largest command payload this wire accepts (1 MiB).
pub const MAX_COMMAND_BYTES: usize = 1024 * 1024;

/// The largest command body: the schema tag plus the largest payload.
pub const MAX_BODY_BYTES: usize = SCHEMA_BYTES + MAX_COMMAND_BYTES;

/// The largest frame this wire accepts, header included.
pub const MAX_FRAME_BYTES: usize = HEADER_BYTES + MAX_BODY_BYTES;

const KIND_NOOP: u8 = 0;
const KIND_COMMAND: u8 = 1;

/// A replicated value: an explicit no-op, or one bounded command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicatedValue {
    /// The value QuePaxa writes into a slot that carries no command (a gap
    /// filler). It is a variant on purpose: no integer is reserved for it, so a
    /// command can never collide with "nothing".
    Noop,
    /// A complete command, carried in the value itself.
    Command(Command),
}

/// One command: the schema the payload is read under, and the payload bytes.
///
/// The payload is opaque here — schema semantics belong to the application that
/// owns the schema — but its length is bounded by [`MAX_COMMAND_BYTES`] at
/// construction and at decode time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    schema: u16,
    payload: Vec<u8>,
}

impl Command {
    /// Build a command, refusing a zero schema (0 is never a schema tag) and an
    /// over-long payload before the payload is copied anywhere else.
    pub fn new(schema: u16, payload: Vec<u8>) -> Result<Self, WireError> {
        if schema == 0 {
            return Err(WireError::ZeroSchema);
        }
        if payload.len() > MAX_COMMAND_BYTES {
            return Err(WireError::BodyTooLong {
                body_len: payload.len(),
                max: MAX_COMMAND_BYTES,
            });
        }
        Ok(Self { schema, payload })
    }

    /// The schema this command's payload is read under (never 0).
    pub fn schema(&self) -> u16 {
        self.schema
    }

    /// The payload bytes.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

/// What went wrong encoding or decoding a frame, or building a signing
/// preimage. Every variant names the rule that refused, not a parser state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// The frame does not begin with [`MAGIC`].
    BadMagic { found: [u8; 4] },
    /// The frame names a version this decoder does not implement.
    UnsupportedVersion { found: u16 },
    /// The kind byte is not one this version defines.
    UnknownKind { found: u8 },
    /// A body length exceeds [`MAX_BODY_BYTES`]. Checked from the header alone,
    /// before any buffer is reserved.
    BodyTooLong { body_len: usize, max: usize },
    /// A no-op declared a body.
    NoopWithBody { body_len: usize },
    /// A command body is shorter than its schema tag.
    CommandBodyTooShort { body_len: usize },
    /// A command's schema tag was 0.
    ZeroSchema,
    /// Fewer bytes are present than the frame states.
    Truncated { needed: usize, had: usize },
    /// A decode that requires the frame to be the whole input found more than
    /// one frame's worth of bytes.
    TrailingGarbage { consumed: usize, total: usize },
    /// A signing-domain field is longer than [`MAX_DOMAIN_FIELD_BYTES`].
    DomainFieldTooLong { field: &'static str, len: usize },
    /// A frame larger than [`MAX_FRAME_BYTES`] was offered to a signature.
    FrameTooLong { len: usize, max: usize },
    /// A streaming decoder was given more unfinished bytes than one frame can
    /// carry.
    BufferTooLong { len: usize, max: usize },
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::BadMagic { found } => {
                write!(f, "frame magic {found:02x?} is not UFDC")
            }
            WireError::UnsupportedVersion { found } => {
                write!(f, "frame version {found} is not {WIRE_VERSION}")
            }
            WireError::UnknownKind { found } => write!(f, "frame kind {found} is not defined"),
            WireError::BodyTooLong { body_len, max } => {
                write!(f, "body length {body_len} exceeds {max}")
            }
            WireError::NoopWithBody { body_len } => {
                write!(f, "a no-op frame declared {body_len} body byte(s)")
            }
            WireError::CommandBodyTooShort { body_len } => {
                write!(f, "a command body of {body_len} byte(s) has no schema tag")
            }
            WireError::ZeroSchema => write!(f, "schema tag 0 is not a schema"),
            WireError::Truncated { needed, had } => {
                write!(f, "frame needs {needed} byte(s), {had} present")
            }
            WireError::TrailingGarbage { consumed, total } => {
                write!(f, "a {consumed}-byte frame was followed by {total} byte(s)")
            }
            WireError::DomainFieldTooLong { field, len } => {
                write!(f, "signing-domain field {field} is {len} byte(s)")
            }
            WireError::FrameTooLong { len, max } => {
                write!(f, "a {len}-byte frame exceeds {max}")
            }
            WireError::BufferTooLong { len, max } => {
                write!(f, "{len} buffered byte(s) exceed one frame's {max}")
            }
        }
    }
}

impl std::error::Error for WireError {}

impl ReplicatedValue {
    /// The canonical bytes of this value.
    pub fn encode(&self) -> Result<Vec<u8>, WireError> {
        let (kind, body): (u8, Vec<u8>) = match self {
            ReplicatedValue::Noop => (KIND_NOOP, Vec::new()),
            ReplicatedValue::Command(command) => {
                let mut body = Vec::with_capacity(SCHEMA_BYTES + command.payload.len());
                body.extend_from_slice(&command.schema.to_be_bytes());
                body.extend_from_slice(&command.payload);
                (KIND_COMMAND, body)
            }
        };
        if body.len() > MAX_BODY_BYTES {
            return Err(WireError::BodyTooLong {
                body_len: body.len(),
                max: MAX_BODY_BYTES,
            });
        }
        let mut out = Vec::with_capacity(HEADER_BYTES + body.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&WIRE_VERSION.to_be_bytes());
        out.push(kind);
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }
}

/// Decode the first frame in `buf`, returning it and how many bytes it used.
///
/// A short buffer is [`WireError::Truncated`], never a partial value: callers
/// that stream can wait for more bytes and call again. A body length larger
/// than [`MAX_BODY_BYTES`] is refused from the header, before the payload
/// would be copied.
pub fn decode_frame(buf: &[u8]) -> Result<(ReplicatedValue, usize), WireError> {
    if buf.len() >= MAGIC.len() && buf[..MAGIC.len()] != MAGIC {
        let mut found = [0u8; 4];
        found.copy_from_slice(&buf[..MAGIC.len()]);
        return Err(WireError::BadMagic { found });
    }
    if buf.len() < HEADER_BYTES {
        return Err(WireError::Truncated {
            needed: HEADER_BYTES,
            had: buf.len(),
        });
    }
    let version = u16::from_be_bytes([buf[4], buf[5]]);
    if version != WIRE_VERSION {
        return Err(WireError::UnsupportedVersion { found: version });
    }
    let kind = buf[6];
    if kind != KIND_NOOP && kind != KIND_COMMAND {
        return Err(WireError::UnknownKind { found: kind });
    }
    let body_len = u32::from_be_bytes([buf[7], buf[8], buf[9], buf[10]]) as usize;
    if body_len > MAX_BODY_BYTES {
        return Err(WireError::BodyTooLong {
            body_len,
            max: MAX_BODY_BYTES,
        });
    }
    let total = HEADER_BYTES + body_len;
    if buf.len() < total {
        return Err(WireError::Truncated {
            needed: total,
            had: buf.len(),
        });
    }
    let body = &buf[HEADER_BYTES..total];
    let value = match kind {
        KIND_NOOP => {
            if body_len != 0 {
                return Err(WireError::NoopWithBody { body_len });
            }
            ReplicatedValue::Noop
        }
        _ => {
            if body_len < SCHEMA_BYTES {
                return Err(WireError::CommandBodyTooShort { body_len });
            }
            let schema = u16::from_be_bytes([body[0], body[1]]);
            if schema == 0 {
                return Err(WireError::ZeroSchema);
            }
            let payload = body[SCHEMA_BYTES..].to_vec();
            ReplicatedValue::Command(Command { schema, payload })
        }
    };
    Ok((value, total))
}

/// Decode exactly one frame: trailing bytes are refused rather than ignored,
/// because a caller that wanted one frame and got two has a framing bug.
pub fn decode_exact(buf: &[u8]) -> Result<ReplicatedValue, WireError> {
    let (value, consumed) = decode_frame(buf)?;
    if consumed != buf.len() {
        return Err(WireError::TrailingGarbage {
            consumed,
            total: buf.len(),
        });
    }
    Ok(value)
}

/// A streaming decoder: push arbitrary chunks, receive every complete frame.
///
/// Partial frames stay buffered; coalesced frames are all returned in order. The
/// buffer never grows past [`MAX_FRAME_BYTES`], so a peer cannot make a decoder
/// hold an unbounded prefix of a frame it will never finish.
#[derive(Debug, Default)]
pub struct FrameDecoder {
    pending: Vec<u8>,
}

impl FrameDecoder {
    /// A decoder with no buffered bytes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `chunk` and return every frame it completed, in order.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<ReplicatedValue>, WireError> {
        self.pending.extend_from_slice(chunk);
        let mut decoded = Vec::new();
        loop {
            match decode_frame(&self.pending) {
                Ok((value, consumed)) => {
                    self.pending.drain(..consumed);
                    decoded.push(value);
                }
                Err(WireError::Truncated { .. }) => break,
                Err(error) => return Err(error),
            }
        }
        if self.pending.len() > MAX_FRAME_BYTES {
            return Err(WireError::BufferTooLong {
                len: self.pending.len(),
                max: MAX_FRAME_BYTES,
            });
        }
        Ok(decoded)
    }

    /// Assert that nothing half-read is left over. A truncated tail is an
    /// error here: the stream ended mid-frame.
    pub fn finish(self) -> Result<(), WireError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(WireError::Truncated {
                needed: HEADER_BYTES,
                had: self.pending.len(),
            })
        }
    }

    /// The bytes of the frame currently being assembled.
    pub fn buffered(&self) -> usize {
        self.pending.len()
    }
}

// ── signature domains ────────────────────────────────────────────────────────

/// The prefix every domain-separated command signature covers, ahead of the
/// deployment/audience/schema/context fields and the frame.
pub const SIGNATURE_DOMAIN_PREFIX: &[u8] = b"unfer/distributed/command/v1";

/// A signing-domain field longer than this is refused instead of hashed.
pub const MAX_DOMAIN_FIELD_BYTES: usize = 128;

/// The deployment a signature is for. A signature over a command for
/// `deployment` is not a signature over the same bytes anywhere else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SigningDomain<'a> {
    /// The deployment the command is authorized in.
    pub deployment: &'a str,
    /// Who may act on it (for example, a service name).
    pub audience: &'a str,
    /// The command schema family.
    pub schema: &'a str,
    /// A free-form context the deployment distinguishes (a lane, a purpose).
    pub context: &'a str,
}

/// Which preimage layout a signature was made under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureVersion {
    /// [`signing_preimage`]: prefix, domain fields, frame — all length-tagged.
    DomainSeparated,
    /// The pre-D10 layout: `sha256(frame)` with no domain at all. Kept so
    /// already-signed history still verifies; new signatures must not use it.
    Legacy,
}

/// The bytes a [`SignatureVersion::DomainSeparated`] signature covers.
///
/// Layout: [`SIGNATURE_DOMAIN_PREFIX`], then each of the four domain fields as
/// `len(u16 BE) | bytes`, then `frame_len(u32 BE) | frame`. Every element is
/// length-tagged, so no two (domain, frame) pairs render the same preimage.
pub fn signing_preimage(domain: &SigningDomain<'_>, frame: &[u8]) -> Result<Vec<u8>, WireError> {
    if frame.len() > MAX_FRAME_BYTES {
        return Err(WireError::FrameTooLong {
            len: frame.len(),
            max: MAX_FRAME_BYTES,
        });
    }
    let mut out = Vec::with_capacity(SIGNATURE_DOMAIN_PREFIX.len() + 8 + 8 * 4 + frame.len());
    out.extend_from_slice(SIGNATURE_DOMAIN_PREFIX);
    for (field, text) in [
        ("deployment", domain.deployment),
        ("audience", domain.audience),
        ("schema", domain.schema),
        ("context", domain.context),
    ] {
        if text.len() > MAX_DOMAIN_FIELD_BYTES {
            return Err(WireError::DomainFieldTooLong {
                field,
                len: text.len(),
            });
        }
        out.extend_from_slice(&(text.len() as u16).to_be_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    out.extend_from_slice(&(frame.len() as u32).to_be_bytes());
    out.extend_from_slice(frame);
    Ok(out)
}

/// The preimage for a stated signature version.
///
/// [`SignatureVersion::Legacy`] is exactly the pre-D10 construction —
/// `sha256(frame)` — so a verifier can still accept history while every new
/// signature is domain separated.
pub fn preimage(
    version: SignatureVersion,
    domain: &SigningDomain<'_>,
    frame: &[u8],
) -> Result<Vec<u8>, WireError> {
    match version {
        SignatureVersion::DomainSeparated => signing_preimage(domain, frame),
        SignatureVersion::Legacy => {
            if frame.len() > MAX_FRAME_BYTES {
                return Err(WireError::FrameTooLong {
                    len: frame.len(),
                    max: MAX_FRAME_BYTES,
                });
            }
            use sha2::{Digest, Sha256};
            Ok(Sha256::digest(frame).to_vec())
        }
    }
}

// ── golden vectors ───────────────────────────────────────────────────────────

/// The value a golden vector decodes to, in a form a `const` can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldenValue {
    /// A no-op frame.
    Noop,
    /// A command with the schema and the payload's lowercase hex.
    Command {
        /// The command's schema tag.
        schema: u16,
        /// The payload bytes, lowercase hex.
        payload_hex: &'static str,
    },
}

/// One pinned encoding: a name, the frame's hex, and what it decodes to.
///
/// These are the wire's contract. [`decode_exact`] must accept each one, and
/// encoding the decoded value must reproduce the bytes exactly — a change that
/// breaks either is a wire break, not a refactor.
#[derive(Debug, Clone, Copy)]
pub struct GoldenVector {
    /// What the vector is.
    pub name: &'static str,
    /// The canonical frame, lowercase hex.
    pub hex: &'static str,
    /// The value it decodes to (schema and payload for a command).
    pub value: GoldenValue,
}

impl GoldenValue {
    /// Build the [`ReplicatedValue`] this vector pins.
    pub fn to_value(&self) -> Result<ReplicatedValue, WireError> {
        match self {
            GoldenValue::Noop => Ok(ReplicatedValue::Noop),
            GoldenValue::Command { schema, payload_hex } => {
                let payload = hex::decode(payload_hex).map_err(|_| WireError::Truncated {
                    needed: payload_hex.len() / 2,
                    had: 0,
                })?;
                Ok(ReplicatedValue::Command(Command::new(*schema, payload)?))
            }
        }
    }
}

/// The pinned encodings, checked in `tests/distributed_wire.rs`.
pub const GOLDEN_VECTORS: &[GoldenVector] = &[
    GoldenVector {
        name: "no-op",
        hex: "5546444300010000000000",
        value: GoldenValue::Noop,
    },
    GoldenVector {
        name: "command schema 1 payload \"hi\"",
        hex: "554644430001010000000400016869",
        value: GoldenValue::Command {
            schema: 1,
            payload_hex: "6869",
        },
    },
    GoldenVector {
        name: "command schema 258 payload [00 ff]",
        hex: "5546444300010100000004010200ff",
        value: GoldenValue::Command {
            schema: 258,
            payload_hex: "00ff",
        },
    },
];
