# Seed note: security postures (H9)

**Category**: feature
**Status**: implemented

`unfer_protocol::posture` defines `SecurityPosture { dangerous, auto, strict }`
with `compose(org_floor, scope)` = stricter wins, plus `ProvenanceSource` labels
and the `Screener` seam + `NOT_SECURITY_SCREENED` notice (never a silent pass).
`uk_posture_get`/`uk_posture_set` are operator-gated via the S22 admin seam
(UK-4501 for bounded callers); the loopback `PostureListener` pauses every
Mutate `uk_*` under strict except the two no-effect turn enders. See
PROTOCOL.md "three portal-only walls".
