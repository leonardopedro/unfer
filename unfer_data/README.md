# unfer_data

Data plane for the unfer federation: chunking, encryption, magnet URIs,
content publishing. `crypto` owns the X25519 keypairs and AES-GCM chunk
encryption, `chunk` the content-addressed chunker (`compute_cid`), `magnet`
the `magnet:` URI scheme, `publisher`/`store` the publish/persist layer,
`release` the byte-exact release manifest (S23 golden gate), and
`blueprint` the encrypted `.cell` blueprint packaging (S27). See
`docs/DATA_PLANE.md`.