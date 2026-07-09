//! Test fixtures for `qfm_text`. See `README.md` for the data
//! provenance.

pub mod tiny {
    //! A 200-token synthetic WikiText-style fixture used by the
    //! corpus-level integration tests. Generated at build time
    //! by the `generate_fixtures.sh` script in the crate root
    //! (deterministic; safe to commit).
    include_bytes!("../testdata/tiny_fixture.bin");
}
