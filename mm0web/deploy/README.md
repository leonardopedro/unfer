# Running `mm0web` on this host

`mm0web` verifies an mm0 development (`mm0-rs compile`, then `mm0-c` — a
warning is a proof hole) and publishes the verified artifact to a Zenodo
project deposit. The exchange service in `australVM/refi` submits developments
to it and records the DOI it earns.

## Build

```sh
cd ~/Projects/unfer
cargo build -p mm0web --release --offline
```

`mm0-rs` and `mm0-c` must be on `PATH` (or named by `MM0WEB_MM0_RS` /
`MM0WEB_MM0_C`); `/healthz` answers 503, not 200, when either is missing.

## Env file

`~/.config/mm0web/mm0web.env`, mode 0600:

```sh
MM0WEB_LISTEN=127.0.0.1:8081
MM0WEB_STATE_DIR=/home/leo/.local/share/mm0web
MM0WEB_PUBLIC_URL=http://127.0.0.1:8081
MM0WEB_CREATOR=ReFi Exchange
# Leave ZENODO_API_KEY unset to verify without archiving: submissions are
# verified and stored locally, and the caller is told plainly that nothing was
# published. Set it (and MM0WEB_ZENODO_BASE for the sandbox) to archive.
#ZENODO_API_KEY=
#MM0WEB_ZENODO_BASE=https://sandbox.zenodo.org/api
```

## Unit

```sh
mkdir -p ~/.config/mm0web ~/.local/share/mm0web
cp deploy/mm0web.service ~/.config/systemd/user/mm0web.service
systemctl --user daemon-reload
systemctl --user enable --now mm0web.service
curl -sS http://127.0.0.1:8081/healthz
```

## Reading a submission

The graph, search and stats routes are **read-only**: they answer from the
local index, cannot accept, reject or publish anything, and are the same over
`/…` (HTML) and `/api/v1/…` (JSON).

```sh
# one proof's dependency graph: the layered figure, its fault list, and the
# declared/referenced/unused accounting under it
GET /graph?project=unfer&id=<proof-id>
GET /api/v1/graph?project=unfer&id=<proof-id>   # the graph itself, all of it

# substring search over stored declarations (name or statement, case-insensitive);
# every hit links at the graph of the proof that declares it
GET /search?q=nat
GET /api/v1/search?q=nat

# totals over every stored proof: projects, states, declarations, references
GET /stats
GET /api/v1/stats
```

Two things the graph deliberately does not do. It never decides anything —
the verdict comes from `mm0-rs`/`mm0-c` alone, and an empty or faulted graph
neither accepts nor rejects a submission; the fault list is a hint about the
*shape* of what was judged. And for a submission the checkers **accepted**, an
undeclared name cannot be a defect, so those are reported as information
(variables, notation tokens, proof-script helpers) rather than faults —the page says so where it lists them. A submission above 500 declarations
stops drawing boxes and says so instead — a figure that size is unreadable —
while the JSON endpoint carries every node.


## Publishing for real

Without `ZENODO_API_KEY` nothing is archived: `POST /api/v1/publish` verifies,
stores the proof locally and answers `502` with `"ZENODO_API_KEY is not set"`,
and the exchange records the proof as `verified` rather than `published`. With
a key, the deposit is built by a background worker and the proof's state moves
to `published` with a DOI; the exchange picks that up on its next poll.

The end-to-end path (create draft → upload → publish → DOI) is known-good
against `tests/mock_zenodo.py`; run `./tests/e2e.sh` for the full matrix. The
mock enforces `Content-Type: application/json` on deposit creation for the same
reason Zenodo does — the real API answered `415` for a JSON body sent without
it, and a mock that accepts anything hides exactly that.

## Verified against the real sandbox (2026-10-29)

The key was set on this host and the exchange's own `memsafe linearity
certificate` (`mm0/linearity_certificate.mm0` + `.mm1`) went through the whole
chain: verified by `mm0-rs` (4 sorts, 33 term/def, 29 ax/thm) and `mm0-c`,
queued, archived as record 627938 with DOI `10.5072/zenodo.627938`, and then
minted as an `attested` certificate against that DOI.

`10.5072` is DataCite's **test** prefix: the record page answers 200 and the
sandbox's API carries the DOI, but `doi.org/10.5072/zenodo.627938` redirects to
DataCite's test-prefix page rather than resolving. A production deposit on
`zenodo.org` (`10.5281/…`) resolves normally; nothing in the code changes for
that — only `MM0WEB_ZENODO_BASE` and the key.
