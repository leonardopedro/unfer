#!/usr/bin/env python3
"""A minimal Zenodo deposit API mock, for exercising mm0web's publish path
end-to-end without a network or an account.

Implements only what `mm0web::store::Zenodo` calls:

    POST   /api/deposit/depositions                       create a draft
    GET    /api/deposit/depositions/<id>                  fetch a draft (404 once published)
    POST   /api/deposit/depositions/<id>/actions/newversion
    PUT    /bucket/<dep>/<name>                           upload a file
    POST   /api/deposit/depositions/<id>/actions/publish  publish → 202 + DOI
    GET    /api/records/<id>                              published record + file links
    GET    /bucket/<dep>/<name>                           download (the `pull` half)

Failure injection, via environment:

    MOCK_FAIL_PUBLISH=n   the first n publish calls answer 500 (tests retries)
    MOCK_FAIL_UPLOAD=n    the first n uploads answer 500
    MOCK_DELAY_UPLOAD=s   sleep s seconds before answering an upload
                          (tests crash-resume: kill the service mid-push)
    MOCK_LOG=1            print every request to stderr
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MOCK_PORT", "8899"))
FAIL_PUBLISH = int(os.environ.get("MOCK_FAIL_PUBLISH", "0"))
FAIL_UPLOAD = int(os.environ.get("MOCK_FAIL_UPLOAD", "0"))
DELAY_UPLOAD = float(os.environ.get("MOCK_DELAY_UPLOAD", "0"))
VERBOSE = os.environ.get("MOCK_LOG") == "1"

state = {
    "next_id": 1,
    "drafts": {},   # id -> {"files": {name: bytes}}
    "records": {},  # id -> {"files": {name: bytes}, "doi": str}
    "publish_failures": 0,
    "upload_failures": 0,
    "counts": {"deposits": 0, "newversions": 0, "publishes": 0, "uploads": 0},
}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if VERBOSE:
            sys.stderr.write("mock: " + fmt % args + "\n")

    # ---------- helpers ----------
    def _send(self, code, body=b"{}", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode())

    def _read_body(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(n) if n else b""

    def _host(self):
        return f"http://127.0.0.1:{PORT}"

    def _new_id(self):
        i = state["next_id"]
        state["next_id"] += 1
        return i

    # ---------- GET ----------
    def do_GET(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        with lock:
            if parts[:2] == ["api", "deposit"] and len(parts) == 4 and parts[3].isdigit():
                did = int(parts[3])
                if did in state["drafts"]:
                    self._json(200, {"id": did, "state": "unsubmitted",
                                     "links": {"bucket": f"{self._host()}/bucket/{did}"}})
                else:
                    self._json(404, {"status": 404, "message": "draft not found"})
                return
            if parts[:2] == ["api", "records"] and len(parts) == 3 and parts[2].isdigit():
                rid = int(parts[2])
                rec = state["records"].get(rid)
                if rec is None:
                    self._json(404, {"status": 404, "message": "record not found"})
                    return
                files = [{"key": k,
                          "links": {"self": f"{self._host()}/bucket/{rid}/{k}"}}
                          for k in rec["files"]]
                self._json(200, {"id": rid, "doi": rec["doi"], "files": files})
                return
            if parts[:1] == ["bucket"] and len(parts) == 3:
                owner, name = int(parts[1]), parts[2]
                src = state["records"].get(owner) or state["drafts"].get(owner)
                data = (src or {}).get("files", {}).get(name)
                if data is None:
                    self._json(404, {"status": 404, "message": "no file"})
                    return
                self._send(200, data, "application/octet-stream")
                return
        self._json(404, {"status": 404, "message": f"no GET {self.path}"})

    # ---------- POST ----------
    def do_POST(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        body = self._read_body()

        # create a draft
        if parts == ["api", "deposit", "depositions"]:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                # Real Zenodo refuses a JSON body without this header. The mock
                # used to accept anything, which is how a missing header only
                # showed up in production.
                self._json(415, {"status": 415,
                                 "message": "Invalid 'Content-Type' header. "
                                            "Expected one of: application/json"})
                return
            with lock:
                did = self._new_id()
                state["drafts"][did] = {"files": {}}
                state["counts"]["deposits"] += 1
            if VERBOSE:
                sys.stderr.write(f"mock: created draft {did}\n")
            self._json(201, {"id": did, "state": "unsubmitted",
                             "links": {"bucket": f"{self._host()}/bucket/{did}"}})
            return

        if (len(parts) == 6 and parts[:3] == ["api", "deposit", "depositions"]
                and parts[3].isdigit() and parts[4] == "actions"):
            did = int(parts[3])
            action = parts[5]

            if action == "newversion":
                with lock:
                    if did not in state["records"]:
                        self._json(400, {"status": 400, "message": "record is not published"})
                        return
                    new_id = self._new_id()
                    state["drafts"][new_id] = {"files": dict(state["records"][did]["files"])}
                    state["counts"]["newversions"] += 1
                self._json(200, {"id": new_id,
                                 "links": {"latest_draft": f"{self._host()}/api/deposit/depositions/{new_id}",
                                           "bucket": f"{self._host()}/bucket/{new_id}"}})
                return

            if action == "publish":
                with lock:
                    if state["publish_failures"] < FAIL_PUBLISH:
                        state["publish_failures"] += 1
                        self._json(500, {"status": 500,
                                         "message": "injected publish failure"})
                        return
                    if did not in state["drafts"]:
                        self._json(400, {"status": 400, "message": "no such draft"})
                        return
                    files = state["drafts"].pop(did)["files"]
                    doi = f"10.5281/zenodo.{did}"
                    state["records"][did] = {"files": files, "doi": doi}
                    state["counts"]["publishes"] += 1
                self._json(202, {"id": did, "doi": doi,
                                 "doi_url": f"https://doi.org/{doi}",
                                 "links": {"self": f"{self._host()}/api/records/{did}"}})
                return

        self._json(404, {"status": 404, "message": f"no POST {self.path}"})

    # ---------- PUT (upload) ----------
    def do_PUT(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        data = self._read_body()
        if len(parts) != 3 or parts[0] != "bucket" or not parts[1].isdigit():
            self._json(404, {"status": 404, "message": f"no PUT {self.path}"})
            return
        if DELAY_UPLOAD:
            time.sleep(DELAY_UPLOAD)
        did, name = int(parts[1]), parts[2]
        with lock:
            if state["upload_failures"] < FAIL_UPLOAD:
                state["upload_failures"] += 1
                self._json(500, {"status": 500, "message": "injected upload failure"})
                return
            draft = state["drafts"].get(did)
            if draft is None:
                self._json(400, {"status": 400, "message": f"draft {did} is not open"})
                return
            draft["files"][name] = data
            state["counts"]["uploads"] += 1
        self._json(200, {"ok": True, "key": name, "size": len(data)})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def stats():
    with lock:
        return dict(state["counts"],
                    open_drafts=len(state["drafts"]),
                    records=sorted(state["records"]),
                    deposits_created=state["counts"]["deposits"])


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)

    # a tiny control plane for the test driver
    class Ctrl(Handler):
        def do_GET(self):
            if self.path.startswith("/__stats"):
                self._json(200, stats())
                return
            Handler.do_GET(self)

    srv.RequestHandlerClass = Ctrl
    print(f"mock zenodo on 127.0.0.1:{PORT} "
          f"(fail_publish={FAIL_PUBLISH}, fail_upload={FAIL_UPLOAD}, "
          f"delay_upload={DELAY_UPLOAD})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
