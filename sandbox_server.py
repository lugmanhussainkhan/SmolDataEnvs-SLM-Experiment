# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml", "pandas", "pyarrow", "numpy", "scipy", "scikit-learn", "python-dotenv"]
# ///
"""Sandbox worker: runs agent_eval.py's bash calls on this machine.

    SANDBOX_TOKEN=... uv run sandbox_server.py --config config.yaml --host <private-ip> [--slots 32]

Stage the task files here first: uv run setup_env.py --config config.yaml --set sandbox.remote=[]

API (JSON over HTTP, header `Authorization: Bearer $SANDBOX_TOKEN`):
    GET    /health                -> protocol, slots, cpus, episodes, running, calls
    POST   /missing               {rows: [{task_id, bucket_prefix, files}]} -> {missing: [task_id]}
    POST   /episodes              {task_id, bucket_prefix, files} -> {id}            (422: files not staged)
    POST   /episodes/<id>/run     {command, timeout} -> stdout, stderr, returncode, timed_out, seconds, queue_seconds
    GET    /episodes/<id>/answer  -> {answer}
    DELETE /episodes/<id>         kills its running command and removes it

This runs arbitrary code: bind to a private address and allow only the GPU machine in the firewall.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_eval import load_runs  # noqa: E402
from sandbox import PROTOCOL, MissingInputs, Sandbox, SandboxError, input_files, precompile  # noqa: E402

MAX_BODY = 64 << 20
EPISODE_PATH = re.compile(r"^/episodes/([0-9a-f]{32})(/run|/answer)?$")


def task_row(body: dict) -> dict:
    prefix = str(body["bucket_prefix"])
    if not prefix or prefix.startswith("/") or ".." in Path(prefix).parts:
        raise ValueError("bad bucket_prefix")
    return {
        "task_id": str(body.get("task_id", "")),
        "bucket_prefix": prefix,
        "files": [str(f) for f in body.get("files") or []],
    }


class Worker:
    def __init__(self, sb: Sandbox, data_dir: str, slots: int, max_timeout: float):
        self.sb, self.data_dir, self.slots, self.max_timeout = sb, data_dir, slots, max_timeout
        self.gate = threading.BoundedSemaphore(slots)
        self.lock = threading.Lock()
        self.episodes: dict[str, dict] = {}  # id -> {"ep", "last", "busy"}
        self.running = self.calls = 0

    def open(self, row: dict) -> str:
        files, missing = input_files(row, self.data_dir)
        if missing:
            raise MissingInputs(f"missing input files: {missing}")
        eid = uuid.uuid4().hex
        ep = self.sb.episode(files)
        with self.lock:
            self.episodes[eid] = {"ep": ep, "last": time.time(), "busy": 0}
        return eid

    def missing(self, body: dict) -> bool:
        try:
            return bool(input_files(task_row(body), self.data_dir)[1])
        except (KeyError, ValueError):
            return True

    def _use(self, eid: str, delta: int) -> dict:
        with self.lock:
            item = self.episodes.get(eid)
            if item is None:
                raise LookupError(eid)
            item["busy"] += delta
            item["last"] = time.time()
            return item

    def run(self, eid: str, command: str, timeout: float) -> dict:
        timeout = max(1.0, min(float(timeout), self.max_timeout))
        item = self._use(eid, +1)
        try:
            t0 = time.time()
            with self.gate:
                queued = time.time() - t0
                with self.lock:
                    self.running += 1
                try:
                    res = item["ep"].run(command, timeout)
                finally:
                    with self.lock:
                        self.running -= 1
                        self.calls += 1
        finally:
            with self.lock:
                item["busy"] -= 1
                item["last"] = time.time()
        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
            "timed_out": res.timed_out,
            "seconds": res.seconds,
            "queue_seconds": round(queued, 3),
        }

    def answer(self, eid: str) -> str | None:
        return self._use(eid, 0)["ep"].answer()

    def close(self, eid: str) -> None:
        with self.lock:
            item = self.episodes.pop(eid, None)
        if item:
            item["ep"].close()

    def reap(self, ttl: float) -> None:
        while True:
            time.sleep(min(60.0, ttl / 4))
            cutoff = time.time() - ttl
            with self.lock:
                stale = [k for k, v in self.episodes.items() if v["busy"] == 0 and v["last"] < cutoff]
                items = [self.episodes.pop(k) for k in stale]
            for item in items:
                item["ep"].close()
            if items:
                print(f"reaped {len(items)} idle episodes", flush=True)

    def health(self) -> dict:
        with self.lock:
            return {
                "protocol": PROTOCOL,
                "slots": self.slots,
                "cpus": os.cpu_count(),
                "episodes": len(self.episodes),
                "running": self.running,
                "calls": self.calls,
                "load": os.getloadavg(),
            }


def make_handler(worker: Worker, token: str) -> type[BaseHTTPRequestHandler]:
    expected = f"Bearer {token}".encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            return data

        def _route(self, method: str) -> tuple[int, dict]:
            body = self._body() if method == "POST" else {}
            if (method, self.path) == ("GET", "/health"):
                return 200, worker.health()
            if (method, self.path) == ("POST", "/missing"):
                return 200, {"missing": [str(r.get("task_id")) for r in body["rows"] if worker.missing(r)]}
            if (method, self.path) == ("POST", "/episodes"):
                return 200, {"id": worker.open(task_row(body))}
            m = EPISODE_PATH.match(self.path)
            if m and method == "POST" and m[2] == "/run":
                if not isinstance(body.get("command"), str):
                    raise ValueError("command must be a string")
                return 200, worker.run(m[1], body["command"], body.get("timeout", 90))
            if m and method == "GET" and m[2] == "/answer":
                return 200, {"answer": worker.answer(m[1])}
            if m and method == "DELETE" and m[2] is None:
                worker.close(m[1])
                return 200, {"ok": True}
            return 404, {"error": f"no route {method} {self.path}"}

        def _handle(self, method: str) -> None:
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected):
                self.close_connection = True
                return self._send(401, {"error": "unauthorized"})
            try:
                self._send(*self._route(method))
            except MissingInputs as exc:
                self._send(422, {"error": str(exc)})
            except LookupError:
                self._send(404, {"error": "unknown episode (closed or reaped)"})
            except SandboxError as exc:
                self._send(500, {"error": f"sandbox: {exc}"})
            except (KeyError, ValueError, TypeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the same config the GPU machine uses (data.dir, sandbox.*)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--host", default="127.0.0.1", help="bind address: this machine's private IP")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--slots", type=int, default=0, help="commands run at once (default: CPU count)")
    ap.add_argument("--max-timeout", type=float, default=600, help="cap on one call's timeout, seconds")
    ap.add_argument("--idle-ttl", type=float, default=1800, help="remove episodes idle this long, seconds")
    ap.add_argument("--scratch", help="parent dir for episode dirs (default: system temp; tmpfs is ideal)")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get("SANDBOX_TOKEN", "")
    if len(token) < 16:
        sys.exit("set SANDBOX_TOKEN (16+ chars, e.g. `openssl rand -hex 24`) in the environment or .env")

    cfg = load_runs(a.config, a.set)[0]
    if not (Path(cfg["data"]["dir"]) / "task_files").is_dir():
        sys.exit(f"{cfg['data']['dir']}/task_files not found: run setup_env.py here first")
    scratch = Path(tempfile.mkdtemp(prefix="smol-worker-", dir=a.scratch))
    sc = cfg["sandbox"]
    try:
        sb = Sandbox(sc["python"], sc["mem_gb"], scratch, sc["threads"], sc["net"])
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"sandbox: {exc}")
    print(f"compiling bytecode for {sb.python} ...", flush=True)
    precompile(sb.python)

    worker = Worker(sb, cfg["data"]["dir"], a.slots or os.cpu_count() or 4, a.max_timeout)
    threading.Thread(target=worker.reap, args=(a.idle_ttl,), daemon=True).start()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(worker, token))
    server.daemon_threads = True
    server.request_queue_size = 256
    print(f"sandbox worker on http://{a.host}:{a.port}: {worker.slots} slots, scratch {scratch}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sb.kill_all()
        server.server_close()
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
