"""Where the model's bash calls run: a local bubblewrap sandbox or remote sandbox_server.py workers.

Inside every command: task files read-only at /home/user/input, writable /workdir for the episode,
fresh /tmp, no network, scrubbed environment, rlimits on memory, CPU time and file size.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

PROTOCOL = "smolenv-sandbox/1"
STREAM_CAP = 20_000
SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc")
MUTABLE_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".h5", ".hdf5"}  # opened read-write by their libraries
ENV = {
    "HOME": "/workdir",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
}


class SandboxError(RuntimeError):
    """The sandbox failed, not the model: the episode is left ungraded."""


class MissingInputs(SandboxError):
    """The task's files are not staged where the episode would run."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    seconds: float
    worker: str | None = None
    queue_seconds: float = 0.0


def input_files(row: dict, data_dir: str) -> tuple[dict[str, Path], list[str]]:
    """({name: path} of the task's files, missing names). An empty `files` list means the whole prefix."""
    prefix = Path(data_dir) / "task_files" / row["bucket_prefix"]
    names = row["files"] or (sorted(p.name for p in prefix.iterdir() if p.is_file()) if prefix.is_dir() else [])
    found = {Path(n).name: prefix / Path(n).name for n in names}
    missing = [n for n, p in found.items() if not p.is_file()] or ([] if found else ["(no files staged)"])
    return {n: p for n, p in found.items() if p.is_file()}, missing


def precompile(python: str) -> None:
    """Compile the data stack's bytecode once; inside the sandbox the environment is read-only."""
    mods = ("pandas", "numpy", "scipy.stats", "sklearn.linear_model", "sklearn.ensemble")
    code = f"import importlib\nfor m in {mods!r}:\n try: importlib.import_module(m)\n except Exception: pass"
    subprocess.run([python, "-c", code], capture_output=True, timeout=900)


def _cap(s: str) -> str:
    return s if len(s) <= STREAM_CAP else s[: STREAM_CAP // 2] + "\n...[truncated]...\n" + s[-STREAM_CAP // 2 :]


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ── local bubblewrap ─────────────────────────────────────────────────────────


class Sandbox:
    def __init__(self, python: str | None, mem_gb: float, scratch: Path, threads: int = 2, net: str = "bwrap"):
        if net not in ("bwrap", "unshare"):
            raise SandboxError(f"sandbox.net must be bwrap or unshare, got {net!r}")
        if net == "unshare" and not shutil.which("unshare"):
            raise SandboxError("unshare not found: sudo apt install util-linux")
        self.net = net
        if not shutil.which("bwrap"):
            raise SandboxError("bwrap not found: sudo apt install bubblewrap")
        self.python = python or sys.executable
        self.mem_kb = int(mem_gb * 1024 * 1024)
        self.env = {**ENV, **{k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        self.scratch = scratch
        self.live: set[subprocess.Popen] = set()
        scratch.mkdir(parents=True, exist_ok=True)
        probe = (
            "import sys,os,json;print(json.dumps({'exe':os.path.abspath(sys.executable),'roots':"
            "[sys.prefix,sys.base_prefix,sys.exec_prefix,sys.base_exec_prefix,"
            "os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))]"
            "+[p for p in sys.path[1:] if p]}))"
        )
        info = json.loads(
            subprocess.run([self.python, "-I", "-c", probe], capture_output=True, text=True, check=True).stdout
        )
        self.exe = info["exe"]  # unresolved on purpose: a venv finds its site-packages through this path
        roots = sorted(
            {os.path.realpath(p) for p in info["roots"] if os.path.isdir(p)}
            | {os.path.dirname(os.path.dirname(self.exe))}
        )
        self.py_roots = [
            r
            for r in roots
            if r != "/"
            and not any(r != o and r.startswith(o + "/") for o in roots)
            and not any(r == s or r.startswith(s + "/") for s in SYSTEM_RO)
        ]

    def episode(self, files: dict[str, Path]) -> "Episode":
        return Episode(self, files)

    def kill_all(self) -> None:
        for proc in list(self.live):
            _kill(proc)


class Episode:
    def __init__(self, sb: Sandbox, files: dict[str, Path]):
        self.sb = sb
        self.procs: set[subprocess.Popen] = set()
        self.dir = sb.scratch / f"ep-{uuid.uuid4().hex[:12]}"
        self.workdir = self.dir / "workdir"
        self.workdir.mkdir(parents=True)
        (self.dir / "copies").mkdir()
        args = ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--hostname", "sandbox"]
        if sb.net == "unshare":
            args.append("--share-net")
        for p in SYSTEM_RO:
            args += ["--ro-bind-try", p, p]
        args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home"]
        for p in sb.py_roots:
            args += ["--ro-bind", p, p]
        for name, src in sorted(files.items()):
            src = Path(src).resolve()
            if src.suffix.lower() in MUTABLE_SUFFIXES:
                shutil.copy2(src, self.dir / "copies" / name)
                args += ["--bind", str(self.dir / "copies" / name), f"/home/user/input/{name}"]
            else:
                args += ["--ro-bind", str(src), f"/home/user/input/{name}"]
        args += ["--bind", str(self.workdir), "/workdir", "--chdir", "/workdir"]
        env = {"PATH": f"{os.path.dirname(sb.exe)}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", **sb.env}
        for k, v in env.items():
            args += ["--setenv", k, v]
        self.args_file = self.dir / "bwrap.args"  # via --args FD: many input files overflow argv
        self.args_file.write_bytes(b"\0".join(a.encode() for a in args) + b"\0")

    def run(self, command: str, timeout: float) -> ExecResult:
        limits = (
            f"ulimit -v {self.sb.mem_kb} 2>/dev/null; ulimit -t {int(4 * timeout + 10)} 2>/dev/null; "
            "ulimit -f 524288 2>/dev/null; "
        )
        fd = os.open(self.args_file, os.O_RDONLY)
        argv = ["bwrap", "--args", str(fd), "--", "/bin/bash", "-c", limits + 'exec /bin/bash -c "$1"', "bash", command]
        if self.sb.net == "unshare":
            argv = ["unshare", "--user", "--map-root-user", "--net", "--", *argv]
        t0, timed_out, proc = time.time(), False, None
        try:
            proc = subprocess.Popen(
                argv,
                cwd=self.dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=(fd,),
            )
            self.sb.live.add(proc)
            self.procs.add(proc)
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
                out, err = proc.communicate()
        finally:
            os.close(fd)
            self.sb.live.discard(proc)
            self.procs.discard(proc)
        err_s = err.decode("utf-8", "replace")
        if err_s.startswith("bwrap:"):
            raise SandboxError(err_s.strip()[:400])
        return ExecResult(
            _cap(out.decode("utf-8", "replace")), _cap(err_s), proc.returncode, timed_out, round(time.time() - t0, 3)
        )

    def answer(self) -> str | None:
        p = self.workdir / "answer.txt"
        if p.is_symlink() or not p.is_file():
            return None
        return p.read_text(errors="replace")[:4096]

    def close(self) -> None:
        for proc in list(self.procs):
            _kill(proc)
        shutil.rmtree(self.dir, ignore_errors=True)


class LocalBackend:
    def __init__(self, sb: Sandbox, data_dir: str):
        self.sb, self.data_dir = sb, data_dir

    def check(self, rows: list[dict]) -> list[str]:
        return [r["task_id"] for r in rows if input_files(r, self.data_dir)[1]]

    def episode(self, row: dict) -> Episode:
        files, missing = input_files(row, self.data_dir)
        if missing:
            raise MissingInputs(f"missing input files: {missing}")
        return self.sb.episode(files)

    def kill_all(self) -> None:
        self.sb.kill_all()

    def close(self) -> None:
        shutil.rmtree(self.sb.scratch, ignore_errors=True)


# ── remote workers ───────────────────────────────────────────────────────────


class RemoteBackend:
    """Spreads episodes over sandbox_server.py workers; each episode stays on the worker that opened it."""

    def __init__(self, urls: list[str], token: str, slack: float = 600):
        if not token:
            raise SandboxError("sandbox.remote is set: put SANDBOX_TOKEN in the environment or .env")
        self.token, self.slack = token, slack
        self.lock = threading.Lock()
        self.slots: dict[str, int] = {}
        self.episodes: set[RemoteEpisode] = set()
        for url in (u.rstrip("/") for u in urls):
            try:
                health = self.request(url, "GET", "/health", timeout=15, retries=1)
            except SandboxError as exc:
                if "HTTP 401" in str(exc):
                    raise SandboxError(f"{url} rejected SANDBOX_TOKEN") from None
                print(f"warning: skipping unreachable worker {url}: {exc}", file=sys.stderr)
                continue
            if health.get("protocol") != PROTOCOL:
                raise SandboxError(f"{url} speaks {health.get('protocol')!r}, expected {PROTOCOL!r}: update the worker")
            self.slots[url] = max(1, int(health.get("slots", 1)))
        if not self.slots:
            raise SandboxError(f"no sandbox worker reachable: {urls}")
        self.urls = list(self.slots)
        self.open: dict[str, int] = dict.fromkeys(self.urls, 0)

    def request(self, url: str, method: str, path: str, body: dict | None = None, timeout: float = 60, retries=3):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url + path, data=data, method=method, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read() or b"{}")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code == 422:
                    raise MissingInputs(f"{url}: {detail}") from None
                if exc.code >= 500 and attempt < retries:
                    time.sleep(1 + attempt)
                    continue
                raise SandboxError(f"{url}{path}: HTTP {exc.code} {detail}") from None
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                if attempt < retries:
                    time.sleep(1 + 2 * attempt)
                    continue
                raise SandboxError(f"{url}{path}: {type(exc).__name__}: {exc}") from None

    def check(self, rows: list[dict]) -> list[str]:
        body = {
            "rows": [{"task_id": r["task_id"], "bucket_prefix": r["bucket_prefix"], "files": r["files"]} for r in rows]
        }
        missing = set()
        for url in self.urls:
            missing |= set(self.request(url, "POST", "/missing", body, timeout=300)["missing"])
        return sorted(missing)

    def episode(self, row: dict, url: str | None = None) -> "RemoteEpisode":
        with self.lock:
            url = url or min(self.urls, key=lambda u: self.open[u] / self.slots[u])
            self.open[url] += 1
        try:
            body = {"task_id": row["task_id"], "bucket_prefix": row["bucket_prefix"], "files": row["files"]}
            eid = self.request(url, "POST", "/episodes", body)["id"]
        except Exception:
            with self.lock:
                self.open[url] -= 1
            raise
        ep = RemoteEpisode(self, url, eid)
        with self.lock:
            self.episodes.add(ep)
        return ep

    def release(self, ep: "RemoteEpisode") -> None:
        with self.lock:
            if ep in self.episodes:
                self.episodes.discard(ep)
                self.open[ep.url] -= 1

    def kill_all(self) -> None:
        for ep in list(self.episodes):
            ep.close()

    def close(self) -> None:
        self.kill_all()


class RemoteEpisode:
    def __init__(self, backend: RemoteBackend, url: str, eid: str):
        self.backend, self.url, self.eid = backend, url, eid

    def run(self, command: str, timeout: float) -> ExecResult:
        # not retried: the command may already have run
        r = self.backend.request(
            self.url,
            "POST",
            f"/episodes/{self.eid}/run",
            {"command": command, "timeout": timeout},
            timeout=timeout + self.backend.slack + 30,
            retries=0,
        )
        return ExecResult(
            r["stdout"], r["stderr"], r["returncode"], r["timed_out"], r["seconds"], self.url, r.get("queue_seconds", 0)
        )

    def answer(self) -> str | None:
        return self.backend.request(self.url, "GET", f"/episodes/{self.eid}/answer")["answer"]

    def close(self) -> None:
        try:
            self.backend.request(self.url, "DELETE", f"/episodes/{self.eid}", timeout=15, retries=1)
        except SandboxError:
            pass  # the worker reaps idle episodes on its own
        finally:
            self.backend.release(self)
