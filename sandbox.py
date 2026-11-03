"""bubblewrap sandbox: task files read-only at /home/user/input, writable /workdir, no network."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

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
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
}


class SandboxError(RuntimeError):
    """The sandbox failed, not the model: the episode is left ungraded."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    seconds: float


def _cap(s: str) -> str:
    return s if len(s) <= STREAM_CAP else s[: STREAM_CAP // 2] + "\n...[truncated]...\n" + s[-STREAM_CAP // 2 :]


class Sandbox:
    def __init__(self, python: str | None, mem_gb: float, scratch: Path):
        if not shutil.which("bwrap"):
            raise SandboxError("bwrap not found: sudo apt install bubblewrap")
        self.python = python or sys.executable
        self.mem_kb = int(mem_gb * 1024 * 1024)
        self.scratch = scratch
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


class Episode:
    def __init__(self, sb: Sandbox, files: dict[str, Path]):
        self.sb = sb
        self.dir = sb.scratch / f"ep-{uuid.uuid4().hex[:12]}"
        self.workdir = self.dir / "workdir"
        self.workdir.mkdir(parents=True)
        (self.dir / "copies").mkdir()
        args = ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--hostname", "sandbox"]
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
        env = {"PATH": f"{os.path.dirname(sb.exe)}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", **ENV}
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
        t0, timed_out = time.time(), False
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
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                out, err = proc.communicate()
        finally:
            os.close(fd)
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
        shutil.rmtree(self.dir, ignore_errors=True)
