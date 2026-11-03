# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.22", "pyyaml", "pandas", "pyarrow", "python-dotenv", "numpy", "scipy",
#                 "scikit-learn", "litellm>=1.50", "mcp"]
# ///
"""Download tasks and task files for the config's splits, then check the sandbox. Safe to re-run.

    uv run setup_env.py --config config.yaml [--splits eval,test] [--check-only]

Layout: <data.dir>/SmolDataEnvs/data/<split>-*.parquet and <data.dir>/task_files/<bucket_prefix>/<file>.
HF_TOKEN is read from the environment or .env.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_eval as ev  # noqa: E402
from sandbox import Sandbox, SandboxError  # noqa: E402

DATASET = "FineEnvs/SmolDataEnvs"
APPARMOR_FIX = """bwrap is blocked. On Ubuntu 23.10+ give it an AppArmor profile:

sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>
profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo apparmor_parser -r /etc/apparmor.d/bwrap
"""


def hf_retry(fn, attempts: int = 12):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            text = str(exc)
            limited = "429" in text or "rate limit" in text.lower()
            transient = any(s in text for s in ("502", "503", "504", "timed out", "Connection"))
            if not (limited or transient) or i == attempts - 1:
                raise
            m = re.search(r"Retry after (\d+) seconds", text)
            wait = int(m[1]) + 2 if m else min(30 * (i + 1), 300)
            print(f"  {'rate limited' if limited else 'transient error'}; retrying in {wait}s", flush=True)
            time.sleep(wait)


def download(data_dir: Path, splits: list[str], revision: str | None, workers: int) -> None:
    import pandas as pd
    from huggingface_hub import download_bucket_files, list_bucket_tree, snapshot_download

    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=revision,
        local_dir=data_dir / "SmolDataEnvs",
        allow_patterns=["data/*.parquet", "README.md"],
    )
    rows = [
        r
        for s in splits
        for p in sorted((data_dir / "SmolDataEnvs" / "data").glob(f"{s}-*.parquet"))
        for r in pd.read_parquet(p).to_dict(orient="records")
    ]
    prefixes = sorted({(r["hf_bucket"], r["bucket_prefix"]) for r in rows})
    print(f"tasks     {len(rows)} in {splits}, {len(prefixes)} data folders")

    cache_path = data_dir / "bucket_listing_cache.json"  # one API call per folder; listings are reused
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    lock = threading.Lock()

    def listing(key):
        ck = "|".join(key)
        if ck not in cache:
            items = hf_retry(lambda: list(list_bucket_tree(key[0], prefix=key[1].rstrip("/") + "/", recursive=True)))
            with lock:
                cache[ck] = {
                    Path(i.path).name: (i.path, getattr(i, "size", 0) or 0)
                    for i in items
                    if getattr(i, "type", None) == "file"
                }
                if len(cache) % 25 == 0:
                    cache_path.write_text(json.dumps(cache))
        return key, cache[ck]

    print(f"listing   {sum('|'.join(k) not in cache for k in prefixes)} folders ({len(cache)} cached)")
    try:
        with ThreadPoolExecutor(min(workers, 4)) as ex:
            remote = dict(ex.map(listing, prefixes))
    finally:
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))

    todo, absent, nbytes = defaultdict(list), set(), 0
    for r in rows:
        key = (r["hf_bucket"], r["bucket_prefix"])
        for name in [Path(str(f)).name for f in r["files"]] or sorted(remote[key]):
            dest = data_dir / "task_files" / r["bucket_prefix"] / name
            if dest.is_file():
                continue
            if name not in remote[key]:
                absent.add(f"{r['bucket_prefix']}/{name}")
                continue
            path, size = remote[key][name]
            if (path, dest) not in todo[key[0]]:
                todo[key[0]].append((path, dest))
                nbytes += size
    n = sum(map(len, todo.values()))
    print(
        f"files     {n} to download ({nbytes / 1e9:.2f} GB)" + (f"; {len(absent)} not in the bucket" if absent else "")
    )

    def fetch(job):
        bucket, files = job
        for _, dest in files:
            dest.parent.mkdir(parents=True, exist_ok=True)
        hf_retry(lambda: download_bucket_files(bucket, files=[(p, str(d) + ".part") for p, d in files]))
        for _, dest in files:
            os.replace(str(dest) + ".part", dest)  # never leave a partial file under the final name
        return len(files)

    batches = [(b, files[i : i + 50]) for b, files in todo.items() for i in range(0, len(files), 50)]
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for k in ex.map(fetch, batches):
            done += k
            print(f"  {done}/{n}", flush=True)


def check(runs: list[dict]) -> bool:
    ok = True
    for run in runs:
        try:
            rows = ev.load_tasks(run["data"])
        except SystemExit as exc:
            print(f"[!!] {run['name']}: {exc}")
            return False
        missing = sum(bool(ev.input_files(r, run["data"]["dir"])[1]) for r in rows)
        print(
            f"[{'ok' if not missing else '!!'}] {run['name']}: {len(rows)} tasks ({run['data']['split']}), "
            f"{missing} with missing input files"
        )
        ok &= not missing

    cfg = runs[0]
    python = cfg["sandbox"]["python"] or sys.executable
    subprocess.run(
        [
            python,
            "-c",
            "import importlib\nfor m in ('pandas','numpy','scipy.stats','sklearn.linear_model',"
            "'sklearn.ensemble'):\n try: importlib.import_module(m)\n except Exception: pass",
        ],
        capture_output=True,
        timeout=900,
    )  # compile bytecode once; the sandbox env is read-only
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.csv"
        probe.write_text("a\n1\n2\n")
        try:
            ep = Sandbox(python, cfg["sandbox"]["mem_gb"], Path(tmp) / "scratch").episode({"probe.csv": probe})
            res = ep.run(
                "python3 -c \"import pandas as pd; print(pd.read_csv('/home/user/input/probe.csv').a.sum())\"; "
                "python3 -c 'import pandas, numpy, scipy, sklearn' && echo packages-ok; "
                "(echo x > /home/user/input/probe.csv) 2>/dev/null || echo input-readonly; "
                "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\" 2>/dev/null "
                "&& echo NET-OPEN || echo net-blocked; echo -n 7 > /workdir/answer.txt",
                120,
            )
            answer = ep.answer()
            ep.close()
        except SandboxError as exc:
            print(f"[!!] sandbox: {exc}")
            if any(s in str(exc) for s in ("uid map", "RTM_NEWADDR", "Permission denied")):
                print(APPARMOR_FIX)
            return False
    lines = res.stdout.split()
    for want, label in [
        ("3", "reads task files"),
        ("packages-ok", "has pandas, numpy, scipy, sklearn"),
        ("input-readonly", "task files are read-only"),
        ("net-blocked", "no network"),
    ]:
        ok &= want in lines
        print(f"[{'ok' if want in lines else '!!'}] sandbox {label}")
    ok &= answer == "7"
    print(f"[{'ok' if answer == '7' else '!!'}] sandbox writes /workdir/answer.txt")
    if not ok and res.stderr.strip():
        print("    sandbox stderr:", res.stderr.strip()[-600:])
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--splits", help="comma list to stage (default: every split the config uses)")
    ap.add_argument("--revision", help="dataset revision to pin")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    runs = ev.load_runs(a.config, a.set)
    if not a.check_only:
        by_dir = defaultdict(list)
        for run in runs:
            if run["data"]["split"] not in by_dir[run["data"]["dir"]]:
                by_dir[run["data"]["dir"]].append(run["data"]["split"])
        for d, splits in by_dir.items():
            splits = a.splits.split(",") if a.splits else splits
            print(f"data dir {d}: splits {splits}")
            download(Path(d), splits, a.revision, a.workers)
    print()
    if not check(runs):
        sys.exit("\nsetup incomplete: fix the [!!] items above")
    print(f"\nready: uv run agent_eval.py --config {a.config}")


if __name__ == "__main__":
    main()
