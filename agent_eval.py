# /// script
# requires-python = ">=3.10"
# dependencies = ["litellm>=1.50", "mcp", "python-dotenv", "pyyaml", "pandas", "pyarrow", "numpy", "scipy",
#                 "scikit-learn"]
# ///
"""Run a tool-calling model on SmolDataEnvs tasks in a sandbox and grade /workdir/answer.txt.

uv run agent_eval.py --config config.yaml
uv run agent_eval.py --config config.yaml --set run.trials=4 --steps test --resume
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grader import grade as grade_answer  # noqa: E402
from sandbox import Sandbox, SandboxError  # noqa: E402

DEFAULTS = {
    "model": {
        "name": None,  # LiteLLM model string, e.g. hosted_vllm/LiquidAI/LFM2.5-350M
        "args": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
        "num_retries": 3,
        "timeout": 600,
    },
    "data": {"dir": "data", "split": "test", "tiers": [], "task_ids": None, "limit": 0},
    "agent": {"max_turns": 12, "exec_timeout": 90, "tool_output_chars": 8000, "system_prompt": None},
    "sandbox": {"python": None, "mem_gb": 16},
    "run": {"trials": 1, "concurrency": 16, "seed": 0, "out": "runs/eval.json", "skip_existing": False},
}

SYSTEM_PROMPT = (
    "You are an autonomous data-analysis agent operating in a sandboxed Linux container. Your only tool is "
    "`bash`. The dataset files are in /home/user/input/. Python 3 + pandas + numpy + scikit-learn + scipy are "
    "pre-installed.\n\nWork the problem step-by-step: first inspect the data (ls, head, shape, dtypes), then "
    "plan, then compute, then submit.\n\nTo submit your final answer you MUST call the `bash` tool to write it "
    'to /workdir/answer.txt, e.g. `echo -n "<value>" > /workdir/answer.txt`. Keep the answer short and '
    "concise. Do NOT end your turn without submitting."
)

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command in the sandbox and return its combined stdout+stderr. The shell is non-stateful "
            "between calls (each call is a fresh `bash -c`). Use it to explore files (ls, head, cat), run Python "
            '(`python3 -c ...`), and write the final answer (`echo -n "<value>" > /workdir/answer.txt`).'
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute."}},
            "required": ["command"],
        },
    },
}


# ── config ───────────────────────────────────────────────────────────────────


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_runs(path: str | None, sets: list[str]) -> list[dict]:
    """One run per `chain:` step (or one run). Precedence: DEFAULTS < config < step < --set."""
    raw = (yaml.safe_load(Path(path).read_text()) if path else None) or {}
    steps = raw.pop("chain", None) or [{}]
    runs = []
    for step in steps:
        cfg = _merge(_merge(DEFAULTS, raw), step)
        for item in sets:
            key, _, value = item.partition("=")
            *parents, leaf = key.strip().split(".")
            node = cfg
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = yaml.safe_load(value)
        cfg["name"] = step.get("name") or (cfg["data"]["split"] if len(steps) > 1 else "run")
        if not cfg["model"]["name"]:
            sys.exit("config: model.name is required")
        runs.append(cfg)
    for field, values in (("name", [c["name"] for c in runs]), ("run.out", [c["run"]["out"] for c in runs])):
        if len(set(values)) != len(values):
            sys.exit(f"config: chain steps need distinct {field}, got {values}")
    return runs


# ── tasks ────────────────────────────────────────────────────────────────────


def load_tasks(data: dict) -> list[dict]:
    import pandas as pd

    paths = sorted((Path(data["dir"]) / "SmolDataEnvs" / "data").glob(f"{data['split']}-*.parquet"))
    if not paths:
        sys.exit(f"no parquet for split {data['split']!r} under {data['dir']}; run setup_env.py first")
    rows = pd.concat([pd.read_parquet(p) for p in paths]).to_dict(orient="records")
    for r in rows:
        r["files"] = [str(f) for f in r["files"]]
    if data["tiers"]:
        rows = [r for r in rows if r["difficulty_tier"] in set(data["tiers"])]
    if ids := data["task_ids"]:
        ids = json.loads(Path(ids).read_text()) if isinstance(ids, str) else ids
        by_id = {r["task_id"]: r for r in rows}
        rows = [by_id[i] for i in ids if i in by_id]
    return rows[: data["limit"]] if data["limit"] else rows


def input_files(row: dict, data_dir: str) -> tuple[dict[str, Path], list[str]]:
    """({name: path} of the task's files, missing names). An empty `files` list means the whole prefix."""
    prefix = Path(data_dir) / "task_files" / row["bucket_prefix"]
    names = row["files"] or (sorted(p.name for p in prefix.iterdir() if p.is_file()) if prefix.is_dir() else [])
    found = {Path(n).name: prefix / Path(n).name for n in names}
    missing = [n for n, p in found.items() if not p.is_file()] or ([] if found else ["(no files staged)"])
    return {n: p for n, p in found.items() if p.is_file()}, missing


def grade(row: dict, answer: str | None) -> dict:
    if not (answer or "").strip():
        return {"reward": 0.0, "grade_method": "no_answer_file"}
    g = grade_answer(row["answer"], answer, reward_mode=row["reward_mode"], question=row["instruction"])
    return {"reward": g.reward, "grade_method": g.method}


# ── episode ──────────────────────────────────────────────────────────────────


def run_tool(ep, name: str, arguments: str, agent: dict) -> tuple[str, dict]:
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        return f"error: tool arguments are not valid JSON: {arguments[:200]}", {"error": "bad_json"}
    if name != "bash":
        return f"error: unknown tool {name!r}; the only tool is `bash`", {"error": "unknown_tool"}
    command = args.get("command") if isinstance(args, dict) else None
    if not isinstance(command, str):
        return "error: bash requires a string `command`", {"error": "missing_command"}
    res = ep.run(command, agent["exec_timeout"])
    text = res.stdout + res.stderr + (f"\n[killed: exceeded {res.seconds:.0f}s]" if res.timed_out else "")
    text = text[: agent["tool_output_chars"]] if text.strip() else f"(empty output, rc={res.returncode})"
    return text, {"returncode": res.returncode, "timed_out": res.timed_out, "seconds": res.seconds}


def _is_context_error(exc: Exception) -> bool:
    import litellm

    text = str(exc).lower()
    return (
        isinstance(exc, litellm.ContextWindowExceededError)
        or "maximum context length" in text
        or ("context length" in text and "exceed" in text)
    )


async def run_episode(cfg: dict, row: dict, trial: int, sb: Sandbox, pool: ThreadPoolExecutor) -> dict:
    import litellm

    loop = asyncio.get_running_loop()
    m, agent = cfg["model"], cfg["agent"]
    rec = {
        "task_id": row["task_id"],
        "trial": trial,
        "tier": row["difficulty_tier"],
        "reward_mode": row["reward_mode"],
        "gold": row["answer"],
        "answer": None,
        "reward": None,
        "grade_method": None,
        "error": None,
        "stop": None,
        "n_turns": 0,
        "n_tool_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "seconds": 0.0,
        "turns": [],
        "messages": [],
    }
    files, missing = input_files(row, cfg["data"]["dir"])
    if missing:
        rec["error"] = f"missing input files: {missing}"
        return rec

    args = dict(m["args"])
    if args.get("temperature", 0) > 0:
        args.setdefault("seed", cfg["run"]["seed"] * 1000 + trial)
    messages = [
        {"role": "system", "content": agent["system_prompt"] or SYSTEM_PROMPT},
        {"role": "user", "content": row["instruction"]},
    ]
    t0 = time.time()
    ep = await loop.run_in_executor(pool, sb.episode, files)
    rec["stop"] = "max_turns"
    try:
        for turn in range(agent["max_turns"]):
            t_call = time.time()
            try:
                resp = await litellm.acompletion(
                    model=m["name"],
                    messages=messages,
                    tools=[BASH_TOOL],
                    tool_choice="auto",
                    num_retries=m["num_retries"],
                    timeout=m["timeout"],
                    **args,
                )
            except Exception as exc:
                if turn > 0 and _is_context_error(exc):
                    rec["stop"] = "context_full"
                    break
                raise
            choice, usage = resp.choices[0], resp.usage
            msg = choice.message
            calls = [
                {
                    "id": tc.id or f"call_{turn}_{i}",
                    "type": "function",
                    "function": {"name": tc.function.name or "", "arguments": tc.function.arguments or ""},
                }
                for i, tc in enumerate(msg.tool_calls or [])
            ]
            messages.append(
                {"role": "assistant", "content": msg.content or "", **({"tool_calls": calls} if calls else {})}
            )
            turn_rec = {
                "turn": turn,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "latency": round(time.time() - t_call, 3),
                "executions": [],
            }
            if reasoning := getattr(msg, "reasoning_content", None):
                turn_rec["reasoning"] = reasoning
            rec["prompt_tokens"] += turn_rec["prompt_tokens"]
            rec["completion_tokens"] += turn_rec["completion_tokens"]
            rec["n_turns"] += 1
            rec["turns"].append(turn_rec)
            if not calls:
                rec["stop"] = "no_tool_call"
                break
            for tc in calls:
                rec["n_tool_calls"] += 1
                text, ex = await loop.run_in_executor(
                    pool, run_tool, ep, tc["function"]["name"], tc["function"]["arguments"], agent
                )
                turn_rec["executions"].append({"tool_call_id": tc["id"], **ex})
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": text}
                )
        rec["answer"] = ep.answer()
        rec.update(grade(row, rec["answer"]))
    except SandboxError as exc:
        rec["error"] = f"sandbox: {exc}"
    except Exception as exc:  # endpoint failed after retries: ungraded, never a silent 0
        rec["error"] = f"llm: {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        await loop.run_in_executor(pool, ep.close)
    rec["seconds"] = round(time.time() - t0, 2)
    rec["messages"] = messages
    return rec


# ── summary ──────────────────────────────────────────────────────────────────


def score_block(results: list[dict], trials: int) -> dict:
    by_task = defaultdict(list)
    for r in results:
        if r["reward"] is not None:
            by_task[r["task_id"]].append(r["reward"])
    tasks = list(by_task.values())
    out = {"tasks": len(tasks), "pass@1": round(sum(sum(v) / len(v) for v in tasks) / len(tasks), 4) if tasks else None}
    k = 2
    while k <= trials:
        vals = [
            1.0
            if len(v) - sum(x >= 1 for x in v) < k
            else 1 - math.comb(len(v) - sum(x >= 1 for x in v), k) / math.comb(len(v), k)
            for v in tasks
            if len(v) >= k
        ]
        out[f"pass@{k}"] = round(sum(vals) / len(vals), 4) if vals else None
        k *= 2
    return out


def summarize(cfg: dict, results: list[dict], wall: float, resumed: int) -> dict:
    graded = [r for r in results if r["reward"] is not None]
    mean = lambda key: round(sum(r[key] for r in graded) / len(graded), 2) if graded else None  # noqa: E731
    trials = cfg["run"]["trials"]
    return {
        "model": cfg["model"]["name"],
        "split": cfg["data"]["split"],
        "episodes": len(results),
        "graded": len(graded),
        "ungraded": len(results) - len(graded),
        **score_block(results, trials),
        "by_tier": {
            t: score_block([r for r in results if r["tier"] == t], trials) for t in sorted({r["tier"] for r in results})
        },
        "by_mode": {
            m: score_block([r for r in results if r["reward_mode"] == m], trials)
            for m in sorted({r["reward_mode"] for r in results})
        },
        "stop_reasons": dict(Counter(r["stop"] for r in graded)),
        "grade_methods": dict(Counter(r["grade_method"] for r in graded)),
        "answered": sum(r["grade_method"] != "no_answer_file" for r in graded),
        "errors": dict(Counter(r["error"].split(":")[0] for r in results if r["error"])),
        "mean_turns": mean("n_turns"),
        "mean_tool_calls": mean("n_tool_calls"),
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
        "resumed_episodes": resumed,
        "wall_seconds": round(wall, 1),
    }


# ── run ──────────────────────────────────────────────────────────────────────


def run_one(cfg: dict, resume: bool = False) -> dict:
    rows = load_tasks(cfg["data"])
    out = Path(cfg["run"]["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial.jsonl")
    trials = cfg["run"]["trials"]
    wanted = {(r["task_id"], t) for r in rows for t in range(trials)}
    done, previous = {}, []
    if resume:  # keep finished episodes from the output and the partial log; rerun errored ones
        if out.is_file():
            previous = json.loads(out.read_text())["results"]
        if partial.is_file():
            previous += [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
        done = {
            (r["task_id"], r["trial"]): r
            for r in previous
            if r["error"] is None and (r["task_id"], r["trial"]) in wanted
        }
    elif partial.is_file():
        partial.unlink()
    todo = [(r, t) for r in rows for t in range(trials) if (r["task_id"], t) not in done]
    print(
        f"{cfg['model']['name']} on {cfg['data']['split']}: {len(rows)} tasks x {trials} trials "
        f"({len(done)} resumed), concurrency {cfg['run']['concurrency']}"
    )

    scratch = Path(tempfile.mkdtemp(prefix="smol-agent-"))
    try:
        sb = Sandbox(cfg["sandbox"]["python"], cfg["sandbox"]["mem_gb"], scratch)
    except Exception as exc:
        shutil.rmtree(scratch, ignore_errors=True)
        sys.exit(f"sandbox: {exc}")

    async def run_all() -> list[dict]:
        pool = ThreadPoolExecutor(max(4, cfg["run"]["concurrency"]))
        sem = asyncio.Semaphore(cfg["run"]["concurrency"])
        n = 0

        async def one(row: dict, trial: int) -> dict:
            nonlocal n
            async with sem:
                rec = await run_episode(cfg, row, trial, sb, pool)
            n += 1
            with partial.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            shown = rec["reward"] if rec["error"] is None else "ERR " + rec["error"][:60]
            print(
                f"[{n:>4}/{len(todo)}] {rec['task_id'][:26]:28} {rec['tier']:6} t{trial} turns={rec['n_turns']:<2} "
                f"stop={str(rec['stop']):12} answer={str(rec['answer'])[:20]!r:24} "
                f"gold={rec['gold'][:16]!r:20} {shown}",
                flush=True,
            )
            return rec

        try:
            return await asyncio.gather(*(one(r, t) for r, t in todo))
        finally:
            pool.shutdown(wait=True)

    t0 = time.time()
    try:
        results = list(done.values()) + asyncio.run(run_all())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    results.sort(key=lambda r: (r["task_id"], r["trial"]))
    summary = summarize(cfg, results, time.time() - t0, len(done))
    shown = copy.deepcopy(cfg)
    for k in shown["model"]["args"]:
        if "key" in k.lower() or ("token" in k.lower() and k != "max_tokens"):
            shown["model"]["args"][k] = "***"
    out.write_text(json.dumps({"config": shown, "summary": summary, "results": results}, indent=1))
    partial.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_mode"}, indent=2))
    print(f"wrote {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--steps", help="comma list of chain step names to run")
    ap.add_argument("--resume", action="store_true", help="skip finished steps, continue partial ones")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    import litellm

    litellm.suppress_debug_info = True

    runs = load_runs(a.config, a.set)
    if a.steps:
        wanted = {s.strip() for s in a.steps.split(",") if s.strip()}
        if unknown := wanted - {c["name"] for c in runs}:
            sys.exit(f"--steps: unknown {sorted(unknown)}; chain has {[c['name'] for c in runs]}")
        runs = [c for c in runs if c["name"] in wanted]

    summaries = []
    for i, cfg in enumerate(runs, 1):
        out = Path(cfg["run"]["out"])
        print(f"\n=== [{i}/{len(runs)}] {cfg['name']}: split={cfg['data']['split']} -> {out}")
        resume = a.resume or cfg["run"]["skip_existing"]
        if resume and out.is_file():
            summary = json.loads(out.read_text()).get("summary") or {}
            if not summary.get("ungraded"):
                print("    output exists, skipping")
                summaries.append((cfg["name"], summary))
                continue
            print(f"    output has {summary['ungraded']} ungraded episodes, rerunning those")
        summaries.append((cfg["name"], run_one(cfg, resume)))

    if len(summaries) > 1:
        print("\n=== chain summary")
        for name, s in summaries:
            if not s:
                print(f"  {name:12} (no summary)")
                continue
            ks = " ".join(f"{k}={s[k]}" for k in s if k.startswith("pass@"))
            tiers = " ".join(f"{t}={v.get('pass@1')}" for t, v in s.get("by_tier", {}).items())
            print(f"  {name:12} {s['split']:6} {ks}  [{tiers}]  graded={s['graded']}/{s['episodes']}")


if __name__ == "__main__":
    main()
