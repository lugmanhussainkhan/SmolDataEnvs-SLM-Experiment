# /// script
# requires-python = ">=3.10"
# dependencies = ["litellm>=1.50", "mcp", "python-dotenv", "pyyaml", "pandas", "pyarrow", "numpy", "scipy",
#                 "scikit-learn"]
# ///
"""Run a tool-calling model on SmolDataEnvs tasks in a sandbox and grade /workdir/answer.txt.

uv run agent_eval.py --config config.yaml
uv run agent_eval.py --config config.yaml --set run.trials=4 --steps test --resume
uv run agent_eval.py --config config.yaml --set run.trials=3 --train
uv run agent_eval.py --config config.yaml --set data.continue_from=stuck_states.jsonl --set run.trials=3 --train
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grader import grade as grade_answer  # noqa: E402
from sandbox import LocalBackend, MissingInputs, RemoteBackend, Sandbox, SandboxError, input_files  # noqa: E402,F401

DEFAULTS = {
    "model": {
        "name": None,  # LiteLLM model string, e.g. hosted_vllm/LiquidAI/LFM2.5-350M
        "args": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
        "api": "completion",
        "num_retries": 3,
        "timeout": 600,
    },
    "data": {"dir": "data", "split": "test", "tiers": [], "task_ids": None, "limit": 0, "continue_from": None},
    "agent": {"max_turns": 12, "exec_timeout": 90, "tool_output_chars": 8000, "system_prompt": None, "replay": True},
    "sandbox": {
        "python": None,
        "mem_gb": 16,
        "threads": 2,  # BLAS/OpenMP threads per command
        "net": "bwrap",
        "remote": [],  # sandbox_server.py URLs; empty = run locally. SANDBOX_TOKEN in env or .env
        "remote_slack": 600,  # seconds a call may wait in a worker's queue
    },
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
        if cfg["model"]["api"] not in ("completion", "responses"):
            sys.exit(f"config: model.api must be completion or responses, got {cfg['model']['api']!r}")
        runs.append(cfg)
    for field, values in (("name", [c["name"] for c in runs]), ("run.out", [c["run"]["out"] for c in runs])):
        if len(set(values)) != len(values):
            sys.exit(f"config: chain steps need distinct {field}, got {values}")
    return runs


# ── tasks ────────────────────────────────────────────────────────────────────


def load_tasks(data: dict) -> list[dict]:
    import pandas as pd

    split = "*" if data["continue_from"] else data["split"]
    paths = sorted((Path(data["dir"]) / "SmolDataEnvs" / "data").glob(f"{split}-*.parquet"))
    if not paths:
        sys.exit(f"no parquet for split {split!r} under {data['dir']}; run setup_env.py first")
    rows = pd.concat([pd.read_parquet(p) for p in paths]).to_dict(orient="records")
    for r in rows:
        r["files"] = [str(f) for f in r["files"]]
        r["id"] = r["task_id"]
    if data["continue_from"]:
        rows = load_states(data["continue_from"], rows)
    if data["tiers"]:
        rows = [r for r in rows if r["difficulty_tier"] in set(data["tiers"])]
    if ids := data["task_ids"]:
        ids = json.loads(Path(ids).read_text()) if isinstance(ids, str) else ids
        by_id = defaultdict(list)
        for r in rows:
            by_id[r["task_id"]].append(r)
        rows = [r for i in ids for r in by_id.get(i, [])]
    return rows[: data["limit"]] if data["limit"] else rows


def load_states(path: str, tasks: list[dict]) -> list[dict]:
    by_id = {r["task_id"]: r for r in tasks}
    rows, missing = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        st = json.loads(line)
        if st["task_id"] not in by_id:
            missing.append(st["task_id"])
            continue
        meta = {k: st.get(k) for k in ("trial", "group", "kind", "cut", "keep_calls", "student_answer")}
        rows.append(
            {**by_id[st["task_id"]], "id": f"{st['task_id']}/t{st['trial']}", "prefix": st["messages"],
             "continued_from": meta}
        )
    if missing:
        sys.exit(f"{path}: {len(missing)} states reference unknown task ids, e.g. {missing[:3]}")
    return rows


def grade(row: dict, answer: str | None) -> dict:
    if not (answer or "").strip():
        return {"reward": 0.0, "grade_method": "no_answer_file"}
    g = grade_answer(row["answer"], answer, reward_mode=row["reward_mode"], question=row["instruction"])
    return {"reward": g.reward, "grade_method": g.method}


# ── episode ──────────────────────────────────────────────────────────────────


def _digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode()).hexdigest()[:16]


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
    ex = {"returncode": res.returncode, "timed_out": res.timed_out, "seconds": res.seconds}
    if res.worker:
        ex.update(worker=res.worker, queue_seconds=res.queue_seconds)
    return text, ex


def _is_context_error(exc: Exception) -> bool:
    import litellm

    text = str(exc).lower()
    return (
        isinstance(exc, litellm.ContextWindowExceededError)
        or "maximum context length" in text
        or ("context length" in text and "exceed" in text)
    )


def replay_prefix(ep, prefix: list[dict], agent: dict) -> dict:
    saved = {m["tool_call_id"]: m["content"] for m in prefix if m["role"] == "tool"}
    t0, calls, mismatched = time.time(), 0, 0
    for m in prefix:
        for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            text, _ = run_tool(ep, tc["function"]["name"], tc["function"]["arguments"], agent)
            calls += 1
            mismatched += saved.get(tc["id"]) != text
    return {"calls": calls, "mismatched": mismatched, "seconds": round(time.time() - t0, 2)}


def _wire(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "train"} for m in messages]


RESPONSES_TOOL = {"type": "function", **BASH_TOOL["function"]}


def _plain(obj) -> dict:
    return obj.model_dump(mode="json", exclude_none=True) if hasattr(obj, "model_dump") else dict(obj)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text") or "" for p in content or [] if isinstance(p, dict))


def _to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    instructions, items = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            instructions.append(_text(m["content"]))
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": _text(m["content"])})
        elif role == "assistant":
            if text := _text(m.get("content")):
                items.append({"role": "assistant", "content": text})
            for tc in m.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                )
        else:
            items.append({"role": role, "content": m["content"]})
    return ("\n\n".join(instructions) or None), items


async def _call_completion(m: dict, messages: list[dict], args: dict) -> tuple[dict, str | None, dict | None]:
    import litellm

    resp = await litellm.acompletion(
        model=m["name"],
        messages=_wire(messages),
        tools=[BASH_TOOL],
        tool_choice="auto",
        num_retries=m["num_retries"],
        timeout=m["timeout"],
        **args,
    )
    choice, usage = resp.choices[0], resp.usage
    msg = choice.message
    assistant = msg.model_dump(exclude_none=True)
    assistant["content"] = msg.content or (None if assistant.get("thinking_blocks") else "")
    usage = usage.model_dump(mode="json", exclude_none=True) if usage else None
    return assistant, choice.finish_reason, usage


async def _call_responses(m: dict, messages: list[dict], args: dict) -> tuple[dict, str | None, dict | None]:
    import litellm

    args = dict(args)
    args.pop("seed", None)
    if "max_tokens" in args:
        args.setdefault("max_output_tokens", args.pop("max_tokens"))
    if (effort := args.pop("reasoning_effort", None)) is not None:
        args.setdefault("reasoning", {"effort": effort})
    instructions, items = _to_responses_input(messages)
    resp = await litellm.aresponses(
        model=m["name"],
        input=items,
        instructions=instructions,
        tools=[RESPONSES_TOOL],
        tool_choice="auto",
        num_retries=m["num_retries"],
        timeout=m["timeout"],
        **args,
    )
    text, reasoning, calls = [], [], []
    for item in map(_plain, resp.output or []):
        kind = item.get("type")
        if kind == "reasoning":
            parts = item.get("content") or item.get("summary") or []
            reasoning.append("".join(p.get("text") or "" for p in parts))
        elif kind == "message":
            text.append("".join(p.get("text") or p.get("refusal") or "" for p in item.get("content") or []))
        elif kind == "function_call":
            calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or ""},
                }
            )
    assistant = {"role": "assistant", "content": "".join(text)}
    if calls:
        assistant["tool_calls"] = calls
    if any(reasoning):
        assistant["reasoning_content"] = "\n\n".join(r for r in reasoning if r)
    details = _plain(resp.incomplete_details) if resp.incomplete_details else {}
    if resp.status == "incomplete":
        finish = "length" if "max_output_tokens" in str(details.get("reason")) else str(details.get("reason"))
    else:
        finish = "tool_calls" if calls else "stop"
    usage = _plain(resp.usage) if resp.usage else None
    if usage:
        usage.setdefault("prompt_tokens", usage.get("input_tokens", 0))
        usage.setdefault("completion_tokens", usage.get("output_tokens", 0))
    return assistant, finish, usage


async def run_episode(
    cfg: dict, row: dict, trial: int, backend, pool: ThreadPoolExecutor, attempt: int = 0
) -> dict:
    loop = asyncio.get_running_loop()
    m, agent = cfg["model"], cfg["agent"]
    prefix = row.get("prefix")
    rec = {
        "task_id": row["task_id"],
        **({"state": row["id"], "continued_from": row["continued_from"]} if prefix else {}),
        "trial": trial,
        "attempt": attempt,
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
    call = _call_responses if m["api"] == "responses" else _call_completion
    args = dict(m["args"])
    if (args.get("temperature") or 0) > 0:
        args.setdefault("seed", cfg["run"]["seed"] * 1000 + trial + 1_000_000 * attempt)
    if prefix:
        messages = copy.deepcopy(prefix)
        max_turns = agent["max_turns"] - sum(m["role"] == "assistant" for m in prefix)
    else:
        messages = [
            {"role": "system", "content": agent["system_prompt"] or SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
        ]
        max_turns = agent["max_turns"]
    t0 = time.time()
    try:
        ep = await loop.run_in_executor(pool, backend.episode, row)
    except SandboxError as exc:  # MissingInputs included
        rec["error"] = f"sandbox: {exc}"
        return rec
    rec["stop"] = "max_turns"
    try:
        if prefix and agent["replay"]:
            rec["replay"] = await loop.run_in_executor(pool, replay_prefix, ep, prefix, agent)
        for turn in range(max_turns):
            t_call = time.time()
            sent = (len(messages), _digest(messages))
            try:
                assistant, finish_reason, usage = await call(m, messages, args)
            except Exception as exc:
                if (turn > 0 or prefix) and _is_context_error(exc):
                    rec["stop"] = "context_full"
                    break
                raise
            usage = usage or {}
            calls = assistant.get("tool_calls") or []
            for tc in calls:
                tc["function"]["name"] = tc["function"].get("name") or ""
                tc["function"]["arguments"] = tc["function"].get("arguments") or ""
            if prefix:
                assistant["train"] = True
            messages.append(assistant)
            turn_rec = {
                "turn": turn,
                "finish_reason": finish_reason,
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0,
                "latency": round(time.time() - t_call, 3),
                "context_messages": sent[0],
                "context_digest": sent[1],
                "usage": usage or None,
                "executions": [],
            }
            if reasoning := assistant.get("reasoning_content"):
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
        rec["answer"] = await loop.run_in_executor(pool, ep.answer)
        rec.update(grade(row, rec["answer"]))
    except SandboxError as exc:
        rec["error"] = f"sandbox: {exc}"
    except Exception as exc:  # endpoint failed after retries: ungraded, never a silent 0
        rec["error"] = f"llm: {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        await loop.run_in_executor(pool, ep.close)
    rec["seconds"] = round(time.time() - t0, 2)
    rec["messages"] = messages
    rec["context_append_only"] = all(
        _digest(messages[: t["context_messages"]]) == t["context_digest"] for t in rec["turns"]
    )
    return rec


# ── summary ──────────────────────────────────────────────────────────────────


def score_block(results: list[dict], trials: int) -> dict:
    by_task = defaultdict(list)
    for r in results:
        if r["reward"] is not None:
            by_task[r.get("state") or r["task_id"]].append(r["reward"])
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


def worker_stats(results: list[dict]) -> dict:
    calls = defaultdict(list)
    for r in results:
        for t in r["turns"]:
            for ex in t["executions"]:
                if ex.get("worker"):
                    calls[ex["worker"]].append(ex)
    return {
        w: {
            "calls": len(v),
            "mean_seconds": round(sum(x["seconds"] for x in v) / len(v), 3),
            "mean_queue_seconds": round(sum(x["queue_seconds"] for x in v) / len(v), 3),
        }
        for w, v in sorted(calls.items())
    }


def summarize(cfg: dict, results: list[dict], wall: float, resumed: int) -> dict:
    graded = [r for r in results if r["reward"] is not None]
    mean = lambda key: round(sum(r[key] for r in graded) / len(graded), 2) if graded else None  # noqa: E731
    trials = cfg["run"]["trials"]
    continued = [r for r in results if r.get("continued_from")]
    return {
        "model": cfg["model"]["name"],
        **({"continue_from": cfg["data"]["continue_from"]} if continued else {}),
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
        **(
            {
                "by_kind": {
                    k: score_block([r for r in continued if str(r["continued_from"]["kind"]) == k], trials)
                    for k in sorted({str(r["continued_from"]["kind"]) for r in continued})
                },
                "by_cut": {
                    c: score_block([r for r in continued if str(r["continued_from"]["cut"]) == c], trials)
                    for c in sorted({str(r["continued_from"]["cut"]) for r in continued})
                },
                "replay_mismatched": sum(bool(r.get("replay", {}).get("mismatched")) for r in continued),
            }
            if continued
            else {}
        ),
        "stop_reasons": dict(Counter(r["stop"] for r in graded)),
        "grade_methods": dict(Counter(r["grade_method"] for r in graded)),
        "answered": sum(r["grade_method"] != "no_answer_file" for r in graded),
        "context_edited": sum(not r.get("context_append_only", True) for r in results),
        "errors": dict(Counter(r["error"].split(":")[0] for r in results if r["error"])),
        "mean_turns": mean("n_turns"),
        "mean_tool_calls": mean("n_tool_calls"),
        "mean_prompt_tokens": mean("prompt_tokens"),
        "mean_completion_tokens": mean("completion_tokens"),
        **({"sandbox": workers} if (workers := worker_stats(results)) else {}),
        "resumed_episodes": resumed,
        "wall_seconds": round(wall, 1),
    }


# ── run ──────────────────────────────────────────────────────────────────────


def make_backend(cfg: dict):
    sc = cfg["sandbox"]
    try:
        if sc["remote"]:
            return RemoteBackend(sc["remote"], os.environ.get("SANDBOX_TOKEN", ""), sc["remote_slack"])
        sandbox = Sandbox(
            sc["python"], sc["mem_gb"], Path(tempfile.mkdtemp(prefix="smol-agent-")), sc["threads"], sc["net"]
        )
        return LocalBackend(sandbox, cfg["data"]["dir"])
    except (SandboxError, subprocess.CalledProcessError) as exc:
        sys.exit(f"sandbox: {exc}")


def _key(rec: dict) -> tuple[str, int]:
    return rec.get("state") or rec["task_id"], rec["trial"]


def _finished(rec: dict) -> bool:
    return rec["error"] is None


def _clean(rec: dict) -> bool:
    return rec["error"] is None and (rec["reward"] or 0) >= 1


def _train_pending(results: list[dict], trials: int) -> int:
    by_task = defaultdict(list)
    for r in results:
        by_task[_key(r)[0]].append(r)
    return sum(not any(map(_clean, v)) and sum(map(_finished, v)) < trials for v in by_task.values())


def run_one(cfg: dict, resume: bool = False, train: bool = False) -> dict:
    rows = load_tasks(cfg["data"])
    out = Path(cfg["run"]["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial.jsonl")
    trials = cfg["run"]["trials"]
    wanted = {(r["id"], t) for r in rows for t in range(trials)}
    done, previous, attempts = {}, [], {}
    if resume:  # keep finished episodes from the output and the partial log; rerun the rest
        if out.is_file():
            previous = json.loads(out.read_text())["results"]
        if partial.is_file():
            previous += [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
        done = {
            _key(r): r
            for r in previous
            if _finished(r) and _key(r) in wanted
        }
        for r in previous:
            key = _key(r)
            attempts[key] = max(attempts.get(key, 0), r.get("attempt", 0) + 1)
    elif partial.is_file():
        partial.unlink()
    if train:
        solved = {k[0] for k, r in done.items() if _clean(r)}
        todo = [(r, t) for r in rows if r["id"] not in solved for t in range(trials) if (r["id"], t) not in done]
    else:
        todo = [(r, t) for r in rows for t in range(trials) if (r["id"], t) not in done]
    print(
        f"{cfg['model']['name']} on {cfg['data']['continue_from'] or cfg['data']['split']}: "
        f"{len(rows)} {'states' if cfg['data']['continue_from'] else 'tasks'} x "
        f"{f'up to {trials} trials until clean' if train else f'{trials} trials'} "
        f"({len(done)} resumed), concurrency {cfg['run']['concurrency']}"
    )

    backend = make_backend(cfg)
    if missing := backend.check([r for r, _ in todo]):
        backend.close()
        where = "the workers (run setup_env.py there)" if cfg["sandbox"]["remote"] else cfg["data"]["dir"]
        sys.exit(f"{len(missing)} tasks have input files missing on {where}, e.g. {missing[:3]}")

    async def run_all() -> list[dict]:
        pool = ThreadPoolExecutor(max(4, cfg["run"]["concurrency"]))
        sem = asyncio.Semaphore(cfg["run"]["concurrency"])
        n = 0

        async def one(row: dict, trial: int) -> dict:
            nonlocal n
            async with sem:
                rec = await run_episode(cfg, row, trial, backend, pool, attempts.get((row["id"], trial), 0))
            n += 1
            with partial.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            shown = rec["reward"] if rec["error"] is None else "ERR " + rec["error"][:60]
            print(
                f"[{n:>4}/{len(todo)}] {row['id'][:30]:32} {rec['tier']:6} t{trial} turns={rec['n_turns']:<2} "
                f"stop={str(rec['stop']):12} answer={str(rec['answer'])[:20]!r:24} "
                f"gold={rec['gold'][:16]!r:20} {shown}",
                flush=True,
            )
            return rec

        async def until_clean(row: dict, trials_left: list[int]) -> list[dict]:
            recs = []
            for t in trials_left:
                recs.append(await one(row, t))
                if _clean(recs[-1]):
                    break
            return recs

        try:
            if train:
                by_row = defaultdict(list)
                for r, t in todo:
                    by_row[r["id"]].append((r, t))
                groups = await asyncio.gather(
                    *(until_clean(v[0][0], [t for _, t in v]) for v in by_row.values())
                )
                return [rec for g in groups for rec in g]
            return await asyncio.gather(*(one(r, t) for r, t in todo))
        finally:
            pool.shutdown(wait=True)

    t0 = time.time()
    try:
        results = list(done.values()) + asyncio.run(run_all())
    except KeyboardInterrupt:
        backend.kill_all()
        saved = sum(1 for line in partial.read_text().splitlines() if line.strip()) if partial.is_file() else 0
        sys.exit(f"\ninterrupted: {saved} episodes saved in {partial}; rerun with --resume to continue")
    finally:
        backend.close()
    results.sort(key=_key)
    summary = summarize(cfg, results, time.time() - t0, len(done))
    if train:
        by_task = defaultdict(list)
        for r in results:
            by_task[_key(r)[0]].append(r)
        firsts = [min((r["trial"] for r in v if _clean(r)), default=None) for v in by_task.values()]
        summary["train"] = {
            "tasks": len(by_task),
            "clean": sum(f is not None for f in firsts),
            "clean_first_try": sum(f == 0 for f in firsts),
            "episodes": len(results),
        }
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
    ap.add_argument(
        "--train",
        action="store_true",
        help="trajectory generation: run.trials is a per-task budget; stop retrying a task once it scores 1",
    )
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
            prev = json.loads(out.read_text())
            summary = prev.get("summary") or {}
            results = prev.get("results", [])
            retry = _train_pending(results, cfg["run"]["trials"]) if a.train else sum(not _finished(r) for r in results)
            if not retry:
                print("    output exists, skipping")
                summaries.append((cfg["name"], summary))
                continue
            print(f"    output has {retry} {'unsolved tasks with budget left' if a.train else 'ungraded episodes'}, rerunning those")
        summaries.append((cfg["name"], run_one(cfg, resume, a.train)))

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
