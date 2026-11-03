"""Strict short-answer grader for SmolDataEnvs.

Rules: the answer must be one value on one line (<= 200 chars). Numeric golds need a single number
that rounds to the gold's precision (capped at 4 dp); x100 / /100 only when the question is about
a percent or a rate. Text golds need a case-insensitive exact match. Lists match item by item in
any order. yes/true/y and no/false/n are equivalent.

    python grader.py runs/a.json runs/b.json [--csv rescored.csv]   # re-grade saved eval runs
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_LEN = 200
_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_THOUSANDS = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_FRACTION = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
_SCALED_QUESTION = re.compile(
    r"percent|%|\b(accuracy|precision|recall|f1|f1-score|score|rate|proportion|share|ratio)\b", re.I
)
_BOOL = {"yes": "yes", "true": "yes", "y": "yes", "no": "no", "false": "no", "n": "no"}


@dataclass
class GradeResult:
    reward: float
    method: str  # exact | numeric | numeric_rounded | numeric_scaled | list | bool | miss | invalid_format


def _clean(s) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().strip("`\"'").strip())
    return s.rstrip(".").strip()


def parse_number(s) -> tuple[float | None, bool, int]:
    """(value, had_percent_sign, decimals) if `s` is exactly one number, else (None, ..)."""
    t = _clean(s).replace("$", "").strip()
    pct = t.endswith("%")
    t = t.rstrip("%").strip()
    if _THOUSANDS.fullmatch(t):
        t = t.replace(",", "")
    if (m := _FRACTION.fullmatch(t)) and float(m[2]):
        return float(m[1]) / float(m[2]), pct, 6
    if not _NUM.fullmatch(t):
        return None, pct, 0
    decimals = 6 if "e" in t.lower() else len(t.split(".")[1]) if "." in t else 0
    return float(t), pct, decimals


def _numeric_method(cand: str, gold: str, question: str) -> str | None:
    c, c_pct, c_dec = parse_number(cand)
    g, _, g_dec = parse_number(gold)
    if c is None or g is None:
        return None
    tol = 0.5 * 10 ** -min(g_dec, 4)
    if abs(c - g) <= tol + 1e-9:
        return "numeric"
    if min(g_dec, 2) <= c_dec < g_dec and abs(c - g) <= 0.5 * 10**-c_dec + 1e-9:
        return "numeric_rounded"
    scaled_tol = 0.5 * 10 ** -max(min(g_dec, 4), 1)
    if (
        c
        and g
        and (c_pct or _SCALED_QUESTION.search(question))
        and any(abs(x - g) <= scaled_tol + 1e-9 for x in (c * 100, c / 100))
    ):
        return "numeric_scaled"
    return None


def _item_match(cand: str, gold: str, question: str) -> bool:
    if _clean(cand).lower() == _clean(gold).lower():
        return True
    return parse_number(gold)[0] is not None and _numeric_method(cand, gold, question) is not None


def question_text(instruction: str) -> str:
    """The question inside a SmolDataEnvs instruction, without the answer-format boilerplate."""
    m = re.search(r"Question:\s*\n(.*?)\n\s*\n", instruction or "", re.S)
    return m[1].strip() if m else (instruction or "")


def grade(gold, candidate, *, reward_mode: str = "", question: str = "") -> GradeResult:
    cand, gold = ("" if candidate is None else str(candidate)).strip(), str(gold).strip()
    question = question_text(question)
    if not cand:
        return GradeResult(0.0, "miss")
    if "\n" in cand or len(cand) > MAX_LEN:
        return GradeResult(0.0, "invalid_format")
    c, g = _clean(cand).lower(), _clean(gold).lower()
    if c == g:
        return GradeResult(1.0, "exact")
    if reward_mode == "exact_bool" or g in _BOOL:
        ok = _BOOL.get(c, "?c") == _BOOL.get(g, "?g")
        return GradeResult(float(ok), "bool" if ok else "miss")
    if parse_number(gold)[0] is not None:
        if parse_number(cand)[0] is None:
            parts = [p for p in cand.split(",") if p.strip()]
            if len(parts) > 1 and all(_numeric_method(p, gold, question) for p in parts):
                return GradeResult(1.0, "list")
            return GradeResult(0.0, "invalid_format")
        method = _numeric_method(cand, gold, question)
        return GradeResult(1.0, method) if method else GradeResult(0.0, "miss")
    gold_items = [p.strip() for p in gold.split(",") if p.strip()]
    cand_items = [p.strip() for p in cand.split(",") if p.strip()]
    if len(gold_items) >= 2 and len(cand_items) == len(gold_items):
        for item in gold_items:
            hit = next((x for x in cand_items if _item_match(x, item, question)), None)
            if hit is None:
                break
            cand_items.remove(hit)
        else:
            return GradeResult(1.0, "list")
    return GradeResult(0.0, "miss")


def _rescore(paths: list[str], csv_path: str | None) -> None:
    import csv
    import json
    from collections import defaultdict

    rows = []
    for path in paths:
        data = json.load(open(path))
        old, new, tiers = defaultdict(list), defaultdict(list), {}
        for r in data["results"]:
            if r.get("reward") is None:
                continue
            question = r["messages"][1]["content"] if len(r.get("messages", [])) > 1 else ""
            g = grade(r["gold"], r["answer"], reward_mode=r["reward_mode"], question=question)
            old[r["task_id"]].append(r["reward"])
            new[r["task_id"]].append(g.reward)
            tiers[r["task_id"]] = r["tier"]
            rows.append(
                [
                    path,
                    r["task_id"],
                    r["trial"],
                    r["tier"],
                    r["reward_mode"],
                    r["gold"],
                    r["answer"],
                    r["reward"],
                    g.reward,
                    g.method,
                ]
            )
        p1 = lambda d, ts: sum(sum(d[t]) / len(d[t]) for t in ts) / max(1, len(ts))  # noqa: E731
        by_tier = {t: p1(new, [k for k in new if tiers[k] == t]) for t in sorted(set(tiers.values()))}
        print(
            f"{path}: logged pass@1={p1(old, list(old)):.3f}  strict pass@1={p1(new, list(new)):.3f}  "
            + "  ".join(f"{t}={v:.3f}" for t, v in by_tier.items())
        )
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["file", "task_id", "trial", "tier", "reward_mode", "gold", "answer", "logged", "strict", "method"]
            )
            w.writerows(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    _rescore(a.runs, a.csv)
