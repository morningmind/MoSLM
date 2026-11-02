"""Build the oracle table: run every expert on every question and record who gets it right.

This is the first experiment in the proposal. If the oracle (best expert per question)
is not clearly above the best single expert, routing has little room to help.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

from .expert import Expert

Scorer = Callable[[str, str], bool]  # (model_output, reference) -> correct?

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def score_last_number(output: str, reference: str) -> bool:
    """GSM8K-style: compare the last number in the output with the reference number."""
    nums = _NUM_RE.findall(output or "")
    ref = _NUM_RE.findall(reference or "")
    if not nums or not ref:
        return False
    try:
        return abs(float(nums[-1].replace(",", "")) - float(ref[-1].replace(",", ""))) < 1e-6
    except ValueError:
        return False


def score_choice_letter(output: str, reference: str) -> bool:
    """MMLU/ARC-style: first standalone A-E letter in the output matches the reference."""
    m = re.search(r"\b([A-E])\b", (output or "").upper())
    return bool(m) and m.group(1) == reference.strip().upper()[:1]


def score_exact(output: str, reference: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return norm(output) == norm(reference)


SCORERS: Dict[str, Scorer] = {
    "number": score_last_number,
    "choice": score_choice_letter,
    "exact": score_exact,
}


def build_oracle_table(
    items: Iterable[dict],
    experts: List[Expert],
    out_csv: str,
    scorers: Optional[Dict[str, Scorer]] = None,
    progress: bool = True,
) -> List[dict]:
    """Run every expert on every item and write one CSV row per (item, expert).

    Each item: {"id": str, "task": "gsm8k", "prompt": str, "reference": str, "scorer": "number"}
    `scorer` names an entry in SCORERS (or your own `scorers` dict), e.g. a code-execution
    scorer for MBPP that you supply yourself.
    """
    all_scorers = dict(SCORERS, **(scorers or {}))
    rows: List[dict] = []
    items = list(items)
    fields = ["question_id", "task", "expert", "correct", "prompt_tokens",
              "completion_tokens", "latency_ms", "cost", "error"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(items, 1):
            scorer = all_scorers[item.get("scorer", "exact")]
            for expert in experts:
                row = {"question_id": item["id"], "task": item.get("task", ""), "expert": expert.name}
                try:
                    resp = expert.run(item["prompt"])
                    out = resp.message.content or ""
                    row.update(
                        correct=int(scorer(out, item["reference"])),
                        prompt_tokens=resp.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens,
                        latency_ms=round(resp.latency_ms, 1),
                        cost=round(expert.cost(resp.usage.total_tokens), 8),
                        error="",
                    )
                except Exception as e:
                    row.update(correct=0, prompt_tokens=0, completion_tokens=0,
                               latency_ms=0, cost=0, error=str(e)[:200])
                writer.writerow(row)
                rows.append(row)
            f.flush()
            if progress and (i % 10 == 0 or i == len(items)):
                print(f"[oracle] {i}/{len(items)} questions done")
    return rows


def summarize_oracle(rows: List[dict]) -> dict:
    """Best single expert vs oracle accuracy, overall and per task."""
    by_q: Dict[str, Dict[str, int]] = defaultdict(dict)
    task_of: Dict[str, str] = {}
    for r in rows:
        by_q[r["question_id"]][r["expert"]] = int(r["correct"])
        task_of[r["question_id"]] = r["task"]

    experts = sorted({r["expert"] for r in rows})

    def stats(qids):
        n = len(qids) or 1
        per_expert = {e: sum(by_q[q].get(e, 0) for q in qids) / n for e in experts}
        oracle = sum(max(by_q[q].values()) for q in qids) / n
        best = max(per_expert, key=per_expert.get)
        return {
            "n": len(qids),
            "per_expert": per_expert,
            "best_expert": best,
            "best_single": per_expert[best],
            "oracle": oracle,
            "headroom_points": round((oracle - per_expert[best]) * 100, 1),
        }

    qids = list(by_q)
    result = {"overall": stats(qids), "by_task": {}}
    for task in sorted(set(task_of.values())):
        result["by_task"][task] = stats([q for q in qids if task_of[q] == task])
    return result
