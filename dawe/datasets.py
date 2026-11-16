"""Files that go to the GPU and come back.

  SFT file:   one row per training item  {"id", "messages": [system, user, assistant]}
  Eval file:  one row per BFCL item      {"id", "harness", "messages": [system, user]}
  Preds file: one row per BFCL item      {"id", "harness", "text"}   (written by the GPU job)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import bfcl
from .harness import Harness, parse_calls, render, target, to_bfcl


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sft_rows(items: list[bfcl.Item], h: Harness) -> list[dict]:
    rows = []
    for it in items:
        msgs, _ = render(h, it)
        rows.append({"id": it.id, "messages": msgs + [{"role": "assistant", "content": target(h, it)}]})
    return rows


def eval_rows(items: list[bfcl.Item], harnesses: list[Harness]) -> list[dict]:
    return [{"id": it.id, "harness": h.name, "messages": render(h, it)[0]} for h in harnesses for it in items]


def score(preds: list[dict], items: list[bfcl.Item]) -> dict:
    """Per harness: accuracy, format-failure rate, and accuracy per BFCL category."""
    by_id = {it.id: it for it in items}
    cache: dict[str, Harness] = {}
    tally: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # harness -> key -> [ok, n]
    fmt_fail: dict[str, int] = defaultdict(int)
    for p in preds:
        it = by_id.get(p["id"])
        if it is None:
            continue
        h = cache.setdefault(p["harness"], Harness.load(p["harness"]))
        _, names = render(h, it)
        calls = parse_calls(h, p["text"])
        ok = calls is not None and bfcl.check(to_bfcl(calls, names), it)
        fmt_fail[h.name] += calls is None
        for key in ("all", it.category):
            tally[h.name][key][0] += ok
            tally[h.name][key][1] += 1
    return {
        name: {
            "accuracy": t["all"][0] / t["all"][1],
            "n": t["all"][1],
            "format_fail": fmt_fail[name] / t["all"][1],
            "by_category": {k: round(v[0] / v[1], 4) for k, v in t.items() if k != "all"},
        }
        for name, t in sorted(tally.items())
    }


def pair_rows(items: list[bfcl.Item], a: Harness, b: Harness) -> list[dict]:
    """Paired traces: the same item and the same correct answer under two harnesses."""
    rows = []
    for it in items:
        ma, _ = render(a, it)
        mb, _ = render(b, it)
        rows.append({
            "id": it.id,
            "category": it.category,
            a.name: ma + [{"role": "assistant", "content": target(a, it)}],
            b.name: mb + [{"role": "assistant", "content": target(b, it)}],
        })
    return rows
