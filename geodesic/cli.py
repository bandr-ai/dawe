"""geodesic command line.

  geodesic fetch                        download BFCL into data/bfcl
  geodesic harnesses                    list harness versions
  geodesic render <harness> <item_id>   show the prompt and correct reply for one BFCL item
  geodesic roundtrip                    check every BFCL item survives render -> reply -> parse -> score
  geodesic build-sft                    write the SFT file (BFCL train split under one harness)
  geodesic build-eval                   write the eval prompts (BFCL test split x harnesses)
  geodesic build-pairs <a> <b>          paired traces: train items under harness a and b
  geodesic report                       one table with every eval in runs/
  geodesic score <preds.jsonl>          score model outputs per harness
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import bfcl
from .datasets import eval_rows, pair_rows, read_jsonl, score, sft_rows, write_jsonl
from .harness import Harness, available, parse_calls, render, target, to_bfcl


def _roundtrip(h: Harness, items: list[bfcl.Item]) -> tuple[int, list[str]]:
    ok, bad = 0, []
    for it in items:
        _, names = render(h, it)
        calls = parse_calls(h, target(h, it))
        if calls is not None and bfcl.check(to_bfcl(calls, names), it):
            ok += 1
        else:
            bad.append(it.id)
    return ok, bad


def report(items: list[bfcl.Item]) -> str:
    """All runs/preds_*.jsonl in one table. The shim row is the v1-trained model scored on v1:
    a perfect translation layer turns any of our harness changes back into v1."""
    rows = {}
    for f in sorted(Path("runs").glob("preds_*.jsonl")):
        rows[f.stem.removeprefix("preds_")] = score(read_jsonl(f), items)
    if "sft_v1" in rows:
        v1 = rows["sft_v1"]["v1"]
        rows["sft_v1 + shim"] = {h: v1 for h in rows["sft_v1"]}
    hs = available()
    out = ["all items (n=204)", f"{'model':22}" + "".join(f"{h:>11}" for h in hs)]
    for name, r in rows.items():
        out.append(f"{name:22}" + "".join(f"{r[h]['accuracy'] * 100:10.1f}%" if h in r else f"{'-':>11}" for h in hs))
    out += ["", "parallel items only (n=48)", f"{'model':22}" + "".join(f"{h:>11}" for h in hs)]
    for name, r in rows.items():
        out.append(
            f"{name:22}"
            + "".join(f"{r[h]['by_category'].get('parallel', 0) * 100:10.1f}%" if h in r else f"{'-':>11}" for h in hs)
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geodesic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("harnesses")
    r = sub.add_parser("render")
    r.add_argument("harness")
    r.add_argument("item_id")
    rt = sub.add_parser("roundtrip")
    rt.add_argument("--harness", nargs="*", default=None)
    s = sub.add_parser("build-sft")
    s.add_argument("--harness", default="v1")
    s.add_argument("--out", type=Path, default=None)
    e = sub.add_parser("build-eval")
    e.add_argument("--harness", nargs="*", default=None)
    e.add_argument("--out", type=Path, default=Path("data/eval/bfcl_test.jsonl"))
    pr = sub.add_parser("build-pairs")
    pr.add_argument("a")
    pr.add_argument("b")
    sub.add_parser("report")
    sc = sub.add_parser("score")
    sc.add_argument("preds", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        for p in bfcl.fetch():
            print(p)
        return 0
    if a.cmd == "harnesses":
        for n in available():
            print(f"{n:12} {Harness.load(n).description}")
        return 0
    items = bfcl.load_all()
    if a.cmd == "build-sft":
        train = [i for i in items if i.split == "train"]
        out = a.out or Path(f"data/sft/{a.harness}_bfcl_train.jsonl")
        write_jsonl(out, sft_rows(train, Harness.load(a.harness)))
        print(f"{out}  ({len(train)} rows)")
        return 0
    if a.cmd == "render":
        it = next(i for i in items if i.id == a.item_id)
        h = Harness.load(a.harness)
        msgs, _ = render(h, it)
        for m in msgs:
            print(f"--- {m['role']}\n{m['content']}")
        print(f"--- assistant (target)\n{target(h, it)}")
        return 0
    if a.cmd == "roundtrip":
        splits = Counter(i.split for i in items)
        print(f"{len(items)} items, split {dict(splits)}, skipped {bfcl.SKIPPED}")
        failed = False
        for n in a.harness or available():
            ok, bad = _roundtrip(Harness.load(n), items)
            print(f"{n:12} {ok}/{len(items)}" + (f"  failed: {bad[:5]}" if bad else ""))
            failed |= bool(bad)
        return 1 if failed else 0
    if a.cmd == "build-eval":
        hs = [Harness.load(n) for n in (a.harness or available())]
        test = [i for i in items if i.split == "test"]
        rows = eval_rows(test, hs)
        write_jsonl(a.out, rows)
        print(f"{a.out}  ({len(rows)} prompts = {len(test)} test items x {len(hs)} harnesses)")
        return 0
    if a.cmd == "build-pairs":
        train = [i for i in items if i.split == "train"]
        out = Path(f"data/pairs/{a.a}__{a.b}.jsonl")
        write_jsonl(out, pair_rows(train, Harness.load(a.a), Harness.load(a.b)))
        print(f"{out}  ({len(train)} pairs)")
        return 0
    if a.cmd == "report":
        print(report(items))
        return 0
    if a.cmd == "score":
        print(json.dumps(score(read_jsonl(a.preds), items), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
