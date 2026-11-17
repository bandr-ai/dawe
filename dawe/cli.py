"""dawe command line.

  dawe skill path | install --dest DIR    locate or install the DAWE skill
  dawe workbench init | freeze       prepare runs with user data, no GPU
  dawe fetch                        download BFCL (pinned revision, hash-checked) into data/bfcl
  dawe splits [--write]             check (or regenerate) splits/bfcl_v3_split.json
  dawe patch ...                    sparse weight patches: create | from-dense | inspect | apply | restore | verify
  dawe telemetry status|on|off      anonymous usage telemetry (see docs/TELEMETRY.md)
  dawe harnesses                    list harness versions
  dawe render <harness> <item_id>   show the prompt and correct reply for one BFCL item
  dawe roundtrip                    check every BFCL item survives render -> reply -> parse -> score
  dawe build-sft                    write the SFT file (BFCL train split under one harness)
  dawe build-eval                   write the eval prompts (BFCL test split x harnesses)
  dawe build-pairs <a> <b>          paired traces: train items under harness a and b
  dawe map-summary <map.json> [<map2.json>]   read a runs/map_*.json: is the harness change concentrated?
  dawe report                       one table with every eval in runs/
  dawe score <preds.jsonl>          score model outputs per harness
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import bfcl, splits, telemetry
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


def _load_harness(name: str) -> Harness:
    try:
        return Harness.load(name)
    except FileNotFoundError:
        raise SystemExit(f"error: unknown harness {name!r}; available: {', '.join(available())}") from None


def _read_preds(path: Path) -> list[dict]:
    try:
        return read_jsonl(path)
    except OSError as exc:
        raise SystemExit(f"error: {exc}") from None


def format_table(title: str, rows: dict[str, dict[str, float | None]], hs: list[str]) -> list[str]:
    """Accuracy table: one row per model, one column per harness (None -> not run)."""
    w = max(len(h) for h in hs) + 1
    out = [title, f"{'model':24}" + "".join(f"{h:>{w}}" for h in hs)]
    for name, r in rows.items():
        out.append(f"{name:24}" + "".join(f"{r[h] * 100:{w - 1}.1f}%" if r.get(h) is not None else f"{'-':>{w}}" for h in hs))
    return out


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
    test = [i for i in items if i.split == "test"]
    n_par = sum(i.category == "parallel" for i in test)
    overall = {n: {h: r[h]["accuracy"] for h in r} for n, r in rows.items()}
    parallel = {n: {h: r[h]["by_category"].get("parallel", 0) for h in r} for n, r in rows.items()}
    return "\n".join(
        format_table(f"all items (n={len(test)})", overall, hs) + [""]
        + format_table(f"parallel items only (n={n_par})", parallel, hs)
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "telemetry":
        return telemetry.main(argv[1:])
    milestone = telemetry.command_name(argv)
    if argv and argv[0] == "patch":  # no dataset needed; keeps the apply path free of BFCL files
        from . import patch

        return telemetry.run(milestone, patch.main, argv[1:])
    if argv and argv[0] in ("skill", "workbench"):
        from . import workbench

        return telemetry.run(milestone, workbench.skill_main if argv[0] == "skill" else workbench.main, argv[1:])
    ap = argparse.ArgumentParser(prog="dawe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("skill", help="path | install --dest DIR: bundled DAWE skill")
    sub.add_parser("workbench", help="init | freeze: prepare a DAWE run with your data")
    sub.add_parser("patch", help="create | from-dense | inspect | apply | restore | verify: sparse weight patches")
    sub.add_parser("telemetry", help="status | on | off | location on|off: anonymous usage telemetry")
    sub.add_parser("fetch")
    sp = sub.add_parser("splits")
    sp.add_argument("--write", action="store_true")
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
    ms = sub.add_parser("map-summary")
    ms.add_argument("maps", type=Path, nargs="+")
    sc = sub.add_parser("score")
    sc.add_argument("preds", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "map-summary":
        from . import mapping

        print(mapping.summarize(a.maps[0]))
        if len(a.maps) > 1:
            print("\n" + mapping.compare(a.maps[0], a.maps[1]))
        return 0
    if a.cmd == "fetch":
        for p in bfcl.fetch():
            print(p)
        return 0
    if a.cmd == "harnesses":
        for n in available():
            print(f"{n:12} {Harness.load(n).description}")
        return 0
    items = bfcl.load_all()
    if a.cmd == "splits":
        if a.write:
            print(splits.write(items))
            return 0
        problems = splits.check(items)
        print("split manifest matches" if not problems else "\n".join(problems))
        return 1 if problems else 0
    if a.cmd == "build-sft":
        train = [i for i in items if i.split == "train"]
        out = a.out or Path(f"data/sft/{a.harness}_bfcl_train.jsonl")
        write_jsonl(out, sft_rows(train, _load_harness(a.harness)))
        print(f"{out}  ({len(train)} rows)")
        return 0
    if a.cmd == "render":
        it = next((i for i in items if i.id == a.item_id), None)
        if it is None:
            print(f"error: unknown item id {a.item_id!r}", file=sys.stderr)
            return 1
        h = _load_harness(a.harness)
        msgs, _ = render(h, it)
        for m in msgs:
            print(f"--- {m['role']}\n{m['content']}")
        print(f"--- assistant (target)\n{target(h, it)}")
        return 0
    if a.cmd == "roundtrip":
        counts = Counter(i.split for i in items)
        print(f"{len(items)} items, split {dict(counts)}, skipped {bfcl.SKIPPED}")
        failed = False
        for n in a.harness or available():
            ok, bad = _roundtrip(_load_harness(n), items)
            print(f"{n:12} {ok}/{len(items)}" + (f"  failed: {bad[:5]}" if bad else ""))
            failed |= bool(bad)
        return 1 if failed else 0
    if a.cmd == "build-eval":
        hs = [_load_harness(n) for n in (a.harness or available())]
        test = [i for i in items if i.split == "test"]
        rows = eval_rows(test, hs)
        write_jsonl(a.out, rows)
        print(f"{a.out}  ({len(rows)} prompts = {len(test)} test items x {len(hs)} harnesses)")
        return 0
    if a.cmd == "build-pairs":
        train = [i for i in items if i.split == "train"]
        out = Path(f"data/pairs/{a.a}__{a.b}.jsonl")
        write_jsonl(out, pair_rows(train, _load_harness(a.a), _load_harness(a.b)))
        print(f"{out}  ({len(train)} pairs)")
        return 0
    if a.cmd == "report":
        print(report(items))
        return 0
    if a.cmd == "score":
        print(json.dumps(score(_read_preds(a.preds), items), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
