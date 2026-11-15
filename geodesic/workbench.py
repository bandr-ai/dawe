"""Local preparation for skill-driven DAWE runs. No model calls or GPU dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path


SKILL = Path(__file__).resolve().parent / "_skills" / "dawe"
if not SKILL.exists():
    SKILL = Path(__file__).resolve().parent.parent / "skills" / "dawe"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_new_directory(out: Path, writer) -> None:
    out = out.resolve()
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dawe-", dir=out.parent) as tmp:
        staged = Path(tmp) / "output"
        staged.mkdir()
        writer(staged)
        # Do not replace a destination created while preparing the output.
        if out.exists():
            raise ValueError(f"Output already exists: {out}")
        staged.rename(out)


def install_skill(destination: Path) -> Path:
    out = destination.expanduser().resolve() / "dawe"
    write_new_directory(out, lambda p: shutil.copytree(SKILL, p, dirs_exist_ok=True))
    return out


def init_workspace(source: Path, model: str, out: Path, *, max_tries: int = 5,
                   budget_usd: float | None = None) -> Path:
    source = source.expanduser().resolve()
    out = out.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Source does not exist: {source}")
    if not model.strip():
        raise ValueError("Supply a checkpoint path or model identifier.")
    if max_tries < 1 or (budget_usd is not None and (not math.isfinite(budget_usd) or budget_usd <= 0)):
        raise ValueError("max-tries and an explicit budget must be positive.")
    if source == out or out in source.parents or (source.is_dir() and source in out.parents):
        raise ValueError("Keep the run directory separate from the source data.")
    config = {
        "schema_version": 1, "source": str(source), "model": model,
        "model_revision": None, "max_tries": max_tries, "budget_usd": budget_usd,
        "status": "needs_inspection", "compute_started": False,
    }

    def write(p):
        (p / "run.json").write_bytes(encoded(config))
        shutil.copytree(SKILL, p / "skill")
        (p / "AGENTS.md").write_text(
            "# DAWE run\n\nRead `skill/SKILL.md` and `run.json`. Inspect the supplied source and checkpoint, "
            "then carry out the DAWE workflow. This workspace contains no model weights.\n\n"
            "Prepare reviewed examples before spending compute. `budget_usd: null` means spending "
            "has not been authorized; an explicit zero-cost inspection can still proceed. "
            "Honor any budget already authorized in the conversation. Record the pinned checkpoint "
            "revision or content fingerprint before editing. Keep evaluation examples outside the "
            "editing directory. Stop GPU resources when finished or at the agreed limit.\n"
        )
        (p / ".gitignore").write_text("*\n!.gitignore\n")

    write_new_directory(out, write)
    return out


def read_examples(path: Path) -> list[dict]:
    rows = []
    ids = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("each example must be an object")
            for key in ("id", "group_id", "source_ref", "target_source"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{key} must be a nonempty string")
            if row["id"] in ids:
                raise ValueError("duplicate example id")
            if row.get("purpose") not in ("repair", "preserve"):
                raise ValueError("purpose must be repair or preserve")
            if row.get("split") not in (None, "development", "evaluation"):
                raise ValueError("split must be development or evaluation")
            msgs = row.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                raise ValueError("messages must contain a prompt and a final assistant target")
            for m in msgs:
                if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
                    raise ValueError("messages need system/user/assistant/tool roles")
                if not isinstance(m.get("content"), str):
                    raise ValueError("message content must be text; render tool schemas in your adapter")
            if msgs[-1]["role"] != "assistant" or not msgs[-1]["content"]:
                raise ValueError("the final message must be a nonempty reviewed assistant target")
            if not any(m["role"] == "user" for m in msgs[:-1]):
                raise ValueError("the prompt must contain a user message")
            decisions = row.get("decisions", [])
            if not isinstance(decisions, list):
                raise ValueError("decisions must be a list")
            for point in decisions:
                if (not isinstance(point, dict) or not isinstance(point.get("kind"), str)
                        or not point["kind"].strip() or not isinstance(point.get("prefix"), str)
                        or not msgs[-1]["content"].startswith(point["prefix"])):
                    raise ValueError("each decision needs a kind and an exact prefix of the assistant target")
            ids.add(row["id"])
            rows.append(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid example on line {line_no}: {exc}") from None
    if not rows:
        raise ValueError("No examples found.")
    return rows


def split_examples(rows: list[dict], evaluation_fraction: float = 0.2) -> tuple[list, list]:
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation-fraction must be between 0 and 1")
    groups: dict[str, list[dict]] = {}
    prompts: dict[str, str] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
        prompt = digest(encoded(row["messages"][:-1]))
        if prompt in prompts and prompts[prompt] != row["group_id"]:
            raise ValueError("An identical prompt occurs in different groups; group related attempts together.")
        prompts[prompt] = row["group_id"]
    if {r["purpose"] for r in rows} != {"repair", "preserve"}:
        raise ValueError("Supply both repair and preserve examples before freezing a run.")
    labels = {}
    for group, members in groups.items():
        explicit = {r["split"] for r in members if r.get("split")}
        if len(explicit) > 1:
            raise ValueError("A group is assigned to both development and evaluation.")
        labels[group] = next(iter(explicit), None)
    if any(label is not None for label in labels.values()):
        if any(label is None for label in labels.values()):
            raise ValueError("Explicit splits must cover every group.")
    else:
        # Stratify by the group's purposes; paired old/new and repeat attempts stay together.
        buckets: dict[tuple, list[str]] = {}
        for group, members in groups.items():
            key = tuple(sorted({r["purpose"] for r in members}))
            buckets.setdefault(key, []).append(group)
        for bucket in buckets.values():
            order = sorted(bucket, key=lambda g: digest(g.encode()))
            n = max(1, min(len(order) - 1, round(len(order) * evaluation_fraction)))
            for i, group in enumerate(order):
                labels[group] = "evaluation" if i < n else "development"
    development = [r for r in rows if labels[r["group_id"]] == "development"]
    evaluation = [r for r in rows if labels[r["group_id"]] == "evaluation"]
    for name, part in (("development", development), ("evaluation", evaluation)):
        if {r["purpose"] for r in part} != {"repair", "preserve"}:
            raise ValueError(f"{name} needs repair and preserve examples; add independent groups or explicit splits.")
    return development, evaluation


def freeze_examples(source: Path, development_dir: Path, evaluation_dir: Path,
                    evaluation_fraction: float = 0.2) -> dict:
    development_dir, evaluation_dir = development_dir.resolve(), evaluation_dir.resolve()
    if (development_dir == evaluation_dir or development_dir in evaluation_dir.parents
            or evaluation_dir in development_dir.parents):
        raise ValueError("Development and evaluation directories must be separate, not nested.")
    if development_dir.exists() or evaluation_dir.exists():
        raise ValueError("Freeze outputs must be new directories.")
    rows = read_examples(source)
    development, evaluation = split_examples(rows, evaluation_fraction)
    result = {}
    # Serialize canonically so identical data in a different source order freezes identically.
    for name, part, out in (("development", development, development_dir), ("evaluation", evaluation, evaluation_dir)):
        data = b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode()
                        for r in sorted(part, key=lambda r: r["id"]))
        meta = {"schema_version": 1, "split": name, "count": len(part), "sha256": digest(data),
                "groups": sorted({r["group_id"] for r in part}),
                "by_purpose": {purpose: sum(r["purpose"] == purpose for r in part)
                               for purpose in ("repair", "preserve")}}

        def write(p, data=data, meta=meta):
            (p / "examples.jsonl").write_bytes(data)
            (p / "manifest.json").write_bytes(encoded(meta))

        write_new_directory(out, write)
        result[name] = meta
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geodesic workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="prepare a skill workspace; never starts compute")
    init.add_argument("--source", type=Path, required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--max-tries", type=int, default=5)
    init.add_argument("--budget-usd", type=float)
    freeze = sub.add_parser("freeze", help="validate reviewed examples and freeze grouped development/evaluation files")
    freeze.add_argument("examples", type=Path)
    freeze.add_argument("--development-dir", type=Path, required=True)
    freeze.add_argument("--evaluation-dir", type=Path, required=True)
    freeze.add_argument("--evaluation-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(init_workspace(args.source, args.model, args.out, max_tries=args.max_tries, budget_usd=args.budget_usd))
        else:
            result = freeze_examples(args.examples, args.development_dir, args.evaluation_dir, args.evaluation_fraction)
            print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def skill_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="geodesic skill")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("path", help="print the bundled SKILL.md location")
    install = sub.add_parser("install", help="copy the skill into a chosen skills directory")
    install.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(SKILL / "SKILL.md" if args.command == "path" else install_skill(args.dest))
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
