"""BFCL (Berkeley Function Calling Leaderboard) items and a call checker.

Items are kept in BFCL's own names. Harnesses rename things on the way in and
map them back on the way out, so scoring always happens in BFCL's namespace.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HF_BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
CATEGORIES = ("simple", "multiple", "parallel")
DEFAULT_DATA_DIR = Path("data/bfcl")

Call = dict[str, Any]  # {"name": str, "arguments": dict}


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    question: str  # the single user turn
    functions: list[dict]  # BFCL function schemas
    ground_truth: list[dict]  # [{fn_name: {arg: [allowed values]}}]

    @property
    def split(self) -> str:
        """Stable train/test split by id hash, 75/25 like Deletion Tuning's SFT setup."""
        h = int(hashlib.sha256(self.id.encode()).hexdigest(), 16) % 100
        return "test" if h < 25 else "train"


def fetch(data_dir: Path = DEFAULT_DATA_DIR) -> list[Path]:
    """Download the BFCL v3 files we use. Skips files that already exist."""
    out = []
    for cat in CATEGORIES:
        for rel in (f"BFCL_v3_{cat}.json", f"possible_answer/BFCL_v3_{cat}.json"):
            dest = data_dir / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{HF_BASE}/{rel}", dest)
            out.append(dest)
    return out


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(category: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[Item]:
    items = _jsonl(data_dir / f"BFCL_v3_{category}.json")
    answers = {a["id"]: a["ground_truth"] for a in _jsonl(data_dir / f"possible_answer/BFCL_v3_{category}.json")}
    out = []
    for it in items:
        turns = it["question"]
        if len(turns) != 1 or len(turns[0]) != 1:
            raise ValueError(f"{it['id']}: expected one single-message turn")
        item = Item(it["id"], category, turns[0][0]["content"], it["function"], answers[it["id"]])
        if problem := _label_problem(item):
            SKIPPED[item.id] = problem
            continue
        out.append(item)
    return out


SKIPPED: dict[str, str] = {}  # item id -> why it was left out


def _label_problem(item: Item) -> str | None:
    """BFCL has a few answers that don't match their own schema. We skip those, never repair them."""
    schemas = {f["name"]: f for f in item.functions}
    for entry in item.ground_truth:
        for name, args in entry.items():
            if name not in schemas:
                return f"answer calls {name!r}, which the item does not offer"
            extra = set(args) - set(schemas[name]["parameters"].get("properties", {}))
            if extra:
                return f"answer uses unknown arguments {sorted(extra)}"
    return None


def load_all(data_dir: Path = DEFAULT_DATA_DIR) -> list[Item]:
    return [it for cat in CATEGORIES for it in load(cat, data_dir)]


# --- gold calls -------------------------------------------------------------


def gold_calls(item: Item) -> list[Call]:
    """One concrete correct answer: the first allowed value for each argument.

    An argument whose only allowed value is "" (meaning "may be omitted") is left out.
    """
    calls = []
    for entry in item.ground_truth:
        ((name, args),) = entry.items()
        concrete = {}
        for arg, allowed in args.items():
            vals = [v for v in allowed if v != ""]
            if vals:
                concrete[arg] = _concrete(vals[0])
        calls.append({"name": name, "arguments": concrete})
    return calls


def _concrete(v: Any) -> Any:
    # Nested dicts in BFCL answers also hold lists of allowed values.
    if isinstance(v, dict):
        out = {}
        for k, allowed in v.items():
            if isinstance(allowed, list):
                vals = [x for x in allowed if x != ""]
                if vals:
                    out[k] = _concrete(vals[0])
            else:
                out[k] = allowed
        return out
    if isinstance(v, list):
        return [_concrete(x) for x in v]
    return v


# --- checker ------------------------------------------------------------------


def _norm_str(s: str) -> str:
    return " ".join(s.lower().replace("'", "").replace('"', "").split())


def _value_ok(got: Any, allowed: list) -> bool:
    return any(_eq(got, a) for a in allowed if a != "")


def _eq(got: Any, want: Any) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return type(got) is type(want) and got == want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return float(got) == float(want)
    if isinstance(want, str) and isinstance(got, str):
        return _norm_str(got) == _norm_str(want)
    if isinstance(want, list) and isinstance(got, (list, tuple)):
        return len(got) == len(want) and all(_eq(g, w) for g, w in zip(got, want))
    if isinstance(want, dict) and isinstance(got, dict):
        # nested BFCL dicts: each key maps to a list of allowed values
        for k, allowed in want.items():
            allowed = allowed if isinstance(allowed, list) else [allowed]
            if k not in got:
                if "" in allowed:
                    continue
                return False
            if not _value_ok(got[k], allowed):
                return False
        return set(got) <= set(want)
    return got == want


def _call_ok(call: Call, entry: dict) -> bool:
    ((name, args),) = entry.items()
    if call.get("name") != name:
        return False
    got = call.get("arguments") or {}
    if set(got) - set(args):
        return False  # unknown argument
    for arg, allowed in args.items():
        if arg not in got:
            if "" not in allowed:
                return False  # required argument missing
            continue
        if not _value_ok(got[arg], allowed):
            return False
    return True


def check(calls: list[Call] | None, item: Item) -> bool:
    """True if the calls match the ground truth. Order does not matter for parallel items."""
    if calls is None or len(calls) != len(item.ground_truth):
        return False
    remaining = list(item.ground_truth)
    for call in calls:
        for i, entry in enumerate(remaining):
            if _call_ok(call, entry):
                del remaining[i]
                break
        else:
            return False
    return True
