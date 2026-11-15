"""The train/test split as a file, so it can be checked without trusting the code that makes it.

Rule (geodesic.bfcl.Item.split): sha256(item id) mod 100 < 25 -> test, else train. No random seed involved.
`splits/bfcl_v3_split.json` lists every id; `geodesic splits` recomputes it from the data and compares.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import bfcl, pins

MANIFEST = Path(__file__).resolve().parent.parent / "splits" / "bfcl_v3_split.json"


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build(items: list[bfcl.Item]) -> dict:
    train = sorted(i.id for i in items if i.split == "train")
    test = sorted(i.id for i in items if i.split == "test")
    return {
        "rule": "test if int(sha256(item_id), 16) % 100 < 25 else train",
        "dataset": {"repo": pins.BFCL_REPO, "revision": pins.BFCL_REVISION, "categories": list(bfcl.CATEGORIES)},
        "skipped": dict(sorted(bfcl.SKIPPED.items())),
        "counts": {"train": len(train), "test": len(test)},
        "sha256": {"train": _digest(train), "test": _digest(test)},
        "train": train,
        "test": test,
    }


def check(items: list[bfcl.Item], path: Path = MANIFEST) -> list[str]:
    """Differences between the committed manifest and what the code produces now (empty = identical)."""
    want = json.loads(path.read_text())
    got = build(items)
    problems = []
    for key in ("rule", "dataset", "skipped", "counts", "sha256"):
        if want[key] != got[key]:
            problems.append(f"{key}: manifest {want[key]!r} != computed {got[key]!r}")
    for side in ("train", "test"):
        if gone := sorted(set(want[side]) - set(got[side])):
            problems.append(f"{side}: {len(gone)} ids in the manifest are not produced now, e.g. {gone[:3]}")
        if new := sorted(set(got[side]) - set(want[side])):
            problems.append(f"{side}: {len(new)} ids produced now are not in the manifest, e.g. {new[:3]}")
    return problems


def write(items: list[bfcl.Item], path: Path = MANIFEST) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(items), indent=1) + "\n")
    return path
