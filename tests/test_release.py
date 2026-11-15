"""The release's pins and files agree with each other and with the data."""

import hashlib
import json
from pathlib import Path

import pytest

from geodesic import bfcl, pins, splits

ROOT = Path(__file__).resolve().parent.parent
needs_data = pytest.mark.skipif(bool(bfcl.verify_data()), reason="run `geodesic fetch` first")


def test_split_rule_is_the_documented_one():
    """Known ids and their buckets, so the rule cannot drift unnoticed even without the data files."""
    def bucket(i):
        return int(hashlib.sha256(i.encode()).hexdigest(), 16) % 100

    manifest = json.loads(splits.MANIFEST.read_text())
    assert manifest["counts"] == {"train": len(manifest["train"]), "test": len(manifest["test"])} == {"train": 595, "test": 204}
    assert all(bucket(i) < 25 for i in manifest["test"]) and all(bucket(i) >= 25 for i in manifest["train"])
    assert not set(manifest["train"]) & set(manifest["test"])
    assert manifest["dataset"]["revision"] == pins.BFCL_REVISION
    assert manifest["sha256"] == {"train": splits._digest(manifest["train"]), "test": splits._digest(manifest["test"])}


@needs_data
def test_data_and_split_match_the_manifest():
    assert bfcl.verify_data() == []
    assert splits.check(bfcl.load_all()) == []


def test_lineage_file_matches_pins():
    assert json.loads((ROOT / "release" / "lineage_sft_v1.json").read_text()) == pins.lineage_sft_v1()


def test_modal_jobs_take_versions_from_pins_only():
    import re

    for f in ("gpu/app.py", "gpu/lab.py"):
        assert not re.search(r'"[A-Za-z0-9_.-]+==[0-9]', (ROOT / f).read_text()), f"{f} hardcodes a package pin"
    assert pins.BASE_REVISION in (ROOT / "docs" / "REPRODUCE.md").read_text()
    assert pins.BFCL_REVISION in (ROOT / "docs" / "REPRODUCE.md").read_text()


def test_pin_tables_in_docs_agree():
    doc = (ROOT / "docs" / "REPRODUCE.md").read_text()
    for name, version in {**pins.TRAIN_PACKAGES, **pins.VLLM_PACKAGES}.items():
        assert f"{name.split('-')[0]}" in doc and version in doc, name


def test_released_patches_match_their_checksums_and_notes():
    from geodesic import patch as gp

    d = ROOT / "release" / "patches"
    for line in (d / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest, name
    s1, s2 = gp.load_patch(d / "s1_eos12.gpatch"), gp.load_patch(d / "s2_plan8_eos12.gpatch")
    assert (s1.changed(), s2.changed()) == (2557, 5064)  # the counts in the agents' notes
    assert gp.touched_rows(s1.edits[0]) == [248046] and gp.touched_rows(s2.edits[0]) == [27, 248046]
    assert s1.manifest["source"]["fingerprint"] == s2.manifest["source"]["fingerprint"]
    assert s1.manifest["source"]["lineage"] == pins.lineage_sft_v1()
    assert s1.size_bytes < 30_000 and s2.size_bytes < 50_000
