"""User-data preparation invariants; no GPU or model downloads."""
import importlib.util
import json
from pathlib import Path

import pytest

from dawe.cli import main
from dawe.workbench import SKILL, freeze_examples, init_workspace, install_skill, read_examples, split_examples


def fixture_rows():
    return read_examples(Path(__file__).resolve().parents[1] / "examples/dawe/examples.jsonl")


def test_freeze_keeps_task_variants_together_and_is_reproducible(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = fixture_rows()
    source.write_text("\n".join(json.dumps(r) for r in rows))
    first = freeze_examples(source, tmp_path / "edit", tmp_path / "eval")
    source.write_text("\n".join(json.dumps(r) for r in reversed(rows)))
    second = freeze_examples(source, tmp_path / "edit2", tmp_path / "eval2")
    assert first == second
    assert not set(first["development"]["groups"]) & set(first["evaluation"]["groups"])
    for name in ("development", "evaluation"):
        assert all(n > 0 for n in first[name]["by_purpose"].values())
    dev = read_examples(tmp_path / "edit/examples.jsonl")
    assert all(sum(r["group_id"] == g for r in dev) == 2 for g in first["development"]["groups"])
    assert not (tmp_path / "edit/evaluation").exists()


def test_explicit_split_preserved_and_conflicts_rejected():
    rows = fixture_rows()
    for r in rows:
        r["split"] = "evaluation" if r["group_id"] == "case-3" else "development"
    _, evaluation = split_examples(rows)
    assert {r["group_id"] for r in evaluation} == {"case-3"}
    rows[-1]["split"] = "development"
    with pytest.raises(ValueError, match="both development and evaluation"):
        split_examples(rows)
    rows[-1]["split"] = "evaluation"
    del rows[0]["split"]
    del rows[1]["split"]
    with pytest.raises(ValueError, match="every group"):
        split_examples(rows)


def test_duplicate_prompts_across_groups_rejected():
    rows = fixture_rows()
    duplicate = dict(rows[0], id="repeat", group_id="different-task")
    with pytest.raises(ValueError, match="identical prompt"):
        split_examples(rows + [duplicate])


def test_insufficient_preservation_or_groups_cannot_look_evaluated():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="both repair and preserve"):
        split_examples([r for r in rows if r["purpose"] == "repair"])
    with pytest.raises(ValueError, match="needs repair and preserve"):
        split_examples(rows[:2])


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(messages=[{"role": "user", "content": "no target"}]),
    lambda r: r.update(target_source=""),
    lambda r: r.update(decisions=[{"kind": "continue", "prefix": "not the target prefix"}]),
    lambda r: r.update(purpose="test"),
])
def test_bad_normalized_examples_rejected(tmp_path, mutation):
    row = fixture_rows()[0]
    mutation(row)
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="line 1"):
        read_examples(p)


def test_freeze_cannot_overwrite_or_nest_outputs(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in fixture_rows()))
    with pytest.raises(ValueError, match="not nested"):
        freeze_examples(source, tmp_path / "edit", tmp_path / "edit/eval")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="new directories"):
        freeze_examples(source, tmp_path / "edit", existing)
    assert not (tmp_path / "edit").exists()


def test_init_is_local_and_preserves_source_and_existing_workspace(tmp_path):
    source = tmp_path / "private.jsonl"
    source.write_text("raw source, deliberately not normalized yet")
    out = init_workspace(source, "/models/owned", tmp_path / "run")
    config = json.loads((out / "run.json").read_text())
    assert config["budget_usd"] is None and config["compute_started"] is False
    assert (out / "skill/SKILL.md").exists()
    assert (out / "skill/scripts/probe.py").exists()
    assert source.read_text() == "raw source, deliberately not normalized yet"
    assert not (out / source.name).exists()
    with pytest.raises(ValueError, match="already exists"):
        init_workspace(source, "/models/owned", out)
    with pytest.raises(ValueError, match="separate"):
        init_workspace(tmp_path, "/models/owned", tmp_path / "run2")
    with pytest.raises(ValueError, match="positive"):
        init_workspace(source, "/models/owned", tmp_path / "run3", budget_usd=float("nan"))


def test_install_does_not_overwrite_and_copies_references(tmp_path):
    out = install_skill(tmp_path)
    assert (out / "references/data.md").exists()
    with pytest.raises(ValueError, match="already exists"):
        install_skill(tmp_path)


def test_new_cli_routes_do_not_load_bfcl(monkeypatch, tmp_path):
    from dawe import bfcl

    def no_dataset():
        pytest.fail("BFCL should not be loaded for user data")

    monkeypatch.setattr(bfcl, "load_all", no_dataset)
    assert main(["skill", "path"]) == 0
    assert main(["skill", "install", "--dest", str(tmp_path / "skills")]) == 0
    assert main(["workbench", "init", "--source", str(tmp_path / "missing"), "--model", "owned", "--out", str(tmp_path / "run")]) == 2


def test_probe_causal_positions_and_alignment():
    spec = importlib.util.spec_from_file_location("dawe_probe", SKILL / "scripts/probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    prompt, reply = "Q:", "<call/>"
    rendered = prompt + reply + "<eos>"
    offsets = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 9), (9, 14)]
    assert probe.decision_indices(offsets, prompt, reply, rendered, [{"prefix": "", "kind": "start"}]) == [(2, "start")]
    assert probe.decision_indices(offsets, prompt, reply, rendered, [{"prefix": reply, "kind": "stop"}]) == [(5, "stop")]
    with pytest.raises(ValueError, match="token boundary"):
        probe.decision_indices(offsets, prompt, reply, rendered, [{"prefix": "<c", "kind": "middle"}])
    with pytest.raises(ValueError, match="Chat template"):
        probe.decision_indices(offsets, prompt, reply, "different template", [{"prefix": "", "kind": "start"}])
