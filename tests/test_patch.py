"""Sparse patches: round trip, refusal of look-alike checkpoints, restoration, tamper detection."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import pytest

from dawe import patch as gp

VOCAB, DIM = 64, 16


def _bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def make_checkpoint(root: Path, seed: int = 0, *, vocab: int = VOCAB, dim: int = DIM, tokenizer: str = "tok-v1", config: str = "cfg-v1") -> Path:
    """A small two-shard checkpoint shaped like a real one: tied embeddings plus a few layers."""
    rng = random.Random(seed)
    root.mkdir(parents=True)
    gp.write_safetensors(root / "model-00001-of-00002.safetensors", {
        "model.language_model.embed_tokens.weight": ("BF16", [vocab, dim], _bytes(rng, vocab * dim * 2)),
        "model.language_model.layers.0.mlp.down_proj.weight": ("BF16", [dim, 2 * dim], _bytes(rng, dim * 2 * dim * 2)),
    }, {"format": "pt"})
    gp.write_safetensors(root / "model-00002-of-00002.safetensors", {
        "model.language_model.layers.1.mlp.down_proj.weight": ("BF16", [dim, 2 * dim], _bytes(rng, dim * 2 * dim * 2)),
        "model.visual.norm.weight": ("F32", [dim], _bytes(rng, dim * 4)),
    }, {"format": "pt"})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (root / "config.json").write_text(json.dumps({"cfg": config, "tie_word_embeddings": True}))
    (root / "tokenizer.json").write_text(json.dumps({"tok": tokenizer}))
    (root / "tokenizer_config.json").write_text("{}")
    (root / "preprocessor_config.json").write_text("{}")
    return root


EMBED = "model.language_model.embed_tokens.weight"


def edit_rows(src: Path, dst: Path, rows: dict[int, int]) -> Path:
    """Copy src to dst, then overwrite whole embedding rows (row -> fill byte pattern seed)."""
    shutil.copytree(src, dst)
    ck = gp.Checkpoint(dst)
    path, start, _ = ck.span(EMBED)
    with path.open("r+b") as f:
        for row, seed in rows.items():
            f.seek(start + row * DIM * 2)
            f.write(_bytes(random.Random(1000 + seed), DIM * 2))
    return dst


@pytest.fixture
def src(tmp_path):
    return make_checkpoint(tmp_path / "src")


@pytest.fixture
def edited(tmp_path, src):
    return edit_rows(src, tmp_path / "edited", {5: 1, 40: 2})


@pytest.fixture
def patch_file(tmp_path, src, edited):
    out = tmp_path / "p.gpatch"
    gp.create_from_dirs(src, edited, out, name="two-rows", description="test", lineage={"base": "toy"})
    return out


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and p.name != gp.PROVENANCE_FILE}


# --- the happy path ------------------------------------------------------------------------------


def test_patch_is_sparse_and_records_only_changes(patch_file, src):
    p = gp.load_patch(patch_file)
    assert [e.name for e in p.edits] == [EMBED]
    e = p.edits[0]
    assert gp.touched_rows(e) == [5, 40]
    assert len(e.indices) <= 2 * DIM  # a row can't change more than DIM weights
    assert patch_file.stat().st_size < VOCAB * DIM * 2  # smaller than the one tensor it edits
    d = gp.describe(p)
    assert d["changed_weights"] == len(e.indices) and d["tensors"][0]["rows"] == [5, 40]
    assert d["lineage"] == {"base": "toy"}


def test_apply_then_restore_is_byte_exact(tmp_path, src, edited, patch_file):
    out = tmp_path / "patched"
    report = gp.apply_patch(patch_file, src, out)
    assert tree_bytes(out) == tree_bytes(edited)  # identical to the checkpoint the patch was made from
    assert (out / gp.PROVENANCE_FILE).exists() and not (src / gp.PROVENANCE_FILE).exists()
    assert report["output_fingerprint"] == gp.load_patch(patch_file).manifest["result"]["fingerprint"]
    assert gp.identify(out).fingerprint == report["output_fingerprint"]

    back = tmp_path / "restored"
    gp.restore_patch(patch_file, out, back)
    assert tree_bytes(back) == tree_bytes(src)
    assert gp.identify(back).fingerprint == gp.load_patch(patch_file).manifest["source"]["fingerprint"]


def test_source_directory_is_never_modified(tmp_path, src, patch_file):
    before = tree_bytes(src)
    gp.apply_patch(patch_file, src, tmp_path / "out", link="symlink")
    assert tree_bytes(src) == before


@pytest.mark.parametrize("link", ["copy", "hardlink", "symlink"])
def test_link_modes_produce_a_loadable_directory(tmp_path, src, edited, patch_file, link):
    out = tmp_path / f"out_{link}"
    gp.apply_patch(patch_file, src, out, link=link)
    assert tree_bytes(out) == tree_bytes(edited)
    # the patched shard is always a real, separate file; untouched files may be shared
    assert not os.path.samefile(out / "model-00001-of-00002.safetensors", src / "model-00001-of-00002.safetensors")
    if link == "hardlink":
        assert os.path.samefile(out / "model-00002-of-00002.safetensors", src / "model-00002-of-00002.safetensors")
    if link == "symlink":
        assert (out / "model-00002-of-00002.safetensors").is_symlink()


def test_patch_bytes_are_deterministic(tmp_path, src, edited, patch_file):
    again = tmp_path / "again.gpatch"
    gp.create_from_dirs(src, edited, again, name="two-rows", description="test", lineage={"base": "toy"})
    assert again.read_bytes() == patch_file.read_bytes()


def test_realistic_size(tmp_path):
    big = make_checkpoint(tmp_path / "big", vocab=2000, dim=512)
    ed = edit_rows(big, tmp_path / "bigedit", {7: 1, 1500: 2})
    p = gp.create_from_dirs(big, ed, tmp_path / "big.gpatch", name="x")
    d = gp.describe(p)
    assert d["changed_weights"] <= 1024
    assert d["file_bytes"] < 12_000 and d["dense_bytes_if_whole_tensors"] > 2_000_000


# --- refusal -------------------------------------------------------------------------------------


def refuses(patch_file, ckpt, tmp_path, match="", **kw):
    with pytest.raises(gp.IncompatibleCheckpoint, match=match):
        gp.apply_patch(patch_file, ckpt, tmp_path / "should-not-exist", **kw)
    assert not (tmp_path / "should-not-exist").exists()
    assert not list(tmp_path.glob(".should-not-exist*"))


def test_refuses_same_name_different_weights(tmp_path, patch_file):
    """Two checkpoints with the same layout, config, tokenizer and the same embedding (as when
    LoRA fine-tuning leaves embeddings alone) but different other weights, like a base model and its fine-tune."""
    sibling = make_checkpoint(tmp_path / "sibling", seed=0)
    ck = gp.Checkpoint(sibling)
    path, start, _ = ck.span("model.language_model.layers.1.mlp.down_proj.weight")
    with path.open("r+b") as f:
        f.seek(start)
        f.write(b"\xff\xfe")
    refuses(patch_file, sibling, tmp_path, "fingerprint differs")
    # --fast skips hashing untouched tensors and so cannot see this; that is why it is not the default
    gp.apply_patch(patch_file, sibling, tmp_path / "fast", full=False)


def test_refuses_different_tokenizer(tmp_path, patch_file):
    refuses(patch_file, make_checkpoint(tmp_path / "t", tokenizer="tok-v2"), tmp_path, "tokenizer files differ: tokenizer.json")


def test_refuses_different_config(tmp_path, patch_file):
    refuses(patch_file, make_checkpoint(tmp_path / "c", config="cfg-v2"), tmp_path, "config files differ: config.json")


def test_refuses_different_shapes(tmp_path, patch_file):
    refuses(patch_file, make_checkpoint(tmp_path / "s", vocab=VOCAB + 8), tmp_path, "names/dtypes/shapes differ")


def test_refuses_unrelated_embedding_values(tmp_path, patch_file):
    refuses(patch_file, make_checkpoint(tmp_path / "u", seed=9), tmp_path, "content differs")


def test_already_patched_says_so(tmp_path, src, patch_file):
    out = tmp_path / "patched"
    gp.apply_patch(patch_file, src, out)
    refuses(patch_file, out, tmp_path, "already patched")


def test_restore_refuses_an_unpatched_checkpoint(tmp_path, src, patch_file):
    with pytest.raises(gp.IncompatibleCheckpoint, match="not patched yet"):
        gp.restore_patch(patch_file, src, tmp_path / "x")


def test_restore_refuses_a_patched_checkpoint_edited_afterwards(tmp_path, src, patch_file):
    out = tmp_path / "patched"
    gp.apply_patch(patch_file, src, out)
    ck = gp.Checkpoint(out)
    path, start, _ = ck.span("model.visual.norm.weight")
    with path.open("r+b") as f:
        f.seek(start)
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(gp.IncompatibleCheckpoint):
        gp.restore_patch(patch_file, out, tmp_path / "back")


def test_output_directory_rules(tmp_path, src, patch_file):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    with pytest.raises(gp.PatchError, match="not empty"):
        gp.apply_patch(patch_file, src, tmp_path / "busy")
    with pytest.raises(gp.PatchError, match="separate"):
        gp.apply_patch(patch_file, src, src)
    with pytest.raises(gp.PatchError, match="separate"):
        gp.apply_patch(patch_file, src, src / "inside")
    (tmp_path / "empty").mkdir()
    gp.apply_patch(patch_file, src, tmp_path / "empty")  # an empty directory is fine
    assert (tmp_path / "empty" / "config.json").exists()


def test_verify_reports_which_side_matches(src, edited, patch_file):
    assert gp.verify_checkpoint(patch_file, src, "source") == []
    assert gp.verify_checkpoint(patch_file, edited, "result") == []
    assert gp.verify_checkpoint(patch_file, src, "result") != []


# --- the patch file itself ---------------------------------------------------------------------------


def _flip(path: Path, offset_from_end: int) -> None:
    b = bytearray(path.read_bytes())
    b[-offset_from_end] ^= 1
    path.write_bytes(bytes(b))


def test_corrupted_payload_is_detected(patch_file):
    _flip(patch_file, 3)
    with pytest.raises(gp.PatchError, match="payload hash mismatch"):
        gp.load_patch(patch_file)


def test_edited_manifest_is_detected(tmp_path, patch_file):
    sh = gp.read_header(patch_file)
    manifest = json.loads(sh.metadata["geodesic_patch"])
    manifest["source"]["fingerprint"] = "0" * 64
    raw = patch_file.read_bytes()
    forged = json.dumps(manifest, sort_keys=True)
    hdr = raw[8:sh.data_start].decode().rstrip()
    header = json.loads(hdr)
    header["__metadata__"]["geodesic_patch"] = forged
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    bad = tmp_path / "forged.gpatch"
    bad.write_bytes(len(blob).to_bytes(8, "little") + blob + raw[sh.data_start:])
    with pytest.raises(gp.PatchError, match="patch_id mismatch"):
        gp.load_patch(bad)


def test_not_a_patch(tmp_path, src):
    with pytest.raises(gp.PatchError, match="no manifest"):
        gp.load_patch(src / "model-00001-of-00002.safetensors")


def test_unsorted_or_out_of_range_edits_are_rejected():
    e = gp.TensorEdit("t", "BF16", (4, 4), [3, 2], b"\0\0\0\0", b"\1\1\1\1")
    with pytest.raises(gp.PatchError, match="increasing"):
        gp.write_patch("/dev/null", {"name": "x", "source": {}, "result": {}}, [e])
    e = gp.TensorEdit("t", "BF16", (4, 4), [16], b"\0\0", b"\1\1")
    with pytest.raises(gp.PatchError, match="out of range"):
        gp.write_patch("/dev/null", {"name": "x", "source": {}, "result": {}}, [e])


# --- other ways to make a patch ----------------------------------------------------------------------------


def test_from_dense_matches_diffing_two_checkpoints(tmp_path, src, edited, patch_file):
    """The old saver wrote whole changed tensors under framework names. Converting must give the same patch content."""
    new_embed = gp.Checkpoint(edited)
    path, s, e = new_embed.span(EMBED)
    dense = tmp_path / "old_style.safetensors"
    gp.write_safetensors(dense, {EMBED: ("BF16", [VOCAB, DIM], path.read_bytes()[s:e])})
    out = tmp_path / "converted.gpatch"
    gp.create_from_dense(src, dense, out, name="two-rows", description="test", lineage={"base": "toy"})
    assert out.read_bytes() == patch_file.read_bytes()


def test_from_dense_resolves_framework_prefixes(tmp_path, src, edited):
    ck = gp.Checkpoint(edited)
    path, s, e = ck.span(EMBED)
    raw = path.read_bytes()[s:e]
    dense = tmp_path / "d.safetensors"
    for name in ("language_model.embed_tokens.weight", EMBED):  # prefix dropped / exact
        gp.write_safetensors(dense, {name: ("BF16", [VOCAB, DIM], raw)})
        assert gp.create_from_dense(src, dense, tmp_path / "x.gpatch", name="x").changed() > 0
    gp.write_safetensors(dense, {"embed_tokens.weight": ("BF16", [VOCAB, DIM], raw)})
    with pytest.raises(gp.PatchError, match="cannot match"):
        gp.create_from_dense(src, dense, tmp_path / "y.gpatch", name="x")


def test_from_dense_rejects_wrong_shape(tmp_path, src):
    dense = tmp_path / "d.safetensors"
    gp.write_safetensors(dense, {EMBED: ("BF16", [VOCAB + 1, DIM], bytes((VOCAB + 1) * DIM * 2))})
    with pytest.raises(gp.PatchError, match="checkpoint has"):
        gp.create_from_dense(src, dense, tmp_path / "x.gpatch", name="x")


def test_create_from_edits_sorts_and_drops_noops(tmp_path, src, edited, patch_file):
    ck = gp.Checkpoint(edited)
    ed = gp.load_patch(patch_file).edits[0]
    idx = list(reversed(ed.indices)) + [0]  # unsorted, plus an index whose "new" value is the old value
    ck_src = gp.Checkpoint(src)
    old0 = gp.gather(ck_src.span(EMBED), 2, [0])
    new = b"".join(ed.new[k * 2:(k + 1) * 2] for k in reversed(range(len(ed.indices)))) + old0
    out = tmp_path / "e.gpatch"
    p = gp.create_from_edits(src, {"model.language_model.embed_tokens.weight": (idx, new)}, out, name="two-rows", description="test", lineage={"base": "toy"})
    assert p.edits[0].indices == ed.indices and p.edits[0].new == ed.new
    assert ck is not None


def test_dense_edits_are_refused_as_non_sparse(tmp_path, src):
    other = make_checkpoint(tmp_path / "other", seed=3)
    with pytest.raises(gp.PatchError, match="not a sparse edit"):
        gp.create_from_dirs(src, other, tmp_path / "x.gpatch", name="x")


def test_identical_checkpoints_make_no_patch(tmp_path, src):
    twin = tmp_path / "twin"
    shutil.copytree(src, twin)
    with pytest.raises(gp.PatchError, match="nothing to patch"):
        gp.create_from_dirs(src, twin, tmp_path / "x.gpatch", name="x")


# --- interoperability with the real safetensors library -------------------------------------------------------


def test_real_safetensors_reads_our_files_and_we_read_theirs(tmp_path):
    np = pytest.importorskip("numpy")
    st = pytest.importorskip("safetensors.numpy")
    rng = np.random.default_rng(0)
    root = tmp_path / "real"
    root.mkdir()
    emb = rng.standard_normal((VOCAB, DIM)).astype("float16")
    other = rng.standard_normal((DIM, DIM)).astype("float32")
    st.save_file({"embed.weight": emb}, str(root / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    st.save_file({"layer.weight": other}, str(root / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    edited = tmp_path / "real_edit"
    shutil.copytree(root, edited)
    emb2 = emb.copy()
    emb2[3] = rng.standard_normal(DIM).astype("float16")
    st.save_file({"embed.weight": emb2}, str(edited / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})

    p = gp.create_from_dirs(root, edited, tmp_path / "r.gpatch", name="real")
    assert gp.touched_rows(p.edits[0]) == [3]
    out = tmp_path / "real_out"
    gp.apply_patch(tmp_path / "r.gpatch", root, out)
    got = st.load_file(str(out / "model-00001-of-00002.safetensors"))["embed.weight"]
    assert np.array_equal(got.view("uint16"), emb2.view("uint16"))
    assert st.load_file(str(out / "model-00002-of-00002.safetensors"))["layer.weight"].tobytes() == other.tobytes()

    # our patch container is a valid safetensors file for the real reader
    from safetensors import safe_open

    with safe_open(str(tmp_path / "r.gpatch"), framework="np") as f:
        assert "geodesic_patch" in f.metadata()
        assert set(f.keys()) == {"index/embed.weight", "old/embed.weight", "new/embed.weight"}
        assert f.get_tensor("index/embed.weight").dtype == np.uint32


# --- command line ------------------------------------------------------------------------------------------------


def test_cli_roundtrip(tmp_path, src, edited, capsys):
    out = tmp_path / "cli.gpatch"
    assert gp.main(["create", "--source", str(src), "--edited", str(edited), "-o", str(out)]) == 0
    assert gp.main(["inspect", str(out)]) == 0
    assert "rows [5, 40]" in capsys.readouterr().out
    assert gp.main(["verify", str(out), str(src)]) == 0
    assert gp.main(["verify", str(out), str(src), "--expect", "result"]) == 1
    assert gp.main(["apply", str(out), "--source", str(src), "--out", str(tmp_path / "a")]) == 0
    assert gp.main(["apply", str(out), "--source", str(tmp_path / "a"), "--out", str(tmp_path / "b")]) == 1
    assert "already patched" in capsys.readouterr().err
    assert gp.main(["restore", str(out), "--patched", str(tmp_path / "a"), "--out", str(tmp_path / "r")]) == 0
    assert tree_bytes(tmp_path / "r") == tree_bytes(src)


def test_cli_from_dense_expect_changed(tmp_path, src, edited):
    ck = gp.Checkpoint(edited)
    path, s, e = ck.span(EMBED)
    dense = tmp_path / "d.safetensors"
    gp.write_safetensors(dense, {EMBED: ("BF16", [VOCAB, DIM], path.read_bytes()[s:e])})
    out = tmp_path / "x.gpatch"
    assert gp.main(["from-dense", "--source", str(src), "--dense", str(dense), "-o", str(out), "--expect-changed", "1"]) == 1
    assert not out.exists()
    n = len(gp.diff_regions(gp.Checkpoint(src).span(EMBED), ck.span(EMBED), 2))
    assert gp.main(["from-dense", "--source", str(src), "--dense", str(dense), "-o", str(out), "--expect-changed", str(n)]) == 0


def test_fast_mode_does_not_claim_an_unmeasured_fingerprint(tmp_path, src, patch_file):
    out = tmp_path / "fast"
    report = gp.apply_patch(patch_file, src, out, full=False)
    assert report["fingerprint_verified"] is False and report["output_fingerprint"] is None
    marker = json.loads((out / gp.PROVENANCE_FILE).read_text())
    assert marker["fingerprint_verified"] is False and marker["output_fingerprint"] is None
    full = gp.apply_patch(patch_file, src, tmp_path / "full")
    assert full["fingerprint_verified"] is True and full["output_fingerprint"]


def test_verify_fast_warns(patch_file, src, capsys):
    assert gp.main(["verify", str(patch_file), str(src), "--fast"]) == 0
    assert "--fast does not hash untouched tensors" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["inspect", "{missing}"], ["apply", "{missing}", "--source", "{missing}", "--out", "{out}"],
                                  ["verify", "{missing}", "{missing}"],
                                  ["from-dense", "--source", "{missing}", "--dense", "{missing}", "-o", "{out}"]])
def test_missing_files_are_errors_not_tracebacks(tmp_path, capsys, argv):
    fill = {"{missing}": str(tmp_path / "nope"), "{out}": str(tmp_path / "out")}
    assert gp.main([fill.get(a, a) for a in argv]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_create_refuses_to_overwrite_without_force(tmp_path, src, edited, capsys):
    out = tmp_path / "fix.gpatch"
    argv = ["create", "--source", str(src), "--edited", str(edited), "-o", str(out)]
    assert gp.main(argv) == 0
    first = out.read_bytes()
    capsys.readouterr()
    assert gp.main(argv) == 1
    assert "already exists; nothing was written" in capsys.readouterr().err
    assert out.read_bytes() == first
    assert sorted(p.name for p in tmp_path.glob("*.gpatch")) == ["fix.gpatch"]  # no numbered sibling


def test_force_replaces_the_existing_file(tmp_path, src, edited):
    out = tmp_path / "fix.gpatch"
    out.write_bytes(b"old")
    assert gp.main(["create", "--source", str(src), "--edited", str(edited), "-o", str(out), "--force"]) == 0
    assert out.read_bytes() != b"old" and sorted(p.name for p in tmp_path.glob("*.gpatch")) == ["fix.gpatch"]


def test_from_dense_refuses_existing_file_and_expect_changed_failure_leaves_it_alone(tmp_path, src, edited):
    ck = gp.Checkpoint(edited)
    path, s, e = ck.span(EMBED)
    dense = tmp_path / "d.safetensors"
    gp.write_safetensors(dense, {EMBED: ("BF16", [VOCAB, DIM], path.read_bytes()[s:e])})
    out = tmp_path / "x.gpatch"
    out.write_bytes(b"precious")
    assert gp.main(["from-dense", "--source", str(src), "--dense", str(dense), "-o", str(out), "--expect-changed", "1"]) == 1
    assert out.read_bytes() == b"precious"
    assert gp.main(["from-dense", "--source", str(src), "--dense", str(dense), "-o", str(out)]) == 1
    assert out.read_bytes() == b"precious"
