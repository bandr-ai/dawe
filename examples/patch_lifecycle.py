"""The whole patch lifecycle on a toy checkpoint, in seconds, with no GPU and no downloads.

    uv run python examples/patch_lifecycle.py

Builds two toy "models" that have the same names, shapes, config and tokenizer, edits two embedding rows of one
of them, makes a sparse patch, applies it, refuses the wrong model, and restores the original byte for byte.
"""

import json
import random
import shutil
import tempfile
from pathlib import Path

from dawe import patch as gp

VOCAB, DIM, EMBED = 4096, 256, "model.language_model.embed_tokens.weight"
tmp = Path(tempfile.mkdtemp(prefix="dawe-demo-"))


def toy_model(root: Path, seed: int) -> Path:
    rng = random.Random(seed)
    root.mkdir()
    gp.write_safetensors(root / "model-00001-of-00002.safetensors",
                         {EMBED: ("BF16", [VOCAB, DIM], rng.randbytes(VOCAB * DIM * 2))})
    gp.write_safetensors(root / "model-00002-of-00002.safetensors",
                         {"model.language_model.layers.0.mlp.down_proj.weight": ("BF16", [DIM, 4 * DIM], rng.randbytes(DIM * 4 * DIM * 2))})
    (root / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    (root / "tokenizer.json").write_text("{}")
    return root


def say(msg: str) -> None:
    print(f"\n== {msg}")


say("a fine-tuned model, and a sibling with identical embeddings but different other weights (think: base vs fine-tune)")
model = toy_model(tmp / "model", seed=1)
sibling = tmp / "sibling"
shutil.copytree(model, sibling)
with (sibling / "model-00002-of-00002.safetensors").open("r+b") as f:
    f.seek(-8, 2)
    f.write(b"\xde\xad\xbe\xef" * 2)

say("edit rows 27 and 3000 of the embedding matrix, then diff the two checkpoints into a sparse patch")
edited = tmp / "edited"
shutil.copytree(model, edited)
path, start, _ = gp.Checkpoint(edited).span(EMBED)
with path.open("r+b") as f:
    for row in (27, 3000):
        f.seek(start + row * DIM * 2)
        f.write(random.Random(row).randbytes(DIM * 2))
p = gp.create_from_dirs(model, edited, tmp / "demo.gpatch", name="demo", description="two embedding rows")
print(gp.format_description(gp.describe(p)).split("config files")[0])

say("apply to the right model: writes a NEW directory, original untouched")
print(json.dumps(gp.apply_patch(tmp / "demo.gpatch", model, tmp / "patched"), indent=1)[:400], "...")

say("apply to the look-alike model: refused, nothing written")
try:
    gp.apply_patch(tmp / "demo.gpatch", sibling, tmp / "wrong")
except gp.IncompatibleCheckpoint as e:
    print(str(e).splitlines()[0], "\n ", str(e).splitlines()[1])
print("output directory exists:", (tmp / "wrong").exists())

say("apply twice: refused, and it says why")
try:
    gp.apply_patch(tmp / "demo.gpatch", tmp / "patched", tmp / "twice")
except gp.IncompatibleCheckpoint as e:
    print(str(e).splitlines()[-1])

say("restore: byte-identical to the original")
gp.restore_patch(tmp / "demo.gpatch", tmp / "patched", tmp / "restored")
same = all((model / n).read_bytes() == (tmp / "restored" / n).read_bytes() for n in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
print("restored == original:", same)
shutil.rmtree(tmp)
