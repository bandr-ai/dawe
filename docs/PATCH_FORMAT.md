# Patch format (`.gpatch`, version 1)

A patch is the list of weights that changed between two checkpoints, plus enough identity information to
refuse any other checkpoint. It is a valid [safetensors](https://github.com/huggingface/safetensors) file, so
any safetensors reader can open it, and `dawe.patch` reads and writes it with the Python standard
library only (no torch, no numpy).

## Why sparse, and how small

The edits in this project touch one or two rows of the tied embedding matrix. The embedding matrix alone is
over 1.2 GB in bf16 (at least 248,047 rows × 2,560 × 2 bytes). The lab used to save the whole changed tensor,
so a "2,557-weight patch" was a 1.2 GB file. A sparse patch stores, per changed tensor, the flat indices and
the old and new values:

```
4 bytes index + 2 bytes old + 2 bytes new  =  8 bytes per changed bf16 weight, plus about 3.5 KB of manifest
```

| Edit | Changed weights | `.gpatch` size | Whole changed tensor |
|---|---|---|---|
| s1 `eos12` (one row, `<|im_end|>`) | 2,557 | **23,632 bytes** | 1,271,398,400 bytes |
| s2 `plan8_eos12` (rows `<` and `<|im_end|>`) | 5,064 | **43,688 bytes** | 1,271,398,400 bytes |

Measured on the exported files in `release/patches/`: the tensor is `model.language_model.embed_tokens.weight`,
BF16 `[248320, 2560]`, 635,699,200 weights. A patch is between 1/29,000 and 1/54,000 of the tensor it edits. On a synthetic edit
of the same shape, create, apply and restore took 1.1 s, 1.8 s and 1.6 s.

## File layout

Tensors (names are literal strings):

| Name | dtype | Contents |
|---|---|---|
| `index/<tensor>` | `U32` (`U64` if the tensor has more than 2³² elements) | strictly increasing flat (row-major) positions that change |
| `old/<tensor>` | the tensor's own dtype | values at those positions in the source checkpoint |
| `new/<tensor>` | the tensor's own dtype | values at those positions after the patch |

`<tensor>` is the tensor's key in the checkpoint's safetensors files. Values are raw little-endian bytes and are
compared bit for bit, so NaNs and signed zeros round-trip exactly.

The v1 wire format retains its original `geodesic_patch` metadata key and `geodesic-patch` format name so released artifacts remain compatible after the DAWE rename. These are serialization identifiers, not the package or CLI name.

Metadata key `geodesic_patch` holds a JSON manifest:

| Field | Meaning |
|---|---|
| `format`, `version` | `geodesic-patch`, `1` |
| `name`, `description` | free text |
| `source.fingerprint` | content identity of the checkpoint the patch applies to (below) |
| `source.layout_sha256` | hash of every tensor's name, dtype and shape |
| `source.n_tensors` | number of tensors in the checkpoint |
| `source.config` | SHA-256 of `config.json` and `generation_config.json` (those that exist) |
| `source.tokenizer` | SHA-256 of `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`, `chat_template.json` (those that exist) |
| `source.lineage` | free-form record of how the source checkpoint was made (base model and revision, training recipe, data) |
| `tensors[]` | per patched tensor: `name`, `dtype`, `shape`, `numel`, `changed`, `source_sha256`, `result_sha256` (hash of the whole tensor before and after) |
| `result.fingerprint` | content identity of the checkpoint after applying |
| `changed_weights` | total changed weights |
| `payload_sha256` | hash of the index, old and new bytes in manifest order |
| `patch_id` | SHA-256 of the canonical manifest (everything above) |

Files are deterministic: the same inputs give the same bytes. There is no timestamp.

## Checkpoint identity

`fingerprint = sha256` of the lines `name \t dtype \t shape \t sha256(tensor bytes)`, one per tensor, sorted
by name. It does not depend on how tensors are split across shard files. It does cover every tensor.

Why every tensor and not just the patched one: LoRA fine-tuning here leaves the embedding matrix alone, so
the rows we edit are **byte-identical in `sft_v1` and in plain `Qwen/Qwen3.5-4B`**. A check on the patched
tensor, the shapes, the config and the tokenizer would accept the base model. Only the full fingerprint tells
them apart. The test suite includes exactly this case.

## Operations

```bash
# create
dawe patch create     --source CKPT --edited CKPT -o x.gpatch --lineage-file release/lineage_sft_v1.json
dawe patch from-dense --source CKPT --dense old_patch.safetensors -o x.gpatch --expect-changed 2557
# look
dawe patch inspect x.gpatch                # tensors, rows touched, identities, lineage
dawe patch verify  x.gpatch DIR --expect source|result
# use
dawe patch apply   x.gpatch --source CKPT --out NEW_DIR
dawe patch restore x.gpatch --patched NEW_DIR --out ORIGINAL_DIR
```

**apply** checks, in order: tensor names/dtypes/shapes; config and tokenizer hashes; each patched tensor's shape,
dtype and content hash; then hashes every other tensor and compares the fingerprint. Any mismatch raises
`IncompatibleCheckpoint` and **nothing is written**. If the directory already matches the patch's *result*, the
error says it is already patched. It then writes `NEW_DIR` (built in a hidden sibling directory and renamed
into place), re-hashes each patched tensor in the output, and writes `geodesic_applied.json` recording which
patch produced it. The input directory is never modified; `NEW_DIR` may not be the input or inside it.

**restore** is the same operation backwards: it requires the input to match the patch's *result* exactly and
writes the original values. The result is byte-identical to the original checkpoint (tested).

Options: `--fast` skips hashing untouched tensors (it prints a warning; see the weakness above).
`--link copy|hardlink|symlink` chooses how untouched files reach `NEW_DIR` (default `copy`: independent, but
uses the disk space of the model; the patched shard is always a fresh copy).

## What this does and does not protect

- It protects against applying a patch to the wrong checkpoint, against a corrupted patch (payload hash), and
  against a manifest edited after creation (`patch_id`).
- It is **integrity, not authenticity**. Anyone can create a patch with a valid-looking manifest. If you
  download patches, verify the file against a checksum or signature published separately.
- Applying a patch tells you the weights are what the patch says. It does not tell you the patched model behaves
  well; that is what the evaluation in [WRITEUP.md](WRITEUP.md) is for.
- The patch changes a tied embedding row, which is also the *input* embedding of that token (for
  `<|im_end|>`, every chat-template turn boundary). See the write-up.

## Python

```python
from dawe import patch
p = patch.load_patch("x.gpatch")
print(patch.format_description(patch.describe(p)))
patch.apply_patch("x.gpatch", "sft_v1/merged", "sft_v1_xml", full=True)
patch.restore_patch("x.gpatch", "sft_v1_xml", "sft_v1_again")
```
