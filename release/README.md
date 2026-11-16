# release/

| Path | What |
|---|---|
| `lineage_sft_v1.json` | how the checkpoint the patches apply to (`sft_v1/merged`) was made; embedded in every patch |
| `patches/s1_eos12.gpatch` | session s1: `v4_format` (XML calls) fix |
| `patches/s2_plan8_eos12.gpatch` | session s2: `v11_plan` (plan-first replies) fix |

## The patches

| File | Weights changed | Rows of `embed_tokens.weight` | Bytes | SHA-256 |
|---|---|---|---|---|
| `s1_eos12.gpatch` | 2,557 | 248046 (`<\|im_end\|>`) | 23,632 | `148edd3bcbe96540e15f082b66a0ef458b6beb131021d275d8829e9f7a712493` |
| `s2_plan8_eos12.gpatch` | 5,064 | 27 (`<`), 248046 (`<\|im_end\|>`) | 43,688 | `c83fa6928329a0314301c5caafc262dd3c85f6bb6af42f0c440d072608416219` |

Both apply to one checkpoint: fingerprint `10bd45fa563a958c20e4864b7ae784e07d87c0c3fa17372d0efa0f7bcd259772`
(`sft_v1/merged`, 723 tensors, one 9,078,620,536-byte `model.safetensors`). They produce
`87f99872382c0ae4…` (s1) and `11dfa310062f10b1…` (s2). The checksums are integrity checks for a download, not signatures
(see [PATCH_FORMAT.md](../docs/PATCH_FORMAT.md#what-this-does-and-does-not-protect)).

```bash
cd release/patches && shasum -a 256 -c SHA256SUMS
dawe patch inspect s1_eos12.gpatch
dawe patch apply   s1_eos12.gpatch --source /path/to/sft_v1/merged --out /path/to/sft_v1_xml
dawe patch restore s1_eos12.gpatch --patched /path/to/sft_v1_xml  --out /path/to/sft_v1_again
```

## How they were made

Converted on 2026-11-16 from the whole-tensor patch files the lab originally wrote
(`agent/s1/patches/eos12.safetensors`, `agent/s2/patches/plan8_eos12.safetensors` on the Modal volume), using
`modal run gpu/lab.py::export` on CPU. The converter diffs each stored tensor against the checkpoint and raised
unless the number of changed weights equalled what the lab recorded when the agent made the patch (2,557 and 5,064). It did not
raise. The agents' notes also say which rows were edited, and the converted patches agree (s1: `<|im_end|>`; s2: `<` and `<|im_end|>`).

## What was verified, and what was not

Verified on the real `sft_v1/merged` checkpoint (CPU only, 2026-11-16):

- both patches `verify` as that checkpoint's patches (full hash of all 723 tensors);
- `apply` writes a new directory whose fingerprint equals the one recorded in the patch; `apply` on that directory is refused with
  "already patched";
- `restore` returns `model.safetensors` **byte-identical** to the original (`cmp`), for both patches;
- the base model `Qwen/Qwen3.5-4B` @ `851bf6e8…` has a **byte-identical** `embed_tokens.weight` (same SHA-256 as `sft_v1`'s), while
  other tensors differ (`layers.0.mlp.down_proj`, `layers.10.mlp.up_proj`, `layers.3.self_attn.q_proj`). So a check on the patched
  tensor alone would have accepted the base model, and the full fingerprint does not.

**Not verified:** the held-out accuracy of a model produced by `dawe patch apply`. The numbers in
[RESULTS.md](../RESULTS.md) come from the original whole-tensor patches. The converted patches change exactly the same number of
weights in the same rows, which is strong evidence, but not a re-run. To close that loop (about 2 H100 evaluations; each is a
billed GPU run, so not done here):

```bash
dawe patch apply s1_eos12.gpatch --source sft_v1/merged --out sft_v1_xml
# upload / mount sft_v1_xml, then: modal run gpu/app.py::gen --model <that model>  and  dawe score
```

## The checkpoint

The patches refuse anything but the exact `sft_v1/merged` above, and that checkpoint is **not published** (a licensing and hosting
decision; see [LICENSES.md](../docs/LICENSES.md#3-model-terms-separate-from-the-code-license)). LoRA training is not guaranteed to
reproduce the same bytes. If yours differs, `apply` refuses; make a patch for your own model by re-running the agent's edit script
(`sessions/s1/edits/eos12.py`) in the lab, or with `dawe patch create`.
