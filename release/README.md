# release/

| Path | What |
|---|---|
| `lineage_sft_v1.json` | how the checkpoint the patches apply to (`sft_v1/merged`) was made; embedded in every patch |
| `patches/` | the sparse `.gpatch` files |

## Status of `patches/`

**Empty for now.** The patches exist on the Modal volume `geodesic-runs` as whole-tensor files
(`agent/s1/patches/eos12.safetensors`, `agent/s2/patches/plan8_eos12.safetensors`) written by the old saver. Convert
them with the exporter, which needs the `sft_v1` checkpoint on the same volume:

```bash
uv sync --extra gpu
modal run gpu/lab.py::export --session s1 --name eos12        # -> release/patches/s1_eos12.gpatch
modal run gpu/lab.py::export --session s2 --name plan8_eos12  # -> release/patches/s2_plan8_eos12.gpatch
uv run geodesic patch inspect release/patches/s1_eos12.gpatch
```

The exporter raises unless the converted patch changes exactly as many weights as the lab recorded when the patch was
made (s1: **2,557**, s2: **5,064**, the counts in the agents' notes). If you convert by hand, pass
`--expect-changed` to `geodesic patch from-dense`. Then record the SHA-256 of each file here:

| File | Weights | Bytes | SHA-256 |
|---|---|---|---|
| `s1_eos12.gpatch` | 2,557 (expected) | *not yet exported* | |
| `s2_plan8_eos12.gpatch` | 5,064 (expected) | *not yet exported* | |

After export, re-run the held-out evaluation on a model produced with `geodesic patch apply` (not the lab's
symlinked directory) and confirm it reproduces the numbers in [RESULTS.md](../RESULTS.md); that closes the loop on
the new format.

## Applying

```bash
geodesic patch apply release/patches/s1_eos12.gpatch --source /path/to/sft_v1/merged --out /path/to/sft_v1_xml
```

This refuses any checkpoint other than the exact `sft_v1/merged` it was made from. Obtain that checkpoint by
training it ([REPRODUCE.md](../docs/REPRODUCE.md#2-baseline-lora-sft-on-v1)); training is not guaranteed to
reproduce the same bytes, so a patch may refuse your retrain. In that case, regenerate the patch for your model by
re-running the agent's edit script (`sessions/s1/edits/eos12.py`) in the lab, which writes a patch bound to your
checkpoint.
