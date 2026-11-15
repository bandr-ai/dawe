# Results

Held-out BFCL test split (204 items; 48 parallel), scored with our own matcher (`geodesic/bfcl.py`), not BFCL's official evaluator. Greedy decoding, thinking off. One run each.

"Held-out" means the agents never saw test items or per-item predictions. The agent briefs did quote aggregate test numbers; see [docs/LAB.md](docs/LAB.md#what-the-agents-were-told-about-the-test-set). At n=204 the 95% interval on a single accuracy is about ±4 to 6 points. Read this with [docs/WRITEUP.md](docs/WRITEUP.md#limitations).

Provenance: these tables were produced by `geodesic report` from `runs/preds_*.jsonl`, which are not committed yet, so they cannot be regenerated from this repository alone. The only edit since is layout (column width); no number was changed. The `v1`, `v4_format`, `v4` parallel and `v6_all` columns for `sft_v1`, `dawe_s1_eos12`, the random control, `ceiling_v4_format` and the shim also agree with the maintainers' own earlier write-up of these runs (2026-11-12 and 2026-11-15). The `_DE` suffix in `dawe_s1_eos12_DE` is a file-name suffix from a later run of the s1 patch on `v11_plan` and `v12_xml_opaque`.

Rows: `base` = Qwen3.5-4B; `sft_v1` = LoRA SFT on v1; `ceiling_X` = SFT directly on harness X (retraining); `dawe_*` = sft_v1 + agent-written weight patch; `dawe_control_rand_eos` = same-norm random edit to the same row as the v4 patch; `sft_v1 + shim` = a lossless format translator (v1 score everywhere).

```
all items (n=204)
model                                v1 v10_positional       v11_plan v12_xml_opaque      v2_rename      v3_schema      v4_format      v5_prompt         v6_all      v7_opaque      v8_native      v9_pycall
base                              71.6%          83.8%          11.3%          82.4%          71.6%          73.5%          81.4%          84.3%          71.1%          68.6%          86.8%          81.4%
ceiling_v11_plan                  90.2%          91.7%          88.7%          84.3%          90.7%          82.4%          82.8%          89.7%          77.9%              -          91.7%          90.2%
ceiling_v4_format                 73.0%              -              -              -          74.5%          75.5%          89.7%          74.0%          84.8%              -              -              -
ceiling_v6_all                    63.2%              -              -              -          66.7%          67.6%          87.7%          77.9%          84.8%              -              -              -
dawe_control_rand_eos             87.7%              -              -              -          87.3%          82.8%          68.1%          87.3%          64.7%              -              -              -
dawe_s1_eos12                     87.7%              -              -              -          87.3%          82.8%          85.8%          87.7%          81.9%              -              -              -
dawe_s1_eos12_DE                      -              -          64.2%          86.3%              -              -              -              -              -              -              -              -
dawe_s2_plan8_eos12               87.7%          83.8%          88.2%          85.8%          87.3%          82.8%          85.8%          87.3%          81.9%          88.7%          89.7%          84.3%
sft_v1                            87.7%          84.3%          20.1%          71.1%          87.3%          82.8%          73.5%          87.3%          64.2%          88.2%          89.7%          83.8%
sft_v1 + shim                     87.7%          87.7%          87.7%          87.7%          87.7%          87.7%          87.7%          87.7%          87.7%          87.7%          87.7%          87.7%

parallel items only (n=48)
model                                v1 v10_positional       v11_plan v12_xml_opaque      v2_rename      v3_schema      v4_format      v5_prompt         v6_all      v7_opaque      v8_native      v9_pycall
base                              47.9%          70.8%          14.6%          72.9%          37.5%          54.2%          70.8%          62.5%          64.6%          33.3%          77.1%          72.9%
ceiling_v11_plan                  85.4%          87.5%          83.3%          79.2%          85.4%          83.3%          77.1%          85.4%          79.2%              -          83.3%          83.3%
ceiling_v4_format                 29.2%              -              -              -          27.1%          39.6%          81.2%           6.2%          79.2%              -              -              -
ceiling_v6_all                     6.2%              -              -              -           8.3%          16.7%          81.2%          33.3%          83.3%              -              -              -
dawe_control_rand_eos             81.2%              -              -              -          81.2%          81.2%           2.1%          81.2%           0.0%              -              -              -
dawe_s1_eos12                     81.2%              -              -              -          81.2%          81.2%          81.2%          83.3%          77.1%              -              -              -
dawe_s1_eos12_DE                      -              -          52.1%          83.3%              -              -              -              -              -              -              -              -
dawe_s2_plan8_eos12               81.2%          72.9%          87.5%          83.3%          81.2%          81.2%          81.2%          81.2%          81.2%          83.3%          83.3%          72.9%
sft_v1                            81.2%          75.0%          39.6%          18.8%          81.2%          81.2%          25.0%          81.2%           0.0%          83.3%          79.2%          75.0%
sft_v1 + shim                     81.2%          81.2%          81.2%          81.2%          81.2%          81.2%          81.2%          81.2%          81.2%          81.2%          81.2%          81.2%
```
