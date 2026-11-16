# Licenses and terms

Code, model, and data are separate things with separate terms. This file lists each one, what was
checked, and what is still open. It is a record of what we found, not legal advice.

Checked on 2026-11-16 against the Hugging Face and PyPI metadata for the exact revisions and versions we pin.

## 1. This repository's code: Apache-2.0

[`LICENSE`](../LICENSE) is the unmodified Apache License 2.0 text. [`NOTICE`](../NOTICE) names the copyright
holder as "The DAWE authors". Replace that with the legal owner (person or organization) before
publishing if it should be a specific entity.

Covers: `dawe/`, `gpu/`, `scripts/`, `tests/`, `harnesses/`, `docs/`, `splits/`, `sessions/` (see item 6),
`release/`, `repro/`.

No third-party source files are vendored or copied into this repository. `dawe/bfcl.py` contains our own
simplified answer checker; it is not BFCL's official scorer and does not reuse its code (see
[WRITEUP.md](WRITEUP.md#limitations)).

## 2. Dependencies (installed, not redistributed)

| Package | Pinned | Used by | License (PyPI metadata unless noted) | Compatible with Apache-2.0 |
|---|---|---|---|---|
| *(none)* | | `dawe` itself, including `dawe patch` | standard library only | n/a |
| pytest | 9.2.0 | tests | MIT | yes |
| safetensors | 0.8.0 | tests (cross-check of our file reader/writer) | Apache-2.0 (classifier) | yes |
| numpy | 2.5.4 | tests | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | yes |
| hatchling | build only | packaging | MIT | yes |
| torch | 2.13.0 | GPU jobs | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | yes |
| transformers | 5.20.2 | GPU jobs | Apache-2.0 | yes |
| peft | 0.21.2 | SFT job | Apache-2.0 | yes |
| accelerate | 1.15.0 | GPU jobs | Apache-2.0 | yes |
| flash-linear-attention | 0.5.2 | GPU jobs (kernels for Qwen3.5's linear attention) | MIT | yes |
| vllm | 0.31.0 | generation jobs | Apache-2.0 | yes |
| hf_transfer | unpinned | GPU jobs (downloads) | Apache-2.0 (upstream repository; the PyPI record has no license field) | yes |
| modal | client, unpinned | launching GPU jobs | Apache-2.0 (the Modal service has its own terms of service) | yes |

None of these are bundled into a release of this repository. Transitive dependencies of the GPU images are
resolved when the images are built; their licenses have **not** been audited. `repro/freeze_*.txt`
lists them (see [REPRODUCE.md](REPRODUCE.md#exact-environment)); audit that list if you
redistribute a container image.

## 3. Model terms (separate from the code license)

| Model | Role | License | Where verified |
|---|---|---|---|
| `Qwen/Qwen3.5-4B` @ `851bf6e8…` | base model, never redistributed here | Apache-2.0 | model card front matter and the repository's `LICENSE` file at that revision |
| `sft_v1/merged` (our LoRA fine-tune merged into the base) | the checkpoint every patch applies to | derivative of the above, so Apache-2.0 conditions carry over: keep Qwen's license text and say that it was modified | **not published**; see below |
| `.gpatch` files | weight differences plus hashes, no tensors from the base | offered under Apache-2.0 by us; only meaningful with the checkpoint above | n/a |

Open items for the maintainers:

- **Whether and where to publish `sft_v1/merged`.** Patches refuse to apply to anything else (that is the
  point of [PATCH_FORMAT.md](PATCH_FORMAT.md)), and LoRA training is not guaranteed to reproduce the same
  bytes, so without the exact checkpoint a reader can only re-derive patches, not apply ours. Publishing it
  is a decision for the maintainers; if they do, include Qwen's `LICENSE`, link the base revision, and describe
  the changes (`release/lineage_sft_v1.json` has the recipe).
- Qwen's license has not been compared line by line against Apache-2.0 beyond the identification above.

## 4. Dataset terms (separate from the code license)

| Dataset | License | Where verified |
|---|---|---|
| Berkeley Function Calling Leaderboard v3 (simple, multiple, parallel), `gorilla-llm/Berkeley-Function-Calling-Leaderboard` @ `61fc0608…` | Apache-2.0 | dataset card front matter at that revision |

- The data files are **not** in this repository. `dawe fetch` downloads them from the pinned revision and
  checks their SHA-256 (`dawe/pins.py`).
- Excerpts that *are* in the repository: `sessions/s1/context/paired_examples.md`,
  `sessions/s2/context/example_v11_plan.md` (a few BFCL items rendered under each harness), and
  `sessions/*/tries/*.json`, `sessions/*/context/baseline_train.json` (train-item ids, model outputs on them).
  They are reproduced under BFCL's Apache-2.0 terms; the BFCL citation is in the [README](../README.md#credits).
- We did not audit where BFCL's own items came from (e.g. the public APIs whose documentation some function
  schemas imitate). The dataset card states Apache-2.0 and lists no separate third-party terms.
- Tokens, prompts, and answers derived from BFCL (SFT files, eval prompts) are generated locally into
  `data/` and are not committed.

## 5. Results and predictions

`RESULTS.md` contains accuracy numbers only. Raw model predictions (`runs/preds_*.jsonl`) are not in the
repository yet; they quote BFCL test items' model outputs, so the same dataset terms apply when they are added.

## 6. Material written by an AI agent

`sessions/*/edits`, `analysis`, `notes.md`, `LOG.md`, and the `SUBMITTED.json` notes were written by an AI
coding agent working inside the lab (see [LAB.md](LAB.md)); the operator reported the agent as Codex
running a model called "GPT-6 Astra", which we cannot independently verify. They are published as experimental records
so readers can inspect exactly what the agent did.

**Open item for the maintainers:** confirm that the agent provider's terms let you release its output under
Apache-2.0. We have not checked those terms. If you cannot, the alternative is to keep `sessions/` under a
"records, all rights reserved" notice and keep only the code under Apache-2.0.

## 7. Credit (not a license matter)

- Qwen team, for Qwen3.5-4B.
- BFCL authors (Yan, Mao, Ji, Zhang, Patil, Stoica, Gonzalez) for the benchmark.
- *The Surprising Effectiveness of Deleting Weights in LLM Post-Training* (arXiv 2611.10866) and
  *Sparks of Weight-Space Programming* (Takehi et al., 2026-11-02) for the ideas this builds on.
