# Reproducing the experiment

Three levels, from free to expensive. Do them in order; each tells you whether the next is worth it.

| Level | Needs | Time / cost | What it verifies |
|---|---|---|---|
| A. Checks | a laptop | minutes, free | the data, the split, the scorer, the harnesses, the patch tools |
| B. Re-derive | Modal account, an H100 | about an hour, a few dollars | baseline model, the failure, the map, a patch, held-out numbers |
| C. Apply our patches | the exact `sft_v1` checkpoint + `release/patches/*.gpatch` | minutes | that the shipped patches reproduce our numbers |

Level C is blocked on one thing the repository does not contain: the `sft_v1` checkpoint (the patches are in `release/patches/`;
see [release/README.md](../release/README.md) and [LICENSES.md](LICENSES.md#3-model-terms-separate-from-the-code-license)).
Level B does not need either, it makes its own.

## Exact environment

| Thing | Pin | Where |
|---|---|---|
| Base model | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `dawe/pins.py` |
| Dataset | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` @ `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, six files with SHA-256 | `dawe/pins.py` (checked by `dawe fetch`) |
| Split | `sha256(item id) mod 100 < 25` is test; 595 train / 204 test; `simple_363` skipped | `splits/bfcl_v3_split.json` (checked by `dawe splits`) |
| Training stack | torch 2.13.0, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2 (Python 3.12) | `dawe/pins.py` → `gpu/app.py` |
| Generation stack | vllm 0.31.0 (Python 3.12) | `dawe/pins.py` → `gpu/app.py`, `gpu/lab.py` |
| SFT recipe | LoRA r=32, alpha 64, every `Linear` in the language model, lr 1e-4, 3 epochs, batch 4×4, seed 0, bf16, loss on reply only | `dawe/pins.py::SFT` |
| Decoding | greedy, max 512 new tokens, thinking off, `max_model_len` 8192 | `dawe/pins.py::GENERATION` |
| Local tooling | `uv.lock` | repo root |

**Not pinned or not yet recorded:**

- Transitive dependencies of the Modal images are resolved at build time. `repro/freeze_*.txt` records what the three images resolve to
  as of 2026-11-16 (regenerate with `modal run gpu/app.py::freeze` and `modal run gpu/lab.py::freeze`); this is not proof of what the
  original runs used. Note the generation image resolves `transformers==5.17.0`, not the training image's 5.20.2. The Modal base image
  (`debian_slim`) is not pinned by digest.
- The BFCL revision above is the current head of the dataset repository (last modified 2026-04-29, before our runs). The
  original fetch used `main`, whose revision was not recorded. The data reproduces the 595 / 204 split and the skipped
  item, which is the evidence that it is the same data; nothing stronger exists.
- GPU training and vLLM generation are not guaranteed bit-reproducible across hardware or driver versions. Expect
  differences of a few items, and a retrained `sft_v1` will almost certainly **not** be byte-identical to ours.

## A. Checks (no GPU)

```bash
uv sync --extra dev
uv run pytest -q                 # 55+ tests: scorer, harness round trips, patch format
uv run dawe fetch            # six files, SHA-256 checked against the pin
uv run dawe splits           # recompute the split, compare with splits/bfcl_v3_split.json
uv run dawe roundtrip        # strict diagnostic; known simple_307 native-format mismatch exits nonzero
uv run dawe harnesses        # list the 12 harnesses
uv run dawe render v4_format parallel_0    # see exactly what the model sees and should say
```

A new split cannot be produced by accident: `dawe splits` fails if the rule, the dataset revision or any id changes.

## B. Re-derive the experiment (Modal)

Setup: `uv sync --extra dev --extra gpu`, `modal setup`. Everything lands in `runs/` (gitignored) and on the
Modal volume `geodesic-runs`. The original research apps and volumes retain their historical names after the DAWE rename so these commands still address the recorded experiment resources. Commands below are in the order we ran them; costs are Modal list prices at the time
(`gpu/app.py::PRICE`, about $4.29 per H100-hour) and are logged to `runs/costs.jsonl`.

### 1. Data and harnesses

```bash
uv run dawe fetch
uv run dawe build-sft                    # data/sft/v1_bfcl_train.jsonl     (595 rows, harness v1)
uv run dawe build-eval                   # data/eval/bfcl_test.jsonl        (204 items x 12 harnesses)
```

### 2. Baseline: LoRA SFT on v1

```bash
modal run gpu/app.py::smoke                  # optional, ~10 min: 10 steps + 20 generations
modal run gpu/app.py::full                   # SFT -> runs/preds_base.jsonl, runs/preds_sft_v1.jsonl
uv run dawe score runs/preds_sft_v1.jsonl
```

This is the failure. Expect `sft_v1` near 87.7% under `v1`, about 73.5% under `v4_format`, about 20.1% under `v11_plan`
(see [RESULTS.md](../RESULTS.md); exact values will differ a little on your hardware).

Retraining ceilings (SFT from the base model directly on a new harness):

```bash
for h in v4_format v6_all v11_plan; do uv run dawe build-sft --harness $h; done
modal run gpu/app.py::ceiling --harnesses v4_format,v6_all,v11_plan
```

Extra harnesses added after the first runs (for example `v7` to `v12`) were scored with
`modal run gpu/app.py::gen --model sft_v1 --eval-file data/eval/extra.jsonl --append`, on an eval file restricted to
the new harnesses (`uv run dawe build-eval --harness v7_opaque v8_native --out data/eval/extra.jsonl`).

### 3. Map the failure

```bash
uv run dawe build-pairs v1 v4_format     # same train items, same answers, two renderings
uv run dawe build-pairs v1 v6_all
modal run gpu/app.py::map_ --pairs v1__v4_format,v1__v6_all
uv run dawe map-summary runs/map_v1__v4_format.json     # is the change concentrated? where?
```

`map-summary` prints, for each harness: how surprised the model is at "another call follows" decisions, how the gradient
spreads over matrices and layers, and per layer how well one direction in hidden space separates "continue" from "stop".
For `v4` the surprise is about 0.001 nats under `v1` and about 0.98 under `v4`; the hidden-state separation appears from
layer 16 on (accuracy 0.95 to 0.99). The summary the agent saw is `sessions/s1/context/map_v1_vs_v4.txt`.

The clean decision-point map the agents actually used, plus the baseline on train items, comes from the lab setup:

```bash
modal deploy gpu/lab.py
modal run gpu/lab.py::setup --session s1                    # v4_format target, keep v1; "continue" decisions only
# s2 (reconstructed from the code and logs; the original shell history was not kept):
modal run gpu/lab.py::setup --session s2 --target v11_plan --keep v1 \
    --kinds plan_start,after_plan --tag plan_ --no-multi-only    # boolean flag spelling depends on your Modal CLI version
```

### 4. The agent session

```bash
mkdir -p /tmp/s1 && cp -r sessions/s1/BRIEF.md sessions/s1/context /tmp/s1/ && cd /tmp/s1
export LAB_SESSION=s1
cp /path/to/repo/scripts/lab.py ./lab   # the brief calls it ./lab; needs `modal` installed
python lab status
```

Point your agent at `/tmp/s1/BRIEF.md`. It will use `lab run | patch | try | submit`. Our sessions used a Codex-based
agent (reported by the operator as GPT-6 Astra); any agent that can run shell commands should work, and results will differ.
You can skip this step and use ours: the edits are in `sessions/s1/edits/eos12.py` and `sessions/s2/edits/plan8_eos12.py`; submit
them to the lab with `python lab patch eos12.py NAME` (which saves a sparse, checkpoint-bound `.gpatch`).

### 5. Held-out evaluation

```bash
modal run gpu/lab.py::final --session s1 --name eos12            # -> runs/preds_dawe_s1_eos12.jsonl
modal run gpu/lab.py::final --session s2 --name plan8_eos12
cp sessions/control/rand.py . && LAB_SESSION=control python lab patch rand.py rand_eos   # the random-edit control
modal run gpu/lab.py::final --session control --name rand_eos
uv run dawe report                                           # the table in RESULTS.md
```

`dawe report` reads every `runs/preds_*.jsonl`, scores it with the committed scorer, and prints both tables.

## C. Apply our patches

```bash
dawe patch apply release/patches/s1_eos12.gpatch --source /path/to/sft_v1/merged --out /path/to/sft_v1_xml
modal run gpu/app.py::gen --model <that checkpoint>          # then score as above
```

The apply step refuses any checkpoint that is not byte-for-byte the one the patch was made from. If you retrained
`sft_v1`, expect a refusal; make a patch for your own model by running the agent's edit script instead
(section B.4) or use `dawe patch create --source YOURS --edited EDITED`.

## What would make this stronger

Committed per-item predictions (so `report` regenerates RESULTS.md and paired tests are possible), the exported
patches, the frozen environment, a second SFT seed, and the controls listed in [EXPERIMENT.md](EXPERIMENT.md#not-yet-done).

## Known roundtrip mismatch

`v8_native` roundtrips 798/799 gold replies; the other eleven harnesses roundtrip 799/799. In `simple_307`,
the string-typed optional `venue` has gold choices including boolean `True`. Native serialization and
schema-based parsing return the string `"True"`, which the scorer rejects. The release tests isolate this
exact case as a strict expected failure: a fix must remove the expected failure, and new failures are errors.
The strict `dawe roundtrip` diagnostic continues to fail on it. This test policy changes neither gold
targets nor scoring and does not revise historical research results.
