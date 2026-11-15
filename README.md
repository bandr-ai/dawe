# geodesic

Edit model weights instead of retraining. First use: a harness change (new call format, plan-first replies, ...)
breaks a fine-tuned model; a small weight edit found by an agent from paired traces repairs it.

**This release is a reproducible experiment, not a product.** It contains the harnesses, the split, the scorer, the
baseline recipe, the data-to-weight map code, the bounded agent lab, everything the agents wrote, a sparse and
checkpoint-bound patch format with apply and restore tools, and the numbers. Read
[docs/WRITEUP.md](docs/WRITEUP.md) for what was found and, just as important, what was not shown.

## Use DAWE with your own data

Give your coding agent this prompt:

```text
Read skills/dawe/SKILL.md and use DAWE on my traces in /path/to/logs.
The model checkpoint is /path/to/checkpoint. Investigate why it stops after
one tool call, derive a small weight patch, and test that existing behavior
still works. Inspect the data first and propose the compute budget before
starting paid GPU work.
```

The agent inspects your source, prepares corrected and preservation examples, maps the failing decisions, writes and tests edits, and returns a checkpoint-bound patch with its evaluation. You can supply other model failures too; a small edit is not guaranteed. Model weights and trustworthy targets or a verifier are required. This is a skill-driven investigation, not an automatic patch-search service.

The skill is bundled in source and Python wheels. Read it directly, locate an installed copy with `geodesic skill path`, or copy it into a skills directory you choose:

```bash
geodesic skill install --dest /path/to/your/skills
geodesic workbench init --source /path/to/logs --model /path/to/checkpoint --out /path/to/dawe-run
```

Preparation is local and starts no compute. The agent adapts the user's data/runtime; the original Modal lab remains BFCL-specific. See [the skill](skills/dawe/SKILL.md) and [the no-GPU example](examples/dawe/README.md). Keep private runs, data, and checkpoints out of Git.

## The result in one table

Qwen3.5-4B, LoRA SFT on BFCL under harness `v1`; 204 held-out items, greedy, one run each
([RESULTS.md](RESULTS.md)):

| | `v1` | `v4_format` (XML calls) | `v11_plan` (plan first) | Weights changed |
|---|---|---|---|---|
| fine-tuned on `v1` | 87.7 | 73.5 | 20.1 | none |
| + agent patch for `v4` | 87.7 | 85.8 | 64.2 | 2,557 |
| + agent patch for `v11` | 87.7 | 85.8 | 88.2 | 5,064 |
| retrain on the new harness only | 73.0 / 90.2 | 89.7 | 88.7 | all |

Single run, 204 items (about ±5 points), our own scorer rather than BFCL's, no unconditional-bias control yet. See
[Limitations](docs/WRITEUP.md#limitations) before quoting these.

## Start here

| I want to... | Go to |
|---|---|
| see the story from baseline to patch to evaluation | [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) |
| check what is and is not claimed | [docs/WRITEUP.md](docs/WRITEUP.md) |
| re-run it | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| use or inspect a patch | [docs/PATCH_FORMAT.md](docs/PATCH_FORMAT.md), [release/](release/) |
| read what the agents did | [sessions/](sessions/) |
| understand the lab's guarantees | [docs/LAB.md](docs/LAB.md) |
| know the licenses and model/dataset terms | [docs/LICENSES.md](docs/LICENSES.md) |

## Quick start (no GPU)

```bash
uv sync --extra dev
uv run pytest -q
uv run geodesic fetch                  # BFCL v3 simple/multiple/parallel, pinned revision, SHA-256 checked
uv run geodesic splits                 # the 595/204 split matches splits/bfcl_v3_split.json
uv run geodesic roundtrip              # correct answers survive every harness
uv run geodesic harnesses              # the 12 harness versions
uv run python examples/patch_lifecycle.py
```

## Patches

A patch is sparse (changed indices and values: tens of KB, not the GB-sized tensor it edits), names the exact checkpoint it
was made for by content hash, refuses any other, writes a new directory, and can be undone byte for byte:

```bash
geodesic patch inspect x.gpatch
geodesic patch apply   x.gpatch --source CHECKPOINT --out NEW_DIR
geodesic patch restore x.gpatch --patched NEW_DIR   --out ORIGINAL_DIR
```

Standard library only; the patch tools do not import torch. The two released patches (23.6 KB and 43.7 KB) are in
[`release/patches/`](release/patches/) with their SHA-256 in [release/README.md](release/README.md); the checkpoint they apply
to is not published (see [docs/LICENSES.md](docs/LICENSES.md)).

## Layout

```
geodesic/        harnesses, BFCL loader and scorer, dataset builders, map reader, splits, pins, patch format
harnesses/       v1 ... v12 as TOML (v1 is the baseline; each change is one or two fields)
splits/          the train/test split as a file
gpu/             Modal jobs: SFT, generation, maps (app.py); the agent lab (lab.py)
scripts/lab.py   the agent's command-line tool
sessions/        briefs, context, analysis, edits, tries, notes, logs for each agent session, plus the random control
release/         patches and the lineage of the checkpoint they apply to
repro/           frozen environments (to be recorded)
examples/        runnable demos
docs/            write-up, walkthrough, reproduction, patch format, lab, licenses
```

## Harness versions (`harnesses/`)

| Version | Change from v1 |
|---|---|
| `v1` | Baseline: JSON schema in system prompt, JSON `tool_calls` output |
| `v2_rename` | Tool names: `math.factorial` -> `tool_math_factorial` |
| `v3_schema` | camelCase argument names, tools shown as signatures |
| `v4_format` | XML `<tool_call>` blocks instead of JSON |
| `v5_prompt` | New system prompt, tools moved into the user message |
| `v6_all` | All of the above |
| `v7_opaque` | Tool names -> per-item IDs (`t_3f9a`), arguments -> `a1, a2, ...` |
| `v8_native` | Qwen3.5's own tool format (`<tool_call><function=...><parameter=...>`) |
| `v9_pycall` | Python call syntax `[f(x=1)]` |
| `v10_positional` | Positional arguments `[f(1, 2)]` |
| `v11_plan` | Reply must start with `<plan>...</plan>`, then the JSON calls |
| `v12_xml_opaque` | v4 XML plus v7 opaque names |

Scoring always happens in BFCL's own names. A call using a name the current harness doesn't offer fails, even if it was the
right name under an older harness. `simple_363` is skipped: its answer calls a function the item doesn't offer.

## Data and split

BFCL v3 simple/multiple/parallel only (799 items, 798 usable), from a pinned dataset revision. Split 75/25 by a hash of the
item id (`sha256(id) mod 100 < 25` is test). Train = SFT on `v1` and everything the
agents saw; test = held-out items under every harness.

## Commands

```
geodesic fetch | splits [--write] | harnesses | render <harness> <id> | roundtrip
geodesic build-sft [--harness H] | build-eval | build-pairs A B | map-summary MAP.json | report | score preds.jsonl
geodesic patch create | from-dense | inspect | apply | restore | verify
geodesic skill path | install --dest DIR
geodesic workbench init --source DATA --model CHECKPOINT --out RUN
geodesic workbench freeze EXAMPLES --development-dir EDIT --evaluation-dir EVAL
```

## Credits

Data: Yan, Mao, Ji, Zhang, Patil, Stoica, Gonzalez, *Berkeley Function Calling Leaderboard* (2024), Apache-2.0.
Model: Qwen3.5-4B, Apache-2.0. Ideas: *The Surprising Effectiveness of Deleting Weights in LLM Post-Training* (arXiv 2611.10866) and
*Sparks of Weight-Space Programming* (Takehi et al., 2026). Details and terms: [docs/LICENSES.md](docs/LICENSES.md).

## License

Code: [Apache-2.0](LICENSE). The model and dataset have their own terms, listed separately in
[docs/LICENSES.md](docs/LICENSES.md). Contents of `sessions/` were written by an AI agent; see the open item there.
