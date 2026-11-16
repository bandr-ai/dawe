# DAWE

**Bring your data and model. Get a tested weight patch.**

DAWE is an agent skill for fixing model behavior with small weight edits. Give it logs, conversations,
tool traces, or corrected examples, point it at your checkpoint, and describe what should change.
The agent prepares the data, investigates the failure, derives candidate edits, and tests that the fix
preserves behavior that already works.

## Use it

Clone the repo and install the local tools (Python 3.11+ and uv):

```bash
git clone https://github.com/bandr-ai/dawe.git
cd dawe
uv sync
```

Open the repo in your coding agent and give it this:

```text
Read skills/dawe/SKILL.md and use DAWE.

My data is in /path/to/data.
My model checkpoint is /path/to/checkpoint.
Fix: [describe the behavior you want to change].
Preserve: [describe what already works].

Handle the data preparation and evaluation. Inspect the files first and
propose a compute budget before paid work. Return a tested patch, the
results, and the scripts used to produce it. Stop GPU resources when done.
```

That's the starting point. You don't need to choose layers, label decision tokens, or convert your logs
by hand. The agent adapts the workflow to your data and model. It needs accessible weights and reliable
correct answers or a verifier; it asks for missing essentials after inspecting what you supplied.

You get a report, evaluation results, and—when an edit passes the checks—a checkpoint-bound `.gpatch`.
Apply it to a new checkpoint directory:

```bash
uv run dawe patch apply /path/to/fix.gpatch --source /path/to/checkpoint --out /path/to/patched-model
```

Use that directory with your compatible inference runtime. Keep the original checkpoint; the patch can
also be undone with `dawe patch restore`.

DAWE is experimental. It accepts different data sources through the agent, but a successful small edit
is not guaranteed for every failure. The worked experiment below repaired structural output failures;
it does not establish general reasoning or knowledge repair.

[Full user guide](docs/USAGE.md) · [The skill](skills/dawe/SKILL.md) · [No-GPU example](examples/dawe/README.md)

## Install the skill elsewhere

You can also copy the bundled skill into your agent's configured skills directory:

```bash
uv run dawe skill install --dest /path/to/your/skills
```

For explicit local preparation, use `dawe workbench init` and `dawe workbench freeze`; the agent can run
these for you. Preparation starts no compute. Keep private data, runs, and checkpoints out of Git.

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
uv run dawe fetch                  # BFCL v3 simple/multiple/parallel, pinned revision, SHA-256 checked
uv run dawe splits                 # the 595/204 split matches splits/bfcl_v3_split.json
uv run dawe roundtrip              # correct answers survive every harness
uv run dawe harnesses              # the 12 harness versions
uv run python examples/patch_lifecycle.py
```

## Patches

A patch is sparse (changed indices and values: tens of KB, not the GB-sized tensor it edits), names the exact checkpoint it
was made for by content hash, refuses any other, writes a new directory, and can be undone byte for byte:

```bash
dawe patch inspect x.gpatch
dawe patch apply   x.gpatch --source CHECKPOINT --out NEW_DIR
dawe patch restore x.gpatch --patched NEW_DIR   --out ORIGINAL_DIR
```

Standard library only; the patch tools do not import torch. The two released patches (23.6 KB and 43.7 KB) are in
[`release/patches/`](release/patches/) with their SHA-256 in [release/README.md](release/README.md); the checkpoint they apply
to is not published (see [docs/LICENSES.md](docs/LICENSES.md)).

## Layout

```
dawe/        harnesses, BFCL loader and scorer, dataset builders, map reader, splits, pins, patch format
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
dawe fetch | splits [--write] | harnesses | render <harness> <id> | roundtrip
dawe build-sft [--harness H] | build-eval | build-pairs A B | map-summary MAP.json | report | score preds.jsonl
dawe patch create | from-dense | inspect | apply | restore | verify
dawe skill path | install --dest DIR
dawe workbench init --source DATA --model CHECKPOINT --out RUN
dawe workbench freeze EXAMPLES --development-dir EDIT --evaluation-dir EVAL
```

## Credits

Data: Yan, Mao, Ji, Zhang, Patil, Stoica, Gonzalez, *Berkeley Function Calling Leaderboard* (2024), Apache-2.0.
Model: Qwen3.5-4B, Apache-2.0. Ideas: *The Surprising Effectiveness of Deleting Weights in LLM Post-Training* (arXiv 2611.10866) and
*Sparks of Weight-Space Programming* (Takehi et al., 2026). Details and terms: [docs/LICENSES.md](docs/LICENSES.md).

## License

Code: [Apache-2.0](LICENSE). The model and dataset have their own terms, listed separately in
[docs/LICENSES.md](docs/LICENSES.md). Contents of `sessions/` were written by an AI agent; see the open item there.
