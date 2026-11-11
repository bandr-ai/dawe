# geodesic

Edit model weights instead of retraining. First use: a harness change (renamed tools,
new schema, new call format, new prompt) breaks a fine-tuned model; fix it with a small
weight edit found from paired traces (DAWE, data-attributed weight editing).

## Status

Working end to end. Results: [RESULTS.md](RESULTS.md). Agent sessions (briefs, logs, edits, notes): [sessions/](sessions/).

- Qwen3.5-4B, LoRA SFT on BFCL (train split) under harness v1; scored on the held-out test split (204 items).
- Two harness changes break it: XML calls (v4, 87.7 → 73.5) and plan-first replies (v11, 87.7 → 20.1).
- An agent (Codex, GPT-6 Astra) given the data→weight map writes a closed-form weight patch, no training:
  - v4: 2,557 weights → 85.8; v11: 5,064 weights → 88.2 (retraining: 88.7); v1 unchanged at 87.7.
  - Same-size random edit to the same row: no fix (68.1).

GPU side: `gpu/app.py` (SFT, generation, map), `gpu/lab.py` (the agent's lab: `scripts/lab.py run|patch|try|submit`,
budgets enforced on the Modal volume, train items only).

```bash
uv sync --extra dev
uv run geodesic fetch          # BFCL v3 simple/multiple/parallel -> data/bfcl
uv run geodesic harnesses      # list harness versions
uv run geodesic render v6_all parallel_0
uv run geodesic roundtrip      # correct answers survive every harness
uv run geodesic build-sft      # BFCL train split under v1 -> data/sft/
uv run geodesic build-eval     # BFCL test split x 6 harnesses -> data/eval/bfcl_test.jsonl
uv run geodesic score preds.jsonl   # accuracy + format failures per harness
uv run pytest -q
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
| `v7_opaque` | Tool names → per-item IDs (`t_3f9a`), arguments → `a1, a2, …` |
| `v8_native` | Qwen3.5's own tool format (`<tool_call><function=…><parameter=…>`) |
| `v9_pycall` | Python call syntax `[f(x=1)]` |
| `v10_positional` | Positional arguments `[f(1, 2)]` |
| `v11_plan` | Reply must start with `<plan>…</plan>`, then the JSON calls |
| `v12_xml_opaque` | v4 XML plus v7 opaque names |

Scoring always happens in BFCL's own names. A call using a name the current harness
doesn't offer fails, even if it was the right name under an older harness.

`simple_363` is skipped: its answer calls a function the item doesn't offer.

## Data

BFCL v3 simple/multiple/parallel only (799 items). Split 75/25 by a hash of the item id,
the same ratio Deletion Tuning used for SFT. Train = SFT on harness v1; test = held-out items under every harness.
