# Prepare a DAWE run without a GPU

These eight synthetic records illustrate the input contract, grouped split, and provenance fields. They are not model results or evidence that a patch works.

From the repo root:

```bash
uv run dawe workbench init --source examples/dawe/examples.jsonl --model /path/to/checkpoint --out /tmp/dawe-example-run
uv run dawe workbench freeze examples/dawe/examples.jsonl \
  --development-dir /tmp/dawe-example-run/edit \
  --evaluation-dir /tmp/dawe-example-evaluation
```

Inspect the two manifests. Related old/new examples stay together, and both partitions have repair and preservation records. No weights were loaded and no compute was started. The synthetic `ping` replies are placeholders for a user's reviewed targets, not an instruction to edit any model.

For a real run, give your coding agent the raw data and ask it to read `skills/dawe/SKILL.md`. It prepares source-specific examples, a scorer and the model adapter, maps failing decisions, and derives candidate edits. The existing BFCL sessions are the worked model-editing example; this fixture tests only local preparation.
