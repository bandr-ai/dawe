# User data and frozen examples

Accept logs, conversations, tool traces, or corrected examples. Inspect their actual structure and implement a normalizer for that source. Do not require users to hand-author this schema.

Each normalized JSONL object has:

- `id`: unique example ID.
- `group_id`: underlying task/instance ID. All repeats, old/new harness versions, and related corrections use the same group.
- `purpose`: `repair` or `preserve`.
- `source_ref`: original file and record/span reference.
- `target_source`: how the correct target was established (supplied correction, deterministic verifier, oracle, etc.). This is provenance, not independent validation by DAWE.
- `messages`: text chat messages ending in the reviewed assistant target. At least one preceding user message. Render native tool objects into the model's supported format in your adapter.
- Optional `decisions`: objects with `kind` and `prefix`, the exact assistant-target text BEFORE a next-token decision. An empty prefix means the start of the reply. The full target means the next template token after the reply; inspect which token that actually is.
- Optional `split`: `development` or `evaluation`. If used, every group must have a split, and group members cannot disagree. Use existing customer evaluation partitions rather than resplitting them. Assign groups seen during source inspection to development; do not call them held-out later.
- Other fields can carry condition names, verifier inputs, acceptable outputs, or task-specific scoring metadata.

Example (one record; enough independent groups are needed to freeze):

```json
{"id":"case-01-new","group_id":"case-01","purpose":"repair","source_ref":"logs/session.jsonl:12","target_source":"reviewed correction","messages":[{"role":"user","content":"Call the ping tool using XML."},{"role":"assistant","content":"<tool_call name=\"ping\"/>"}],"decisions":[{"kind":"open_xml","prefix":""}]}
```

Freeze using the lightweight CLI:

```bash
dawe workbench freeze normalized.jsonl \
  --development-dir /path/to/dawe-run/edit \
  --evaluation-dir /path/to/evaluator/held-out
```

Without explicit splits, it deterministically assigns groups, stratified by repair/preserve membership, targeting 20% evaluation. Both partitions must contain repair and preserve examples; a tiny dataset may need more independent groups. The helper rejects identical prompts assigned to different groups, but cannot detect semantic near-duplicates. Inspect those yourself. It does not validate that targets are true or that labels were reviewed.

Only copy `edit/` onto the editing worker. Do not mount the original normalized file, raw source directory, or held-out directory there. A local coding agent can still access files outside `edit/`; this is a cooperative separation, not an access-control guarantee.

Freeze the scorer and save its code/hash. Prefer executable task checks to string equality when multiple answers or formats are valid. Score generated outputs, not just gold-prefix token probabilities. Record zero valid examples or scorer failures as errors, never as 100% success.
