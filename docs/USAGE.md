# Use DAWE with your model and data

Give a coding agent your traces, the exact model checkpoint, and a description of what is failing. The agent follows the [DAWE skill](../skills/dawe/SKILL.md) to investigate the failure, derive a small weight edit when the evidence supports one, and evaluate it against behavior you want to preserve.

The starting point is simple: open this repository in your coding agent and give it the data path, checkpoint path, and desired fix. It handles normalization, scoring, model inspection, editing, and evaluation. The sections below explain what happens behind that prompt; you do not need to carry out each preparation step yourself.

## 1. Bring these inputs

| Input | What to provide |
|---|---|
| Model | The exact checkpoint you serve, preferably unquantized Transformers safetensors, plus its tokenizer and chat-template settings. An API endpoint alone cannot be weight-edited. |
| Failure examples | Raw logs, conversations, tool traces, or corrected examples. Say what is wrong: for example, “after switching to XML, it stops after the first of two required tool calls.” |
| Correct behavior | Reviewed corrections, tool schemas plus an executable verifier, or another trustworthy way to establish the answer. Failed traces alone do not identify the correct answer. |
| Behavior to preserve | Independent examples of things that already work, such as single-call requests and the old output format. |
| Run constraints | Available hardware, maximum candidates, a compute budget, and acceptable preservation regressions. These can be settled after local inspection. |

You do not need to choose layers, label decision tokens, or format every log into a special schema. The agent does that after inspecting your source. If reliable targets or accessible weights are missing, it reports the gap before attempting edits.

Keep data and checkpoints outside the source repository. A possible local layout is:

```text
~/dawe/                    # this repository
~/dawe-inputs/logs/            # raw data, kept intact
~/models/my-model/             # exact source checkpoint
~/dawe-runs/xml-repair/        # agent workspace, created separately
~/dawe-evaluation/xml-repair/  # frozen evaluation data, outside editing workspace
```

## 2. Install and give the agent the task

Use Python 3.11+ and uv. From a source checkout containing `skills/dawe/SKILL.md`:

```bash
git clone https://github.com/bandr-ai/dawe.git
cd dawe
uv sync
uv run dawe skill path
```

Give your coding agent this prompt, replacing the paths and failure description:

```text
Read skills/dawe/SKILL.md and use it on my data.

Data: /absolute/path/to/dawe-inputs/logs
Model: /absolute/path/to/models/my-model
Run directory: /absolute/path/to/dawe-runs/xml-repair

After our harness switched to XML, the model often ends its turn after
the first tool call even when the request needs two calls. Corrected
examples and tool schemas are in the data folder. Preserve single-call
behavior and the old JSON format.

Inspect the files first. Prepare a scorer and separate development and
evaluation groups. Explain the proposed compute and acceptance criteria
before starting paid work; no paid compute is authorized yet.

Then, within the agreed limits, diagnose the failing decisions, derive
and test small closed-form weight edits, and evaluate the chosen patch.
Return REPORT.md, the patch if successful, the scripts and per-item
results. Stop any GPU resources you start when finished.
```

If you already have an agreed budget or available local GPU, include that instead. Set a numeric preservation tolerance before candidate selection. Model size and runtime determine memory and cost; the original experiment's costs are not a quote for your run.

Reading the skill directly is enough. Optionally copy it into the skills directory your agent uses:

```bash
uv run dawe skill install --dest /absolute/path/to/your/skills
```

This creates `your/skills/dawe/` and refuses to overwrite an existing copy. Agent discovery depends on your agent's skills configuration.

## 3. What happens during the run

1. **Inspect and prepare.** The agent reads a small source sample, checks the checkpoint/runtime, and creates a run workspace. It recovers reviewed targets and writes a source-specific normalizer and task scorer. You review unresolved questions about the intended behavior.
2. **Freeze the evidence.** Related attempts, old/new formats, and corrections stay in the same group. Examples already inspected go into development. A separate evaluation set contains both repair and preservation cases. Evaluation contents and scores stay out of candidate selection; directory separation is cooperative, not filesystem isolation.
3. **Measure the baseline.** The agent generates actual replies under your serving template and decoding settings, scores repair and preservation behavior, and records the model fingerprint. Local preparation has not launched a model job; this stage requires the agreed compute.
4. **Map and edit.** It identifies specific failing token decisions and inspects model states. It derives small candidate edits from development evidence, such as a closed-form readout correction, and scores generated outputs after each try. It starts each candidate from the original checkpoint. There is no optimizer-based fine-tuning in this workflow.
5. **Evaluate once selected.** It freezes the chosen edit and scorer, then compares baseline and patched generations on evaluation examples using identical settings. It reports repair gains and preservation regressions separately, with counts and per-item results.
6. **Deliver and stop.** If acceptance criteria pass, it exports a checkpoint-bound `.gpatch` and verifies apply/restore. Otherwise it returns the diagnosis and failed attempts. It stops at the candidate/budget cap and shuts down resources either way.

For reference, the preparation commands the agent uses are:

```bash
uv run dawe workbench init \
  --source /absolute/path/to/dawe-inputs/logs \
  --model /absolute/path/to/models/my-model \
  --out /absolute/path/to/dawe-runs/xml-repair

# After the agent has normalized and reviewed examples:
uv run dawe workbench freeze /absolute/path/to/normalized.jsonl \
  --development-dir /absolute/path/to/dawe-runs/xml-repair/edit \
  --evaluation-dir /absolute/path/to/dawe-evaluation/xml-repair
```

`init` writes `run.json`, `AGENTS.md`, a copy of the skill, and a `.gitignore`; it does not copy data, load weights, or start compute. `freeze` validates and splits the agent-prepared records; it does not establish whether the targets are correct. Output directories must be new. Budget and try-count settings record limits for the agent; they do not enforce a provider spending cap.

You can try just these preparation steps with the [synthetic no-GPU example](../examples/dawe/README.md). That example does not produce a trained model or a working patch.

## 4. What you get back and how to use it

Expect a `REPORT.md` explaining the exact checkpoint, data/split hashes, scorer, baseline and patched counts, preservation regressions, compute used, and limitations. Keep the normalizer, scorer, edit code, try log, and per-item evaluation results with it. A successful investigation also returns a frozen `.gpatch`.

Inspect and apply that patch to the exact source checkpoint:

```bash
uv run dawe patch inspect /absolute/path/to/candidate.gpatch
uv run dawe patch apply /absolute/path/to/candidate.gpatch \
  --source /absolute/path/to/models/my-model \
  --out /absolute/path/to/models/my-model-patched
```

Applying writes a new checkpoint directory. Point your existing compatible inference runtime at that directory and run your serving checks before using it in production. Applying a patch does not deploy it. Keep the original checkpoint; a patch for it is not portable to another model or even a different fine-tune of the same base.

To reconstruct the source weights from the patched checkpoint:

```bash
uv run dawe patch restore /absolute/path/to/candidate.gpatch \
  --patched /absolute/path/to/models/my-model-patched \
  --out /absolute/path/to/models/my-model-restored
```

The [patch format guide](PATCH_FORMAT.md) describes compatibility and verification. The two published research patches require our exact `sft_v1` checkpoint, which is not published; use your own run to create a patch for your model.

## What this can promise today

The worked experiment repaired premature stopping and plan-opening failures on one model and benchmark subset. It does not establish that arbitrary reasoning errors, missing knowledge, or every harness change admit a tiny edit. The bundled collector targets supported Transformers text chat templates; other architectures, quantized weights, and custom runtimes require adaptation and may be unsupported.

The original idea—subtract two separately trained models to obtain a harness delta and transfer it to a model trained on another task—has not been demonstrated here. This workflow derives an edit from the specific model's states and trustworthy targets. It can use new-harness examples; it is not a zero-new-harness-data transfer result. Read the [write-up](WRITEUP.md) for the measured evidence and remaining controls.
