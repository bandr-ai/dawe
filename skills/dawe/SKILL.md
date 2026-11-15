---
name: dawe
description: Investigate failures in a user's model using their traces or corrected examples, map failing decisions to model states, derive small weight edits, and evaluate and export a checkpoint-bound patch. Use for data-attributed weight editing with accessible model weights; API-only models can be diagnosed but cannot be patched.
---

# DAWE: from failing examples to a weight patch

The user supplies data and a model. You do the source-specific work: inspect the data, build a reliable scorer, identify the failure, inspect model states, derive an edit, test it, and deliver a patch with evidence. They should not need to pre-label layers or know the DAWE machinery. Do not substitute a fine-tuning run for weight editing.

This skill is an experimental workflow, not a promise that every behavior has a small fix. The published Geodesic experiment repaired structural output failures in one model. Use that evidence as a worked example, not a universal prescription to edit EOS or two embedding rows.

## Start with what the user provided

Inspect a small sample of their files and the existing model/runtime before asking questions. Discover available corrections, verifier code, tool schemas, repeated attempts, and known-good examples. Keep raw files intact. Do not print credentials or upload source data to another service without authorization.

You need the intended failure, accessible weights for the exact checkpoint, trustworthy target behavior, and a way to score it. Ask only for missing essentials that the files and conversation cannot resolve. An API endpoint alone does not provide weights. A failed trace alone does not establish a correct answer: recover one from an executable verifier, supplied corrections, or a reliable oracle; otherwise report that gap instead of inventing gold labels.

Create a run directory separate from the data:

```bash
geodesic workbench init --source /path/to/data --model /path/to/checkpoint --out /path/to/dawe-run
```

This only writes local preparation files. Use the user's existing compute authorization. If none exists, finish data inspection and the concrete compute plan before asking for a budget immediately before a paid run. Record model and agent costs separately. Do not start a cloud GPU or leave one warm merely to prepare the workspace.

## Prepare and freeze the evidence

Read [the data contract](references/data.md). Write a small source-specific normalizer, not a universal parser guessed from one sample. Preserve source references and how each target was established. Add repair examples and independent examples of behavior to preserve. Group alternate formats and repeated attempts at the same underlying task together. Select a task scorer before seeing patch results; exact text match is only appropriate when the full output really must match.

Assign groups whose contents you inspected during preparation to development, using explicit splits when necessary. Freeze development and evaluation files with `geodesic workbench freeze`. Keep the evaluation directory outside the editing workspace and off editing workers. Fit, inspect errors, and select candidates on development data only. Never place evaluation examples, predictions, aggregate scores, or source files containing evaluation items in the editor's context. This is a cooperative workflow: directory separation does not sandbox an agent with filesystem access. Record any earlier exposure honestly.

## Diagnose before editing

Read [the lab guide](references/lab.md). Establish a generated baseline on repair and preservation examples, with the actual chat template and inference settings. Separate protocol/format failures from wrong answers and record task success as well as token likelihoods.

Locate concrete failing decisions. Use teacher-forced targets to compare gold next tokens, hidden states, and, when useful, gradients at those points. The bundled `scripts/probe.py` collects all-layer states at explicit target prefixes for supported Transformers text chat templates; inspect its token alignment checks and use a model-specific adapter when they fail. It is a local model job and can use substantial memory. Do not run it until compute is authorized. Pin the checkpoint/revision and environment in the run record.

Build a concise map: example/source ID → failing decision/token → state or gradient evidence → candidate tensor. Check whether the desired information exists in late states before choosing a readout edit. Gradient size alone does not prove causal location. Keep preservation states in the fitting constraints. Do not copy a published patch onto a different checkpoint.

## Write and try the edit

Create `analysis/`, `edits/`, `tries/`, and `LOG.md` in the editing workspace. Explain the hypothesis in the log before each try. Derive closed-form edits (ridge regression, projections, SVD, constrained sparse writes) from development evidence. Analysis may use gradients; do not use an optimizer, gradient descent, LoRA training, or silently generated extra training data and call it the same method.

Use fresh source weights for each try. Save each edit as readable code and record the exact changed tensors and weight count after storage-dtype rounding. Inspect tied embeddings, normalization, quantization, and architecture before writing. Keep the source checkpoint intact. Capture changed tensors or an edited checkpoint in a format accepted by Geodesic's patch tools.

Track every model job, candidate, elapsed compute, and known cost against the agreed budget in `run.json` and a ledger. The local preparation commands do not enforce a cloud spending ceiling. Use provider limits or an external worker ledger when needed; stop before another job would exceed the authorized cap. A simple source scan is not a sandbox for agent code.

Generate and score each candidate on both repair and preservation development examples, including free-running contexts. Teacher-forced next-token improvements alone are not success. Keep the best candidate under the predeclared preservation tolerance. Stop at the try/budget cap, when a satisfactory candidate is found, or when the evidence no longer supports the hypothesis. A failed investigation is a valid outcome; do not spend indefinitely to force a patch.

## Freeze, evaluate, and deliver

Freeze the chosen patch and scorer before opening the evaluation set. Run baseline and patched inference with identical settings. Record per-item results, counts, repair gains, preservation regressions, and the data/split hashes. If you inspect evaluation failures and edit again, that split is now development data; use fresh evaluation examples or disclose the contamination.

Package with the existing commands (install/use Geodesic in the same environment):

```bash
geodesic patch from-dense --source CHECKPOINT --dense changed-tensors.safetensors -o candidate.gpatch
# Or: geodesic patch create --source CHECKPOINT --edited EDITED_CHECKPOINT -o candidate.gpatch
geodesic patch inspect candidate.gpatch
geodesic patch apply candidate.gpatch --source CHECKPOINT --out PATCHED_CHECKPOINT
geodesic patch restore candidate.gpatch --patched PATCHED_CHECKPOINT --out RESTORED_CHECKPOINT
```

`changed-tensors.safetensors` contains the NEW complete values of changed tensors, not deltas. The public artifact is sparse and bound to the exact source checkpoint. Verify apply/restore before handing it over. Do not use `--fast` to bypass the full checkpoint identity check. For unsupported checkpoint layouts or quantized weights, report the limitation rather than modifying packed bytes blindly.

Deliver `REPORT.md`, the frozen `.gpatch` when one worked, the edit and normalizer/scorer code, development try logs, and evaluation results. State what ran, what did not, the remaining failures, and whether acceptance criteria passed. Keep private datasets and full checkpoints out of Git. Stop any GPU resources you started. Do not deploy a model, publish data, or make a repository public merely because a patch was generated.
