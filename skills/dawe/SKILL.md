---
name: dawe
description: Run a data-attributed weight-editing investigation on a user's model and traces, from source-specific data preparation and model-state diagnosis through controlled candidate experiments, preservation checks, and evaluation of an exported checkpoint-bound patch. Use when accessible model weights and trustworthy targets or a verifier are available; API-only models can be diagnosed but cannot be weight-patched.
---

# DAWE: investigate, edit, test, and ship a weight patch

The user brings data, an exact checkpoint, and the behavior to change or preserve. Own the investigation: build the data and evaluation adapters, reproduce the behavior, map decisions to model states, derive candidate edits, run controlled experiments, and verify the exported patch in inference. Do not make the user design the experiment or give them only a plan and a `.gpatch` file.

DAWE is an experimental workflow, not a universal repair algorithm. The worked experiment fixed structural output failures in one checkpoint; it does not prove that arbitrary errors have a small fix. Use its analyses as examples, not as instructions to edit EOS in every model. Do not substitute fine-tuning for weight editing.

## 1. Inspect and define the run

Inspect a small source sample, model configuration, tokenizer, serving template, and available hardware before asking for missing information. Discover corrections, tool schemas, verifiers, old/new harness versions, and working cases. Keep source data intact and private. Failed traces do not establish correct answers: use reviewed corrections, executable checks, or a reliable oracle, and record which one supplied each target.

The first `dawe` milestone command prints a one-time anonymous-usage notice to stderr. Relay it to the user in your own words: say what is collected (no paths, names, or data), and recommend keeping it on, because it is the only way the maintainers learn whether DAWE is being used and where it fails. Ask once, never repeat. If the user wants it off, run `dawe telemetry off` immediately and do not argue; `dawe telemetry location off` turns off only the IP-derived location. See `docs/TELEMETRY.md` in the repository.

Create a separate workspace:

```bash
dawe workbench init --source DATA --model CHECKPOINT --out RUN
```

This writes preparation files, not a model runner or an automatic search. Use the user's existing runtime and compute authorization. Finish local inspection and a concrete compute plan before requesting any missing paid-compute authorization. Include model loading, mapping, baseline generation, controls, candidate tries, exported-patch evaluation, and shutdown in the estimate. Track agent-model costs separately from GPU costs. No cloud GPU is needed just to prepare files.

Read [the investigation playbook](references/investigation.md) before starting model work. Write `PLAN.md` with the checkpoint/revision, harness difference, failure categories, primary metric, minimum repair gain, preservation tolerance, candidate cap, compute cap, generation settings, and reserved evaluation budget. Use existing acceptance criteria; resolve missing numeric thresholds before selecting candidates. Choose a bounded first experiment, not an unbounded search.

## 2. Build trustworthy data and scoring

Read [the data contract](references/data.md). Write and retain a source-specific normalizer and scorer. Separate protocol validity from semantic task success: valid XML with the wrong function or arguments is still a failure. Exercise the scorer on known-correct, known-wrong, truncated, and malformed outputs before using it to compare edits. Record counts and parser errors; empty or failed runs cannot score as success.

Group repeated attempts, old/new formats, and corrections of the same task together. Use development fit and validation groups for fitting and candidate selection; reserve separate evaluation groups with both repair and preservation cases. Data already inspected belongs in development. Freeze using `dawe workbench freeze`; it checks structure and grouping, not truth of labels. Hash the scorer and split manifests.

Keep evaluation contents, predictions, and scores out of the editing context and worker. Do not mount raw files containing evaluation items on the editing worker. This separation is cooperative, not a filesystem sandbox. If evaluation information was already exposed, record what and when rather than claiming it was hidden.

## 3. Reproduce and map the failing decisions

Read [the runtime and mapping guide](references/lab.md). Build a generation runner using the exact served checkpoint, chat template, tool serialization, stop rules, and decoding settings. Generate an unedited baseline on repair and preservation development cases; retain per-item outputs and scorer results. First rule out a template, parser, truncation, or runtime mismatch that should be fixed in software.

Localize concrete decisions: premature stopping, malformed arguments, incorrect tool choice, or a missing opening token need different hypotheses. Compare paired old/new traces where available. Inspect teacher-forced next-token ranks, logits, hidden states, and selected gradients at those positions. Include successful continuation and genuine stopping examples, not just failures.

The bundled `scripts/probe.py` collects all-layer states at explicit target prefixes for supported Transformers text chat templates. It does not generate baseline outputs, fit edits, or establish causality. Check token alignment; write a model-specific adapter when needed. Do not launch it outside the compute authorization.

Produce `DIAGNOSIS.md` and a provenance map: source/example ID → failing decision and prefix → state/logit evidence → candidate tensor. Test whether the desired distinction generalizes across development groups. Probe accuracy and gradient magnitude are evidence for a hypothesis, not proof of causal location. If the model has no useful signal for a readout correction, change the bounded hypothesis or return a diagnosis.

## 4. Run the editing lab

Maintain `analysis/`, `edits/`, `tries/`, `LOG.md`, and a compute ledger. Each try follows **hypothesis → derive → screen → generate → score → keep/reject**. Read the candidate and control procedures in [the investigation playbook](references/investigation.md).

Derive edits from development evidence using closed-form regression, projections, SVD, or constrained sparse writes as appropriate. Gradients may inform analysis; do not use an optimizer, gradient-descent training, or LoRA and call it the same procedure. Choose edit scale and regularization on development validation, not on evaluation results. The published -12/+8 targets are not defaults for a new model.

Start each independent candidate from fresh source weights. If composing edits, declare the composition and retain isolated component candidates. Record exact tensors, norm/rank or support constraints, seeds, numerical precision, and changed-value counts **after storage-dtype rounding**. Inspect tied embeddings and quantization before editing; never patch packed bytes blindly. Keep source weights intact.

Use cheap decision-level screens to reject unsupported edits, then run free generation on both repair and preservation development cases. Better gold-token likelihood alone is not acceptance. Compare an unchanged/no-op baseline and controls appropriate to the mechanism: for a readout direction, budget a same-support/same-norm random edit and a matched constant-logit-bias control; for multiple components, test components separately. Mark omitted controls as not run and restrict causal claims accordingly. Controls are not automatically run by the CLI.

Record every candidate and control, including rejected ones, with settings, commands, per-item results, denominators, errors, and compute used. Prefer the smallest passing edit under the agreed preservation tolerance. Stop at success, the try/compute cap, or a failed hypothesis. Reserve enough budget to test the exported artifact; a lab-only success is not a delivered tested patch.

## 5. Freeze and verify the artifact

Read [the evaluation and delivery protocol](references/evaluation.md) before final evaluation. Freeze the selected edit, scorer, settings, and acceptance criteria before opening evaluation data. Evaluate a baseline and the **checkpoint produced by applying the exported patch**, rather than reporting scores from only the in-memory candidate.

```bash
dawe patch from-dense --source CHECKPOINT --dense changed-tensors.safetensors -o candidate.gpatch
# Or: dawe patch create --source CHECKPOINT --edited EDITED_CHECKPOINT -o candidate.gpatch
dawe patch inspect candidate.gpatch
dawe patch apply candidate.gpatch --source CHECKPOINT --out PATCHED_CHECKPOINT
dawe patch restore candidate.gpatch --patched PATCHED_CHECKPOINT --out RESTORED_CHECKPOINT
```

`patch create` and `patch from-dense` refuse to overwrite: if `candidate.gpatch` exists they exit with an error and write nothing. Choose a new `-o` name for each candidate (for example `candidate-2.gpatch`) and use that same name in every later command, or pass `--force` to replace the file on purpose.

`changed-tensors.safetensors` contains the NEW complete changed tensors, not deltas. Use full checkpoint identity checks; do not bypass them with `--fast`. Check apply/restore, unchanged tensors, and restored checkpoint identity. Reload the applied checkpoint in the intended compatible inference runtime before scoring it. Check actual tied-weight behavior after reload. Report unsupported layouts or runtime conversions rather than silently changing the source model.

Evaluate repair and preservation separately, with identical baseline/patched settings, item IDs, valid counts, format/task metrics, regressions, and remaining failures. Paired per-item outcomes matter even when aggregate accuracy is unchanged. Use extra repetitions or uncertainty estimates when sampling or small score differences affect the conclusion, within the budget. If you inspect evaluation failures and revise the edit, that set becomes development; obtain fresh evaluation data or disclose the exposure.

## 6. Deliver a reproducible outcome and stop

Return `REPORT.md` with a clear verdict: accepted, rejected, or incomplete. Include the exact checkpoint identity; data, split, scorer and patch hashes; baseline/candidate/control matrix; repair gains and preservation regressions; compute used; and limitations. The [delivery protocol](references/evaluation.md) specifies the artifact record and reproduction commands.

An accepted result includes the `.gpatch`, normalizer, scorer, inference and edit scripts, dependency versions, all try records, and per-item baseline/patched evaluation results. Supply exact apply, inference, score, restore, and reproduction commands with paths that exist. Keep private data/checkpoints out of Git; hand them back privately within the authorized environment.

If the patch failed, return the investigation and its evidence, without labeling a failed candidate as a successful repair. If final inference was not run, call the result incomplete, not a tested patch. Stop GPU resources you started on success, failure, interruption, or budget exhaustion and record shutdown status. Do not stop unrelated infrastructure. Model deployment or publication is separate from producing this artifact.
