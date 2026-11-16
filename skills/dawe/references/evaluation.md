# Evaluate the shipped patch and deliver the investigation

Read before selecting/finalizing the candidate. The deliverable is an evaluated checkpoint-bound artifact with its experiment record, not just a promising in-memory edit.

## Close selection before final evaluation

Freeze selected edit code, source identity, actual written tensor values, development try history, scorer, generation settings, and acceptance criteria. Hash the selected patch and record the selection time. An evaluator runs the reserved evaluation partition without exposing its items/results to candidate selection. On one local machine this remains a cooperative policy, not secure isolation.

Generate a source baseline and the patched model under identical evaluation settings. Reuse baseline predictions only when the source identity, dataset manifest, template, inference settings, scorer, and relevant runtime versions match; document reuse. A reported old aggregate is not a replacement for those predictions.

## Exercise the actual delivery path

1. Export `candidate.gpatch` from the source and edited checkpoint, or complete NEW values of the changed tensors. `patch from-dense` does not take deltas. Retain an edit script that can recreate those values from the development evidence.
2. Inspect metadata and sparse support. Check the expected tensor names, actual value count, source identity, and finite values. A claim of “two rows changed” must match the serialized differences after rounding.
3. Apply to a new checkpoint directory using full identity checks. Do not use `--fast`. Verify the applied changed values against the selected candidate and unchanged tensors against the source. Use the patch verifier/metadata and content hashes; never infer equality from filenames alone.
4. Start a fresh inference process that loads this applied checkpoint with the intended tokenizer/runtime, rather than reusing a mutated model already in memory. Verify config, tied embeddings/output rows, and adapters are loaded as intended. Test the actual supported loading path; a conversion to another runtime is a separate step with its own checks.
5. Generate and score the frozen evaluation set on this loaded checkpoint. In-memory lab scores belong in the development record; use the applied-checkpoint scores for delivery claims.
6. Restore the applied checkpoint to a new directory and verify that its source checkpoint fingerprint matches the original. The patch format verifies model artifacts; do not claim recovery of arbitrary unrelated filesystem metadata. Keep a small reload/generation smoke test for restore when feasible within the budget.

DAWE's patch CLI implements packaging and identity checks. The skill's agent must implement generation, scoring, candidate-to-export comparison, and any model-specific loading checks. Do not imply that exporting automatically performs evaluation.

## Results to retain

Save one record per example with ID/group, purpose, condition, model variant, output, termination reason, token count, parser status, task score, and any runtime error. Keep private prompts/targets under the user's data policy; hashes alone cannot reproduce an evaluation without access to the original inputs.

Report separately:

- Repair and preservation passed/total counts, accuracy, and percentage-point change.
- Relevant slices and protocol validity as well as semantic correctness.
- Paired wins, losses, unchanged cases, and the new failures introduced by the patch.
- Sample sizes, repeated-run/seed settings, and uncertainty where material. A few items' difference on a small set does not establish equivalence to retraining.
- Controls, their comparable development settings, and whether they were run. Do not call a state-dependent mechanism established when the constant-bias comparison was omitted.

Every expected example needs a result or an explicit error. Count failed inference/parsing attempts under the predeclared scoring policy; do not drop them from the denominator after seeing results. If nondeterminism matters, repeat within the authorized budget or state that only a single run was measured.

Once evaluation failures inform a new edit, treat that partition as development and freeze a fresh evaluation set. If none exists, report exploratory results and the exposure rather than implying a new held-out test.

## Report and reproduction bundle

Use `REPORT.md` to answer, in order:

1. **Verdict:** accepted, rejected, or incomplete, with the measured acceptance criteria. An export-only artifact is incomplete as a tested repair.
2. **Inputs and contract:** source checkpoint/revision and fingerprint, data provenance, split/group counts and hashes, known exposure, serving setup, scorer and acceptance criteria.
3. **Diagnosis and edit:** concrete failures, decision/state evidence, what was changed, why, actual tensors/counts, and alternative explanations. Do not turn probe correlation into a causal claim.
4. **Experiment matrix:** baseline, every candidate/control, and the chosen artifact's final evaluation. Separate development and evaluation results; include omitted cells and reasons.
5. **Delivery verification:** patch hash, apply/restore commands and results, fresh-process runtime load, and final inference/scoring evidence. State any gap.
6. **Costs and limitations:** actual or unknown costs, remaining failures, unsupported architectures, unrun controls, private artifacts needed for reproduction, and resource shutdown status.

Include:

```text
REPORT.md                     verdict, counts, evidence, limitations
reproduce.sh                  exact commands in execution order
run.json, PLAN.md             input/runtime identity, agreed limits
prepare/                      normalizer and scorer
analysis/, edits/             diagnosis and candidate derivations
tries/, LOG.md                full development history, controls, selection
selected/candidate.gpatch     accepted artifact; label rejected ones clearly
selected/manifest.json        artifact/source/script hashes and versions
evaluation/                   per-item baseline/applied-patch outputs and scores
environment.txt, costs.jsonl   dependency versions, commands and usage
```

The manifest should identify the source, patch, scorer, edit script, data manifests, and environment used; a dictionary of SHA-256 values is sufficient. The CLI does not generate this full report bundle for the agent. The reproduction script must use real paths or documented input variables, fail on errors, and not start paid compute unless separately authorized. Record model and dataset revisions, dtype, seeds, and inference flags. Do not copy full checkpoints or private examples into a public repository.

If the criteria failed, return the measured diagnosis and rejected candidates; no successful repair is claimed. If compute was exhausted before exported-path inference, return an incomplete outcome and the exact remaining step. Never relabel the original DAWE research metrics as results of the new run.

## Resource cleanup

On every exit path, stop the GPU jobs/resources created for this run, verify shutdown through the provider/runtime, and record what was stopped and what could not be confirmed. Do not stop pre-existing customer services. A running endpoint, scheduled job, or warm worker is not cleaned up merely because a local command finished. Model deployment, dataset publication, and public artifact releases require their own task authorization.
