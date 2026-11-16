# The DAWE investigation lab

Read this before model work. This is the reusable procedure behind the original sessions, including the work the original agent received precomputed. It is not a claim that those sessions ran every control described here.

## Run contract and workspace

In `PLAN.md`, make the intended experiment concrete:

- Exact checkpoint/revision and tokenizer; weight dtype/quantization, tied matrices, loader and inference versions.
- Data sources, trusted target provenance, old/new harness settings, known exposure, and grouping/split policy.
- Repair metric and minimum gain, preservation metric and maximum regression, and failure slices that must not regress. Record whether thresholds are percentage points, relative changes, or item counts.
- Baseline generation settings, stop strings/token IDs, context/output limits, scorer version, and candidate selection rule.
- Authorized hardware, candidate limit, maximum cost/time, resources owned by the run, and reserved final-evaluation compute. Existing local hardware is not authorization to rent a GPU.

Keep raw inputs separate. A useful workspace is:

```text
RUN/
  run.json, PLAN.md, LOG.md
  prepare/          normalizer, scorer, source/provenance inventory
  edit/             frozen development examples and manifest
  baseline/         development predictions, per-item scores, runtime settings
  maps/             decision records, model states, summaries
  analysis/         scripts for diagnosis and screening
  edits/            readable candidate edit functions
  tries/            one record plus predictions/scores per candidate/control
  selected/         frozen candidate, source fingerprint, patch and hashes
  evaluation/       results from an evaluator after candidate selection
  REPORT.md, reproduce.sh, environment.txt, costs.jsonl
EVALUATOR/           frozen evaluation inputs; not mounted on editing workers
```

These are output conventions for the agent, not directories automatically populated by `workbench init`. An evaluator can write final results back only after selection is closed. The editor must not use them to choose another candidate.

## Reproduce the actual serving failure

Build source-specific adapters for three distinct jobs: rendering model inputs, generating replies, and scoring outputs. Reuse the serving runtime when possible. Record how corrections/tool objects become model text; the generic normalizer accepts text messages, not arbitrary native tool APIs.

On development examples, save the raw model output, termination reason, token count, parser status, semantic task result, and failure category. Compare old/new formats on the same tasks when available. Slice by the actual failure: single/multi-call, continuation/true-stop, valid/invalid arguments, long/short context, or another source-supported category.

Check whether failure is explained by the wrong chat template, an overly short generation cap, mismatched stop tokens, or an incompatible parser before changing weights. If correcting software meets the user's objective, report that path and its measured behavior; do not force a weight edit to justify using DAWE.

## From data to candidate weights

1. **Locate decisions.** Identify the exact target prefix before the earliest consequential divergence. Record token IDs and the causal state position that predicts the next token. Verify teacher forcing and free generation use the same prompt/template. An observed generated prefix may differ from the gold prefix: keep both and label which produced a state.
2. **Compare logits.** Record target/rival token scores and ranks at failing decisions, correct decisions, and preservation examples. A lower end-of-turn probability is helpful only where the target really requires continuation.
3. **Inspect states.** Collect paired states under old/new conditions where possible. Fit simple probes using one development group partition, validate on different groups, and retain those IDs. Do not split token positions from the same example across fit and validation.
4. **Use gradients selectively.** If needed, compare whole-reply gradients with decision-only gradients. Large gradients can reflect syntax noise and do not identify the best intervention by themselves. Logit-lens analyses on intermediate layers require the architecture's correct normalization/readout.
5. **Propose an intervention.** Write what should change, where, why those states support it, expected preservation behavior, and what observation would falsify it. Check ties, normalization, parameter names, and dtype before writing.

The bundled probe only implements part of this mapping. Build and retain the missing runtime adapters and analyses. In a source checkout, `sessions/s1/analysis/`, `sessions/s2/analysis/`, their edit scripts, and `docs/WALKTHROUGH.md` show concrete examples. They are BFCL/Qwen-specific. An installed skill does not include those folders; this playbook and `lab.md` remain self-contained.

## Derive and screen candidates

For a supported readout hypothesis, let rows of `X` be causal hidden states and `Y` be desired token-logit shifts. Solve a regularized linear system in float32 or better, using `solve` rather than explicitly constructing a matrix inverse. See `lab.md` for the example. Include zero-shift preservation constraints; choose state weights, regularization, and shift scales on development validation. A hidden size of d implies a d×d Gram matrix, so account for its memory and solve cost on large models.

Other diagnoses may justify low-rank projections or sparse writes to other tensors. Do not call unrelated mechanisms equivalent. If the representation is not recoverable with the chosen intervention, stop that hypothesis rather than increasing edit magnitude indefinitely.

Check condition numbers/solver failures, finite values, change count after conversion to storage dtype, and predicted shifts on both fit and validation states. Tiny edits can round to zero; large tied-row edits can change input processing as well as output scores. Save the unrounded derivation and the actual written values.

A candidate screen is not a free-running evaluation. Reject obvious failures cheaply, then generate complete replies, including termination, argument correctness, and downstream tool behavior where relevant. Use an isolated tool executor or verifier for examples whose tools have external side effects; do not execute real customer actions merely to score a model.

## Candidate record

For every candidate/control, retain a machine-readable record with:

- ID, parent/source checkpoint hash, hypothesis and edit-script hash, parameters and random seed.
- Tensor names, changed-value count, norm, rank/support description, dtype, and tied-weight implications.
- Development fit/validation manifest hashes, inference/scorer settings and code hashes.
- Per-slice passed/total counts, parser/runtime errors, result paths, repair gain and preservation changes.
- Commands, elapsed time, known GPU/agent costs, and decision (reject/keep/select) with reason.

Every independent candidate begins from source weights. A composed candidate explicitly names its components and is also evaluated from the source checkpoint. Do not let mutations leak between tries or compare candidates generated with different templates or token budgets without labeling that difference.

## Controls and what they establish

Plan useful controls within the budget, rather than adding a long benchmark suite after a candidate works.

| Comparison | Use | Interpretation limit |
|---|---|---|
| Unchanged checkpoint / no-op path | Every run | Detects runtime, scorer, and export-path differences. |
| Same-support, same-norm random edit | A directional/sparse edit | Tests whether the selected direction matters more than a perturbation of that size. One random seed is weak evidence. |
| Constant logit bias to the same token(s) | A state-dependent readout claim | Tests a simpler serving-time intervention; match strength on development, never evaluation. It is a control, not a weight patch. |
| Individual components | A combined edit | Shows which components help and whether composition regresses preservation. |
| Prompt/template fix or format shim | A harness-format failure | A practical alternative, scored on the same tasks. An ideal translator ceiling is not an implemented shim. |
| Retraining | A claim relative to retraining, with separate authorization | Record initialization, data and compute; retraining from base and continued SFT are different baselines. |

For a readout repair, budget the random-direction and constant-bias comparisons early. If the user chooses a smaller budget, record omissions and avoid claims those comparisons would support. Report the full matrix of measured cells, marking missing cells `not run`. Do not import the original experiment's scores as a baseline for a customer's checkpoint.

## Select and stop

Select using development repair improvement subject to preservation tolerance, then edit size and complexity. Keep counts and item-level regressions even if net accuracy is unchanged. Use a predetermined tie-break rule or disclose judgment used. Evaluation scores do not enter selection.

Update the cost ledger after each model job, including mapping, model-load, idle, control, and scoring time. Reserve final apply/reload/evaluation compute. Cooperative ledgers do not enforce a provider cap: use resource timeouts and provider limits when available. Stop before the next job would exceed authorization; do not spend the evaluation reserve on more search.

Stop when the target is met, the hypothesis is unsupported, or the budget/candidate limit is reached. Return rejected or incomplete results honestly. Follow `evaluation.md` for the exported artifact and final decision.
