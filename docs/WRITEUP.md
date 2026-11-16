# DAWE: from data to tested weight patches

A fine-tuned model can break when the harness around it changes. We tested whether a small, closed-form
edit to the weights, found by an agent from paired traces, can repair it, as an alternative to retraining.
This page says what we did, what we found, and how much the evidence supports. Every number is from
[`RESULTS.md`](../RESULTS.md); the experiment is described step by step in [WALKTHROUGH.md](WALKTHROUGH.md) and
can be re-run from [REPRODUCE.md](REPRODUCE.md).

## Try it on your own data

The workflow is now available as a [repository skill](../skills/dawe/SKILL.md). Give a coding agent your raw traces,
the exact checkpoint, and a description of the failure. Supply reviewed corrections or a verifier, plus examples of
behavior to preserve. The agent prepares the data and scorer, freezes development/evaluation groups, measures the
baseline, maps failing decisions to model states, and derives and tests small closed-form edits within an agreed budget.
If an edit passes repair and preservation checks, it returns a checkpoint-bound `.gpatch`, a report, and the scripts
and per-item results needed to inspect it. An unsuccessful run returns its diagnosis rather than a promised patch.

DAWE is a repository skill: users bring data and a checkpoint, and the agent handles the investigation.
The CLI handles local preparation and patch packaging. An agent still implements the source- and model-specific investigation. Preparation starts no GPU work;
model inspection, generation, and candidate evaluation need compatible compute. See [the user guide](USAGE.md) for
installation, a copyable prompt, the run stages, and applying/restoring your patch.

The skill makes the method available; it adds no new model-editing results to the measurements below. Its preparation
helpers and packaging were tested locally, but its generic collector has not been validated in an end-to-end run on a
new user's model.

## Relation to the original idea

The starting idea was a “git diff for models”: learn a harness change from two models, then apply that difference to a
model trained on another task. **That cross-task weight-delta transfer has not been tested.** What we did instead was
derive a small edit from paired harness traces and hidden states of one fine-tuned checkpoint, then measure its effect.
The experiment uses new-harness targets; it does not demonstrate adaptation without new-harness data. The released
patches are bound to one exact checkpoint, so shipping a harness release with patches for many models remains a research
direction. Today a user runs the investigation on their own checkpoint and gets a patch specific to it.

## Summary

- Setup: Qwen3.5-4B, LoRA SFT on BFCL function-calling items rendered under one harness (`v1`: JSON schema in the
  prompt, JSON `tool_calls` out). 595 train items, 204 held-out test items, greedy decoding, one run per cell.
- Twelve harness variants were scored. Renaming tools, opaque tool ids, a new system prompt, moving the tool list, Qwen's
  native tool format, and Python-style calls cost between 0 and about 5 points (noise at this size is similar). Two changes to
  the **shape of the reply** broke the model: XML tool calls (`v4`: 87.7% to 73.5%) and plan-first replies (`v11`: 87.7% to
  20.1%). Variants that include XML (`v6`, `v12`) broke too.
- For each, an AI coding agent was given a map from data to weights (paired old/new traces, hidden states and gradients
  at the decision points) and a budgeted GPU lab, with no optimizer allowed. It submitted one closed-form edit to one or
  two rows of the tied embedding matrix: 2,557 weights for `v4`, 5,064 for `v11`.
- Held-out result: `v11` goes 20.1% to 88.2% (retraining on `v11`: 88.7%); `v4` goes 73.5% to 85.8% (retraining on `v4`:
  89.7%); `v1` stays 87.7% in both. A same-norm random edit to the same row as the `v4` patch gives 68.1% on `v4`.

What this does **not** show is listed under [Limitations](#limitations); the important ones are one model, one benchmark
subset, one seed, no unconditional-bias control, and a test set that was held out but not hidden from the briefs.

## Results

Accuracy on the 204 held-out items (counts in parentheses; 95% Wilson interval half-width is 4 to 6 points at this size, so
differences under about 5 points are within noise for a single run):

| Model | `v1` (old harness) | `v4_format` (XML) | `v11_plan` (plan-first) | Weights changed |
|---|---|---|---|---|
| `sft_v1` (LoRA on `v1`) | 87.7 (179) | 73.5 (150) | 20.1 (41) | none |
| + s1 patch `eos12` | 87.7 (179) | **85.8** (175) | 64.2 (131) | 2,557 |
| + s2 patch `plan8_eos12` | 87.7 (179) | 85.8 (175) † | **88.2** (180) | 5,064 |
| same-norm random edit, same row as `eos12` | 87.7 | 68.1 (139) | not run | one row |
| retrain from base on `v4` only | 73.0 | 89.7 (183) | not run | all |
| retrain from base on `v11` only | 90.2 | 82.8 | 88.7 (181) | all |
| base Qwen3.5-4B, no SFT | 71.6 | 81.4 (166) | 11.3 | none |
| ideal format translator in front of `sft_v1` (upper bound for any shim) | 87.7 | 87.7 | 87.7 | none |

† The s2 edit was fit using `v4` hidden states as well (it includes the s1 end-of-turn suppression), so its `v4` score is
not evidence of transfer to a harness it never saw.

Other rows in `RESULTS.md`: the s1 patch on `v12` (XML calls with opaque tool names, which it was not fit on) goes
71.1% to 86.3%; the s2 patch on `v6` and `v12` goes to 81.9% and 85.8%, from 64.2% and 71.1%. Both of those
harnesses contain the `v4` reply format.

Multi-call (parallel) items, 48 of the 204: `v4` 25.0% to 81.2% with the s1 patch (retrain on `v4`: 81.2%); `v11` 39.6% to 87.5%
with the s2 patch (retrain: 83.3%).

How to read the comparison with retraining: on `v11` the patch is level with retraining. On `v4` it is 8 items
(3.9 points) short of retraining and 4 items short of `v1`. That is inside single-run noise, but it is not "matches".
Per-item predictions are needed for a paired test; they are not in the repository yet.

## What the patches are

Both patches add a vector to a few rows of `model.language_model.embed_tokens.weight`, which is tied to the output
layer. Row `<|im_end|>` (end of turn) gets a correction `delta = (XᵀX + λI)⁻¹ Xᵀy`, a ridge regression from the final-layer
hidden state to a target logit shift: about −12 at positions where another call must follow and 0 at positions where
the turn should end. The s2 patch also raises `<` (token 27) by about +8 at the start of a reply when the plan must open. The
shift is `h · delta` for the hidden state `h`, so it depends on the state; it is not a constant bias. (We have not run the
constant-bias control; see below.)

The per-position averages the agents report (for example −11.97 at `v4` continuations, −0.01 at `v4` stops) are
measured on the same states the regression was fit on.

Why this location: in both failures the model reaches the decision point with a hidden state that separates "another call
follows" from "stop" (held-out linear accuracy 0.95 to 0.99 from layer 16 on, under `v4`), but the end-of-turn token still wins. So
the fix is at the readout. That is an interpretation supported by a linear probe on train items plus the patch working; it is
not a causal analysis.

Which part of the map mattered: the gradient part of the map spread over many matrices, with layer 0 holding the most
(20% of the whole-reply gradient for `v4`), and the agent's edit is not there. What the agent used was the hidden-state part
of the map.

The edit is not purely an output change. Embeddings are tied, so the patched row is also the input embedding for the same
token, and `<|im_end|>` appears in every chat-template turn. `v1` accuracy not moving (87.7 to 87.7) says the effect on
reading is small on these prompts; we did not isolate it (for example by patching an untied copy of the output layer).

## Size and portability

The lab originally saved the whole edited tensor, which for these edits is a file of over 1.2 GB. Patches are now sparse
(changed indices plus old and new values, bound to the exact checkpoint by content hashes): 23,632 bytes for the s1 edit and
43,688 bytes for the s2 edit ([PATCH_FORMAT.md](PATCH_FORMAT.md), files in [`release/patches/`](../release/patches/)).
A patch applies only to the exact `sft_v1` checkpoint it was made from, can be undone byte for byte, and refuses a
look-alike, including the plain base model (whose embedding rows are identical to `sft_v1`'s).

Cost: the agents' lab ledgers show $0.79 (s1) and $0.63 (s2) of GPU. That is the agent's own usage only. The maintainers'
running total for the whole project (SFT, maps, ceilings, agent lab, control and held-out scoring) was about **$5.30**
when the s1 results were written up, of which the random control and the two test-set scorings were about $0.27. That figure
predates the s2 session, so s2's $0.63 and its evaluations come on top, and it excludes the agent's own model usage
(Codex on a ChatGPT plan). The itemized log (`runs/costs.jsonl`) was not committed, so there is no exact final total.

## Limitations

1. **Scope.** One model (Qwen3.5-4B), one benchmark subset (BFCL v3 simple/multiple/parallel, 799 items, one skipped), one
   SFT run, one generation per cell, 204 test items. No variance across SFT seeds or agent attempts. Results are specific to
   structural reply changes fixed at the output layer. They say nothing about errors in argument values or semantics.
2. **Our scorer is not BFCL's.** `dawe/bfcl.py` is a simplified matcher written for this project (order-insensitive for
   parallel calls, case and quote tolerant strings, int/float equality). It has not been validated against the official
   evaluator, so these accuracies are not comparable to leaderboard numbers.
3. **The test set was held out, not hidden.** The agents never saw test items or predictions, and patches were chosen on train
   scores, but the briefs quote aggregate test numbers (baselines, retraining targets, and for s2 the s1 patch's test score) and
   the lab image contains the full dataset files. Details: [LAB.md](LAB.md#what-the-agents-were-told-about-the-test-set).
4. **Train scores are inflated and the maps come from memorized items.** `sft_v1` was trained on the 595 train items (99.5% under
   `v1`). The hidden states used to fit the patches come from those items. Train accuracy of the final patches (93.9% for
   `v4`, 97.8% for `v11`) is well above held-out accuracy (85.8%, 88.2%).
5. **Controls are incomplete.** Done: a same-norm random edit to the same row for `v4` (68.1%, below the unpatched 73.5%,
   so a perturbation of that size in that row can hurt). Not done: an unconditional-bias control (same tokens, same average
   shift, no dependence on the hidden state), a random control for `v11`, a second SFT seed. Until the unconditional-bias
   control is run, "it is more than a logit bias" is a hypothesis; the per-position averages are consistent with it but are
   in-sample.
6. **The retraining baselines are narrow.** "Retrain" means SFT from the base model on one new harness, with the same recipe.
   It is not continued training from `sft_v1`, and not a mix of old and new harness data. The drop of `v1` to 73.0% for the
   `v4`-only model is therefore a statement about single-harness retraining, and it does not hold for `v11`, where the retrained
   model scores 90.2% on `v1`. A mixed-data retrain was not run.
7. **A translation shim is not an alternative we beat.** The shim row is the upper bound for a perfect shim (87.7% everywhere).
   The patches do not exceed it; the argument for a patch over a shim is operational (ships with the model, no serving-path
   code), not accuracy.
8. **Agent sessions are not independent.** s2 started from s1's patch and notes. Each task was attempted once.
9. **Cooperative lab.** Budget and data rules are guard rails, not isolation ([LAB.md](LAB.md)).
10. **The reported accuracies belong to the original whole-tensor patches.** The released sparse `.gpatch` files were converted from
    those (the converter checks that exactly 2,557 and 5,064 weights change, matching what the lab recorded) and verified against the
    real checkpoint ([release/README.md](../release/README.md)), but the held-out evaluation was not re-run on a model produced by
    `dawe patch apply`. Raw held-out predictions are not in the repository either, so the tables in `RESULTS.md` cannot yet be
    regenerated from committed files; they match the maintainers' own earlier write-ups of the same runs.
11. **Environment.** Top-level package versions, the model revision and the dataset revision are pinned, and `repro/freeze_*.txt`
    lists the resolved versions of all three images as built on 2026-11-16 (not necessarily as of the original runs; the generation image
    uses `transformers` 5.17.0 while training used 5.20.2). The Modal base image is not pinned by digest, and GPU training and vLLM
    generation are not guaranteed bit-reproducible.

## What the data supports

- Under this setup, changing the shape of the reply (XML calls, plan-first) hurts the fine-tuned model much more than renames,
  a new prompt or Qwen's native format.
- A closed-form edit to one or two embedding rows, fit on 595 train items, moved held-out accuracy from 73.5% to 85.8% (`v4`)
  and from 20.1% to 88.2% (`v11`) without moving `v1`, and the same-sized random edit did not help. Each is a single run on 204 items.
- It is small enough to review: the diff is a list of indices and values in two rows of one tensor, checkable by hand
  with `dawe patch inspect`.

## Credit

- BFCL: Yan, Mao, Ji, Zhang, Patil, Stoica, Gonzalez, *Berkeley Function Calling Leaderboard*, 2024: the data.
- *The Surprising Effectiveness of Deleting Weights in LLM Post-Training* (arXiv 2611.10866): learned binary masks that zero under
  0.05% of a model's gate-projection weights match full fine-tuning and forget less. Their open question about combining
  edits motivates stacking patches, which we have not tested.
- *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, 2026-11-02): LLMs reading and writing weights as
  code, the idea behind giving an agent the weights.
- Qwen team: Qwen3.5-4B.
- The ridge regression is standard; the contribution is the map, the lab and the measurements above, not the algebra.

## Not yet done

Unconditional-bias control; `v11` random control; second SFT seed; mixed-data retrain; paired significance tests from per-item
predictions; patch stacking; failures that are not a single end-of-turn decision.
