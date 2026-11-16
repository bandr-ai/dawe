# Adapting a post-trained model when the harness changes

Post-training ties a model to more than a task. It also exposes the model to a particular harness:
the tool descriptions, reply format, and control flow used during training. Change that harness
and the trained behavior may no longer hold.

We tested whether an editing agent could repair some of these failures directly in weight space.
Starting with Qwen3.5-4B post-trained on JSON tool calls, **Astra/Fable derived edits to one or two
weight rows** that improved performance under two new reply formats. XML accuracy rose from
73.5% to 85.8%. Plan-first accuracy rose from 20.1% to 88.2%. Accuracy under the original harness
stayed at 87.7% with either patch.

The edits changed 2,557 and 5,064 stored weight values, respectively. They were calculated from
examples and model states without an optimizer-based fine-tuning loop. These are single-run
results on one model and 204 held-out items; they suggest a useful repair is possible, with several
controls still needed.

**DAWE packages this investigation as a repository skill. Bring your data and model. Astra/Fable
investigates your data, derives small weight edits, and delivers tested patches.**

![Your data and model → Astra/Fable → tested patch.](assets/dawe-workflow.gif)

## A harness change can break an otherwise working model

We trained a LoRA adapter on 595 BFCL function-calling items rendered under a JSON harness. We
then evaluated the resulting checkpoint under twelve harness variants, keeping the underlying
requests and target tool calls fixed.

Renaming tools and changing prompts had smaller effects. Changing the shape of the reply caused
larger drops: XML tool calls reduced accuracy from 87.7% to 73.5%, while requiring a plan before
the calls reduced it to 20.1%.

One XML failure was especially concrete. A request required two calls. The model wrote the first
correctly, then ended its turn:

```text
Required                        Generated

<tool_call>first</tool_call>     <tool_call>first</tool_call>
<tool_call>second</tool_call>    [ends turn]
```

This narrows the problem. At the stopping point, does the model still represent that another call
is needed? If so, an edit to the weights that score the end-of-turn token might change the decision.

## From the failed decision to a weight edit

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
explores LLMs reading and writing neural network weights. We apply that idea to a specific problem:
adapting a post-trained model to a changed harness.

Astra/Fable is the editing agent. Qwen3.5-4B is the model being repaired. The agent received paired
old/new harness traces, a map of model states at the relevant decisions, and a bounded GPU lab.
It could write analysis code, derive edits, and evaluate candidates. Optimizer-based training was
not allowed.

For XML, the hidden-state analysis found a strong signal separating positions where another call
should follow from positions where the turn should end. Yet the end-of-turn token still won at
some continuation positions. This suggested trying a correction at the output readout.

The agent fit a vector `δ` from final-layer hidden states to desired changes in that token's logit.
Writing the states as rows of `X` and the desired shifts as `y`, the fit was ridge regression:

```text
δ = (XᵀX + λI)⁻¹Xᵀy

w_end ← w_end + δ
```

The targets were about −12 where another call should follow and 0 at real stopping points.
At generation time, the added logit shift is `h · δ`, where `h` is the current hidden state. The
correction therefore depends on the state at that position.

For plan-first replies, the agent also fit a correction to the row for `<`, helping the model begin
the required reply. The XML edit changed one row; the plan-first edit changed two. Candidate
selection used development scores, followed by evaluation of generated replies on held-out items.

These rows belong to a tied input/output embedding matrix. Editing them can affect both token
scoring and input embeddings. The results support the readout explanation, but do not isolate its
causal contribution. The [technical account](EXPERIMENT.md#what-the-patches-are) gives the details.

## What changed in generation?

We used greedy decoding with the same generation settings before and after each edit. The scores
below use our own function-call matcher on 204 held-out items, rather than the official BFCL
evaluator.

| Reply format | Original checkpoint | Patched checkpoint | Retrained from base | Weights changed |
|---|---:|---:|---:|---:|
| XML tool calls | 73.5% | 85.8% | 89.7% | 2,557 |
| Plan before tool calls | 20.1% | 88.2% | 88.7% | 5,064 |

Original-harness accuracy remained **87.7%** under either patch. On the 48 items requiring multiple
calls, XML accuracy increased from 25.0% to 81.2%.

A random edit to the same XML row, matched in norm, reduced XML accuracy to 68.1%. The direction
of the edit mattered in this comparison. We have not run a constant logit-bias control, so we do
not yet know how much the state-dependent correction adds over simply discouraging early stopping.

The retraining baselines start from the base model and train on one new harness. They do not cover
continued training from the old checkpoint or mixed-harness training. The plan-first edit came
close to its retraining baseline in this run; this sample does not establish equivalence.

## A patch that can ship with the harness

The changed positions and their old/new values fit into files of roughly **24 KB and 44 KB**. Each
`.gpatch` is bound to the exact source checkpoint by content hashes. It can be inspected, applied,
and reversed without distributing another full checkpoint.

The reported accuracies came from the original lab edits. We verified conversion to sparse patches
and apply/restore against the checkpoint. Held-out generation through the exported-patch path has
not been rerun.

Our starting idea was to derive a harness update from the difference between two trained models,
then transfer it to a model trained on another task. **That transfer remains untested.** The edits
here use new-harness examples and apply to one checkpoint. Shipping harness releases with patches
for a range of models is the direction we want to test next.

## Using DAWE

Open the repository in Astra/Fable and point it at the [DAWE skill](../skills/dawe/SKILL.md):

```text
Read skills/dawe/SKILL.md and use DAWE.
My data is in /path/to/data. My model is /path/to/checkpoint.
Fix: [what broke]. Preserve: [what already works].
Inspect the data first and propose a compute budget before paid work.
Return the tested patch, evaluation results, and scripts.
```

The agent needs accessible weights and reliable targets or a verifier. It prepares the data,
measures the failure, and investigates candidate edits. A patch is returned when repair and
preservation checks pass; otherwise the report records what failed. The [user guide](USAGE.md)
covers setup and patch commands.

## Testing on larger customer models

We’re now testing DAWE with customers on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
and [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), with work extending
to larger checkpoints up to roughly 100B parameters. These customer evaluations are ongoing;
the results above are from the Qwen3.5-4B experiment, not the larger models. We’ll report the
larger-model results once evaluation is complete.

For now, the published measurements cover one checkpoint, one benchmark subset, and single runs.
Test items were held out, but aggregate test scores appeared in the agents’ briefs. Another
training seed and further controls are still needed. See the
[full limitations](EXPERIMENT.md#limitations) before interpreting these results more broadly.

The customer work tests whether the same procedure holds across models and harness changes.
The practical question stays the same: when the harness evolves, how much of the adaptation can
fit into a small, tested weight patch?

---

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [Deletion Tuning](https://arxiv.org/abs/2611.10866). See the [experiment details](EXPERIMENT.md),
[results](../RESULTS.md), [walkthrough](WALKTHROUGH.md), and [reproduction instructions](REPRODUCE.md)
for the experimental record.
