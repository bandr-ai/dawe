# Your harness moved on. Can your model follow?

You post-train a model for your agent. It learns the tools, follows the reply format, and works well
in your evals. Then you ship a better harness.

Maybe tool calls now use XML. Maybe every reply needs a plan before the first action. The task has
barely changed, but the model starts failing. A checkpoint that worked with yesterday's harness no
longer works with today's.

**DAWE asks whether some of those failures can be repaired with a small weight edit.** Bring your data
and model. Astra/Fable investigates failures, derives small weight edits, and delivers tested patches.

![DAWE: your data and model flow to Astra/Fable for investigation, weight editing, evaluation, and a tested patch.](assets/dawe-workflow.gif)

## The harness is part of what the model learns

A harness is the code around a model: the tools it can use, how they are described, the instructions
it receives, and how its replies are read. Post-training happens inside that arrangement. We should
not assume the resulting behavior will survive every change to it.

Here is a small example. A request needs two tool calls. Under the old JSON format, the model makes
both. Switch the reply format to XML and it writes the first call, then ends its turn.

```text
What the new harness needs       What the model does

<tool_call>first</tool_call>     <tool_call>first</tool_call>
<tool_call>second</tool_call>    [ends turn]
```

The syntax changed, and a useful behavior broke. The question becomes more specific: at the moment
it stops, is the model missing the information that another call is needed, or is that information
present but losing to the end-of-turn token?

That distinction gives us something to investigate.

## Give the editing agent evidence

Weight-space programming explores how an LLM can read and change another network's weights. We take
that idea into one practical setting: **a post-trained model adapting to an evolving harness**.

The editing agent—Astra/Fable—gets the checkpoint, examples of the failure, trustworthy target
behavior, and examples of behavior that should keep working. The model being repaired is a separate
model; in our worked experiment, it was Qwen3.5-4B.

The workflow connects a failure to a place in the model:

1. Generate replies and identify exactly where they go wrong.
2. Inspect model states at those decisions.
3. Derive a small candidate edit from that evidence.
4. Generate again and check both the repair and the behavior to preserve.

The agent has tools, model runs, and a limited budget. It writes analysis and edit code. It does not
run an optimizer-based fine-tuning loop. A better next-token score is only a clue; the patched model
has to produce better replies.

## In this case, the model knew more than it said

We started with Qwen3.5-4B, post-trained on function-calling examples in a JSON harness. We evaluated
twelve harness variants. Renames and prompt changes caused relatively small differences. XML calls
and plan-first replies caused much larger drops.

For the XML failure, later model states contained a strong signal separating “another call follows”
from “the turn should end.” Yet the model still selected the end-of-turn token too early.

The agent used those states to calculate a correction to the weight row for that token. The intended
change was simple: make ending the turn less likely when another call should follow, while leaving
real stopping points and the old format alone. The plan-first patch also adjusted the row for the
opening `<` token.

These were closed-form fits, derived from the examples rather than learned through gradient-descent
training. The XML patch changed 2,557 stored weight values in one row. The plan-first patch changed
5,064 values across two rows. The [technical account](EXPERIMENT.md#what-the-patches-are) explains the
fit, the tied embeddings, and why the readout explanation is an interpretation rather than a causal
proof.

## Did the patched model work?

We tested on 204 held-out items, using the same generation settings before and after the edit.

| Reply format | Before the patch | After its patch | Changed weights |
|---|---:|---:|---:|
| XML tool calls | 73.5% | 85.8% | 2,557 |
| Plan, then tool calls | 20.1% | 88.2% | 5,064 |

Old-format accuracy stayed at **87.7%** with either patch. A same-sized random edit to the XML patch's
row reduced XML accuracy to 68.1%, so simply perturbing that row did not reproduce the improvement.

Retraining from the base model on each new harness reached 89.7% for XML and 88.7% for plan-first.
The XML edit fell short; the plan-first edit was close in this run. This small experiment does not
establish equivalence to retraining.

The patches can be stored as roughly 24 KB and 44 KB files containing the changed positions and their
old/new values. Each is bound to the exact checkpoint it repairs, can be inspected, and can be undone.
The reported model scores came from the original lab edits; we verified the sparse conversion and
apply/restore machinery, but have not rerun held-out inference through the exported-patch path.

## What this result means

It is possible for a harness change to expose a narrow failure that a small weight edit can repair.
In this experiment, the editing agent used traces and model states to find such an edit while
preserving old-format accuracy on the evaluated items.

There is still a lot we do not know. This is one model, one benchmark subset, and single runs. We use
our own scorer. The test items were held out, but aggregate test scores appeared in the agents'
briefs. Important controls—including a constant logit-bias comparison and another training seed—are
missing. The [full limitations](EXPERIMENT.md#limitations) and [results](../RESULTS.md) are available
alongside the code.

We have also **not** demonstrated taking the difference between two trained models and transferring
it to a model trained on another task. These edits use new-harness examples and are specific to their
source checkpoint. Shipping a harness release with patches for many models is a direction to test.

## Bring your own failure

DAWE is available as a [repository skill](../skills/dawe/SKILL.md). Open the repo in Astra/Fable and give
it the data path, checkpoint path, and the behavior you want changed:

```text
Read skills/dawe/SKILL.md and use DAWE.
My data is in /path/to/data. My model is /path/to/checkpoint.
Fix: [what broke]. Preserve: [what already works].
Inspect the data first and propose a compute budget before paid work.
Return the tested patch, evaluation results, and scripts.
```

The agent handles source-specific preparation, diagnosis, editing, and evaluation. It needs accessible
weights and reliable targets or a verifier. It returns a report and, when an edit passes the checks,
a `.gpatch` for your exact model. An unsuccessful investigation returns its findings. The [user guide](USAGE.md)
explains setup and applying a patch. The generic workflow still needs an independent end-to-end run
on another user's model; the worked experiment is the evidence we have today.

The next useful test is a different model, a different harness change, and a clearly measured outcome.
If the harness keeps evolving, we want to understand when a model can follow with a patch.

---

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and the work on [Deletion Tuning](https://arxiv.org/abs/2611.10866). The harness experiment and its
measurements are our focus here. See the [experiment details](EXPERIMENT.md), [walkthrough](WALKTHROUGH.md),
and [reproduction instructions](REPRODUCE.md) for the complete record.
