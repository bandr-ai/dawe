# Geodesic launch: write-up and positioning (draft)

## One line

**Your harness changed and your fine-tuned model broke. Ship a weight patch, not a retrain.**

Alternates: "git diff for model weights" · "harness release + model patch" · "hotfix the weights".

## The story in 4 beats

1. **The problem.** Agent teams fine-tune a model on their harness (tool format, prompts, schemas),
   then change the harness every week. The fine-tuned model breaks, and the usual fix is retraining.
2. **What actually breaks.** We changed a BFCL harness 12 ways. Renaming tools, opaque IDs, new prompts,
   even Qwen's own native tool format: no damage. Changing the *shape of the reply* broke it:
   XML calls (87.7 → 73.5) and "plan first" replies (87.7 → 20.1).
3. **The fix.** We give a frontier agent (Codex, GPT-6 Astra) a map from data to weights (which weights and
   hidden states drive the failing decision, from paired old/new traces) and a lab with a fixed budget.
   No optimizer allowed. It reads the map, derives a closed-form edit, tests it on train items, submits a patch.
4. **The result.** Kilobyte patches, under $1 of GPU each, that match retraining on the new harness and
   leave the old one untouched.

## Headline numbers (hidden test, 204 BFCL items)

| | old harness v1 | XML calls (v4) | plan first (v11) | weights changed | GPU cost |
|---|---|---|---|---|---|
| fine-tuned on v1 | 87.7 | 73.5 | 20.1 | – | – |
| **+ DAWE patch** | **87.7** | **85.8** | **88.2** | 5,064 (2 rows) | $0.79 + $0.63 |
| retrain on new harness | 73.0 (v4 model) · 90.2 (v11 model) | 89.7 | 88.7 | all 4B | ~$0.50–0.70 each |
| random edit, same size & place | 87.7 | 68.1 | – | 2,552 | – |

Supporting points:
- Retraining on v4 recovers v4 but drops v1 from 87.7 to 73.0. The patch keeps both.
- One patch, four harnesses: the v11 patch also fixes v4 (85.8), v6 (81.9) and XML+opaque (85.8).
- The v4 patch transfers unchanged to a harness it never saw (XML + opaque names: 71.1 → 86.3).
- Multi-call items, v11: 39.6 → 87.5 (retraining: 83.3).
- The agent needed 4 and 2 tries (of 25). Every edit is a readable Python file in `sessions/`.

## Mechanism (for the technical thread / blog)

Both failures were "the model knows but doesn't say": the hidden state at the decision point already
separates "another call / start a plan" from "stop", but the output row for the token ignores it.
The agent solved one ridge regression from the mapped hidden states to target logits and added the result
to two rows of the (tied) embedding matrix: `<` at the plan start, end-of-turn after the plan / between calls.
The correction is conditional: large at the mapped failure points, ~0 at JSON and true-stop points.

## Objections we should answer up front

| They'll say | Our answer |
|---|---|
| "Just write a format translator (shim)." | You can for some changes, and it's our upper bound row. It's hand-written code per change that lives in the serving path. The patch ships with the model and was written by an agent in minutes. |
| "That's just a logit bias." | It's conditional on the hidden state, not constant. **Run the constant-bias control before launch** (below) so we can show the number. |
| "n=204, one run." | True. Say it. Add a second seed if we want to claim more. |
| "Retraining is cheap." | Here it's ~$0.70, but it rewrites the whole model, needs data and a pipeline, and lost 15 points on the old harness (v4). The patch is a 2-row diff you can review and revert. |
| "Only works for easy readout bugs." | Both fixes are at the output layer, and we say so. Deeper failures (wrong argument values) are next. |

## What not to claim

- Not "shim-proof" or "general behaviour editing". Claim: structural harness changes, output-layer patches.
- No SOTA language. One model (Qwen3.5-4B), one benchmark (BFCL simple/multiple/parallel).
- Don't call DAWE a new algorithm for the patch math (it's ridge regression). The contribution is the
  data→weight map + agent loop + the finding that these patches match retraining.

## Credit

- Weight-space programming blog (rikiyat.github.io/blog/weight-space-programming): the agent-edits-weights idea.
- Deletion Tuning (arXiv 2611.10866): learned weight masks for unlearning; their open question on combining
  edits motivates the stacking result.
- BFCL (Berkeley Function Calling Leaderboard) for data and grading.

## Draft thread

1/ Every agent team fine-tunes on their harness, then changes the harness every week.
The model breaks. Everyone retrains.
We tried shipping a weight patch instead. 🧵

2/ First: what actually breaks a fine-tuned model? We changed a BFCL harness 12 ways.
Renamed tools, opaque tool IDs, new prompt, even Qwen's native tool format: no damage.
Change the shape of the reply: 87.7% → 73.5% (XML calls), → 20.1% ("plan first").

3/ So we built a data→weight map: paired traces (same task, old vs new harness) → which weights and
hidden states drive the failing decision.
Then handed it to a frontier agent (GPT-6 Astra via Codex) with a GPU lab and a fixed budget. No optimizer allowed.

4/ It found the bug: the model *knows* another call is coming (it's in the hidden state) but the output
for "end of turn" ignores it. It solved one least-squares equation and edited 2 rows of one matrix.

5/ Result on held-out BFCL:
plan-first harness 20.1% → 88.2% (retraining: 88.7%)
XML harness 73.5% → 85.8%
old harness: 87.7% → 87.7%
5,064 weights. $0.63 of GPU. 2 tries.

6/ Retraining on the new harness recovers it but costs you the old one (87.7 → 73.0).
A random edit of the same size in the same place fixes nothing (68.1%).
And one patch fixed 4 harness versions, including one it never saw.

7/ Caveats: one 4B model, 204 test items, one run, output-layer fixes.
Next: failures that aren't a single decision, and stacking patches across releases.
Code + every agent edit, log and note: [repo link]

## Before launch (cheap, recommended)

1. **Constant-bias control** (~$0.30): same tokens, same average shift, but unconditional. Expect it to break
   stops (v1 drops). This is the answer to "it's just a logit bias".
2. **Random control for v11** (~$0.15).
3. Optional: a second SFT seed + rerun of both patches (~$1.50) to show it isn't one lucky model.
4. Make the repo public (or a cleaned copy), add a results figure.
