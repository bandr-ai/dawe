# DAWE task s2: patch the weights so a v1-trained model works under harness v11_plan

You are editing the weights of a language model by hand, the way an engineer patches a binary:
read it, find the circuit, change the numbers. No training.

## The situation

- Model: Qwen3.5-4B (32 layers; hybrid: linear-attention "GatedDeltaNet" layers, with full attention
  every 4th layer; hidden 2560, MLP 9216). LoRA fine-tuned (merged) on BFCL function calling under
  **harness v1** (`sft_v1`). Output embedding is tied to the input embedding.
- v1: tools as JSON schema; the answer is only `{"tool_calls": [{"name": ..., "arguments": {...}}]}`.
- **v11_plan** (new harness): same, but the reply MUST start with one line `<plan>...</plan>` saying which
  functions it will call, then a newline, then the same JSON. A reply without a leading plan is rejected.
  See `context/v11_plan.toml`, `context/example_v11_plan.md`, and the parser in `context/harness_reference.py`
  (`require_plan`, `PLAN_HELP`, `_PLAN`). The plan text itself is not graded, only that it is there and
  the calls after it are right.
- The model fails under v11_plan. Hidden test set: 87.7% under v1, **20.1%** under v11_plan.
  Out of 204 test replies: 80 skip the plan and answer in plain v1 JSON (rejected), 76 write a plan and
  then **end the turn** with no calls, 48 are correct.
- Train items (what you can test on), unpatched: **v1 99.5%, v11_plan 18.3%** (`context/baseline_train.json`).
- References: retraining from scratch on v11_plan data scores 88.7% on it (and 90.2% on v1).
- **A prior DAWE patch exists** (`context/prior_patch_eos12.py`, notes in `context/prior_patch_notes.md`):
  made for a different harness (XML calls), it suppresses premature end-of-turn. Applied unchanged to
  v11_plan it already lifts hidden-test accuracy from 20.1% to 64.2% — it fixes "plan then stop" but
  not "skips the plan". You may study it, reuse its idea, or combine with it; say what you did.
- **Goal:** a weight patch that brings v11_plan up (toward ~88%) **while keeping v1 at ~99.5% train**.
  Smaller, more targeted, more explainable patches are better. Report how many weights you changed.

## Data → weight map (DAWE context)

`context/map_plan.txt`: gradient of ONLY the two decision tokens under v11_plan:
- `plan_start`: the first reply token must open `<plan>` (model prefers `{`), -log p = 0.98
- `after_plan`: after `</plan>` the calls must follow (model prefers end of turn), -log p = 1.91

On the GPU, `ctx.maps()` lists files and `ctx.load_map(name)` loads them. Relevant ones:
- `grad_dp_plan_v11_plan.safetensors` — gradient at those two decision points (per matrix)
- `states_dp_plan_v11_plan.pt` — hidden states there: `{"plan_start": [595, 33, 2560], "after_plan": [595, 33, 2560]}`
  (index 0 = embeddings, i = output of layer i-1)
- older v4 files (`grad_dp_v4_format`, `states_v4_format.pt`, `states_v1.pt`, ...) from the prior patch.
- You can compute any v1 hidden states yourself with `ctx.example(id, "v1")` + `ctx.forward`.

## Your tools (run from this folder)

```
./lab status                    budget used so far
./lab run analysis/foo.py       run Python on the GPU with the model loaded; print() output comes back
./lab patch edits/bar.py NAME   run edit(model, ctx) from that file; the changed weights are saved as patch NAME
./lab try NAME                  score patch NAME on all 595 TRAIN items under v1 and v11_plan (costs 1 try)
./lab submit NAME notes.md      choose your final patch, with a short write-up
```

In `run` and `patch` code you get:
- `model` — the HF model, bf16, on GPU. `torch` is imported.
- `ctx.P` — dict of language-model parameters by name, e.g. `model.language_model.layers.20.mlp.down_proj.weight`
- `ctx.orig(name)` — the unedited tensor
- `ctx.items` — train items (id, category, question, n_calls). Categories: simple, multiple, parallel.
- `ctx.harnesses` — ["v1", "v11_plan"]
- `ctx.example(item_id, harness)` — gold conversation tokenized: `ids`, `text`, `reply_start`,
  `points` = list of (token index d, kind) where the token at d+1 is a decision;
  kinds: `plan_start`, `after_plan`, `continue` (another call follows), `stop` (end of turn)
- `ctx.forward(ex, hidden=True)` — logits [T, V] and hidden states [33, T, 2560]
- `ctx.point_scores(item_ids, harness)` — at each decision point: p(gold next token), greedy ok?, top tokens.
  Cheap, unbudgeted check. `lab try` is the real one.
- `ctx.maps()`, `ctx.load_map(name)`, `ctx.tok`, `ctx.n_layers`
- Forward hooks, `.backward()` for analysis, linear algebra, SVDs, etc. are fine.

A patch file defines `def edit(model, ctx):` and changes weights in place (runs under no_grad).
Weights are restored after every `run`/`patch`.

## Rules (enforced)

- No optimizers or training loops (`torch.optim`, Adam, SGD, schedulers are rejected). You design the edit.
  Closed-form computations (least squares, projections, rank-1 updates you derive) are fine.
- No text generation in `run` (use `lab try`).
- Only train items. Don't look for or use test data.
- Budgets: **25 tries, 120 runs, $5 of GPU**. They can't be reset. Each try ≈ $0.15. Be economical:
  use `point_scores` to screen ideas before spending a try.
- Weights are bf16: tiny changes may round away; check `changed` counts in the patch report.

## What to hand back

1. `./lab submit NAME notes.md` with your best patch.
2. `notes.md`: the mechanism you found, what you changed (tensors, number of weights, why), the try history,
   and v1 / v11_plan train scores of the final patch (overall and parallel).
Keep a running log in `LOG.md` as you go (hypothesis → experiment → result).
