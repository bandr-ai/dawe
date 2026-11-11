# DAWE task: patch the weights so a v1-trained model works under harness v4

You are editing the weights of a language model by hand, the way an engineer patches a binary:
read it, find the circuit, change the numbers. No training.

## The situation

- Model: Qwen3.5-4B (32 layers; hybrid: linear-attention "GatedDeltaNet" layers, with full attention
  every 4th layer; hidden size 2560, MLP 9216). Fine-tuned with LoRA (merged) on BFCL function-calling
  data rendered in **harness v1**. We call it `sft_v1`.
- Harness = how tools and calls are formatted around the model. Files: `context/v1.toml`,
  `context/v4_format.toml`, and the renderer/parser `context/harness_reference.py`.
  - v1: tools as JSON schema, answer as `{"tool_calls": [{"name": ..., "arguments": {...}}]}`
  - v4_format: same, but the answer must be XML blocks: `<tool_call name="f">\n<arg name="x">VALUE</arg>\n</tool_call>`,
    one block per call, joined by a newline.
- The product shipped harness v4. The model now fails under it:
  - Test set (hidden from you): sft_v1 scores 87.7% under v1 but 73.5% under v4.
  - On items that need several calls ("parallel"), it drops from 81% (v1) to 25% (v4).
  - The failure: under v4 it writes the **first** `<tool_call>` block correctly and then **stops**
    (end of turn) instead of writing the next block.
  - For reference: the base model (before fine-tuning) gets 81.4% under v4; retraining from scratch on v4
    data gets 89.7% under v4 but falls to 73.0% under v1. A perfect format translator gets 87.7% everywhere.
- **Goal:** a weight patch that brings v4 accuracy up (toward ~88%) **while keeping v1 at ~87.7%**.
  Smaller, more targeted, more explainable patches are better. Report how many weights you changed.

## Data → weight map (DAWE context), already computed

`context/map_v1_vs_v4.txt` summarizes it:
- Surprise at the "another call follows" decision (the token after `</tool_call>` when more calls are due):
  under v1 ~0.001 nats, under v4 ~0.98 nats. Stop decisions are fine in both.
- Hidden states: a single direction separates "continue" vs "stop" at decision points.
  Under v1 it is readable from layer 0; under v4 it only appears from layer ~16 on (held-out acc 0.95-0.99),
  and the v4 direction has cosine ~0.5 with the v1 direction at layers 16-20.
- Gradients of the SFT loss on v4 gold replies (whole reply) concentrate in early layers (layer 0
  out_proj alone 20%); this whole-reply gradient is mixed with formatting noise (e.g. quoting).
- A cleaner map: gradient of ONLY the "continue" decision tokens: `grad_dp_v4_format.safetensors`
  and `grad_dp_v1.safetensors` (see `context/map_dp.txt` if present).

On the GPU, `ctx.maps()` lists the files and `ctx.load_map(name)` loads them:
- `grad_v1.safetensors`, `grad_v4_format.safetensors` — whole-reply gradients, per weight matrix
- `grad_dp_v1.safetensors`, `grad_dp_v4_format.safetensors` — continue-decision gradients only
- `states_v1.pt`, `states_v4_format.pt` — hidden states at decision points:
  `{"continue": [n, 33, 2560], "stop": [n, 33, 2560]}` (index 0 = embeddings, i = output of layer i-1)

Paired examples (same item, same answer, both harnesses): `context/paired_examples.md`.
Baseline on train items (unpatched), from `lab try`: **v1 99.5%, v4 79.0%** (parallel: v1 99.3%, v4 36.2%).
Train v1 is near 100% because the model was trained on these items; aim to keep it there.
Details and failure examples: `context/baseline_train.json`.

## Your tools (run from this folder)

```
./lab status                    budget used so far
./lab run analysis/foo.py       run Python on the GPU with the model loaded; print() output comes back
./lab patch edits/bar.py NAME   run edit(model, ctx) from that file; the changed weights are saved as patch NAME
./lab try NAME                  score patch NAME on all 595 TRAIN items under v1 and v4_format (costs 1 try)
./lab submit NAME notes.md      choose your final patch, with a short write-up
```

In `run` and `patch` code you get:
- `model` — the HF model (`Qwen3_5ForConditionalGeneration`), bf16, on GPU. `torch` is imported.
- `ctx.P` — dict of language-model parameters by name, e.g.
  `model.language_model.layers.20.mlp.down_proj.weight`
- `ctx.orig(name)` — the unedited tensor
- `ctx.items` — the train items (id, category, question, n_calls). Categories: simple, multiple, parallel.
- `ctx.example(item_id, harness)` — the gold conversation tokenized: `ids`, `text`, `reply_start`,
  `points` = list of (token index d, continue?) where d is the position whose next token is the decision
- `ctx.forward(ex, hidden=True)` — logits [T, V] and hidden states [33, T, 2560]
- `ctx.point_scores(item_ids, harness)` — at each decision point: p(gold next token), whether greedy picks it, top tokens.
  This is your cheap, unbudgeted check. `lab try` is the real one.
- `ctx.maps()`, `ctx.load_map(name)`, `ctx.tok`, `ctx.n_layers`
- You may register forward hooks, compute gradients with `.backward()` for analysis, do linear algebra, SVDs, etc.

A patch file defines `def edit(model, ctx):` and changes weights in place (it runs under no_grad).
Weights are restored after every `run`/`patch`, so experiments don't leak into each other.

## Rules (enforced)

- No optimizers or training loops (`torch.optim`, Adam, SGD, schedulers are rejected). You design the edit.
  Closed-form computations (least squares, projections, rank-1 updates you derive) are fine.
- No text generation in `run` (use `lab try`).
- Only train items. Don't look for or use test data.
- Budgets: **25 tries**, 120 runs/patches, $12 of GPU. They can't be reset.
- Weights are bf16: tiny changes may round away; check `changed` counts in the patch report.

## What to hand back

1. `./lab submit NAME notes.md` with your best patch.
2. `notes.md`: what you found (the mechanism), what you changed (which tensors, how many weights, why),
   the try history (what worked, what didn't), and v1/v4 train scores of the final patch.
Keep a running log in `LOG.md` as you go (hypothesis → experiment → result).
