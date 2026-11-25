# Adapting Qwen3.8-27B to Python tool calls

A closed-form edit to **5,104 weights** raised Qwen3.8-27B from **39–42 to 60 out of 164** on the Python tool-call harness. A rank-16 LoRA fit from the same source conversations scored **63**. The patch gained **11.9 percentage points** over the mean of two base runs, with a 95% interval of **+3.0 to +20.8**. The three-task gap between the patch and LoRA is uncertain.

## Held-out results

Each score is the number of tasks solved out of 164. Slashes indicate repeat runs. Dashes indicate formats we did not run for that variant.

| Variant | Python call list | Native tool calls | Tagged JSON |
|---|---:|---:|---:|
| Original Qwen3.8-27B | 39 / 42 | 66 / 64 | 83 / 78 |
| Prompt hint | 51 | — | — |
| Constant logit bias on `[` (+1) | 45 | — | — |
| Random edit to the same row, matched norm | 44 | — | — |
| **DAWE patch** | **60** | **65** | **76 / 80** |
| DAWE patch with the prompt hint | 57 | — | — |
| Full LoRA | 63 | 67 | 70 |
| LoRA fit on the correction-heavy subset | 43 | 60 | 65 |

| Paired comparison | Difference (pp) | 95% interval (pp) |
|---|---:|---:|
| DAWE patch vs original, Python | +11.9 | +3.0 to +20.8 |
| DAWE patch vs prompt hint, Python | +5.5 | −0.6 to +11.9 |
| Full LoRA vs DAWE patch, Python | +1.8 | −4.9 to +9.4 |
| DAWE patch vs original, native | 0.0 | −3.0 to +3.0 |
| DAWE patch vs original, tagged JSON | −1.5 | −4.7 to +1.5 |
| Full LoRA vs original, tagged JSON | −6.4 | −15.2 to +2.1 |

The intervals use 10,000 paired bootstrap samples over the 40 test scenarios, averaging repeat runs where available. They reflect variation across scenarios. The patch and LoRA each had one Python test run. The full LoRA's tagged JSON score also came from one run, and its interval includes zero.

## What we tried

Qwen often put prose before an otherwise parseable Python call list. The Python decoder expected the list by itself. We collected Qwen's own conversations on 488 fit tasks and used the call-list suffix from those replies as the correction. Valid calls and replies that should end without a call provided examples of behavior to keep.

- **DAWE.** We fit a ridge regression on saved model states to raise the score of the opening `[` token in contexts that called for a list, with preservation examples anchoring its score elsewhere. The edit changed 5,104 BF16 values in row 58 of `lm_head.weight`. We selected it on the development split.
- **LoRA.** We used the same source conversations and correction rule, with loss over assistant replies. The full run used 485 usable conversations, 116.7 million trainable adapter parameters, rank 16, and one epoch. A second run on 191 correction-heavy conversations produced more unnecessary calls and solved 43 Python tasks.

We split all 800 BFCL v4 Multi-Turn tasks by scenario into 488 fit, 148 development, and 164 test tasks. The runs used the BFCL state and response checkers, greedy decoding with thinking off, and vLLM 0.33.0 on one H200. The source checkpoint was [`Qwen/Qwen3.8-27B` at `3756181`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0). The raw runs and checkpoints are outside this repository.

The measured gain is for a specific Python formatting failure on one model and benchmark. The LoRA comparison used one untuned configuration. The native and tagged JSON checks cover those two tool formats.
