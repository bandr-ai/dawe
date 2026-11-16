# Walkthrough: baseline model, failure, map, patch, held-out evaluation

One failure followed from start to finish: the XML harness (`v4_format`) and the patch the agent wrote for it (session
`s1`). The plan-first harness (`v11_plan`, session `s2`) works the same way. Everything quoted here is in the repository;
commands marked **GPU** need Modal ([REPRODUCE.md](REPRODUCE.md)).

Try the free part first. It needs no GPU and no download except BFCL:

```bash
uv sync --extra dev && uv run pytest -q
uv run dawe fetch && uv run dawe splits && uv run dawe roundtrip
uv run python examples/patch_lifecycle.py     # build, apply, refuse a look-alike, restore, on a toy model
```

## 1. Baseline model

`sft_v1` is Qwen3.5-4B with a LoRA fine-tune on the 595 train items of BFCL (simple, multiple, parallel), rendered with harness
`v1` and merged into the weights. The split is a hash of the item id (`splits/bfcl_v3_split.json`), so train and test are
the same on every machine. **GPU:** `modal run gpu/app.py::full`.

What `v1` looks like (`uv run dawe render v1 parallel_0`, last lines):

```
--- user
Play songs from the artists Taylor Swift and Maroon 5, with a play time of 20 minutes and 15 minutes respectively, on Spotify.
--- assistant (target)
{"tool_calls": [{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}, {"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}}]}
```

On the 204 held-out items `sft_v1` scores 87.7% under `v1` (`RESULTS.md`). Scoring is in `dawe/bfcl.py`: the calls the
model made, mapped back to BFCL's names, must match the ground truth (our scorer, not the official one).

## 2. The failure

Same item under `v4_format` (`uv run dawe render v4_format parallel_0`):

```
--- assistant (target)
<tool_call name="spotify.play">
<arg name="artist">"Taylor Swift"</arg>
<arg name="duration">20</arg>
</tool_call>
<tool_call name="spotify.play">
<arg name="artist">"Maroon 5"</arg>
<arg name="duration">15</arg>
</tool_call>
```

The same facts in a different reply shape. Held-out accuracy under `v4_format`: 73.5%. The loss is concentrated in multi-call
items: on the 595 train items `sft_v1` is at 99.3% on parallel items under `v1` and 36.2% under `v4_format`
(`sessions/s1/context/baseline_train.json`). The agent's first inspection (`sessions/s1/LOG.md`) saw what happens: the model
writes the first `<tool_call>` block correctly, then ends its turn. At the position after the first block the probability of
end-of-turn was 0.991 and 0.754 on the first two parallel items it looked at.

Other harness changes did not do this (`RESULTS.md`: renames 87.3%, new prompt 87.3%, Qwen-native 89.7%, all within noise of
87.7%), which is why reply shape is the interesting variable.

## 3. The map

The map ties the failure to places in the model. Built from paired traces (same item, same answer, two harnesses;
`uv run dawe build-pairs v1 v4_format`), then (**GPU**) `modal run gpu/app.py::map_`, then
`uv run dawe map-summary runs/map_v1__v4_format.json`. What the agent was shown (`sessions/s1/context/map_v1_vs_v4.txt`,
excerpt):

```
Surprise at 'another call follows' (avg -log p of the next gold token; higher = more surprised)
  v1         continue 0.001   stop 0.000   (points: 260 continue, 595 stop)
  v4_format  continue 0.975   stop 0.000   (points: 260 continue, 595 stop)
  ...
Continue-vs-stop signal in hidden states, per layer (held-out accuracy; cosine of the two directions)
  layer       v1 acc    v4_format acc  cos(dir)
      0         0.99             0.70      0.00
     16         1.00             0.95      0.53
     20         1.00             0.99      0.54
     32         1.00             0.99      0.17
```

Three readings:

1. The only decision that goes wrong is "another call follows" (surprise 0.975 nats under `v4`, vs 0.001 under `v1`). Stops are fine.
2. Under `v4`, a single direction in the hidden state separates "continue" from "stop" from about layer 16 on (0.95 to 0.99
   held-out accuracy). The information is there at the end of the network.
3. The whole-reply gradient (not shown) is spread over many matrices, the largest share in layer 0 (20%). **The agent did not edit
   there.** It used reading 2, the hidden states, which the map also saves (`states_v4_format.pt`, `states_v1.pt`).

The brief also offered a cleaner gradient map restricted to the "continue" decision tokens (`grad_dp_*.safetensors`).

## 4. The patch

The agent ran 4 candidates (`sessions/s1/tries/`), all the same closed-form fit with a different target strength `a`, scored by
generation on the 595 **train** items under `v1` and `v4`:

| Candidate | Weights changed | `v1` train | `v4` train | `v4` parallel train |
|---|---|---|---|---|
| unpatched | 0 | 99.5 | 79.0 | 36.2 |
| `eos8` | 2,555 | 99.5 | 93.3 | 92.8 |
| `eos6` | 2,552 | 99.5 | 93.4 | 91.5 |
| `eos12` (submitted) | 2,557 | 99.5 | 93.9 | 94.1 |
| `eos16` | 2,559 | 99.5 | 93.8 | 94.1 |

The submitted edit, `sessions/s1/edits/eos12.py` (comments and diagnostics trimmed):

```python
def edit(model, ctx):
    xs, ys = [], []
    for harness in ['v4_format', 'v1']:
        states = ctx.load_map('states_' + harness + '.pt')           # final-layer hidden states at decision points
        for kind in ['continue', 'stop']:
            x = states[kind][:, -1, :].float().cuda()
            xs.append(x)
            ys.append(torch.full((len(x),), -12.0 if harness == 'v4_format' and kind == 'continue' else 0.0, device=x.device))
    x, y = torch.cat(xs), torch.cat(ys)
    gram = x.T @ x
    ridge = 0.01 * gram.diag().mean()
    delta = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1], device=x.device), x.T @ y)   # ridge regression, closed form
    eos = ctx.tok.convert_tokens_to_ids('<|im_end|>')
    w = ctx.P['model.language_model.embed_tokens.weight']
    w[eos].copy_((w[eos].float() + delta).to(w.dtype))                # add to ONE row
```

Read it as: "make the logit of end-of-turn about 12 lower when the hidden state looks like `v4`-continue, and leave it
alone at `v4`-stop and at every `v1` decision." No optimizer, no training loop.

What gets saved is a sparse patch (`release/patches/s1_eos12.gpatch`; the toy demo above shows the same thing on a small model):

```bash
dawe patch inspect release/patches/s1_eos12.gpatch
#   file size      23,632 bytes  (whole changed tensors would be 1,271,398,400)
#   weights changed 2,557
#   model.language_model.embed_tokens.weight  BF16[248320, 2560]: 2,557 of 635,699,200  rows [248046]
dawe patch apply   release/patches/s1_eos12.gpatch --source sft_v1/merged --out sft_v1_xml
dawe patch restore release/patches/s1_eos12.gpatch --patched sft_v1_xml  --out sft_v1_back   # == sft_v1/merged, byte for byte
```

(`rows [248046]` is `<|im_end|>`.) Applying to anything other than the
exact `sft_v1` checkpoint is refused, including the plain base model ([PATCH_FORMAT.md](PATCH_FORMAT.md)).

## 5. Held-out evaluation

After the agent submitted, we (not the agent) ran the 204 test items under all 12 harnesses with the patch applied
(**GPU:** `modal run gpu/lab.py::final --session s1 --name eos12`, then `uv run dawe report`):

| | `v1` | `v4_format` | `v6_all` | `v12_xml_opaque` | `v11_plan` |
|---|---|---|---|---|---|
| `sft_v1` | 87.7 | 73.5 | 64.2 | 71.1 | 20.1 |
| + `eos12` | 87.7 | **85.8** | 81.9 | 86.3 | 64.2 |
| same-norm random edit, same row | 87.7 | 68.1 | 64.7 | not run | not run |
| retrained on `v4` only | 73.0 | 89.7 | 84.8 | not run | not run |

(`v6_all` and `v12_xml_opaque` also use the XML reply; `v12` was not used in fitting.)

The train-to-held-out drop (93.9% to 85.8% on `v4`) is expected: the hidden states used to fit the edit come from items
`sft_v1` was trained on. See [WRITEUP.md](WRITEUP.md#limitations) for what this evidence does and does not establish,
including the controls that have not been run (unconditional logit bias, second seed).

## 6. Do it yourself

| To check | Do |
|---|---|
| the data and split are what we used | `dawe fetch && dawe splits` |
| the scorer accepts our gold answers under every harness | `dawe roundtrip` |
| the table in `RESULTS.md` | needs per-item predictions (not committed yet); `REPRODUCE.md` section B regenerates them |
| what the agent did | `sessions/s1/{BRIEF.md,LOG.md,analysis,edits,tries,notes.md}` |
| the patch tooling | `uv run python examples/patch_lifecycle.py`, `uv run pytest tests/test_patch.py` |
