# Mapping and editing with the user's runtime

## Supported starting point

Accessible, unquantized Transformers safetensors checkpoints with text chat inputs are the easiest first case. Geodesic preparation has no GPU dependency. The bundled state collector needs PyTorch, Transformers, and a model-compatible runtime installed separately. Choose versions compatible with the user's checkpoint rather than replacing an existing working environment.

From the skill folder, after preparing development examples and authorizing compute:

```bash
python scripts/probe.py --model /path/to/checkpoint \
  --development /path/to/run/edit --out /path/to/run/maps \
  --device cuda --template-options '{"enable_thinking":false}'
```

Pass `--revision COMMIT` for a remote model and use the same revision for baseline generation. For local weights, record a Geodesic content fingerprint when creating the patch. `--max-length` and `--max-points` bound the collector's input size; they do not bound provider dollars. A run fails instead of truncating gold targets silently. Do not use evaluation examples with this command.

The collector produces `states.pt` with `[point, layer, hidden]` states and per-point provenance in `points.json`, including target and greedy token IDs and target NLL. `summary.json` records the model, template settings and development manifest hash. Load locally generated tensors with `torch.load(..., weights_only=True)`. The collector does not generate outputs, fit a patch, compute gradients, or prove the mapped information is causal. If the model needs a different wrapper/chat template, write an adapter and preserve the same provenance and checks.

## Reusing the experiment

In a Geodesic source checkout, `sessions/s1/` and `sessions/s2/` show the actual analyses and edits. `gpu/lab.py` and `scripts/lab.py` implement the original Modal loop: run → patch → try → submit. They are BFCL-specific and share the app/volume names used for that experiment. Do not deploy them unchanged for a customer's data or overwrite those volumes. Adapt the dataset loader, exact model loader, chat template, decision detector, scorer, and compute limits; use distinct resources. The published `docs/LAB.md` explains the original guardrails and their limits.

A local run can use the same loop without Modal: analysis scripts load development states; an edit function writes to a freshly loaded source model; the generation runner scores candidates under identical settings. Capture new changed-tensor values with safetensors and use `geodesic patch from-dense` to create the portable artifact. Retain every script and candidate score. The agent implements the source/runtime-specific adapter; there is no generic one-command automatic patch search.

## A readout hypothesis to test

If a linear probe separates required continuation from real stopping in late states, a readout correction may expose information the model already has. For states `X` and desired logit changes `Y`, solve:

```python
gram = X.T @ X
ridge = 0.01 * gram.diag().mean()
delta = torch.linalg.solve(gram + ridge * torch.eye(X.shape[1], device=X.device), X.T @ Y)
```

Fit in float32 or better and check conditioning. Set desired changes to zero on relevant preservation states. Hold out development groups to check that the direction generalizes before spending a full generation try. Update the actual output rows (check whether weights are tied); rounded writes may differ from the mathematical delta. The published amplitudes (-12 and +8) were tuned for one experiment: derive candidates from this model's logits rather than assuming those constants transfer.

If the desired information is absent, readout edits may not help. Inspect earlier states or another constrained edit, or stop with the diagnosis. Do not claim a mask, ridge readout, weight-diff transfer, and optimizer training are the same mechanism.

## Report and stop conditions

Before tries, record maximum candidates, compute authorization, evaluation metric, preservation tolerance, and generation limits. Include model-load and idle time in cost accounting. A JSON ledger writable by agent code is cooperative accounting, not a hard provider cap. Configure provider limits when running remotely. End with a measured patch or a diagnosis and explicit failure; stop remote resources either way.
