# DAWE s2: conditional plan-start and continuation readout repair

## Mechanism and derivation

The supplied decision-point map isolates two format decisions: opening a plan instead of starting JSON, and continuing into calls instead of ending the turn after the plan. Final hidden states distinguish these decisions well enough for a small linear readout correction. We raise the `<` token's score at plan-start states and suppress `<|im_end|>` at after-plan states, with near-zero correction on mapped ordinary JSON decisions. This reuses the prior conditional EOS repair idea and its XML continuation/preservation states, but recomputes the correction jointly with the new plan states rather than applying the prior patch unchanged.

Let X stack the final-layer states from states_dp_plan_v11_plan.pt (595 plan-start and 595 after-plan), states_v1.pt (260 continue and 595 stop), and states_v4_format.pt (260 continue and 595 stop). The available states_dp_plan_v1.pt is empty and contributes no rows. For each of the 2,900 rows, Y has two columns: column 1 is +a at plan-start and zero elsewhere; column 2 is -12 at after-plan and XML continue, and zero elsewhere. Compute once:

    Delta = (X.T X + lambda I)^(-1) X.T Y
    lambda = 0.01 * mean(diag(X.T X))

Add Delta's columns to the `<` (token 27) and `<|im_end|>` (token 248046) rows of model.language_model.embed_tokens.weight, then round to bf16. No optimizer, training loop, adapters, gradient descent, or transformer-layer edits are used. Because embeddings are tied, these rows also affect input token embeddings; full generation tests therefore matter beyond the linear readout diagnostics.

For a=8, lambda=278.564728; column norms are .0840255 and .469773. In-map average corrections are +7.99931 for `<` at plan-start and -11.99792 for EOS after-plan. On mapped v1 continue/stop, mean corrections for both columns are within .0013 logits of zero. Those are in-map diagnostics, not independent generalization estimates.

## Screens and try history

Each candidate was screened with ctx.point_scores before its generation try. The first screen used 12 train items spanning all three categories; the second used seven including the first candidate's remaining plan-skipping failure. Both retain sampled JSON call decisions and remove EOS as the top after-plan token. Double-newline continuations are accepted by the supplied parser even when the exact gold next token is a single newline, so exact-token greedy accuracy at this point is not the format metric.

Baseline (provided): v1 99.495798% overall / 99.34% parallel; v11_plan 18.319328% overall / 23.68% parallel.

1. plan6_eos12: 5,051 changed scalar weights in one tensor. v1 99.495798% overall / 99.34% parallel; v11_plan 97.478992% overall / 97.37% parallel. One format failure (multiple_153 skipped the plan). Exposed remaining train failures mostly involve arguments or call selection.
2. plan8_eos12: 5,064 changed scalar weights in one tensor. v1 99.495798% overall / 99.34% parallel; v11_plan 97.815126% overall / 97.37% parallel. Zero format failures under both harnesses; 13 remaining v11_plan semantic failures. Selected for submission.

## Scope and limitations

Only the supplied maps and permitted train examples were used. No hidden-test data or predictions were accessed. The patch preserves measured v1 accuracy; v1 reply-start states are absent from the supplied preservation map, so this preservation is established by the full train generation evaluation rather than a start-state constraint. The parser accepts multiline plans and extra whitespace: this is a repair for the supplied harness metric and does not guarantee that every generated plan occupies exactly one physical line. Remaining semantic errors and hidden-test generalization are outside what these train scores establish.

## Final submission

Selected patch: **plan8_eos12**, source `edits/plan8_eos12.py`. Exactly **5,064 scalar weights in one tensor** change after bf16 rounding, confined to the two embedding/readout rows described above. The reported relative whole-tensor norm change is .001453. Final train scores on all 595 items: **v1 99.495798% overall / 99.34% parallel; v11_plan 97.815126% overall / 97.37% parallel**. V1 is unchanged from baseline and both harnesses have zero format failures. Stop after two screened candidates to avoid spending GPU budget on semantic errors outside the targeted format repair.

Budget before submission: **2/25 tries, 5/120 runs, approximately $0.63/$5** (includes the $0.2355 opening ledger balance). Exact final `./lab status` is recorded in LOG.md.
