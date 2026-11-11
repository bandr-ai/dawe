# DAWE experiment log

Read BRIEF.md and all seven context files. Baseline train: v1 99.496% (parallel 99.34%), v4 78.992% (parallel 36.18%). Initial budget: 0 tries, 0 runs, $0.1561.

## Hypothesis 1
The late residual stream already contains the missing continue signal; a targeted output-row edit can expose it without disturbing internal computation or JSON. Inspect parameter names, state scales, and decision logits before selecting a closed-form patch. No optimization or training will be used.

Inspection (run 1, $0.16 total): saved final states have norms ~157–162; output is tied to the embedding matrix. XML parallel_1 and parallel_2 choose EOS with probability .991 and .754 at required continuations, but final stops are correct. Only supplied v1/v4 maps were loaded.

Experiment eos8: solve a regularized linear system for a single EOS row correction, targeting -8 logits on v4 continue states and 0 on v4 stops and v1 decisions. Ridge = .01 times mean covariance diagonal. This is a single closed-form weight edit, not iterative training.
Patch eos8 created: 2,555 changed weights in one embedding tensor (EOS row). In-map corrections: v4 continue -7.978±.209; v4 stop -.009±.131; v1 continue -.0004±.053; v1 stop .0001±.051. First full train try running.

Try 1 eos8: v1 .9949579832 overall, .9934 parallel (unchanged); v4 .9327731092 overall, .9276 parallel. V4 simple .9247 versus baseline .9281: one extra simple failure. No format failures. Budget 1 try, 2 runs, $0.30. Most exposed failures are pre-existing value-format/content errors. Next: reserved-state diagnostic and smaller -6 correction to reduce collateral effects.

Run 3 reserved-state diagnostic: fit on 80% of each state group, reserve every fifth state. Reserved corrections (mean±SD): v4 continue -8.031±.546 (least suppression -4.798), v4 stop -.034±.237, v1 continue -.024±.184, v1 stop -.002±.098. This supports a transferable direction within training states, not hidden-test performance; the split is by state rather than item.

Patch eos6: 2,552 weights, one tensor, delta norm .2714 versus .3619 for eos8. Try 2 running.

Try 2 eos6: v1 .9949579832 overall / .9934 parallel (unchanged); v4 .9344537815 overall / .9145 parallel; simple .9349; multiple .9536; no format failures. Budget 2 tries, 4 runs, $0.49. Smaller correction improves aggregate by one item versus eos8, but loses two parallel successes. Test a=12 to bound the strength tradeoff.

Patch eos12: 2,557 weights in one tensor, delta norm .5428. Try 3 running. This is the same analytic direction with a larger scalar, to test whether residual missing calls need stronger EOS suppression.

Try 3 eos12: v1 .9949579832 overall / .9934 parallel; v4 .9394957983 overall / .9408 parallel, simple .9315, multiple .9536, no format failures. Best so far; 36 target failures. Budget 3 tries, 5 runs, $0.64. One final strength check at a=16 before submission.

Patch eos16: 2,559 weights in one tensor, delta norm .7238. Try 4 running; final candidate evaluation.

Try 4 eos16: v1 .9949579832 overall / .9934 parallel; v4 .9378151261 overall / .9408 parallel, simple .9281, multiple .9536; no format failures. Budget 4 tries, 6 runs, $0.79. Stronger suppression has no further parallel benefit and loses one simple success. Select eos12: best overall score and tied best parallel score, with fewer changes than eos16. Final notes completed; submitting eos12.

Submission confirmed: `submitted eos12`. Final `./lab status`: 4/25 tries, 6/120 runs, $0.7949/$12. No hidden evaluation data or predictions accessed; all local work stayed within s1.
