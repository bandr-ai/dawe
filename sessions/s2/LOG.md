# DAWE s2 experiment log

Read BRIEF.md and all context files. Only train data accessed.
Initial budget: 0 tries, 0 runs, $0.2355 / $5.

Hypothesis: the final hidden state already separates plan-start and after-plan decisions from ordinary JSON. Two conditional output-row corrections can favor `<` at plan start and suppress EOS after plan, while preserving v1. First inspect map shapes, tokenization, and baseline point scores; then derive a ridge projection, with zero targets on preservation states.

Run 1: inspected available maps and baseline decisions. `<plan>` tokenizes as `<`, `plan`, `>`; EOS competes with newline after the plan. The v1 decision points supplied by ctx cover call continuation/closing syntax, not reply starts.
Run 2: solved a two-column ridge system over plan start/after-plan, v1 continue/stop, and XML continue/stop. Targets: +6 on `<` at plan start; -12 on EOS after plan and at XML continuation; zero elsewhere. The states_dp_plan_v1 map is empty. Correction norms .0630 and .4698. On 12 train items spanning categories, all plan starts choose `<`; EOS no longer leads after-plan (double newline is harmless to parser). All sampled JSON call decisions stay correct. Next: generation try to detect effects at unprotected v1 starts and generated plan states.

Try 1 — plan6_eos12 (5,051 weights / 1 tensor): v1 99.495798% overall, 99.34% parallel (unchanged); v11_plan 97.478992% overall, 97.37% parallel, 1/595 format failure. Cost ledger: 1 try, 3 runs, $0.44. The 12 exposed TRAIN failures show mostly argument/semantic differences, plus multiple_153 skipping the plan. Hypothesis: +8 plan-start bias fixes that remaining format failure with no v1 cost. Screen this candidate before the second try.
Run 4: +8 screen passes; the previously plan-skipping multiple_153 now chooses `<` with p=.7773. Sampled call decisions preserved. A local script creation attempt used unavailable `python`; retried using sed, with no GPU call or budget charged for the failed local attempt.

Run 5 created plan8_eos12: exactly 5,064 weights in one tensor, two rows; relative tensor norm .001453.
Try 2 — plan8_eos12: v1 99.495798% overall / 99.34% parallel, unchanged. v11_plan 97.815126% overall / 97.37% parallel, zero format failures; 13 remaining failed items. Ledger: 2 tries, 5 runs, approximately $0.63. Selected this patch: it removes the remaining format failure and improves aggregate accuracy by two items, preserving parallel and v1 scores. No additional candidates needed for the targeted format mechanism.
Completed notes.md with derivation, exact changed count, try history, scores, reuse of prior EOS idea, and limitations (including parser acceptance of multiline plans and lack of hidden-test evidence).

Submission confirmed: `./lab submit plan8_eos12 notes.md` returned `submitted plan8_eos12`.
Final `./lab status`: {"tries": 2, "runs": 5, "usd": 0.6333, "limits": {"tries": 25, "runs": 120, "usd": 5}}.
