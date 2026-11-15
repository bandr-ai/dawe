# The agent lab

A coding agent is given a task brief, a data-to-weight map, and a small GPU "lab". It writes an edit to the
weights of a fine-tuned model, spends a limited number of evaluations on **train** items, and submits one patch.
Everything it wrote is in `sessions/`.

## Pieces

| Piece | File | Role |
|---|---|---|
| Brief | `sessions/<s>/BRIEF.md` | the task, rules and tool list the agent was given |
| Context | `sessions/<s>/context/` | harness files and renderer/parser, baseline train scores, map summaries, paired examples, prior patch (s2) |
| Map | `runs/map_*.json`, volume `/runs/sft_v1/maps/` | per-matrix gradients and hidden states at decision points (`gpu/app.py::weight_map`, `gpu/lab.py::decision_map`) |
| Tools | `scripts/lab.py` (client), `gpu/lab.py` (Modal app `geodesic-lab`) | `status`, `run`, `patch`, `try`, `submit` |
| Output | `sessions/<s>/edits/*.py`, `analysis/*.py`, `tries/*.json`, `LOG.md`, `notes.md`, `SUBMITTED.json` | what the agent produced |

Tool semantics: `run` executes analysis code on the GPU with `model` and `ctx` loaded; `patch` runs `edit(model, ctx)`
and saves the diff as a [sparse patch](PATCH_FORMAT.md) (and restores the weights); `try` generates on all 595
train items under the kept and the target harness and returns scores and a few failures; `submit` records the
final choice. Test items are never loaded into the Lab; we run `gpu/lab.py::final` ourselves after submission.

## Budgets

`LIMITS = {"tries": 25, "runs": 120, "usd": 5.0}` in `gpu/lab.py`, tracked in a ledger on the Modal volume.
Both sessions stayed far inside them: s1 used 4 tries, 6 runs, $0.79; s2 used 2 tries, 5 runs, $0.63.

The s1 brief and ledger say `$12` of GPU; s2's say `$5` and so does the code now. The limit was lowered between the
sessions and the original value was not recorded anywhere else, so treat the s1 figure as "$12 as written in
its brief".

## What is enforced, and what is not

The rules in the briefs ("no optimizers", "no generation", "only train items", "budgets can't be reset") are
enforced as follows:

| Rule | Mechanism | Strength |
|---|---|---|
| No optimizers / generation / test data in agent code | regex over the source (`BANNED` in `gpu/lab.py`), and `model.generate` replaced by a stub | stops accidents; trivially bypassed by code that builds the names dynamically |
| Train items only | the Lab loads only train items into `ctx.items` / `ctx.example` | **by convention**: the whole BFCL directory (including test items) is in the image at `/root/data/bfcl`, and the split is a public hash, so agent code could read test items |
| Budgets | ledger JSON on the volume checked before each call | **by convention**: agent code runs in-process with the volume mounted read-write, so it could edit the ledger |
| Weights restored after every call | the Lab compares against a saved copy and copies it back | enforced for in-memory changes |

So this is a **cooperative lab**: a guard rail that makes honest mistakes visible, not an isolation boundary
against an adversarial agent. The logs show both agents stayed within the rules, but the logs are written by
the agents. If you reuse the lab with an untrusted agent, remove test data from the image, run agent code in a
separate process or container without the ledger mounted, and use a read-only volume.

## What the agents were told about the test set

The briefs state aggregate **test-set** numbers so the agent could aim at a target:

- s1: sft_v1's test scores under v1 and v4 (87.7, 73.5), the parallel-item drop, the base model's score (81.4), and the
  retraining numbers (89.7 / 73.0).
- s2: sft_v1's test scores under v1 and v11 (87.7, 20.1), a breakdown of the 204 test replies by failure type
  (80 skip the plan, 76 stop after the plan, 48 correct), the retraining score (88.7), and the **s1 patch's test score
  under v11 (64.2)**.

No test item text or per-item prediction was provided, and patches were chosen by train-item scores only. Still, the
test set was not hidden in the strict sense; the aggregate numbers shaped the goal and, for s2, the plan. We call
the test split **held-out**, not hidden.

## Agent independence

s2 received s1's patch code and notes as context and was told to reuse or combine it if useful; its
edit builds directly on s1's (it reuses the same -12 end-of-turn suppression and refits with the new states). The two
sessions are not independent samples of "what an agent does with this lab". Each task was attempted once, so there
is no estimate of how often the agent finds a fix.

## Running it

See [REPRODUCE.md](REPRODUCE.md#4-the-agent-session). The agent needs: the brief, `context/`, the client
(`scripts/lab.py`, called `./lab` in the briefs), and a session name.
