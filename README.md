# Leave It

A morning filter for a backlog you have decided not to read.

You point it at a stream of items. A fine-tuned 4B open-weight model answers one
question per item — **does this need a human today?** Everything it defers is held
by a durable Temporal fuse rather than dropped, so the decision to ignore something
survives a crash, a reboot, or a closed laptop lid.

Built for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19).

## The claim, and the number behind it

Evaluated on a **held-out future window** — items created after everything the model
trained on. At an operating point that wrongly defers at most 5% of the items that
genuinely needed a maintainer within 24 hours:

| arm | deferred | of pool | AUC |
|---|---|---|---|
| fine-tuned 4B (Tinker LoRA) | 73 / 731 | 10.0% | **0.665** |
| tfidf + hand features | 78 / 731 | 10.7% | 0.663 |
| tfidf + logistic regression | 57 / 731 | 7.8% | 0.634 |
| base 4B, same prompt, no LoRA | 49 / 731 | 6.7% | 0.516 |
| text length only | 28 / 731 | 3.8% | 0.426 |
| already carries a label | 23 / 731 | 3.1% | 0.393 |
| perfect ranker (ceiling) | 496 / 731 | 67.9% | 1.000 |

Read it honestly: the fine-tune does **not** dominate the best classical arm on ranking
quality (0.665 vs 0.663 is a tie), and the classical arm defers 5 more items at exactly the
5% budget. The fine-tune does win against the baseline the Tinker category specifies — the
same model, same prompt, no LoRA — moving deferral from 49 to 73 items at AUC 0.516 → 0.665.

Cost axis, measured on the same 40 items:

| arm | verdict produced | tokens per item |
|---|---|---|
| base 4B, thinking enabled | 15 / 40 | 696 |
| fine-tuned 4B, verdict as first token | 40 / 40 | 2 |

Numbers are regenerated, never typed: `python3 eval/policy.py eval/scores_*.json`
and `python3 app/build_site.py`.

Accuracy is deliberately **not** reported. An arm that says "needs you" to
everything is 32% "accurate" here and saves the user precisely nothing.

## Where the data comes from

`data/fetch_issues.py` pulls real issues from real repositories over GitHub GraphQL,
including per-comment timestamps and author associations, so the label can be
*maintainer* attention rather than *anybody replied*. In large repos, passing users
pile on with "+1 me too"; counting that as attention would teach the model the wrong
thing.

Labels:

- `needs_today` — a MEMBER/OWNER/COLLABORATOR commented, or the item was closed,
  within 24 hours of arrival.
- Items newer than 24h + margin are dropped, not labelled "no": an outcome that has
  not happened yet is censoring, not signal.
- The split is **temporal**: train on older items, test on the newest window.

## The guarantee, in code

The model supplies one number. `app/core.py` decides what that number is allowed to
mean:

1. nothing is deleted, hidden forever, or moved;
2. every deferral gets a fuse, and the fuse is a server-side timer, not a `sleep`;
3. one outside signal unseals an item immediately;
4. the ledger is append-only, so any verdict can be re-read and argued with.

`app/run_demo.sh` executes the proof: three seals are opened, the worker is killed
with `kill -9` while all three fuses are pending, an `on_attention` signal is
delivered to a sealed item **while no worker process exists at all**, and then the
worker comes back. The fuses still fire on the wall-clock schedule they were
promised, and the signal that was delivered to nobody is applied on restart.

## Layout

```
data/     fetch real issues -> labels -> temporal split
train/    Tinker LoRA: build the arm, score the held-out window
eval/     the single policy every arm is judged by, plus classical baselines
app/      deterministic seal layer, Temporal workflow + worker, demo, site builder
docs/     the generated demo page (self-contained, no external requests)
```

## Run it

```bash
python3 data/fetch_issues.py 10
python3 data/build_dataset.py
python3 eval/baselines.py
python3 train/lm.py train tuned 400     # needs TINKER_API_KEY
python3 train/lm.py score tuned
python3 eval/policy.py eval/scores_*.json
cd app && ./run_demo.sh                  # needs the temporal CLI
```

## Honest limits

- Three repositories, one 5-day held-out window: this is a demonstration, not a
  benchmark.
- The label is "a maintainer touched it", which is not the same as "it mattered".
- A mis-deferral is the expensive error, so the policy is asymmetric on purpose: it
  buys patience rather than quiet.

## Licence

MIT for the code. Issues fetched from public repositories remain under their own
projects' licences; the derived dataset is used here only for evaluation.
