# Eval runs: typed GLiNER2 graph, 100 questions, 5 sessions per question

Everything below runs LongMemEval S on 100 questions, capped at 5 sessions each. Reconciliation used PR #373's typed model with the hand vocabulary.

## Scoring bug (#387)
Every run before `7aa430b` paired deepeval's results with questions by position, but deepeval returns them in completion order. So each question got another question's scores.
- Run totals are roughly valid.
- Per-question and per-type results from those runs are not.
- Only the `fx-*` runs were scored correctly.

## Sonnet 4.5 judge (before the fix: totals only)
| run | coverage |
|---|---|
| graph agent over the typed graph (`gliner2-hand-100q-cap5*`) | 15, 12 |
| text search (`text-search-*`) | 33, 34 |
| hybrid, turns only (`hybrid-turnsonly-cap5*`) | 37, 38, 38 |
| hybrid, turns + graph, round 1 (`hybrid-all-cap5*`) | 42, 40, 44 |
| hybrid, turns + graph + user_facts, round 2 graph (`r2-hybrid-all`) | 44 |

## OpenAI gpt-4o judge, fixed scoring (`fx-*`)
Used after the Anthropic credit ran out. Judge and agent share a provider. The OpenAI credit ran out mid-matrix, so each hybrid configuration has one run only.

| run | coverage |
|---|---|
| text search | 24, 27 |
| text search + question date | 26, 24 |
| hybrid, turns only | 25 |
| hybrid, turns + graph + user_facts | 23 |
| hybrid + question date | 23 |

Under this judge the strategies are indistinguishable. It is also harsh on abstention: 2–3 of 8 correct, against Sonnet's 7. The graph's measured lift, a consistent +4 to +6 over turns-only across three runs each, rests on the Sonnet runs from before the fix. It needs re-measuring with a judge that separates the strategies.
