# Which memory benchmark should the eval corpus derive from?

Research findings for [context-graph#308](https://github.com/memgraph/ai-toolkit/issues/308), part of [Map: Context-graph emergence pipeline](https://github.com/memgraph/ai-toolkit/issues/297).

Investigated 2026-08-24. All claims cited to primary sources (papers, official repos, dataset cards). Where something could not be verified primarily, it says so.

## Headline finding

**No existing benchmark ships both realistic agent-harness activity and a question/ground-truth answer key.** The field splits cleanly along one axis, and nothing sits on both sides of it:

- Benchmarks **with** questions + gold answers are **conversational** (user/assistant chat turns) or **web-agent** (browser actions). They exercise `sessions-graph`'s text path, partially exercise `actions-graph`, and never touch `skills-graph`.
- Datasets **with** real coding-agent tool calls — the closest match to what context-graph actually ingests — are **SFT training corpora with no questions at all**. They supply realistic content to ingest, not an answer key.

Practical consequence: the corpus will be assembled from at least two upstream sources with different roles, and a substantial residual gap stays with our own synthetic authoring. See [Residual gap](#residual-gap).

## Comparison

| Dataset | What it ships | Licence | Q + gold answers? | context-graph components exercised |
|---|---|---|---|---|
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (v1) | 500 questions over user/assistant chat haystacks (~40 sessions / ~115k tokens for `_S`; ~500 sessions for `_M`) | **MIT** ([repo metadata](https://github.com/xiaowu0162/LongMemEval)) | Yes | `sessions-graph` only |
| [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | 451 questions, 1,870 web-agent task trajectories | **Apache-2.0** ([repo](https://github.com/xiaowu0162/LongMemEval-V2), [dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)) | Yes | `sessions-graph` + partial `actions-graph` |
| [LoCoMo](https://github.com/snap-research/locomo) | 10 conversations, annotated for QA + event summarization | **CC BY-NC 4.0** — *non-commercial* ([LICENSE.txt](https://github.com/snap-research/locomo/blob/main/LICENSE.txt)) | Yes | `sessions-graph` only |
| [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 80,036 coding-agent trajectories (SWE-agent framework) | CC BY 4.0, plus per-repo licences | **No** | `actions-graph` (content only) |
| [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 207,489 coding-agent trajectories (OpenHands + SWE-agent) | CC BY 4.0, with additional MIT/Apache-2.0/BSD permissions | **No** | `actions-graph` (content only) |

## Per-candidate detail

### LongMemEval v1 — best licence, text path only

Record schema, per the [official README](https://github.com/xiaowu0162/LongMemEval/blob/main/README.md):

- `question_id`, `question_type`, `question`, `answer`, `question_date`
- `haystack_session_ids`, `haystack_dates`, `haystack_sessions` — chat history as turns of `{"role": user|assistant, "content": ...}`, with evidence turns flagged `has_answer: true`
- `answer_session_ids` — evidence sessions, enabling session-level scoring

`question_type` covers single-session-user, single-session-assistant, single-session-preference, temporal-reasoning, knowledge-update, multi-session, and abstention. Distributed via HuggingFace (`xiaowu0162/longmemeval-cleaned`). Three variants: `_S` (~115k tokens), `_M` (~500 sessions), and `oracle` (evidence sessions only).

**Fit**: converts cleanly to `Golden` — `question`→`input`, `answer`→`expected_output`, evidence turns→`context`. The haystack sessions inject as text content and would produce `Chunk`/entity nodes and `Episode`s. Produces **zero** `Action`, `Agent`, or `Skill` nodes. The `abstention` question type is a genuinely useful category we would probably not have thought to author — questions whose correct answer is "that isn't in memory."

**MIT is the cleanest licence in this survey** — no restriction on commercial use or on redistributing converted derivatives.

### LongMemEval-V2 — best available for the action path

Published May 2026 ([arXiv:2605.12493](https://arxiv.org/abs/2605.12493)). 451 manually curated questions against haystacks of up to 500 trajectories / 115M tokens, built over WebArena and WorkArena environments (Magento shopping, shopping admin, Postmill forum, ServiceNow).

Full schema, from the dataset's own [`SCHEMA.md`](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/blob/main/SCHEMA.md):

```
questions.jsonl:     id, domain, environment, question_type, question,
                     image, answer, eval_function
trajectories.jsonl:  id, domain, environment, goal, outcome, start_url, states[]
  states[]:          state_index, step, url, action, thought,
                     accessibility_tree, screenshot
```

**Why it matters here**: its five question categories are *static state recall, dynamic state tracking, workflow knowledge, environment gotchas, premise awareness*. The middle three are recognisably **procedural-memory** shaped — the closest external validation available for the mined-`Procedure` concept in `skills-graph`, which currently has no evaluation story at all.

**Fit caveats, and they are real**:
- `action` is a **string** (a browser action), not a structured tool invocation with a name, input object, and result. There is no `tool_use_id`-style correlation between a call and its result, so the call/result pairing that `actions-graph` models has to be synthesised during conversion, not read off.
- No subagent structure anywhere → `Agent` nodes and the `SPAWNED` edge stay unexercised.
- The dataset is 7.12 GB, overwhelmingly screenshots. context-graph ingests no images, so only the text fields matter — a text-only conversion is a small fraction of that.
- `goal` and `outcome` per trajectory map neatly onto a `Session` plus its `Episode`, which is a genuinely good fit.

Apache-2.0 permits commercial use and redistribution of derivatives.

### LoCoMo — ruled out on licence

The benchmark the surrounding ecosystem standardises on (Mem0, Zep and others report against it), which makes its exclusion worth stating explicitly rather than silently.

`LICENSE.txt` in the official repo is **Creative Commons Attribution-NonCommercial 4.0 International**, verified verbatim; GitHub's own metadata reports the licence as `NOASSERTION`/"Other", so the repo file is the authoritative source. The README states no licence at all.

**NonCommercial is disqualifying for this use.** Memgraph is a commercial vendor and this corpus would be used to develop and evaluate a commercial product — that is the case CC BY-NC excludes. This is a licence judgement, not a legal opinion; if LoCoMo is wanted, it needs an actual legal review or a licence grant from Snap Research, not a workaround.

For the record, its shape: `sample_id`, `conversation` (with `session_<n>`, `session_<n>_date_time`, `speaker_a`/`speaker_b`, turns carrying `speaker`/`dia_id`/`text`/`img_url`/`blip_caption`), `event_summary` as annotated ground truth, and `qa` pairs with `question`/`answer`/`category`/`evidence`. Ten conversations, themselves LLM-agent-generated from personas.

An unverified lead worth flagging rather than asserting: a discrepancy was glimpsed between a committed LoCoMo results JSON (91.56%) and a widely-quoted 92.5% figure. Not chased, since the licence rules the dataset out regardless. Noted only so it is not rediscovered as though it were new.

### SWE coding-agent trajectory corpora — content, not a benchmark

Two large, permissively-licensed options: [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (80,036 trajectories; fields `instance_id`, `model_name`, `target`, `trajectory`, `exit_status`, `generated_patch`, `eval_logs`) and [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) (207,489 trajectories across OpenHands and SWE-agent).

The NVIDIA corpus is the more structured of the two: conversation history with `system`/`user`/`assistant`/`tool` roles, plus a `tools` field carrying the tool definitions available during execution as JSON strings (`"type": "function"` objects with `name` and `parameters`). Whether individual assistant steps carry discrete `tool_calls` with separate name/arguments fields was **not** confirmable from the dataset card alone — it needs a direct look at actual records before any conversion work is scoped on it.

**These are the closest thing available to context-graph's real input shape** — genuine coding-agent tool calls, file edits, and environment observations. They ship **no questions and no gold answers**, so they can only ever supply realistic *content*; the answer key would be ours to author. That is a real option (plant known facts in real trajectories, ask about them) but it is synthetic authoring with borrowed content, not adopting a benchmark.

## Recommendation

**Adopt two, decline one, hold one in reserve:**

1. **LongMemEval v1 (MIT) for the text/semantic path.** Cleanest licence in the survey, real questions with gold answers, and a session structure that converts to `Golden` almost mechanically. Start here — it is the lowest-friction way to get the eval loop producing a real score.
2. **LongMemEval-V2 (Apache-2.0) for the action path**, text fields only, ignoring the screenshot bundles. It is the only surveyed source with both agent trajectories *and* an answer key, and its workflow/gotcha question categories are the only external check available on the procedural-memory direction. Budget for real conversion work: browser action strings must be synthesised into call/result pairs.
3. **Decline LoCoMo** on CC BY-NC, notwithstanding its status as the ecosystem default.
4. **Hold the SWE trajectory corpora in reserve** as content for our own authored questions, once the loop works on 1 and 2. They are the best available proxy for context-graph's true input shape, and CC BY 4.0 permits the use.

## Residual gap

What no surveyed benchmark covers, and synthetic authoring must therefore carry:

1. **`skills-graph` is entirely uncovered.** Nothing ships skill or `SKILL.md` usage data. Skill usage, skill surfacing, and mined `Procedure`s have no external evaluation source whatsoever — 100% synthetic. LME-V2's workflow/gotcha questions are the nearest analogue and they are only an analogue.
2. **Subagent nesting is entirely uncovered.** No surveyed dataset contains subagent spawn traces, so `Agent` nodes and the `SPAWNED` inference rule — the subject of two closed maps ([#275](https://github.com/memgraph/ai-toolkit/issues/275), [#288](https://github.com/memgraph/ai-toolkit/issues/288)) — cannot be exercised by any of them.
3. **No source carries `tool_use_id`-correlated call/result pairs** in the Event Protocol's shape. Every candidate needs conversion, and for LME-V2 that conversion is genuinely lossy rather than mechanical.
4. **The "organizational signal" framing is absent everywhere — the largest gap.** Every benchmark surveyed is *single-user personal memory* (LongMemEval, LoCoMo) or *single-task completion* (LME-V2, SWE traces). None models many people writing into and reading from one shared knowledge graph. The multiplayer axis of this project's thesis has no benchmark behind it, and the question style [#299](https://github.com/memgraph/ai-toolkit/issues/299) actually asked for — what an employee would ask their org's data layer — appears in none of them.

Point 4 is the one that should shape expectations: adopted benchmarks can validate that recall works mechanically, but they cannot validate the thesis. Questions testing shared, cross-person organizational knowledge have to be authored, and that is the part of the corpus with no shortcut.

## Sources

- [LongMemEval repo](https://github.com/xiaowu0162/LongMemEval) · [README](https://github.com/xiaowu0162/LongMemEval/blob/main/README.md) · [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- [LongMemEval-V2 repo](https://github.com/xiaowu0162/LongMemEval-V2) · [dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) · [SCHEMA.md](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/blob/main/SCHEMA.md) · [arXiv:2605.12493](https://arxiv.org/abs/2605.12493)
- [LoCoMo repo](https://github.com/snap-research/locomo) · [LICENSE.txt](https://github.com/snap-research/locomo/blob/main/LICENSE.txt) · [README](https://github.com/snap-research/locomo/blob/main/README.MD)
- [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
