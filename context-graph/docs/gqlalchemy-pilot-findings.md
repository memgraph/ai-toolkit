# GQLAlchemy pilot: infrastructure findings and rerun gates

Recorded 2026-10-09. This report concerns six bounded, sequential, ChatGPT-authenticated
Codex sessions using `gpt-6-luna` at medium reasoning. Each had a 120-second wall limit
and a 14-tool-start limit. GQLAlchemy was pinned to
`54f9b751519d1c13238f44a54bdbdc1894e2c9e4`. No upstream issues were closed or code
changes published. The pilot is **invalid as a memory comparison**.

The task was opt-in NetworkX multigraph export for GQLAlchemy #375, followed by a
fresh-session assessment of batching #246 and #319. All arms were intended to get
the same frozen GitHub resources, starting code, model and testing instructions.
Memory extraction used default HyGM, `summarize=False`, and no ontology derivation
or paid LLM API. Resources were fetched once through resources-graph.

## Findings

| Finding | Evidence and impact | Ownership / next step |
|---|---|---|
| MCP reads denied | Two plain-arm preflight sessions encountered approval-required resource reads under noninteractive `never` policy. That arm was excluded. | Harness configuration. Explicitly authorize the intended read tools and verify them from the actual Codex session. |
| Agent could not reach test Memgraph | Agent sandbox denied fixture access while the external evaluator could connect. Agent and evaluator therefore had different testing capabilities. | Harness configuration. Enable required network access, then verify a harmless query from each arm before issue work. |
| Isolated sessions missed native capture | Project hook files were written, but the graph follow-up had no captured first-session history. Recall returned empty. Capture was recovered afterward from Codex JSONL through the Event Protocol. | Hook discovery needs a live reproduction using the same isolated `CODEX_HOME`, plugin installation and launch settings. Do not infer a universal Codex hook failure from this run. |
| Doctor does not prove hook discovery | `_check_runtime()` invokes the adapter with a probe payload directly. `_check_mcp()` checks imports and registered tools. These checks can pass without demonstrating that a live Codex process discovered hooks or called MCP. | AI Toolkit improvement: distinguish dependency/adapter checks from an opt-in live capture-and-recall check. |
| Background embedding is not controlled by `auto_reconcile` | `SessionsGraphConnector._on_session_end()` and `_on_turn_end()` spawn detached `embed` jobs. `auto_reconcile=False` only disables detached reconciliation. Offline replay overlapped embedding with synchronous processing; the Docker VM exhausted memory and killed the experiment graph. | AI Toolkit improvement: explicit embedding scheduling control and bounded/coalesced jobs. OOM is observed; the relative contributions of embedding, extraction and other containers are not isolated. |
| Runner continued after reconciliation errors | The pilot runner caught reconciliation errors and would proceed to the next graph-memory task. This could silently compare an arm without functioning memory. | Harness defect. Local runner now aborts after processing failure and checks captured sessions' extraction/embedding status before a follow-up. |
| A submission's own regression failed | Both patches passed the independent acceptance check. File-memory submission: 18 tests passed. Graph-memory submission: 17 original tests passed, one new test failed by treating stored `id` properties as internal node IDs. | Candidate quality issue. A passing independent check does not excuse a failing submitted test; human review is still pending. |
| Graph follow-up exhausted its tool allowance | It stopped at 14 tool starts and produced no final report. | Budget outcome, not a toolkit defect. Count interrupted work and report it as incomplete. |
| Dollar cap could not be enforced | Session usage was observable; incremental subscription/credit charges and dollar conversion were not established. | Experiment limitation. Report dollar cost as unknown. Do not substitute API token prices for Codex subscription costs. |
| Evaluator isolation was cooperative | Prompts prohibited reading evaluator files, but filesystem read isolation was not enforced. | Harness limitation. Enforce isolation before treating a larger run as controlled benchmark evidence. |

The graph was recovered. A later direct recall smoke returned 13 turns and 16
facts; this demonstrates recovered local retrieval, not memory use during the
agent run. The first session completed extraction under ontology version 0; the
second remains pending. There are zero Episode nodes, consistent with narrative
summaries being disabled; Episode count is not the readiness criterion.

Released runtime versions used: agent-context-graph 0.4.0, actions-graph 0.4.0,
sessions-graph 0.7.0, skills-graph 0.2.0, resources-graph 0.1.0, hygm 0.2.0,
memgraph-toolbox 0.2.0, unstructured2graph 0.7.0, and mcp-memgraph 0.4.1.
The Memgraph MAGE image was pinned to
`sha256:97383094cd266c1091e69ead8db91e162b0feb356cae438dd0422ce8559d514f`.

The scheduling and doctor observations were also checked against main at
`15e73c9`, after setup/configuration fixes in PR #481. Those fixes should not be
duplicated. This report proposes follow-up work; it does not change runtime behavior.

## Proposed focused implementation PRs

1. **Control background embedding independently.** Provide a public SDK option
   for callers doing synchronous/offline processing, preserving current defaults.
   Cover session end and turn end. Add bounded/coalesced scheduling so repeated
   events cannot launch unlimited workers for the same session. The exact config
   interface and cross-process coordination need design before implementation.
   Verify persistence and successful explicit embedding against real Memgraph;
   process-spawn policy can be tested separately without mocking Cypher strings.
2. **Verify live capture and recall.** An opt-in diagnostic launches the actual
   harness in a disposable workspace/graph with a known sentinel, then verifies
   session/action provenance and sentinel recall in a fresh session. It must use
   the target isolated config and install path, state any model usage beforehand,
   and distinguish an empty result from missing capture, pending processing,
   tool denial or connection failure. Keep the ordinary doctor lightweight.

Memory/retrieval work should coordinate with [map #297](https://github.com/memgraph/ai-toolkit/issues/297).
Diagnostic visibility also relates to [map #374](https://github.com/memgraph/ai-toolkit/issues/374).
Neither proposal needs a broad ontology redesign.

## Gates before another comparison

- Run a tiny capture-and-recall infrastructure smoke using the exact future
  Codex launch settings. First session plants a unique experimental fact; the
  next fresh session must retrieve it through MCP with matching provenance.
  Verify resource reads and agent-side fixture access in the same smoke.
- Use one processing path. For replay, avoid unverified simultaneous native
  capture, suppress detached embedding through a supported interface when
  available, and finish extraction/embedding before launching the next task.
  Fail closed on missing capture, processing errors or incomplete statuses.
- Monitor peak Docker/host memory while processing representative content.
  Serial scheduling is a mitigation, not evidence that OOM is fixed.
- Enforce evaluator separation, freeze acceptance criteria, and require both
  compatibility tests and an arm-blinded human review. The ordinary-context arm
  must be present in the comparison.
- Reserve infrastructure smoke usage inside a new explicit session allowance.
  Preserve the original six-session ledger, sequential execution and wall/tool
  limits. Do not restart the same runner to reset the total allowance. Dollar
  costs stay unknown until the billing conversion can be observed.

The next run should be a small infrastructure validation, followed by a paired
pilot only if these gates pass. Expanding to hundreds of issues now would magnify
configuration failures rather than measure the value of memory.

## Local evidence and current checks

Raw JSONL, evaluation outputs and graph snapshots are retained locally under
`context-graph/experiments/gqlalchemy/runs/20261009T115946Z/` in the originating
checkout. They are excluded from Git; this report is a portable transcription,
not a published reproducible benchmark dataset. No credentials or raw session
transcripts are included here.

The local runner's three real-process supervisor tests pass. Its new readiness
gate was exercised against the recovered real graph: it rejects the pending
second session. Syntax/lint and targeted type checks pass. These checks do not
establish live Codex readiness. Additional model launches remain disabled while
the blockers are unresolved.

## Follow-up validation on 2026-10-09

The user authorized two additional bounded infrastructure sessions. A durable
ledger reserved each launch before execution; the original six-session pilot
ledger was preserved. No comparative issue-solving sessions were launched.

Both fresh GPT-6 Luna sessions read frozen issue #375 through the resource MCP
tool and queried `RETURN 1 AS ok` against a new disposable Memgraph from inside
the sandbox. The first completed in 11.92 seconds with three tool starts; the
second in 16.85 seconds with four tool starts. The second initially sent a
`query` argument to the resource tool, recovered by supplying `address`, and
completed successfully. Explicit approval configuration resolved the earlier
noninteractive tool denial.

Actual first-session JSONL was replayed through the Event Protocol, followed by
serial default-HyGM extraction and embedding with summaries disabled. The second
fresh session retrieved a unique fixture name through the recall MCP tool and
reported the correct first-session ID. Independent assertions verified that the
name and provenance appeared in the tool result, the second prompt did not contain
the name, and the two thread IDs differed. No repository note carried the name.

Both sessions subsequently reached completed reconciliation and embedding
statuses. The Memgraph container's sampled peak was 899.5 MiB under a 2 GiB
container limit; no OOM occurred. This is sampled container memory, not peak
host memory or proof of stability under larger content. Copied authentication
was removed and the validation container stopped; previous containers were kept.

Extraction reported one nonconformant relation for the first session and two for
the second. Second-session processing also logged an already-existing typed
constraint error while ultimately completing. These require integrity review
before making claims about extracted graph-fact quality. Successful recall here
does not isolate the contribution of extracted facts from stored session turns.

**Result:** explicit replay capture, agent-side connectivity, authorized MCP
reads, serial processing and fresh-session recall now pass this small smoke.
Native hook discovery remains unverified. A small paired pilot can use this replay
path once evaluator isolation and quality-review gates are addressed; the full
benchmark remains disabled. Dollar charges are still unknown.

Local evidence: `runs/validation-20261009/validation-summary.json`, launch ledger,
per-session JSONL and sampled container-memory log under the experiment directory.
