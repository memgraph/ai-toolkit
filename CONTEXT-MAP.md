# Context Map

One `CONTEXT.md` per bounded context. Read the ones relevant to your topic.

| Context | Glossary | ADRs | Scope |
|---|---|---|---|
| unstructured2graph | `unstructured2graph/CONTEXT.md` | `unstructured2graph/docs/adr/` | Files/URLs/text → `Chunk` nodes + pluggable entity/relation extraction |
| Agent Context Graph | `context-graph/agent-context-graph/CONTEXT.md` | `context-graph/agent-context-graph/docs/adr/` | Event hub: normalizes harness activity into events, routes to connectors |
| Actions Graph | `context-graph/actions-graph/CONTEXT.md` | — | Tool calls/results as action nodes, for observability |
| Skills Graph | `context-graph/skills-graph/CONTEXT.md` | — | Agent Skills usage per session |
| Sessions Graph | `context-graph/sessions-graph/CONTEXT.md` | — | Memory writes/recall, session reconciliation (formerly "Memory Graph") |
| Resources Graph | `context-graph/resources-graph/CONTEXT.md` | — | External resources (GitHub first) the agent touched: private Touches, shared canonical Resources (implemented and published as `resources-graph`; design map #454) |
| Context Graph Eval | `context-graph/eval/CONTEXT.md` | — | Measures whether memory output answers recall questions |

`context-graph/memory-graph/CONTEXT.md` is a redirect to Sessions Graph, not a separate context.
