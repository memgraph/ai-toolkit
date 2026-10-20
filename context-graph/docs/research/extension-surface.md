# Context Graph extension surface

What an outside developer can change to build and configure their own memory
on Context Graph, as of `main` @ `d9faea5` (2026-10-19). Paths are relative to
`context-graph/` unless noted. Each knob is classified:

- **Config** — `~/.config/context-graph/config.toml` (or `CONTEXT_GRAPH_CONFIG`)
- **Plug-in** — a separate pip package registers through an entry point; no fork
- **Python API** — inject in code, but only if you write your own driver; hooks/CLI won't use it
- **Hardcoded** — fork required

> **Status (same branch):** the hardcoding below is now config-driven where it needed no new plugin
> mechanism — one connector table (`memgraph_context_graph/connectors.py`) and `[capture] connectors`
> (gap 1, apart from the plugins' `hooks.json`, which keep their flags until a CLI release), `[extraction]`
> shared by reconcile/derive (gap 3, config side), `[llm] derive_model` (gap 8), and a per-model recall
> query prefix (gap 5, prefix side). Entry points for third-party connectors/backends are deferred.
> Gap 4 is closed by the capture policy (`memgraph_context_graph/capture_policy.py`), applied in
> `AgentLink.emit` and configured through `[capture]` and `[redact]`; it is config, not a plugin protocol.

## Summary

| Layer | Plug-in / Config today | Python API only | Hardcoded |
|---|---|---|---|
| Capture (hooks → events) | runtimes (`memgraph_context_graph.runtimes`) | `GraphConnector` via `AgentLink.add_connector` | connector registry, `EventType` enum, redaction/truncation (none exists) |
| Storage schema | spaces spec `labels`/`edge_types` | — | each component's labels, constraints, Cypher |
| Ontology | `[ontology] path` YAML, `derive = extend\|off` | `ontology.supply(HygmModel)` (OWL, LLM, hand-built) | User⇒Person rule, core/catch-all labels |
| Extraction | — | any `ExtractionBackend` into `Reconciler.reconcile` | backend + GLiNER2 checkpoint/thresholds/chunking on CLI/hook path |
| Learned types (derivation) | `[llm]` keys | `derivation.run(llm=, observer=)` | provider list, default models, gate constants, `DerivationLimits` |
| Embeddings | `[recall] embedding_model` | `model=` | provider (MAGE `embeddings.text`), bge query prefix, passage size |
| Recall | `[recall]` lanes and widths | `recall(config=RecallConfig)` | lane Cypher, RRF, prompts, no reranker hook |
| Memory files | `[memory] backend = context-graph\|native` | `SessionsGraph` subclass | `MemoryStore` (no protocol), limits, `/memories` root |
| Model-facing tools | `memgraph_context_graph.tools` entry point | — | tool hint shown only if its `connector` is in the closed list |
| Multi-user | `edition`, spaces TOML spec | `memgraph_context_graph.editions` | edition names `community`/`enterprise` only |
| Install / setup | setup flags (runtimes, url, user, memory) | — | connector list, agent table, MCP registration per agent |

## The three seams that already work

1. **Tools** — `memgraph-context-graph/src/memgraph_context_graph/tools.py:44-89`.
   Implement `name, description, input_schema, connector, session_hint, call(arguments, config) -> ToolResult`,
   optionally `available(config)` and `session_context(config, payload)`, and register under
   `[project.entry-points."memgraph_context_graph.tools"]`. `mcp_server.py` serves every
   loaded tool. Copy `resources-graph/src/resources_graph/tool.py` (reads its space via `spaces_for(config)`).
2. **Runtimes** — `hooks/runtime_plugin.py:27-78`, `RuntimeCLIPlugin` protocol, or declaratively via
   `adapters/_spec.py` (`RuntimeSpec`, `SpecPlugin`; module is underscore-private). Works for
   `hook run/init` and `doctor`; *not* for `setup`/`uninstall` (`setup/agents.py:56`, `adapters/agent_mcp.py:186`).
3. **Ontology** — hygm `ManualStrategy` YAML (`hygm/src/hygm/strategies/manual.py`):
   ```yaml
   entity_types:
     - {label: Person, description: "another named individual", identity: global}
     - {label: Organization, description: "a company", identity: global}
   relation_types:
     - {label: works_for, description: "is employed by", start_labels: [User, Person], end_labels: [Organization]}
   ```
   Point `[ontology] path` at it; it is re-synced (by hash) before every CLI reconcile and stored as
   a versioned `(:OntologyVersion)`. `derive = off` uses it verbatim; `extend` pins it and lets the LLM grow it.

## Contracts a custom memory has to honour

- **Join node**: `MERGE (:Session {session_id})`. Only sessions-graph owns `(:User)` and `HAD_SESSION`.
  Documented in prose only; no shared schema module (`:Session(session_id)` index created by two packages).
- **Event Protocol**: `memgraph-context-graph/src/memgraph_context_graph/events.py` — 12 closed event types
  (SessionStart/End, TurnEnd, ToolStart/End, AgentStart/End, LLMStart/End, Handoff, Message, ErrorOccurred).
- **Connector**: `protocols.py:17-48` — `on_event`, optional `supports`, optional `context_before_tool`.
- **Extraction backend**: `unstructured2graph/src/unstructured2graph/extraction_backend.py:25-44` —
  `workspace_label` + `async aingest_chunk(memgraph, chunk)`. The *real* contract is wider than the
  protocol: recall's fact lanes read `MENTIONED_IN.sources`, `r.source_id = action_id`, `r.valid_at/role/text/confidence`,
  `n.text`, and entities need `entity_type` + `file_path = chunk.hash`. Miss these and recall returns nothing.
  Expose `.ontology` on the backend or enforcement falls back to unstructured2graph's LightRAG-style
  `DEFAULT_ONTOLOGY`, not hygm's default.

## Gaps, ranked by how much they block "build your own memory"

1. **No connector entry point.** `hooks/runner.py:460` `_CONNECTOR_ADDERS` (+ `_SCHEMA_COMPONENTS :118`,
   doctor `cli.py:522-597`) is a closed dict; unknown `--connector` raises. A pip package can serve tools
   but cannot capture events in the hook path. Connector selection also lives in plugin `hooks.json`
   (repeated 13× Claude / 7× Codex), not config — a user can't change it without editing a file that
   updates overwrite.
2. **Config file drops unknown sections.** `adapters/_identity.py:385` `_render_config` rebuilds the file
   from `HookConfig`; any `config set`/setup run deletes a third-party `[mymemory]` section. No public API
   to read a plugin's own section. Parser is a hand-rolled TOML subset (strings only).
3. **Extraction backend not selectable outside Python.** `sessions-graph/.../core.py:973-986`
   always builds `GLiNER2Backend` with defaults; CLI `reconcile` passes none. An injected backend also
   gets `ontology_version = None` and is **ignored by derivation** re-extraction/relabelling
   (`derivation.py:280,364`).
4. **No capture-time redaction/filter/truncation hook.** actions-graph stores full tool I/O and
   `event.metadata` (`actions-graph/src/actions_graph/connector.py:229-236`).
5. **Embedding provider fixed to MAGE**, and recall's `QUERY_INSTRUCTION` (`recall.py:43`) is bge-specific,
   so changing `embedding_model` silently degrades queries. No vector index — exhaustive cosine.
6. **Recall is not composable**: no pluggable lane, reranker, or prompt override (`recall.py:147-299`).
7. **Memory store has no protocol** — `MemoryStore` is concrete; `memory_backend.py:131-145` and
   `anthropic_memory.py` call `SessionsGraph.memory_store` directly. `MEMORY_BACKENDS` is closed.
8. **Derivation LLM**: provider chosen by env key (Anthropic, then OpenAI), no `[llm] model`, gate constants
   and `DerivationLimits` not reachable from sessions-graph.
9. **resources-graph's `PlatformAdapter` seam is half-built** — `ADAPTERS = (GitHubAdapter(),)` constant,
   `Address.from_key` rejects non-GitHub, store methods write GitHub labels. Jira/Linear needs a fork.
10. **Setup/edition closures**: setup's agent table and connector list, edition names limited to two.
11. **Viewer** has no panel registration; custom labels don't surface.

Duplication worth folding while fixing #1: default connector list/pins appear in ≥8 places
(`setup/tool.py`, `setup/agents.py:30`, `hooks/cli.py:143`, `scripts/install.sh`, 2× `bootstrap.sh`,
2× `hooks.json`, 2× `SKILL.md`).

## Suggested seams (smallest first)

1. Preserve unknown config sections on rewrite + `config.section(name) -> dict` public reader.
2. `memgraph_context_graph.connectors` entry point: factory `(memgraph_target, config) -> GraphConnector`,
   optional `setup_schema()` and doctor probe; replace `_CONNECTOR_ADDERS`/`_SCHEMA_COMPONENTS` with it.
3. `[capture] connectors = "..."` in config; hooks.json calls the hook without `--connector` flags.
   Collapses the 8-way duplication.
4. `[extraction] backend = "<entry point name>"` (+ GLiNER2 `model`, thresholds) resolved through a
   `memgraph_context_graph.extraction_backends` group; derivation uses the same resolved backend.
5. A capture filter/redactor protocol applied in `AgentLink.emit` before any connector sees the event.
6. `[llm] derive_model`; `QUERY_INSTRUCTION` keyed by embedding model.
7. Later: `MemoryStore` protocol, pluggable recall lanes, resources `PlatformAdapter` entry point.
