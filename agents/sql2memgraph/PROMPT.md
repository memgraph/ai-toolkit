# SQL → Memgraph Migration Agent Prompt

## TL;DR

You are working inside the `agents/sql2memgraph` package, a UV-managed Python project that turns relational schemas (MySQL/PostgreSQL) into Memgraph graph schemas and data. The primary entry point is `main.py`, which wires configuration, environment validation, database analyzers, and the HyGM graph-modeling subsystem. Changes usually touch:

- `core/` — Orchestrates the migration workflow (`migration_agent.py`) and HyGM graph modeling (`hygm/`).
- `database/` — Connectors and analyzers for the source RDBMSs.
- `query_generation/` — Cypher generation helpers.
- `utils/` — Environment setup, connection probes, and CLI helpers.

Always maintain the CLI experience (`main.py`) and respect the line-length < 79 char lint rule.

## Tech Stack & Tooling

- Python 3.10+, managed with [uv](https://github.com/astral-sh/uv).
- Memgraph as the target graph database (Bolt connection).
- Optional LLM features powered by OpenAI (LangChain / LangGraph patterns inside `core/hygm`).
- Testing: `pytest` under `tests/` (integration heavy, uses mocks for DB analyzers).

## Core Concepts

- **HyGM (Hypothetical Graph Modeling)** lives in `core/hygm/hygm.py` and exposes modeling modes via `ModelingMode`:
  - `AUTOMATIC` – one-shot graph generation.
  - `INCREMENTAL` – table-by-table confirmation flow that now includes the full interactive refinement loop.
- **GraphModelingStrategy**: `DETERMINISTIC` (rule-based) and `LLM_POWERED` (needs an LLM+API key).
- **SQLToMemgraphAgent** (`core/migration_agent.py`) coordinates schema analysis, HyGM modeling, query generation, execution, and validation.
- **Database analyzers** in `database/` introspect MySQL/PostgreSQL schemas and emit a normalized metadata structure consumed by HyGM.
- **Query generation** in `query_generation/` converts the graph model + metadata into Cypher migrations, indexes, and constraints.
- **Database data interfaces** in `database/models.py` define the canonical `TableInfo`, `ColumnInfo`, `RelationshipInfo`, and `DatabaseStructure` data classes. These objects flow from analyzers into HyGM via the `to_hygm_format()` helpers, ensuring consistent schema metadata for every modeling mode.
- **Graph schema structures** in `core/hygm/models/graph_models.py` (e.g., `GraphModel`, `GraphNode`, `GraphRelationship`) capture the in-memory graph representation HyGM produces and later serializes to schema format.
- **LLM structured output models** in `core/hygm/models/llm_models.py` (`LLMGraphModel`, `LLMGraphNode`, `LLMGraphRelationship`) describe the contract for AI-generated schemas and include `to_graph_model()` utilities to convert LLM responses into the standard `GraphModel` objects.
- The `GraphModel` serialization format matches the canonical spec in `core/schema/spec.json`, so any changes to the schema data classes should be mirrored against that document.
- Source tracking helpers in `core/hygm/models/sources.py` annotate nodes, relationships, properties, indexes, and constraints with origin metadata. Preserve these when modifying `GraphModel` so downstream migrations retain the link back to the originating tables or user-applied changes.

## Entry Points & CLI Flow

- Run with `uv run main.py` (banner, env checks, connection probes, then the migration workflow).
- CLI prompts include:
  - Graph modeling mode (automatic / incremental with interactive refinement).
  - Modeling strategy (deterministic / AI-powered).
  - Confirmation dialogs during automatic or incremental flows.
  - Mid-session prompts that let users launch the interactive refinement loop after accepting or modifying a table during incremental runs.
- Environment validation happens before migration; failures raise `MigrationEnvironmentError` or `DatabaseConnectionError` from `utils/`.

## Configuration & Environment

- `.env` (or env vars) must provide:
  - `SOURCE_DB_*` (host, port, name, user, password, type [`mysql|postgresql`]).
  - `MEMGRAPH_*` connection details.
  - `OPENAI_API_KEY` for LLM features; omit or leave empty to disable LLM strategy.
- Memgraph must run with `--schema-info-enabled=true` for schema validation.

## Testing & Validation

- Install deps: `uv sync` (or `uv pip install -e .`).
- Run targeted tests: `uv run python -m pytest tests/test_integration.py -v`.
- Keep graph-modeling logic covered via integration tests; they rely on mocked analyzers.
- Observe linting: adhere to 79-character lines and existing logging conventions (`logging` module).

## Development Tips

- Update `PROMPT.md` when project layout or workflows change.
- Prefer existing abstractions: use `SQLToMemgraphAgent` methods, HyGM strategies/helpers, and database adapters.
- For new modeling flows, ensure `ModelingMode` and CLI choices stay in sync.
- Preserve user-facing prompts/emojis—they guide the interactive experience.
- When adding LLM-dependent features, guard them when API keys or clients are missing.
- Document new commands or config expectations in this prompt and `README.md` if user-facing.

## Useful Commands

```bash
# Sync dependencies
uv sync

# Run the CLI
uv run main.py

# Run tests
uv run python -m pytest tests -v
```

## When Generating Code

- Mention which files you change and why, referencing modules above.
- Explain how to rerun the CLI or relevant tests after modifications.
- Provide small follow-up suggestions if more validation is needed.
- Keep output concise but cover context so the next agent run has everything it needs.
