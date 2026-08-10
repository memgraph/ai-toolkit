# skills-graph

A small library to persist, retrieve and evolve AI skills in [Memgraph](https://memgraph.com).

> Part of the [Context Graph](../README.md) family — usually wired into `agent-context-graph` so that skill usage across Claude Code / Codex sessions is recorded automatically. This README covers using it directly.

## Graph Model

```
(:Skill {name, description, content, license, compatibility,
         allowed_tools, metadata, created_at, updated_at, source_path?})
(:Skill)-[:DEPENDS_ON]->(:Skill)
(:Session)-[:USED_SKILL {first_access, last_access, access_count, actions}]->(:Skill)
```

`USED_SKILL` is written from a `MERGE (:Session {session_id})` — skills-graph MERGEs the shared Session node it does not own (see the [Context Graph map](../CONTEXT-MAP.md)). `source_path` is set only when a skill is created by observing a local `SKILL.md` read.

## Quick Start

```python
from skills_graph import SkillGraph, Skill

# Connect (uses MEMGRAPH_URL, MEMGRAPH_USER, MEMGRAPH_PASSWORD env vars by default)
sg = SkillGraph()

# Prepare the database schema (constraints + indexes)
sg.setup()

# Store a skill.
# name: lowercase letters/digits/hyphens only, 1-64 chars, no leading/trailing/
# consecutive hyphens (an invalid name raises SkillValidationError).
sg.add_skill(
    Skill(
        name="memgraph-cypher",
        description="Writing Cypher queries for Memgraph",
        content="# Cypher for Memgraph\n\nUse MATCH, CREATE, MERGE ...",
    )
)

# Retrieve by name
skill = sg.get_skill("memgraph-cypher")

# Search
sg.search_by_name("memgraph")

# Dependencies
sg.add_dependency("advanced-cypher", "memgraph-cypher")
deps = sg.get_dependencies("advanced-cypher")

# List all
all_skills = sg.list_skills()

# Update
sg.update_skill("memgraph-cypher", content="updated content")

# Delete
sg.delete_skill("memgraph-cypher")
```

## Agent Context Graph integration

Wire `SkillGraphConnector` into an `AgentLink` and skill usage is recorded automatically from any runtime adapter's event stream (Claude Code, Codex, OpenAI SDK):

```python
from skills_graph import SkillGraph
from skills_graph.connector import SkillGraphConnector
from agent_context_graph import AgentLink
from agent_context_graph.adapters.claude import ClaudeAdapter

link = AgentLink()
link.add_connector(SkillGraphConnector(SkillGraph()))

adapter = ClaudeAdapter(link, session_id="s-1")
hooks = adapter.get_runtime_hooks()
```

Requires the `agent-context-graph` extra: `pip install "skills-graph[agent-context-graph]"`.

The connector detects skill usage from `TOOL_START`/`TOOL_END`/`MESSAGE` events — direct tool names like `get_skill`, MCP-style names like `mcp__skills__get_skill`, and local reads of an Agent Skills definition (`.../skills/<name>/SKILL.md`), which covers runtimes where using a skill appears as a file read rather than a dedicated tool call. Search/list results that surface skill objects are also recorded through `USED_SKILL`.

## Installation

```bash
pip install skills-graph
# or, for the connector:  pip install "skills-graph[agent-context-graph]"
```

Needs a running Memgraph (default `bolt://localhost:7687`):

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

From a workspace checkout, `uv sync` installs it with its siblings.

## Testing

```bash
uv run pytest tests/ -v
```
