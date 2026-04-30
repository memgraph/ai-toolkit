# skills-graph

A small library to persist, retrieve and evolve AI skills in [Memgraph](https://memgraph.com).

## Graph Model

```
(:Skill {name, description, content, created_at, updated_at})
(:Tag {name})
(:Skill)-[:HAS_TAG]->(:Tag)
(:Skill)-[:DEPENDS_ON]->(:Skill)
```

## Quick Start

```python
from skills_graph import SkillGraph, Skill

# Connect (uses MEMGRAPH_URL, MEMGRAPH_USER, MEMGRAPH_PASSWORD env vars by default)
sg = SkillGraph()

# Prepare the database schema (constraints + indexes)
sg.setup()

# Store a skill
sg.add_skill(Skill(
    name="memgraph-cypher",
    description="Writing Cypher queries for Memgraph",
    content="# Cypher for Memgraph\n\nUse MATCH, CREATE, MERGE ...",
    tags=["cypher", "memgraph"],
))

# Retrieve by name
skill = sg.get_skill("memgraph-cypher")

# Search
sg.search_by_tags(["cypher"])
sg.search_by_name("memgraph")

# Dependencies
sg.add_dependency("advanced-cypher", "memgraph-cypher")
deps = sg.get_dependencies("advanced-cypher")

# List all
all_skills = sg.list_skills()

# Update
sg.update_skill("memgraph-cypher", content="updated content", tags=["cypher"])

# Delete
sg.delete_skill("memgraph-cypher")
```

## Codex Hook Integration

When used through `agent-context-graph`'s Codex hook adapter, `SkillGraphConnector` records direct tool names like `get_skill` and Codex MCP-style names like `mcp__skills__get_skill`. Search/list tool results can be Python lists or JSON tool content containing skill objects with `name` fields.

## Installation

```bash
uv sync
```

## Testing

```bash
uv run pytest tests/ -v
```
