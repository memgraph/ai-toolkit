# Skills Graph

Store reusable Agent Skills in Memgraph and track which sessions and subagents use them.

Part of [Context Graph](../README.md). For automatic capture in a coding agent, use the [Context Graph installer](../README.md#getting-started). The Python API below lets you manage skill definitions directly.

## Installation

```bash
pip install skills-graph
# With the runtime connector:
pip install 'skills-graph[memgraph-context-graph]'
```

Requires a running Memgraph, defaulting to `bolt://localhost:7687`. Direct Python use reads `MEMGRAPH_URL`, `MEMGRAPH_USER`, and `MEMGRAPH_PASSWORD`; hook integrations read the Context Graph config file.

## Quick start

```python
from skills_graph import SkillGraph, Skill

sg = SkillGraph()
sg.setup()

sg.add_skill(
    Skill(
        name="memgraph-cypher",
        description="Writing Cypher queries for Memgraph",
        content="# Cypher for Memgraph\n\nUse MATCH, CREATE, MERGE ...",
    )
)

skill = sg.get_skill("memgraph-cypher")

sg.search_by_name("memgraph")

all_skills = sg.list_skills()

sg.update_skill("memgraph-cypher", content="updated content")

sg.delete_skill("memgraph-cypher")
```

Skill names use 1–64 lowercase letters, digits, or hyphens, with no leading, trailing, or consecutive hyphens. Invalid names raise `SkillValidationError`.

## Track usage automatically

Wire `SkillGraphConnector` into an `AgentLink` and skill usage is recorded automatically from any runtime adapter's event stream (Claude Code, Codex, OpenAI SDK):

```python
from skills_graph import SkillGraph
from skills_graph.connector import SkillGraphConnector
from memgraph_context_graph import AgentLink

link = AgentLink()
skills = SkillGraph()
skills.setup()
link.add_connector(SkillGraphConnector(skills))
```

For SDK application wiring, see the [SDK quick starts](../memgraph-context-graph/docs/sdk-memory.md).

The connector detects skill usage from `TOOL_START`/`TOOL_END`/`MESSAGE` events — direct tool names like `get_skill`, MCP-style names like `mcp__skills__get_skill`, Claude Code's `Skill` tool, and reads of an Agent Skills definition (`.../skills/<name>/SKILL.md`) by a file-read tool or a read program such as `cat` or `sed`, which covers runtimes where using a skill appears as a file read rather than a dedicated tool call. A `SKILL.md` that is only mentioned, staged, searched or written is not usage, and neither is other Markdown with skill-like frontmatter. Search/list results that surface skill objects are also recorded through `USED_SKILL`.

## Graph model

```
(:Skill {name, description, content, license, compatibility,
         allowed_tools, metadata, created_at, updated_at, source_path?})
(:Skill)-[:DEPENDS_ON]->(:Skill)
(:Session|:Agent)-[:USED_SKILL {first_access, last_access, access_count, actions}]->(:Skill)
```

`USED_SKILL` starts at the subagent's `(:Agent)` when actions-graph has created it, otherwise at a `MERGE (:Session {session_id})` — skills-graph MERGEs the shared Session node it does not own (see the [Context Graph map](../CONTEXT-MAP.md)). `source_path` is set only when a skill is created by observing a local `SKILL.md` read.

Dependencies between stored skills use `sg.add_dependency(skill_name, depends_on)` and `sg.get_dependencies(skill_name)`.

## Development

From the repository root:

```bash
./scripts/dev-memgraph.sh test skills-graph
```
