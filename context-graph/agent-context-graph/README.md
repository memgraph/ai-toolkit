# Agent Context Graph

Connect any agent runtime to any context-graph component.

Agent Context Graph is a lightweight adapter layer that decouples runtime-specific hooks from graph storage. It routes a common event protocol from runtime adapters to graph connectors, so you can mix and match SDKs and graph components.

```
Runtime Adapter  ->  Event Protocol  ->  Graph Connector(s)
(Claude,         (ToolStart,         (SkillGraphConnector,
 OpenAI)          ToolEnd, ...)       custom connectors, ...)
```

Runtime plugins are the distribution layer for host-specific hook wiring. They install hooks, skills, and setup helpers for a runtime, then call Agent Context Graph. They are not graph components and should not encode graph-specific meaning.

## Installation

For command-hook runtimes such as Codex and Claude Code, prefer a user-level tool install:

```bash
uv tool install agent-context-graph --with "skills-graph[agent-context-graph]"
```

Or use the plugin bootstrap scripts; they fall back to `uvx` if the tool is not installed yet.

For SDK usage inside an application:

```bash
pip install agent-context-graph
```

With runtime adapters:

```bash
pip install agent-context-graph[claude]
pip install agent-context-graph[openai]
```

Graph connectors live in the graph packages that persist the data. For the skills graph connector:

```bash
pip install skills-graph[agent-context-graph]
```

## Quick Start

### Claude Agent SDK

```python
from agent_context_graph import AgentLink
from agent_context_graph.adapters.claude import ClaudeAdapter
from claude_agent_sdk import ClaudeAgentOptions, query
from skills_graph import SkillGraph
from skills_graph.connector import SkillGraphConnector

# 1. Set up graph storage
skills = SkillGraph()
skills.setup()

# 2. Wire up the link
link = AgentLink()
link.add_connector(SkillGraphConnector(skills))

# 3. Create adapter
adapter = ClaudeAdapter(
    link,
    session_id="my-session",
    session_kwargs={"model": "claude-sonnet-4-20250514"},
)

# 4. Use with Claude Agent SDK
async for message in query(
    prompt="Review the available skills",
    options=ClaudeAgentOptions(hooks=adapter.get_runtime_hooks()),
):
    print(message)
```

### OpenAI Agents SDK

```python
from agent_context_graph import AgentLink
from agent_context_graph.adapters.openai import OpenAIAdapter
from agents import Agent, Runner, function_tool
from skills_graph import SkillGraph
from skills_graph.connector import SkillGraphConnector

# 1. Set up graph storage
skills = SkillGraph()
skills.setup()


# 2. Define a tool whose name matches the SkillGraphConnector defaults
@function_tool
def get_skill(name: str) -> str:
    skill = skills.get_skill(name)
    if skill is None:
        return f"Skill '{name}' not found."
    return f"{skill.name}: {skill.description}\n{skill.content}"


# 3. Wire up the link
link = AgentLink()
link.add_connector(SkillGraphConnector(skills))

# 4. Create adapter
adapter = OpenAIAdapter(
    link,
    session_id="my-session",
    session_kwargs={"model": "gpt-4o-mini"},
)

# 5. Run with hooks
agent = Agent(
    name="Skill Assistant",
    instructions="Use get_skill when the user asks for a named skill.",
    tools=[get_skill],
    model="gpt-4o-mini",
)
result = await Runner.run(
    agent,
    "Get the skill called 'cypher-basics'",
    hooks=adapter.get_runtime_hooks(),
)

# 6. Signal end (OpenAI SDK doesn't have a stop hook)
adapter.end_session()
```

### Command Hook Runtimes

Some agent applications run hooks as external commands instead of in-process SDK callbacks. Runtime adapters should keep the product-specific JSON mapping at the edge, emit the shared `Event` protocol, and leave graph persistence in connectors such as `SkillGraphConnector`.

The installed command is runtime-dispatched:

```bash
agent-context-graph hook <command> [options]
```

Implemented:

| Runtime | Adapter | Hook Shape |
|---------|---------|------------|
| OpenAI Codex | `CodexHooksAdapter` | Command receives one JSON object on `stdin` |
| Claude Code | `ClaudeCodeHooksAdapter` | Command receives one JSON object on `stdin` |

### First-Time Plugin Setup

For Codex and Claude Code plugins, the recommended first-run path is the bootstrap command. It installs the runtime package, checks Memgraph, installs the graph connector extra, and runs `doctor`.

Prerequisites:

- `uv` on `PATH`.
- Memgraph running and reachable over Bolt. Defaults are `bolt://localhost:7687`, empty user/password, and database `memgraph`.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `MEMGRAPH_URL` | `bolt://localhost:7687` | Bolt URL — set to a remote host for non-local Memgraph |
| `MEMGRAPH_USER` | `""` | Bolt username (service account) |
| `MEMGRAPH_PASSWORD` | `""` | Bolt password or OAuth token |
| `MEMGRAPH_DATABASE` | `memgraph` | Target database name |
| `AGENT_CONTEXT_GRAPH_USER_ID` | _(none)_ | Human identity stored on Memory nodes — required for sessions-graph to associate memories with a user |

If Memgraph is not running locally, start it first:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

`uv` manages Python for the tool. If uv-managed Python downloads are blocked in your environment, install Python 3.10+ and rerun bootstrap.

For Codex:

```bash
agent-context-graph bootstrap --runtime codex --connector skills-graph
```

For Claude Code:

```bash
agent-context-graph bootstrap --runtime claude-code --connector skills-graph
```

Expected successful doctor output looks like:

```text
OK agent-context-graph executable: ...
OK agent-context-graph: ...
OK connector:skills-graph: installed=...; memgraph=reachable
OK runtime:codex: strict hook smoke passed
```

Use the matching runtime value when checking Claude Code:

```text
OK runtime:claude-code: strict hook smoke passed
```

The plugin wrapper scripts call the same bootstrap command. If `agent-context-graph` is not installed yet, they fall back to `uvx`:

```bash
./scripts/bootstrap.sh
```

### OpenAI Codex Plugin

Codex hook configuration can be installed as a user-level Codex plugin.

The runtime-plugin flow is:

```text
Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

The plugin installs Codex hook wiring. The Codex runtime adapter normalizes the hook payload. Graph connectors such as `SkillGraphConnector` decide what those events mean in their graph.

Plugin source:

```text
context-graph/plugins/agent-context-graph-codex
```

Register the public Git-backed marketplace:

```bash
codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins
```

Then install or enable `context-graph` from the Codex plugin UI.

Check the installed hook environment with:

```bash
agent-context-graph doctor --runtime codex --connector skills-graph --connector actions-graph --connector sessions-graph
```

Keep graph credentials in the process environment, not in plugin hook files. Runtime hooks use `memgraph-toolbox` defaults unless the Codex process has `MEMGRAPH_*` variables set.

### Claude Code Plugin

Claude Code hook configuration can be installed as a Claude Code plugin.

The runtime-plugin flow is:

```text
Claude Code Plugin -> Claude Code Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

For a public Git-backed marketplace install, add the marketplace inside Claude Code:

```text
/plugin marketplace add memgraph/ai-toolkit
```

Then install:

```text
/plugin install context-graph@context-graph-plugins
```

Check the installed hook environment with:

```bash
agent-context-graph doctor --runtime claude-code --connector skills-graph --connector actions-graph --connector sessions-graph
```

### Source Development

For source development and per-project experiments, you can generate local Codex hook files:

```bash
agent-context-graph setup codex --project-dir "$PWD" --setup-schema
```

This writes local, ignored files:

```text
.codex/config.toml
.codex/hooks.json
```

See [Command Hook Reference](docs/command-hooks.md) for manual setup, non-default Memgraph values, smoke tests, and generated hook JSON details.

### Multiple Graph Components

```python
from agent_context_graph import AgentLink
from agent_context_graph.adapters.claude import ClaudeAdapter
from skills_graph import SkillGraph
from skills_graph.connector import SkillGraphConnector

skills = SkillGraph()

link = AgentLink()
link.add_connector(SkillGraphConnector(skills))
link.add_connector(MyGraphConnector(...))

adapter = ClaudeAdapter(link, session_id="s-1")
hooks = adapter.get_runtime_hooks()
```

Connectors are owned by the graph packages because each graph package knows its own schema and persistence rules.

## Architecture

### Event Protocol

All runtime adapters emit runtime-agnostic `Event` dataclasses:

| Event | When |
|-------|------|
| `SessionStartEvent` | Agent session begins |
| `SessionEndEvent` | Agent session ends |
| `ToolStartEvent` | Before tool/function call |
| `ToolEndEvent` | After tool/function returns |
| `AgentStartEvent` | Agent/subagent begins |
| `AgentEndEvent` | Agent/subagent finishes |
| `LLMStartEvent` | Before LLM call |
| `LLMEndEvent` | After LLM response |
| `HandoffEvent` | Agent hands off to another |
| `MessageEvent` | User/assistant/system message |
| `ErrorOccurredEvent` | Error during execution |

### Runtime Adapters

| Adapter | Runtime Source | Hook Mechanism |
|---------|----------------|----------------|
| `ClaudeAdapter` | Claude Agent SDK | Dict of `HookMatcher` callbacks |
| `OpenAIAdapter` | OpenAI Agents SDK | `RunHooksBase` subclass |
| `CodexHooksAdapter` | OpenAI Codex | Command hooks reading JSON from stdin |

### Graph Connectors

| Connector | Graph Component | Events Handled |
|-----------|----------------|----------------|
| `SkillGraphConnector` | skills-graph | Tool events matching skill access/search operations |

Additional graph connectors should live in the packages that own those graph schemas.

### Adding a New Runtime Adapter

Implement `RuntimeAdapter`:

```python
from agent_context_graph import AgentLink, ToolStartEvent
from agent_context_graph.protocols import RuntimeAdapter


class MyRuntimeAdapter(RuntimeAdapter):
    def __init__(self, link: AgentLink, session_id: str):
        self._link = link
        self._session_id = session_id

    def get_runtime_hooks(self):
        # Return whatever your runtime expects.
        ...

    def _on_tool_call(self, name, args):
        self._link.emit(
            ToolStartEvent(
                session_id=self._session_id,
                tool_name=name,
                tool_input=args,
            )
        )
```

### Adding a New Graph Component

Implement `GraphConnector` in the graph package:

```python
from agent_context_graph import EventType
from agent_context_graph.protocols import GraphConnector


class MyGraphConnector(GraphConnector):
    def supports(self, event):
        return event.event_type in {EventType.TOOL_START, EventType.TOOL_END}

    def on_event(self, event):
        # Write to your graph component.
        ...
```

## License

MIT
