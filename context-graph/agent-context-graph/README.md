# Agent Context Graph

Connect any agent SDK to any context-graph component.

Agent Context Graph is a lightweight adapter layer that decouples SDK-specific hooks from graph storage. It routes a common event protocol from SDK adapters to graph connectors, so you can mix and match SDKs and graph components.

```
SDK Adapter  ->  Event Protocol  ->  Graph Connector(s)
(Claude,         (ToolStart,         (SkillGraphConnector,
 OpenAI)          ToolEnd, ...)       custom connectors, ...)
```

## Installation

```bash
pip install agent-context-graph
```

With SDK adapters:

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
    session_kwargs={"model": "claude-sonnet-4-20250514", "tags": ["review"]},
)

# 4. Use with Claude Agent SDK
async for message in query(
    prompt="Review the available skills",
    options=ClaudeAgentOptions(hooks=adapter.get_sdk_hooks()),
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
    hooks=adapter.get_sdk_hooks(),
)

# 6. Signal end (OpenAI SDK doesn't have a stop hook)
adapter.end_session()
```

### Command Hook Runtimes

Some agent applications run hooks as external commands instead of in-process SDK callbacks. Runtime adapters should keep the product-specific JSON mapping at the edge, emit the shared `Event` protocol, and leave graph persistence in connectors such as `SkillGraphConnector`.

The installed command is runtime-dispatched:

```bash
agent-context-graph-hook <runtime> [runtime options]
```

Implemented:

| Runtime | Adapter | Hook Shape |
|---------|---------|------------|
| OpenAI Codex | `CodexHooksAdapter` | Command receives one JSON object on `stdin` |

Planned:

| Runtime | Adapter | Notes |
|---------|---------|-------|
| Claude Code | `ClaudeCodeHooksAdapter` | TODO: command-hook adapter for Claude Code JSON input/output and `.claude/settings.local.json` setup |

### OpenAI Codex Hooks

Codex hook configuration is local environment wiring, so this repository ignores `.codex/`. Each developer should create their own local `.codex` files or use a user-level Codex config.

1. Make `skills-graph` able to reach Memgraph, then initialize and seed your skill graph once:

```bash
export MEMGRAPH_URL="bolt://localhost:7687"
export MEMGRAPH_USER=""
export MEMGRAPH_PASSWORD=""
```

```python
from skills_graph import SkillGraph

skills = SkillGraph()
skills.setup()
```

2. Install the hook command and the graph connector in the same Python environment:

```bash
python -m venv ~/.venvs/agent-context-graph-hooks
~/.venvs/agent-context-graph-hooks/bin/python -m pip install \
  "agent-context-graph" \
  "skills-graph[agent-context-graph]"
```

For source development in this workspace, use this command instead of the venv binary:

```bash
cd /path/to/ai-toolkit
uv run --package skills-graph --extra agent-context-graph \
  python -m agent_context_graph.hooks.cli codex --connector skills-graph
```

3. Enable Codex hooks in your local workspace:

```toml
# .codex/config.toml
[features]
codex_hooks = true
```

4. Point Codex at the hook command. Replace `COMMAND` with the absolute command from step 2, for example `/Users/me/.venvs/agent-context-graph-hooks/bin/agent-context-graph-hook codex --connector skills-graph`.

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear",
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "PermissionRequest": [
      {
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "Stop": [
      {
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ]
  }
}
```

The adapter records Codex `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PermissionRequest`, and `Stop` payloads. MCP tool names such as `mcp__skills__get_skill` are normalized by `skills-graph` to the underlying `get_skill` operation.

Smoke test the command:

```bash
printf '{"hook_event_name":"Stop","session_id":"test"}' | COMMAND
```

The expected output is:

```json
{"continue": true}
```

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
hooks = adapter.get_sdk_hooks()
```

Connectors are owned by the graph packages because each graph package knows its own schema and persistence rules.

## Architecture

### Event Protocol

All SDK adapters emit SDK-agnostic `Event` dataclasses:

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

### SDK Adapters

| Adapter | SDK | Hook Mechanism |
|---------|-----|----------------|
| `ClaudeAdapter` | Claude Agent SDK | Dict of `HookMatcher` callbacks |
| `OpenAIAdapter` | OpenAI Agents SDK | `RunHooksBase` subclass |
| `CodexHooksAdapter` | OpenAI Codex | Command hooks reading JSON from stdin |

### Graph Connectors

| Connector | Graph Component | Events Handled |
|-----------|----------------|----------------|
| `SkillGraphConnector` | skills-graph | Tool events matching skill access/search operations |

Additional graph connectors should live in the packages that own those graph schemas.

### Adding a New SDK

Implement `SDKAdapter`:

```python
from agent_context_graph import AgentLink, ToolStartEvent
from agent_context_graph.protocols import SDKAdapter

class MySDKAdapter(SDKAdapter):
    def __init__(self, link: AgentLink, session_id: str):
        self._link = link
        self._session_id = session_id

    def get_sdk_hooks(self):
        # Return whatever your SDK expects.
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
