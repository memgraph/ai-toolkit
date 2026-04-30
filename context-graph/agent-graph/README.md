# Agent Graph

Connect any agent SDK to any context-graph component.

Agent Graph is a lightweight adapter layer that decouples **SDK-specific hooks** from **graph storage**. It routes a common event protocol from SDK adapters to graph connectors, so you can mix and match freely.

```
SDK Adapter  ──▶  Event Protocol  ──▶  Graph Connector(s)
(Claude, OpenAI)     (ToolStart,       (ActionsConnector,
                      ToolEnd, …)       SkillsConnector, …)
```

## Installation

```bash
pip install agent-graph
```

With SDK and graph extras:
```bash
pip install agent-graph[claude,actions]     # Claude + actions-graph
pip install agent-graph[openai,actions]     # OpenAI Agents + actions-graph
pip install agent-graph[claude,openai,actions,skills]  # Everything
```

## Quick Start

### Claude Agent SDK

```python
from actions_graph import ActionsGraph
from agent_graph import AgentLink
from agent_graph.adapters.claude import ClaudeAdapter
from agent_graph.connectors.actions import ActionsConnector

# 1. Set up graph storage
graph = ActionsGraph()
graph.setup()

# 2. Wire up the link
link = AgentLink()
link.add_connector(ActionsConnector(graph))

# 3. Create adapter
adapter = ClaudeAdapter(
    link,
    session_id="my-session",
    session_kwargs={"model": "claude-sonnet-4-20250514", "tags": ["review"]},
)

# 4. Use with Claude Agent SDK
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Review main.py",
    options=ClaudeAgentOptions(
        hooks=adapter.get_sdk_hooks(),
        allowed_tools=["Read", "Glob", "Grep"],
    ),
):
    print(message)
```

### OpenAI Agents SDK

```python
from actions_graph import ActionsGraph
from agent_graph import AgentLink
from agent_graph.adapters.openai import OpenAIAdapter
from agent_graph.connectors.actions import ActionsConnector
from agents import Agent, Runner

# 1. Set up graph storage
graph = ActionsGraph()
graph.setup()

# 2. Wire up the link
link = AgentLink()
link.add_connector(ActionsConnector(graph))

# 3. Create adapter
adapter = OpenAIAdapter(
    link,
    session_id="my-session",
    session_kwargs={"model": "gpt-4o"},
)

# 4. Run with hooks
agent = Agent(name="Reviewer", instructions="Review code for bugs.")
result = await Runner.run(
    agent,
    "Check this code for issues",
    hooks=adapter.get_sdk_hooks(),
)

# 5. Signal end (OpenAI SDK doesn't have a stop hook)
adapter.end_session()
```

### Multiple Graph Components

```python
from actions_graph import ActionsGraph
from skills_graph import SkillGraph
from agent_graph import AgentLink
from agent_graph.adapters.claude import ClaudeAdapter
from agent_graph.connectors.actions import ActionsConnector
from agent_graph.connectors.skills import SkillsConnector

# Set up graphs
actions = ActionsGraph()
skills = SkillGraph()

# Wire everything through one link
link = AgentLink()
link.add_connector(ActionsConnector(actions))
link.add_connector(SkillsConnector(skills))

# One adapter feeds both connectors
adapter = ClaudeAdapter(link, session_id="s-1")
hooks = adapter.get_sdk_hooks()
# → Tool calls go to ActionsConnector
# → Skill-related calls also go to SkillsConnector
```

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

### Graph Connectors

| Connector | Graph Component | Events Handled |
|-----------|----------------|----------------|
| `ActionsConnector` | actions-graph | Session, Tool, Agent, Message, Error, LLM |
| `SkillsConnector` | skills-graph | Tool events matching skill operations |

### Adding a New SDK

Implement `SDKAdapter`:

```python
from agent_graph.protocols import SDKAdapter

class MySDKAdapter(SDKAdapter):
    def __init__(self, link: AgentLink, session_id: str):
        self._link = link
        self._session_id = session_id

    def get_sdk_hooks(self):
        # Return whatever your SDK expects
        ...

    def _on_tool_call(self, name, args):
        self._link.emit(ToolStartEvent(
            session_id=self._session_id,
            tool_name=name,
            tool_input=args,
        ))
```

### Adding a New Graph Component

Implement `GraphConnector`:

```python
from agent_graph.protocols import GraphConnector

class MyGraphConnector(GraphConnector):
    def supports(self, event):
        return event.event_type in {EventType.TOOL_START, EventType.TOOL_END}

    def on_event(self, event):
        # Write to your graph component
        ...
```

## License

MIT
