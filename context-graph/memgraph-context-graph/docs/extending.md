# Architecture and extensions

[Back to Memgraph Context Graph](../README.md)

## Multiple graph components

```python
from memgraph_context_graph import AgentLink
from memgraph_context_graph.sdk import ContextGraphMemory
from memgraph_context_graph.sdk.claude import ClaudeAgentMemory
from skills_graph import SkillGraph
from skills_graph.connector import SkillGraphConnector

skills = SkillGraph()

link = AgentLink()
link.add_connector(SkillGraphConnector(skills))
link.add_connector(MyGraphConnector(...))

# Capture goes through your link instead of the config file's connectors.
options = ClaudeAgentMemory(ContextGraphMemory(link=link)).options()
```

Connectors are owned by the graph packages because each graph package knows its own schema and persistence rules.

## Architecture

### Event Protocol

All runtime adapters emit runtime-agnostic `Event` dataclasses:

| Event | When |
|-------|------|
| `SessionStartEvent` | Agent session begins |
| `SessionEndEvent` | Agent session ends |
| `TurnEndEvent` | Agent finished one turn; the session stays open |
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
| `ClaudeAgentMemory` (`memgraph_context_graph.sdk.claude`) | Claude Agent SDK | `HookMatcher` callbacks, translated by the `claude-code` rules |
| `OpenAIAgentsMemory` (`memgraph_context_graph.sdk.openai`) | OpenAI Agents SDK | A `RunHooks` subclass |
| Command-hook adapters | Claude Code, Codex, Copilot CLI, Cursor, OpenCode, Antigravity CLI, Grok Build | Command hooks reading JSON from stdin (see [Command Hook Runtimes](runtime-setup.md#command-hook-runtimes) and [docs/command-hooks.md](command-hooks.md)) |

### Graph Connectors

| Connector | Graph Component | Events Handled |
|-----------|----------------|----------------|
| `SkillGraphConnector` | [skills-graph](../../skills-graph) | Tool/message events matching skill access/search operations |
| `ActionsGraphConnector` | [actions-graph](../../actions-graph) | Session, tool, message, subagent, and error events → action nodes |
| `SessionsGraphConnector` | [sessions-graph](../../sessions-graph) | `SessionStartEvent`/`SessionEndEvent`/`TurnEndEvent` → `(:User)`, `(:Session)`, `HAD_SESSION`; embeds each ended turn and marks it for reconciliation, which only a session end starts |

By default, the plugin enables all three through the config file's `[capture] connectors` setting. Each connector lives in the package that owns its graph schema; additional connectors should too.

### Adding a New Runtime Adapter

Implement `RuntimeAdapter`:

```python
from memgraph_context_graph import AgentLink, ToolStartEvent
from memgraph_context_graph.protocols import RuntimeAdapter


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

### Adding a New Command-Hook Runtime Adapter

The pattern above fits **in-process** runtimes — something that calls your Python code directly (an SDK callback, an embedded framework). **Command-hook** runtimes are different: the harness invokes an external command with a JSON payload on stdin (Claude Code, Codex), rather than calling into your process.

Command-hook runtimes provide four pieces:

1. **A payload translator.** Same idea as `RuntimeAdapter.get_runtime_hooks()`, but the input is the harness's raw JSON payload rather than a native callback. Map each of the harness's hook event names to the matching `Event` subclass and call `link.emit(...)`.
2. **A hook-config generator** (`build_hooks_config(command)`). Builds whatever config shape the harness expects for wiring hooks, pointing every hook at your CLI entry point.
3. **A response function** (`response_for_payload(payload)`). Returns the JSON the harness expects back on stdout, or `None`.
4. **A native doctor probe** (`probe_payload`). This must use the runtime's own event name and field shape; not every runtime calls session completion `Stop`.

Built-in command-hook adapters use the declarative `RuntimeSpec`,
`SpecAdapter`, and `SpecPlugin` in `adapters/_spec.py`; a runtime supplies
event rules, field aliases, hook responses, metadata keys, hook configuration,
and its probe, and `SpecPlugin` derives the Runtime Registration from them. The stdin-loading, connector-construction, and CLI parsing
remain shared in `hooks/runner.py`. External packages can either use that
module or implement `RuntimeAdapter` directly, then publish a Runtime
Registration through the entry-point group.

A registration includes the adapter and probe:

```python
# my_package/adapter.py
from dataclasses import dataclass, field

from memgraph_context_graph.protocols import RuntimeAdapter


class MyCommandHookAdapter(RuntimeAdapter):
    def __init__(self, link, session_id: str | None = None):
        self._link = link
        self._session_id = session_id

    def get_runtime_hooks(self):
        return build_hooks_config("my-runtime hook run my-runtime")

    def handle_payload(self, payload: dict) -> None:
        for event in self._events_from_payload(payload):
            self._link.emit(event)

    def _events_from_payload(self, payload: dict):
        # Map the harness's own hook_event_name / payload shape to Event subclasses.
        ...


def build_hooks_config(command: str, *, timeout: int = 30) -> dict:
    # Return whatever config format your harness expects, every hook pointing at `command`.
    ...


def response_for_payload(payload: dict) -> dict | None:
    # Return the JSON your harness expects back, or None.
    ...


@dataclass(frozen=True)
class MyRuntimePlugin:
    name: str = "my-runtime"
    adapter_class: type = MyCommandHookAdapter
    probe_payload: dict = field(default_factory=lambda: {"hook_event_name": "SessionEnd", "session_id": "doctor"})

    def response_for_payload(self, payload: dict) -> dict | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict:
        return build_hooks_config(command, timeout=timeout)

    # init(project_dir, connectors, **kwargs) is optional: omit it when there is no project-local file to write.


PLUGIN = MyRuntimePlugin()
```

Then register it in your own package's `pyproject.toml` — this is the entire integration, no fork or PR against this repo required:

```toml
[project.entry-points."memgraph_context_graph.runtimes"]
my-runtime = "my_package.adapter:PLUGIN"
```

Once installed, `doctor --runtime my-runtime`, `hook run my-runtime`, and `hook init my-runtime` (if `init` is implemented) work like the built-in registrations. See `runtime_plugin.py` for the full protocol and `pyproject.toml`'s own `[project.entry-points."memgraph_context_graph.runtimes"]` for registration examples.

### Adding a New Graph Component

Implement `GraphConnector` in the graph package:

```python
from memgraph_context_graph import EventType
from memgraph_context_graph.protocols import GraphConnector


class MyGraphConnector(GraphConnector):
    def supports(self, event):
        return event.event_type in {EventType.TOOL_START, EventType.TOOL_END}

    def on_event(self, event):
        # Write to your graph component.
        ...
```

Then register it, so hooks, schema setup and `doctor` can find it:

```python
from memgraph_context_graph.connectors import ConnectorSpec

CONNECTOR = ConnectorSpec(
    name="my-graph",
    graph=MyGraph,  # called with a Memgraph client, or its connection keyword arguments
    build=lambda graph, config: MyGraphConnector(graph),  # config: the config file's HookConfig
    check=None,  # optional: (config) -> (ok, detail) for doctor, beyond a reachable Memgraph
)
```

```toml
[project.entry-points."memgraph_context_graph.connectors"]
my-graph = "my_graph.connector:CONNECTOR"
```

Once installed, `hook run <runtime> --connector my-graph`, `[capture] connectors = ["my-graph"]` and
`doctor` work like the built-in connectors. `setup` only installs the connectors listed in
`memgraph_context_graph.connectors.INSTALLABLE`; install your own package beside the tool yourself.

Give the component's graph class a `forget_session(session_id) -> dict[str, int]`
that deletes what it hangs off that Session and returns counts, and add it to
`FORGET_ORDER` in `memgraph_context_graph/forget.py`, so `forget` reaches it.
