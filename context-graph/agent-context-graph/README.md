# Agent Context Graph

Connect any agent runtime to any context-graph component.

Agent Context Graph is a lightweight adapter layer that decouples runtime-specific hooks from graph storage. It routes a common event protocol from runtime adapters to graph connectors, so you can mix and match SDKs and graph components.

```
Runtime Adapter  ->  Event Protocol  ->  Graph Connector(s)
(Claude,         (ToolStart,         (SkillGraphConnector,
 OpenAI)          ToolEnd, ...)       custom connectors, ...)
```

Runtime plugins are the distribution layer for host-specific hook wiring. They install hooks, skills, and setup helpers for a runtime, then call Agent Context Graph. They are not graph components and should not encode graph-specific meaning.

> **Just want to capture agent sessions?** Start with the [Context Graph guide](../README.md) — it walks through wiring the connectors end to end. This README covers the adapter layer itself and the in-process SDK path.

## Installation

For command-hook runtimes, prefer a user-level tool install:

```bash
uv tool install "agent-context-graph[mcp]" --with "skills-graph[agent-context-graph]"
```

The `mcp` extra is what `agent-context-graph mcp` serves [recall](#recall-memory-for-the-harnesss-model) with.

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
| GitHub Copilot CLI | `CopilotCLIHooksAdapter` | Command receives one JSON object on `stdin` |
| Cursor | `CursorHooksAdapter` | Command receives one JSON object on `stdin` |
| OpenCode | `OpenCodeHooksAdapter` | V2 plugin normalizes callbacks into JSON for the command |
| Antigravity CLI | `AntigravityCLIHooksAdapter` | Command receives one JSON object on `stdin` |
| Grok Build | `GrokHooksAdapter` | Command receives one JSON object on `stdin` |

Codex and Claude Code have distributable Runtime Plugins. Copilot CLI, Cursor,
OpenCode, Antigravity CLI, and Grok Build use project-local wiring generated by `hook init`:

```bash
agent-context-graph hook init copilot-cli
agent-context-graph hook init cursor
agent-context-graph hook init opencode
agent-context-graph hook init antigravity-cli
agent-context-graph hook init grok
```

See the [command-hook reference](docs/command-hooks.md) for generated paths,
payload differences, and current upstream hook documentation.

### First-Time Plugin Setup

For Codex and Claude Code plugins, the recommended first-run path is the bootstrap command. It installs the runtime package (with the connector extras), checks Memgraph, and runs `doctor`.

Prerequisites:

- `uv` on `PATH`. (`uv` manages Python for the tool; if uv-managed Python downloads are blocked, install Python 3.10+ and rerun bootstrap.)
- Memgraph running and reachable over Bolt. Defaults are `bolt://localhost:7687`, empty user/password, database `memgraph`. If it isn't running locally:

  ```bash
  docker run --rm -p 7687:7687 memgraph/memgraph-mage
  ```

**1. Bootstrap all three connectors** (this is what the installed plugin wires into its hooks):

```bash
# Codex
agent-context-graph bootstrap --runtime codex \
  --connector skills-graph --connector actions-graph --connector sessions-graph

# Claude Code
agent-context-graph bootstrap --runtime claude-code \
  --connector skills-graph --connector actions-graph --connector sessions-graph
```

The plugin wrapper script runs the same command (and falls back to `uvx` if the tool isn't installed yet):

```bash
./scripts/bootstrap.sh
```

**2. Configure identity and connection.** Bootstrap writes `~/.config/context-graph/config.toml`; hooks read their configuration from that file at runtime (see [Configuration](#configuration) — env vars are **not** read at hook time). Set your identity, which sessions-graph requires:

```bash
agent-context-graph config set identity.user_id "your-name"
```

The Memgraph connection defaults to `bolt://localhost:7687`. For a remote or HA instance:

```bash
agent-context-graph config set memgraph.url "neo4j://<coordinator-host>:7687"
agent-context-graph config set memgraph.user "<user>"
agent-context-graph config set memgraph.password        # prompts; stored 0600
agent-context-graph config set memgraph.database "memgraph"
```

**3. Verify:**

```bash
agent-context-graph config show
agent-context-graph doctor --runtime claude-code \
  --connector skills-graph --connector actions-graph --connector sessions-graph
```

Expected successful doctor output (use `--runtime codex` for Codex):

```text
OK agent-context-graph executable: ...
OK agent-context-graph: ...
OK config: identity.user_id set
OK memgraph: reachable
OK connector:skills-graph: installed=...; memgraph=reachable
OK connector:actions-graph: installed=...; memgraph=reachable
OK connector:sessions-graph: installed=...; memgraph=reachable
OK embeddings: BAAI/bge-small-en-v1.5 (384 dimensions) inside Memgraph
OK mcp: serves recall
OK runtime:claude-code: strict hook smoke passed
```

> **Reconciliation is a separate step.** The connectors capture session activity, but turning a session's text into extracted entities (a `:Person`/`:Organization` graph) is done out-of-band — see [sessions-graph § reconciliation](../sessions-graph/README.md#session-reconciliation). By default a finished session is marked `reconciliation_status = 'pending'` and `sessions-graph reconcile --pending` extracts it.

### Configuration

Bootstrap and the `config` command write `~/.config/context-graph/config.toml` (mode `0600`). **Hook subprocesses resolve their configuration from CLI flags and this file only — never from environment variables** (they don't inherit your shell), per [ADR 0002](docs/adr/0002-config-file-only-hook-resolution.md).

```toml
[identity]
user_id = "your-name"

[memgraph]
url = "bolt://localhost:7687"
user = ""
password = ""
database = "memgraph"

[llm]
openai_api_key = ""
anthropic_api_key = ""

[reconcile]
auto_reconcile = true

[recall]
embedding_model = "BAAI/bge-small-en-v1.5"
turns_k = "8"
```

`[llm]` and `[reconcile]` are only relevant if you enable sessions-graph's
auto-trigger reconciliation (see
[sessions-graph § reconciliation](../sessions-graph/README.md#session-reconciliation)).
`[reconcile]` is omitted entirely from a freshly-bootstrapped file — absent
means "never configured," distinct from an explicit `auto_reconcile = false`.
`[recall]` is optional too: without it, recall runs with the widths the
benchmark measured; see [Recall](#recall-memory-for-the-harnesss-model).
`[ontology]` is optional as well: `path` names a schema file the user's
sessions are extracted under and `derive` is `extend` (the default) or
`off`; see sessions-graph's [Ontology versions](../sessions-graph/README.md#ontology-versions).

Manage it with:

```bash
agent-context-graph config show
agent-context-graph config get memgraph.url
agent-context-graph config set <key> <value>
# keys: identity.user_id, memgraph.{url,user,password,database},
#       llm.{openai_api_key,anthropic_api_key}, reconcile.auto_reconcile,
#       recall.embedding_model, ontology.{path,derive}, memory.backend
```

Environment variables (`MEMGRAPH_URL`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD`, `MEMGRAPH_DATABASE`, `AGENT_CONTEXT_GRAPH_USER_ID`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are consulted **only at bootstrap time** — if set, `bootstrap` persists them into the config file. Exporting them later has no effect on running hooks; use `config set` instead.

`reconcile.auto_reconcile` is the one exception: `bootstrap` never captures
`SESSIONS_GRAPH_AUTO_RECONCILE` from the environment, and re-running
`bootstrap` preserves whatever it's currently set to rather than resetting it.
Unlike the keys above, nobody has that env var exported for an unrelated
reason — it's only ever set via `agent-context-graph config set
reconcile.auto_reconcile true`.

### Recall: memory for the harness's model

The hooks write sessions into Memgraph; recall reads them back. With
sessions-graph installed, `agent-context-graph mcp` serves a `recall(question)`
tool over stdio MCP, and the Codex and Claude Code plugins bundle that server,
so installing a plugin is all it takes. The tool returns context, not an answer:
the dated messages and facts from the user's own sessions that match the
question, behind a header with the rules for reading them. The harness's model
answers from those rows.

At session start the hook adds one line to the model's context saying the tool
exists. It never pushes retrieved content; the model calls `recall` when a
question needs memory.

A connector may also add a line just before a tool runs
(`GraphConnector.context_before_tool`) — resources-graph's Nudge that what the
tool is about to fetch is already in memory. It is advice only: the call always
proceeds unchanged, and a connector that fails adds nothing. Only Claude Code
and Codex show it (`PreToolUse` `additionalContext` with no permission
decision); the other harnesses' pre-tool hooks can only allow, deny or rewrite
a call, so they get the session-start line alone.

The same search from a shell:

```bash
agent-context-graph recall "what did we decide about the deploy window?"
agent-context-graph recall --json "..."    # the turns and facts as JSON
```

Whose memory is searched comes from `identity.user_id` in the config file, never
from the tool call, so a model can only read its own user's sessions.
`[recall]` overrides the search's lanes and widths, all optional:

| Key | Default | What it sets |
|---|---|---|
| `embedding_model` | `BAAI/bge-small-en-v1.5` | The model Memgraph embeds with; set with `config set recall.embedding_model` |
| `lanes` | all five | Comma-separated subset of `turns,text,entities,facts,user_facts` |
| `turns_k`, `text_k` | `8`, `8` | Messages found by vector and by text search |
| `entities_k`, `edges_per_entity` | `15`, `6` | Entities matched, and facts read from each |
| `facts_k`, `fact_turns_k` | `15`, `8` | Facts matched, and the messages they were read from |
| `user_fact_types`, `user_facts_k` | `2`, `30` | Relation types gathered whole, for counting questions |
| `turn_chars` | `1500` | Characters shown per message |

Recall needs Memgraph with MAGE (`memgraph/memgraph-mage`) for its vector lanes;
without it, it answers from text search and says so. `doctor` checks both
(`embeddings` and `mcp`).

**What the model sees.** A tool result carries the rendered text only. Claude
Code and Codex both show the model a result's `structuredContent` JSON
*instead of* its text when both are present, and the text is the form recall
was benchmarked in; the JSON is what `--json` prints.

### Memory: Context Graph as the harness's memory

Claude Code and Codex each keep their own memory. Context Graph can replace it
for a user who opts in:

```bash
agent-context-graph config set memory.backend context-graph
```

With that set, `agent-context-graph mcp` also serves a `memory` tool with the
Claude API memory tool's commands (`view`, `create`, `str_replace`, `insert`,
`delete`, `rename` on `/memories`). The model keeps its curated long-term memory there:
- `/memories/` for what applies everywhere;
- `/memories/projects/<project>/` for one codebase.

Every file is a `(:Memory)` node in the user's graph, shared across harnesses
and sessions. Whose memory it is comes from `identity.user_id`, never from
the call. Without the setting the tool isn't listed at all, so the harness's
own memory stays in charge. Harnesses list tools once per session, so the
change shows from the next session.

At session start the hook loads the memory, the way the harness's built-in
memory would have. It adds:
- an index of the root and the current project's files, one line each with
  the file's `description:` frontmatter or first line, up to 200 lines;
- the full text of any file with `pin: true` in its frontmatter;
- the current project's folder.

The index is generated from the graph, so the model never maintains an index
file and it can't go stale. The project folder is keyed by the checkout's git
`origin` remote (e.g. `github.com_memgraph_ai-toolkit`), so every clone of a
repository shares one folder. Without a remote, the repository's directory
name is used. See
[ADR 0006](docs/adr/0006-memory-tool-writes-explicit-memory.md) for why this is
the one write the model makes over MCP.

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

Register the public Git-backed marketplace and install the plugin:

```bash
codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins
codex plugin add context-graph@context-graph-plugins
```

Both are non-interactive; `codex plugin add` installs and enables the plugin in one step (`codex plugin list --json` confirms `"enabled": true`).

Check the installed hook environment with:

```bash
agent-context-graph doctor --runtime codex --connector skills-graph --connector actions-graph --connector sessions-graph
```

Graph credentials live in `~/.config/context-graph/config.toml` (written by `bootstrap`/`config set`), not in plugin hook files or the process environment — hooks read that file at runtime. See [Configuration](#configuration).

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
| `ClaudeAdapter` | Claude Agent SDK | Dict of `HookMatcher` callbacks |
| `OpenAIAdapter` | OpenAI Agents SDK | `RunHooksBase` subclass |
| Command-hook adapters | Claude Code, Codex, Copilot CLI, Cursor, OpenCode, Antigravity CLI, Grok Build | Command hooks reading JSON from stdin (see [Command Hook Runtimes](#command-hook-runtimes) and [docs/command-hooks.md](docs/command-hooks.md)) |

### Graph Connectors

| Connector | Graph Component | Events Handled |
|-----------|----------------|----------------|
| `SkillGraphConnector` | [skills-graph](../skills-graph/) | Tool/message events matching skill access/search operations |
| `ActionsGraphConnector` | [actions-graph](../actions-graph/) | Session, tool, message, subagent, and error events → action nodes |
| `SessionsGraphConnector` | [sessions-graph](../sessions-graph/) | `SessionStartEvent`/`SessionEndEvent`/`TurnEndEvent` → `(:User)`, `(:Session)`, `HAD_SESSION`; embeds each ended turn and marks it for reconciliation, which only a session end starts |

The installed plugin wires **all three** (`--connector skills-graph --connector actions-graph --connector sessions-graph`). Each connector lives in the package that owns its graph schema; additional connectors should too.

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

from agent_context_graph.protocols import RuntimeAdapter


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

    # init(project_dir, connectors, **kwargs) is optional -- omit it if your
    # runtime has no project-local hook-config file to generate (matching
    # ClaudeCodeHooksAdapter's own plugin, which doesn't define one yet).


PLUGIN = MyRuntimePlugin()
```

Then register it in your own package's `pyproject.toml` — this is the entire integration, no fork or PR against this repo required:

```toml
[project.entry-points."agent_context_graph.runtimes"]
my-runtime = "my_package.adapter:PLUGIN"
```

Once installed, `agent-context-graph bootstrap --runtime my-runtime`, `doctor --runtime my-runtime`, `hook run my-runtime`, and `hook init my-runtime` (if `init` is implemented) work like the built-in registrations. See `runtime_plugin.py` for the full protocol and `pyproject.toml`'s own `[project.entry-points."agent_context_graph.runtimes"]` for registration examples.

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
