# Memory for SDK apps

[Back to Memgraph Context Graph](../README.md)

An app built on the Claude Agent SDK or the OpenAI Agents SDK can make Context Graph its memory in-process. `memgraph_context_graph.sdk` gives it what hooks and the MCP server give a coding agent:

- **Capture:** the SDK's own hooks record the session through the configured connectors. It is embedded and reconciled like a coding-agent session.
- **Context:** the memory index and where the user left off in this project.
- **Tools:** `recall` and `memory`.

The memory is the same one the user's coding agents use. The Memgraph connection, capture policy and connectors come from the config file, as they do for hooks. The app sets only `user_id` (whose memory it is) and `cwd` (which project).

## Install and configure

Install the extra for your SDK, plus the memory and activity components:

```bash
# Claude Agent SDK
pip install 'memgraph-context-graph[claude]' 'sessions-graph[memgraph-context-graph]' 'actions-graph[memgraph-context-graph]'
# OpenAI Agents SDK
pip install 'memgraph-context-graph[openai]' 'sessions-graph[memgraph-context-graph]' 'actions-graph[memgraph-context-graph]'
```

Use a running Memgraph with MAGE. Set its connection in the [Context Graph config file](configuration.md); SDK integrations use the same file as hooks. Model-provider credentials for the agent itself follow that SDK's configuration.

## Create the memory connection

```python
from memgraph_context_graph.sdk import ContextGraphMemory

memory = ContextGraphMemory(user_id="alice")
```

## Claude Agent SDK

The SDK's hook callbacks get Claude Code's own hook payloads, so the `claude-code` rules translate them, under `source_sdk = "claude-agent-sdk"`. Python callbacks get no `SessionStart` or `SessionEnd`, which changes two things:

- The memory context goes in with the session's first prompt.
- The app ends the session itself.

```python
from claude_agent_sdk import ClaudeSDKClient
from memgraph_context_graph.sdk.claude import ClaudeAgentMemory

claude = ClaudeAgentMemory(memory)
async with ClaudeSDKClient(claude.options(model="claude-sonnet-5-5")) as client:
    await client.query("Which package manager does this project use?")
    async for message in client.receive_response():
        ...
claude.end_session()
```

`options()` does four things:

- adds the hooks;
- adds an in-process `context-graph` MCP server and permits its tools;
- turns Claude Code's own auto memory off;
- keeps any hooks, servers and tools you pass in.

`options()` also turns the Context Graph Claude Code plugin off for the session, because the SDK loads your user settings by default and the plugin's hooks would capture every event a second time.

## OpenAI Agents SDK

Capture reads the run lifecycle (`RunHooks`):

- The user's message comes from the run's input.
- Each model response's text is recorded as an assistant message.
- Tools and handoffs come from their own hooks.

The memory context goes into dynamic instructions, and the tools are function tools.

```python
from agents import Agent, Runner
from memgraph_context_graph.sdk.openai import OpenAIAgentsMemory

wired = OpenAIAgentsMemory(memory, session_id="chat-42")  # one id per conversation
agent = Agent(name="Assistant", instructions=wired.instructions("Be concise."), tools=wired.tools())
result = await Runner.run(agent, "Which package manager does this project use?", hooks=wired.hooks())
wired.end_session()
```

The runnable [Claude example](../examples/sdk_claude_agent.py) and [OpenAI example](../examples/sdk_openai_agents.py) each run two sessions: the first is told something and the second has to know it.
