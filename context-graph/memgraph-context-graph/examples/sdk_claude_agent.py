"""Two Claude Agent SDK sessions sharing Context Graph memory: the second knows what the first was told.

Run with a configured Context Graph (``memgraph-context-graph setup``, or
``CONTEXT_GRAPH_CONFIG`` pointing at a config file) and a logged-in ``claude``::

    uv run --package memgraph-context-graph --extra claude python examples/sdk_claude_agent.py
"""

from __future__ import annotations

import asyncio
import sys

from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, TextBlock, ToolUseBlock

from memgraph_context_graph.sdk import ContextGraphMemory
from memgraph_context_graph.sdk.claude import ClaudeAgentMemory

MODEL = "claude-haiku-5-5"


async def session(memory: ClaudeAgentMemory, prompt: str) -> None:
    print(f"\n> {prompt}")
    async with ClaudeSDKClient(memory.options(model=MODEL, setting_sources=[], max_turns=6)) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)
                    elif isinstance(block, ToolUseBlock):
                        print(f"  [tool] {block.name} {block.input}")
    memory.end_session()


async def main(user_id: str) -> None:
    memory = ClaudeAgentMemory(ContextGraphMemory(user_id=user_id))
    await session(memory, "For this project I always want pnpm, never npm. Please remember that.")
    await session(memory, "Which package manager should I use to add a dependency here?")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "sdk-demo"))
