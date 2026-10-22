"""Two OpenAI Agents SDK conversations sharing Context Graph memory: the second knows what the first was told.

Run with a configured Context Graph (``memgraph-context-graph setup``, or
``CONTEXT_GRAPH_CONFIG`` pointing at a config file) and ``OPENAI_API_KEY`` set::

    uv run --package memgraph-context-graph --extra openai python examples/sdk_openai_agents.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from agents import Agent, Runner
from memgraph_context_graph.sdk import ContextGraphMemory
from memgraph_context_graph.sdk.openai import OpenAIAgentsMemory

MODEL = "gpt-5-mini"


async def conversation(memory: ContextGraphMemory, prompt: str) -> None:
    print(f"\n> {prompt}")
    wired = OpenAIAgentsMemory(memory, session_id=f"openai-demo-{uuid.uuid4()}")
    agent = Agent(
        name="Assistant",
        model=MODEL,
        instructions=wired.instructions("You are a concise coding assistant."),
        tools=wired.tools(),
    )
    result = await Runner.run(agent, prompt, hooks=wired.hooks())
    print(result.final_output)
    wired.end_session()


async def main(user_id: str) -> None:
    memory = ContextGraphMemory(user_id=user_id)
    await conversation(memory, "For this project I always want pnpm, never npm. Please remember that.")
    await conversation(memory, "Which package manager should I use to add a dependency here?")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "sdk-demo"))
