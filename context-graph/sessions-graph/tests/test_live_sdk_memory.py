"""Live: a real model in each SDK uses Context Graph as its memory, and the graph shows it.

Billed, so it runs only with ``LIVE_SDK_E2E=1`` and the SDK's key (the
``test-live-sdk`` workflow sets both). In two separate sessions the model is
told a fact and saves it, then asked for it. The assertions are on what lands
in Memgraph: the user and both sessions, each session's turns, the memory file
and the session that wrote it, and the turns found by ``recall`` once the
detached embed job has run.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("memgraph_context_graph")
pytest.importorskip("actions_graph")

from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.sdk import ContextGraphMemory

pytestmark = pytest.mark.skipif(os.environ.get("LIVE_SDK_E2E") != "1", reason="live, billed: set LIVE_SDK_E2E=1")

TELL = "Remember this for future sessions: my favourite database is Memgraph. Save it to your memory now."
ASK = "What is my favourite database? Check your memory, then answer in one word."
#: Long enough for the detached ``sessions-graph embed`` a turn end starts.
EMBED_DEADLINE_SECONDS = 120


@pytest.fixture
def memory_for(memgraph, tmp_path, monkeypatch):
    """A ContextGraphMemory per user on the test Memgraph; the background jobs run for real."""
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    _identity.write_config(memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    yield lambda user_id: ContextGraphMemory(
        user_id=user_id, cwd=str(tmp_path), connectors=["actions-graph", "sessions-graph"]
    )
    _identity._reset_cache()


def _assert_remembered(memgraph, memory, user_id, sessions, answer):
    told, asked = sessions
    assert told != asked, "each conversation is its own session"
    assert "memgraph" in answer.lower(), f"the second session didn't know: {answer!r}"

    rows = memgraph.query(
        "MATCH (:User {user_id: $u})-[:HAD_SESSION]->(s:Session) RETURN s.session_id AS id", {"u": user_id}
    )
    assert {row["id"] for row in rows} >= {told, asked}

    for session_id, prompt in ((told, TELL), (asked, ASK)):
        turns = memgraph.query(
            "MATCH (:Session {session_id: $s})-[:HAS_ACTION]->(m:Message) "
            "RETURN labels(m) AS labels, m.text AS text ORDER BY m.timestamp",
            {"s": session_id},
        )
        assert [t["text"] for t in turns if "UserMessage" in t["labels"]][:1] == [prompt]
        assert any("AssistantMessage" in t["labels"] for t in turns), f"no reply captured in {session_id}"

    written = memgraph.query(
        "MATCH (:User {user_id: $u})-[:HAS_MEMORY]->(m:Memory) "
        "OPTIONAL MATCH (s:Session)-[:PRODUCED_MEMORY]->(m) RETURN m.content AS content, s.session_id AS by",
        {"u": user_id},
    )
    assert any("memgraph" in (row["content"] or "").lower() and row["by"] == told for row in written), written

    deadline = time.monotonic() + EMBED_DEADLINE_SECONDS
    while True:
        text, is_error = memory.call_tool("recall", {"question": "What is my favourite database?"})
        if not is_error and f"session {told}" in text:
            break
        assert time.monotonic() < deadline, f"recall never found the first session's turns:\n{text}"
        time.sleep(5)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_claude_agent_sdk_remembers_across_sessions(memgraph, memory_for):
    from claude_agent_sdk import ClaudeSDKClient, ResultMessage

    from memgraph_context_graph.sdk.claude import ClaudeAgentMemory

    memory = memory_for("live-claude-agent-sdk")
    claude = ClaudeAgentMemory(memory)

    async def converse(prompt):
        async with ClaudeSDKClient(claude.options(model="haiku", setting_sources=[], max_turns=8)) as client:
            await client.query(prompt)
            result = None
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    result = message
        await claude.aend_session()
        assert result is not None and not result.is_error, result
        return result.session_id, result.result or ""

    told, _ = await converse(TELL)
    asked, answer = await converse(ASK)

    _assert_remembered(memgraph, memory, "live-claude-agent-sdk", (told, asked), answer)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_agents_sdk_remembers_across_sessions(memgraph, memory_for):
    from agents import Agent, Runner
    from memgraph_context_graph.sdk.openai import OpenAIAgentsMemory

    memory = memory_for("live-openai-agents")

    async def converse(prompt, session_id):
        wired = OpenAIAgentsMemory(memory, session_id=session_id)
        agent = Agent(
            name="Assistant",
            model="gpt-4.1-mini",
            instructions=wired.instructions("You are a concise assistant with long-term memory."),
            tools=wired.tools(),
        )
        result = await Runner.run(agent, prompt, hooks=wired.hooks(), max_turns=8)
        await wired.aend_session()
        return session_id, str(result.final_output)

    told, _ = await converse(TELL, "live-openai-1")
    asked, answer = await converse(ASK, "live-openai-2")

    _assert_remembered(memgraph, memory, "live-openai-agents", (told, asked), answer)
