"""End-to-end test: AgentLink → SkillGraphConnector → SkillGraph → Memgraph.

Requires:
- Memgraph at bolt://localhost:7687 (override with env vars)
- agent-graph package installed (optional dependency of skills-graph)
- OpenAI tests additionally require OPENAI_API_KEY
"""

import asyncio
import os

import pytest

pytest.importorskip("agent_graph", reason="agent-graph not installed")

from agent_graph import AgentLink
from agent_graph.adapters.claude import ClaudeAdapter
from agent_graph.events import ToolEndEvent
from skills_graph import Skill, SkillGraph
from skills_graph.connector import SkillGraphConnector


@pytest.fixture()
def sg():
    """SkillGraph connected to a live Memgraph, cleaned before each test."""
    sg = SkillGraph()
    sg._db.query("MATCH (n) DETACH DELETE n")
    sg.setup()
    yield sg
    sg._db.query("MATCH (n) DETACH DELETE n")


@pytest.fixture()
def wired(sg):
    """Full wired stack: adapter → link → connector → graph."""
    link = AgentLink()
    connector = SkillGraphConnector(sg)
    link.add_connector(connector)
    adapter = ClaudeAdapter(link, "test-session", auto_session=False)
    return {"sg": sg, "link": link, "connector": connector, "adapter": adapter}


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_tool_start_creates_used_skill_relationship(wired):
    """Simulating get_skill via adapter creates USED_SKILL edge."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    # Seed a skill in the graph
    sg.add_skill(Skill(name="pdf-processing", description="PDF tools", content="..."))

    # Simulate a get_skill tool call through the Claude adapter
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        adapter._pre_tool_use(
            {
                "tool_name": "get_skill",
                "tool_input": {"name": "pdf-processing"},
                "tool_use_id": "tu-1",
            },
            "tu-1",
            {},
        )
    )

    # Verify the relationship was created
    rows = sg._db.query(
        """
        MATCH (sess:Session {session_id: $sid})-[r:USED_SKILL]->(sk:Skill {name: $name})
        RETURN r.access_count AS cnt, r.actions AS actions
        """,
        params={"sid": "test-session", "name": "pdf-processing"},
    )
    assert len(rows) == 1
    assert rows[0]["cnt"] == 1
    assert rows[0]["actions"] == ["get_skill"]


def test_multiple_accesses_increment_count(wired):
    """Repeated tool calls to the same skill increment the counter."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="cypher-basics", description="Cypher intro", content="..."))

    loop = asyncio.get_event_loop()
    for _ in range(3):
        loop.run_until_complete(
            adapter._pre_tool_use(
                {
                    "tool_name": "get_skill",
                    "tool_input": {"name": "cypher-basics"},
                    "tool_use_id": "tu-x",
                },
                "tu-x",
                {},
            )
        )

    rows = sg._db.query(
        """
        MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name})
        RETURN r.access_count AS cnt
        """,
        params={"sid": "test-session", "name": "cypher-basics"},
    )
    assert len(rows) == 1
    assert rows[0]["cnt"] == 3


def test_tool_end_records_search_results(wired):
    """ToolEndEvent from list_skills records each returned skill."""
    sg = wired["sg"]
    link = wired["link"]

    sg.add_skill(Skill(name="s1", description="d", content="c"))
    sg.add_skill(Skill(name="s2", description="d", content="c"))

    # Emit a ToolEndEvent directly (simulating post-tool result)
    link.emit(
        ToolEndEvent(
            session_id="test-session",
            tool_name="list_skills",
            result=[{"name": "s1"}, {"name": "s2"}],
        )
    )

    rows = sg._db.query(
        """
        MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(sk:Skill)
        RETURN sk.name AS name, r.actions AS actions
        ORDER BY sk.name
        """,
        params={"sid": "test-session"},
    )
    assert len(rows) == 2
    assert rows[0]["name"] == "s1"
    assert rows[0]["actions"] == ["list_skills_result"]
    assert rows[1]["name"] == "s2"


def test_non_skill_tool_ignored(wired):
    """Tools unrelated to skills must not create any relationships."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        adapter._pre_tool_use(
            {
                "tool_name": "Read",
                "tool_input": {"path": "/etc/hosts"},
                "tool_use_id": "tu-2",
            },
            "tu-2",
            {},
        )
    )

    rows = sg._db.query("MATCH ()-[r:USED_SKILL]->() RETURN count(r) AS cnt")
    assert rows[0]["cnt"] == 0


def test_full_session_lifecycle(wired):
    """Simulate a complete session: start → tool calls → stop."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="docker-skill", description="Docker helpers", content="..."))
    sg.add_skill(Skill(name="git-skill", description="Git helpers", content="..."))

    loop = asyncio.get_event_loop()

    # User prompt (ignored by connector)
    loop.run_until_complete(adapter._user_prompt_submit({"prompt": "Show me the docker skill"}, None, {}))

    # get_skill(docker-skill)
    loop.run_until_complete(
        adapter._pre_tool_use(
            {
                "tool_name": "get_skill",
                "tool_input": {"name": "docker-skill"},
                "tool_use_id": "tu-10",
            },
            "tu-10",
            {},
        )
    )
    loop.run_until_complete(
        adapter._post_tool_use(
            {
                "tool_name": "get_skill",
                "tool_use_id": "tu-10",
                "tool_response": "Docker helpers content",
            },
            "tu-10",
            {},
        )
    )

    # update_skill(git-skill)
    loop.run_until_complete(
        adapter._pre_tool_use(
            {
                "tool_name": "update_skill",
                "tool_input": {"name": "git-skill"},
                "tool_use_id": "tu-11",
            },
            "tu-11",
            {},
        )
    )

    # Stop
    loop.run_until_complete(adapter._stop({}, None, {}))

    # Verify both skills were recorded
    rows = sg._db.query(
        """
        MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(sk:Skill)
        RETURN sk.name AS name, r.actions AS actions
        ORDER BY sk.name
        """,
        params={"sid": "test-session"},
    )
    assert len(rows) == 2
    assert rows[0]["name"] == "docker-skill"
    assert rows[1]["name"] == "git-skill"


# ------------------------------------------------------------------
# OpenAI Agents SDK e2e tests
# ------------------------------------------------------------------

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@requires_openai_key
@pytest.mark.asyncio
async def test_openai_agent_get_skill(sg):
    """Real OpenAI agent calls get_skill tool → USED_SKILL edge in Memgraph."""
    from agents.run import RunConfig

    from agent_graph import AgentLink
    from agent_graph.adapters.openai import OpenAIAdapter
    from agents import Agent, Runner, function_tool
    from skills_graph.connector import SkillGraphConnector

    # Seed a skill
    sg.add_skill(Skill(name="cypher-basics", description="Intro to Cypher queries", content="MATCH (n) RETURN n"))

    # Define a function tool that mirrors the SkillGraph API
    @function_tool
    def get_skill(name: str) -> str:
        """Get a skill by name from the skill graph.

        Args:
            name: The name of the skill to retrieve.
        """
        skill = sg.get_skill(name)
        if skill:
            return f"Skill '{skill.name}': {skill.description}\n{skill.content}"
        return f"Skill '{name}' not found."

    # Wire the full stack
    link = AgentLink()
    connector = SkillGraphConnector(sg)
    link.add_connector(connector)
    adapter = OpenAIAdapter(link, session_id="openai-e2e-get", auto_session=False)

    agent = Agent(
        name="Skill Assistant",
        instructions=(
            "You help users retrieve skills. "
            "When asked about a skill, use the get_skill tool with the exact skill name. "
            "Return the skill content."
        ),
        tools=[get_skill],
        model="gpt-4o-mini",
    )

    result = await Runner.run(
        agent,
        "Get the skill called 'cypher-basics'",
        run_config=RunConfig(hooks=adapter.get_sdk_hooks()),
    )
    adapter.end_session()

    assert result.final_output is not None

    # Verify the USED_SKILL relationship was created
    rows = sg._db.query(
        """
        MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(sk:Skill {name: $name})
        RETURN r.access_count AS cnt, r.actions AS actions
        """,
        params={"sid": "openai-e2e-get", "name": "cypher-basics"},
    )
    assert len(rows) == 1
    assert rows[0]["cnt"] >= 1
    assert "get_skill" in rows[0]["actions"]
