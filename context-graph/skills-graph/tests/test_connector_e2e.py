"""End-to-end test: AgentLink → SkillGraphConnector → SkillGraph → Memgraph.

Requires Memgraph at bolt://localhost:7687 (override with env vars).
"""

import asyncio

import pytest

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
