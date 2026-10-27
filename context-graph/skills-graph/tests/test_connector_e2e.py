"""End-to-end test: AgentLink → SkillGraphConnector → SkillGraph → Memgraph.

Requires:
- Memgraph at bolt://localhost:7687 (override with env vars)
- memgraph-context-graph package installed (optional dependency of skills-graph)
- OpenAI tests additionally require OPENAI_API_KEY
"""

import os

import pytest

pytest.importorskip("memgraph_context_graph", reason="memgraph-context-graph not installed")

from memgraph_context_graph import AgentLink
from memgraph_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter
from memgraph_context_graph.adapters.codex import CodexHooksAdapter
from memgraph_context_graph.events import ToolEndEvent
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
    """Full wired stack: Claude Code / Agent SDK hook payloads → link → connector → graph."""
    link = AgentLink()
    connector = SkillGraphConnector(sg)
    link.add_connector(connector)
    adapter = ClaudeCodeHooksAdapter(link, session_id="test-session")
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
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "get_skill",
            "tool_input": {"name": "pdf-processing"},
            "tool_use_id": "tu-1",
        }
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


def test_tool_call_inside_subagent_attaches_used_skill_to_agent_not_session(wired):
    """A skill access inside a subagent links (:Agent)-[:USED_SKILL]->(:Skill), not the Session (#280)."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="pdf-processing", description="PDF tools", content="..."))
    # Simulates an Agent node actions-graph's connector would already have created
    # for this subagent (start_agent runs before any of its tool calls fire).
    sg._db.query(
        "CREATE (:Agent {agent_id: $agent_id, agent_type: 'Explore', started_at: '2026-01-01T00:00:00+00:00'})",
        params={"agent_id": "agent-1"},
    )
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "get_skill",
            "tool_input": {"name": "pdf-processing"},
            "tool_use_id": "tu-agent-1",
            "agent_id": "agent-1",
        }
    )

    agent_rows = sg._db.query(
        "MATCH (:Agent {agent_id: $agent_id})-[r:USED_SKILL]->(:Skill {name: $name}) RETURN count(r) AS cnt",
        params={"agent_id": "agent-1", "name": "pdf-processing"},
    )
    assert agent_rows[0]["cnt"] == 1

    session_rows = sg._db.query(
        "MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name}) RETURN count(r) AS cnt",
        params={"sid": "test-session", "name": "pdf-processing"},
    )
    assert session_rows[0]["cnt"] == 0


def test_tool_call_with_agent_name_but_no_agent_node_falls_back_to_session(wired):
    """An agent with no Agent node -- actions-graph's connector, which creates
    them, isn't wired -- must fall back to the Session, not silently drop the
    whole USED_SKILL record."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="pdf-processing", description="PDF tools", content="..."))
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "get_skill",
            "tool_input": {"name": "pdf-processing"},
            "tool_use_id": "tu-no-agent-node",
            "agent_id": "top-level-agent-with-no-node",
        }
    )

    session_rows = sg._db.query(
        "MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name}) RETURN count(r) AS cnt",
        params={"sid": "test-session", "name": "pdf-processing"},
    )
    assert session_rows[0]["cnt"] == 1


def test_multiple_accesses_increment_count(wired):
    """Repeated tool calls to the same skill increment the counter."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="cypher-basics", description="Cypher intro", content="..."))
    for _ in range(3):
        adapter.handle_payload(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "get_skill",
                "tool_input": {"name": "cypher-basics"},
                "tool_use_id": "tu-x",
            }
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
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/etc/hosts"},
            "tool_use_id": "tu-2",
        }
    )

    rows = sg._db.query("MATCH ()-[r:USED_SKILL]->() RETURN count(r) AS cnt")
    assert rows[0]["cnt"] == 0


def test_add_skill_tool_supported_but_missing_skill_is_not_recorded(wired):
    """add_skill is recognized, but pre-tool tracking cannot match a Skill that does not exist yet."""
    sg = wired["sg"]
    adapter = wired["adapter"]
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "add_skill",
            "tool_input": {"name": "new-skill"},
            "tool_use_id": "tu-add",
        }
    )

    rows = sg._db.query("MATCH ()-[r:USED_SKILL]->() RETURN count(r) AS cnt")
    assert rows[0]["cnt"] == 0


def test_delete_skill_tool_records_existing_skill(wired):
    """delete_skill is recognized while the Skill still exists at pre-tool time."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="old-skill", description="Old helper", content="..."))
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "delete_skill",
            "tool_input": {"name": "old-skill"},
            "tool_use_id": "tu-delete",
        }
    )

    rows = sg._db.query(
        """
        MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name})
        RETURN r.access_count AS cnt, r.actions AS actions
        """,
        params={"sid": "test-session", "name": "old-skill"},
    )
    assert rows[0]["cnt"] == 1
    assert rows[0]["actions"] == ["delete_skill"]


def test_full_session_lifecycle(wired):
    """Simulate a complete session: start → tool calls → stop."""
    sg = wired["sg"]
    adapter = wired["adapter"]

    sg.add_skill(Skill(name="docker-skill", description="Docker helpers", content="..."))
    sg.add_skill(Skill(name="git-skill", description="Git helpers", content="..."))

    # User prompt (ignored by connector)
    adapter.handle_payload({"hook_event_name": "UserPromptSubmit", "prompt": "Show me the docker skill"})

    # get_skill(docker-skill)
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "get_skill",
            "tool_input": {"name": "docker-skill"},
            "tool_use_id": "tu-10",
        }
    )
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "get_skill",
            "tool_use_id": "tu-10",
            "tool_response": "Docker helpers content",
        }
    )

    # update_skill(git-skill)
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "update_skill",
            "tool_input": {"name": "git-skill"},
            "tool_use_id": "tu-11",
        }
    )

    # Stop
    adapter.handle_payload({"hook_event_name": "Stop"})

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


def _used_skills(sg):
    return sg._db.query(
        """
        MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(sk:Skill)
        RETURN sk.name AS name, sk.source_path AS source_path, r.actions AS actions
        ORDER BY sk.name
        """,
        params={"sid": "test-session"},
    )


def _hook(adapter, tool_name, tool_input, cwd):
    adapter.handle_payload(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": f"tu-{tool_name}",
            "cwd": str(cwd),
        }
    )


def test_claude_code_skill_tool_call_records_used_skill(wired, tmp_path, monkeypatch):
    """Claude Code's Skill tool is recorded, with the installed SKILL.md filling the node; its result adds nothing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    skill_file = tmp_path / ".agents" / "skills" / "grilling" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: grilling\ndescription: Grill the user\n---\n", encoding="utf-8")
    adapter = wired["adapter"]

    _hook(adapter, "Skill", {"skill": "grilling"}, tmp_path)
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Skill",
            "tool_input": {"skill": "grilling"},
            "tool_response": {"success": True, "commandName": "grilling"},
            "tool_use_id": "tu-Skill",
            "cwd": str(tmp_path),
        }
    )

    assert _used_skills(wired["sg"]) == [
        {"name": "grilling", "source_path": str(skill_file), "actions": ["invoke_skill"]}
    ]


def test_every_skill_file_read_in_one_command_records_usage(wired, tmp_path):
    for name in ("first", "second"):
        (tmp_path / "skills" / name).mkdir(parents=True)
        (tmp_path / "skills" / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\n", encoding="utf-8"
        )

    _hook(wired["adapter"], "Bash", {"command": "cat skills/first/SKILL.md; cat skills/second/SKILL.md"}, tmp_path)

    assert [row["name"] for row in _used_skills(wired["sg"])] == ["first", "second"]


def test_codex_reading_a_skill_file_records_used_skill(sg, tmp_path):
    """Codex has no skill tool: using a skill is a shell read of its SKILL.md, as its hooks report it."""
    link = AgentLink()
    link.add_connector(SkillGraphConnector(sg))
    adapter = CodexHooksAdapter(link, session_id="test-session")
    skill_file = tmp_path / ".agents" / "skills" / "code-review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: code-review\ndescription: Review a branch\n---\n", encoding="utf-8")

    for command in (
        f"cat {skill_file} >/dev/null; rg --files -g AGENTS.md",
        f"find {tmp_path} -path '*/code-review/SKILL.md' -print 2>/dev/null",
    ):
        adapter.handle_payload(
            {
                "hook_event_name": "PreToolUse",
                "session_id": "test-session",
                "tool_name": "Bash",
                "tool_input": {"command": command},
                "tool_use_id": "call-1",
                "cwd": str(tmp_path),
            }
        )

    assert _used_skills(sg) == [{"name": "code-review", "source_path": str(skill_file), "actions": ["read_skill_file"]}]


def test_activity_that_is_not_skill_usage_records_nothing(wired, tmp_path):
    """Memory files, staging, searching, authoring and delegating a SKILL.md path are not Skill Usage."""
    memory_file = tmp_path / "memory" / "feedback.md"
    memory_file.parent.mkdir()
    memory_file.write_text("---\nname: feedback\ndescription: Prefer small PRs\n---\n", encoding="utf-8")
    adapter = wired["adapter"]

    _hook(adapter, "Read", {"file_path": str(memory_file)}, tmp_path)
    adapter.handle_payload(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": str(memory_file)},
            "tool_response": {"type": "text", "file": {"content": memory_file.read_text(encoding="utf-8")}},
            "tool_use_id": "tu-Read",
            "cwd": str(tmp_path),
        }
    )
    _hook(adapter, "Bash", {"command": "git add skills/release/SKILL.md"}, tmp_path)
    _hook(adapter, "Bash", {"command": "git diff -- skills/release/SKILL.md"}, tmp_path)
    _hook(adapter, "Bash", {"command": "rg -n name skills/release/SKILL.md"}, tmp_path)
    _hook(adapter, "Write", {"file_path": "skills/release/SKILL.md", "content": "---"}, tmp_path)
    _hook(adapter, "Agent", {"prompt": "read skills/release/SKILL.md", "subagent_type": "Explore"}, tmp_path)

    assert _used_skills(wired["sg"]) == []


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
    from agents import Agent, Runner, function_tool
    from memgraph_context_graph import AgentLink
    from memgraph_context_graph.sdk import ContextGraphMemory
    from memgraph_context_graph.sdk.openai import OpenAIAgentsMemory
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
    wired = OpenAIAgentsMemory(ContextGraphMemory(link=link, memory_tool=False), session_id="openai-e2e-get")

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
        hooks=wired.hooks(),
    )
    wired.end_session()

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
