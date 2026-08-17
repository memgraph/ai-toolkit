"""End-to-end tests that run against a live Memgraph instance.

Requires Memgraph to be reachable at bolt://localhost:7687 (default).
Override with MEMGRAPH_URL, MEMGRAPH_USER, MEMGRAPH_PASSWORD env vars.
"""

import pytest

from skills_graph import Skill, SkillGraph


@pytest.fixture()
def sg():
    """SkillGraph connected to a live Memgraph, cleaned before each test."""
    sg = SkillGraph()
    # Clean up any leftover graph nodes from prior runs
    sg._db.query("MATCH (n) DETACH DELETE n")
    sg.setup()
    yield sg
    sg._db.query("MATCH (n) DETACH DELETE n")


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------


def test_setup_creates_constraints_and_indexes(sg):
    constraints = sg._db.query("SHOW CONSTRAINT INFO;")
    assert any(c["label"] == "Skill" and c["properties"] == ["name"] for c in constraints)
    indexes = sg._db.query("SHOW INDEX INFO;")
    assert any(i["label"] == "Skill" and i["property"] == ["name"] for i in indexes)


def test_drop_removes_constraints_and_indexes(sg):
    sg.drop()
    try:
        constraints = sg._db.query("SHOW CONSTRAINT INFO;")
        assert not any(c["label"] == "Skill" for c in constraints)
        indexes = sg._db.query("SHOW INDEX INFO;")
        assert not any(i["label"] == "Skill" for i in indexes)
    finally:
        sg.setup()  # restore schema this test intentionally tore down


# ------------------------------------------------------------------
# Full lifecycle
# ------------------------------------------------------------------


def test_add_and_get_skill(sg):
    skill = Skill(
        name="pdf-processing",
        description="Extract PDF text, fill forms, merge files.",
        content="# PDF Processing\n\nUse pdfplumber to extract text.",
        license="Apache-2.0",
        compatibility="Requires Python 3.10+",
        metadata={"author": "example-org", "version": "1.0"},
        allowed_tools=["Bash(git:*)", "Read"],
    )
    sg.add_skill(skill)

    retrieved = sg.get_skill("pdf-processing")
    assert retrieved is not None
    assert retrieved.name == "pdf-processing"
    assert retrieved.description == skill.description
    assert retrieved.content == skill.content
    assert retrieved.license == "Apache-2.0"
    assert retrieved.compatibility == "Requires Python 3.10+"
    assert retrieved.metadata == {"author": "example-org", "version": "1.0"}
    assert retrieved.allowed_tools == ["Bash(git:*)", "Read"]


def test_update_skill(sg):
    sg.add_skill(Skill(name="s1", description="original", content="v1"))

    updated = sg.update_skill("s1", description="changed", content="v2")
    assert updated is not None
    assert updated.description == "changed"
    assert updated.content == "v2"


def test_update_skill_sets_spec_fields(sg):
    sg.add_skill(Skill(name="s1", description="d", content="c"))

    updated = sg.update_skill(
        "s1",
        license="MIT",
        compatibility="Python 3.10+",
        metadata={"v": "2"},
        allowed_tools=["Bash(git:*)"],
    )
    assert updated.license == "MIT"
    assert updated.compatibility == "Python 3.10+"
    assert updated.metadata == {"v": "2"}
    assert updated.allowed_tools == ["Bash(git:*)"]

    fetched = sg.get_skill("s1")
    assert fetched.license == "MIT"
    assert fetched.metadata == {"v": "2"}


def test_delete_skill(sg):
    sg.add_skill(Skill(name="s1", description="to delete", content="body"))
    assert sg.delete_skill("s1") is True
    assert sg.get_skill("s1") is None
    assert sg.delete_skill("s1") is False


def test_list_skills(sg):
    sg.add_skill(Skill(name="a1", description="first", content="body"))
    sg.add_skill(Skill(name="b2", description="second", content="body"))

    skills = sg.list_skills()
    names = [s.name for s in skills]
    assert "a1" in names
    assert "b2" in names


def test_search_by_name(sg):
    sg.add_skill(Skill(name="cypher-basics", description="d", content="c"))
    sg.add_skill(Skill(name="advanced-cypher", description="d", content="c"))
    sg.add_skill(Skill(name="rust-guide", description="d", content="c"))

    results = sg.search_by_name("cypher")
    names = [s.name for s in results]
    assert "cypher-basics" in names
    assert "advanced-cypher" in names
    assert "rust-guide" not in names


def test_dependencies(sg):
    sg.add_skill(Skill(name="base-skill", description="foundation", content="c"))
    sg.add_skill(Skill(name="advanced-skill", description="builds on base", content="c"))

    sg.add_dependency("advanced-skill", "base-skill")

    deps = sg.get_dependencies("advanced-skill")
    assert len(deps) == 1
    assert deps[0].name == "base-skill"

    dependents = sg.get_dependents("base-skill")
    assert len(dependents) == 1
    assert dependents[0].name == "advanced-skill"

    sg.remove_dependency("advanced-skill", "base-skill")
    assert sg.get_dependencies("advanced-skill") == []


# ------------------------------------------------------------------
# Skill usage
# ------------------------------------------------------------------


def test_record_skill_usage_matches_existing_skill_by_default(sg):
    sg.add_skill(Skill(name="cypher-basics", description="d", content="c"))

    sg.record_skill_usage(
        session_id="usage-session",
        skill_name="cypher-basics",
        action="get_skill",
        timestamp="2026-04-30T00:00:00+00:00",
    )

    rows = sg._db.query(
        "MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name}) "
        "RETURN r.access_count AS cnt, r.actions AS actions",
        params={"sid": "usage-session", "name": "cypher-basics"},
    )
    assert len(rows) == 1
    assert rows[0]["cnt"] == 1
    assert rows[0]["actions"] == ["get_skill"]


def test_record_skill_usage_attaches_to_agent_container_when_given(sg):
    """USED_SKILL attaches to the Agent, not the Session, when usage happened
    inside a subagent -- the either-container pattern HAS_ACTION also uses."""
    sg.add_skill(Skill(name="cypher-basics", description="d", content="c"))
    sg._db.query(
        "CREATE (:Agent {agent_id: $agent_id, agent_type: 'Explore', started_at: '2026-01-01T00:00:00+00:00'})",
        params={"agent_id": "agent-1"},
    )

    sg.record_skill_usage(
        session_id="usage-session",
        skill_name="cypher-basics",
        action="get_skill",
        timestamp="2026-04-30T00:00:00+00:00",
        container_agent_id="agent-1",
    )

    agent_rows = sg._db.query(
        "MATCH (:Agent {agent_id: $agent_id})-[r:USED_SKILL]->(:Skill {name: $name}) RETURN count(r) AS c",
        params={"agent_id": "agent-1", "name": "cypher-basics"},
    )
    assert agent_rows[0]["c"] == 1

    session_rows = sg._db.query(
        "MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill) RETURN count(r) AS c",
        params={"sid": "usage-session"},
    )
    assert session_rows[0]["c"] == 0


def test_record_skill_usage_falls_back_to_session_when_agent_does_not_exist(sg):
    """Some adapters (e.g. OpenAI Agents SDK) set an agent name for every running
    agent, not just genuine nested subagents, and no Agent node for it may exist
    at all. A hard MATCH would silently drop the whole usage record; it must
    fall back to the Session."""
    sg.add_skill(Skill(name="cypher-basics", description="d", content="c"))

    sg.record_skill_usage(
        session_id="usage-session",
        skill_name="cypher-basics",
        action="get_skill",
        timestamp="2026-04-30T00:00:00+00:00",
        container_agent_id="top-level-agent-with-no-node",
    )

    rows = sg._db.query(
        "MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name}) RETURN count(r) AS c",
        params={"sid": "usage-session", "name": "cypher-basics"},
    )
    assert rows[0]["c"] == 1


def test_record_skill_usage_can_create_missing_skill(sg):
    sg.record_skill_usage(
        session_id="usage-session",
        skill_name="memgraph-console",
        action="read_skill_file",
        timestamp="2026-04-30T00:00:00+00:00",
        create_missing=True,
        description="Use mgconsole",
        content="# Skill",
        source_path="/tmp/skills/memgraph-console/SKILL.md",
        metadata={"source": "local_skill_file"},
    )

    skill = sg.get_skill("memgraph-console")
    assert skill is not None
    assert skill.description == "Use mgconsole"

    rows = sg._db.query(
        "MATCH (:Session {session_id: $sid})-[r:USED_SKILL]->(:Skill {name: $name}) RETURN count(r) AS c",
        params={"sid": "usage-session", "name": "memgraph-console"},
    )
    assert rows[0]["c"] == 1
