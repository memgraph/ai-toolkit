"""End-to-end tests that run against a live Memgraph instance.

Requires Memgraph to be reachable at bolt://localhost:7687 (default).
Override with MEMGRAPH_URL, MEMGRAPH_USER, MEMGRAPH_PASSWORD env vars.
"""

import pytest
from skill_graph import SkillGraph, Skill


@pytest.fixture()
def sg():
    """SkillGraph connected to a live Memgraph, cleaned before each test."""
    sg = SkillGraph()
    # Clean up any leftover skill/tag nodes from prior runs
    sg._db.query("MATCH (n) DETACH DELETE n")
    sg.setup()
    yield sg
    sg._db.query("MATCH (n) DETACH DELETE n")


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
        tags=["pdf", "extraction"],
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
    assert set(retrieved.tags) == {"pdf", "extraction"}


def test_update_skill(sg):
    sg.add_skill(Skill(name="s1", description="original", content="v1"))

    updated = sg.update_skill("s1", description="changed", content="v2", tags=["new"])
    assert updated is not None
    assert updated.description == "changed"
    assert updated.content == "v2"
    assert updated.tags == ["new"]


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


def test_search_by_tags(sg):
    sg.add_skill(
        Skill(name="s1", description="d", content="c", tags=["python", "graph"])
    )
    sg.add_skill(Skill(name="s2", description="d", content="c", tags=["python"]))
    sg.add_skill(Skill(name="s3", description="d", content="c", tags=["rust"]))

    results = sg.search_by_tags(["python"])
    names = [s.name for s in results]
    assert "s1" in names
    assert "s2" in names
    assert "s3" not in names

    results = sg.search_by_tags(["python", "graph"])
    assert len(results) == 1
    assert results[0].name == "s1"


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
    sg.add_skill(
        Skill(name="advanced-skill", description="builds on base", content="c")
    )

    sg.add_dependency("advanced-skill", "base-skill")

    deps = sg.get_dependencies("advanced-skill")
    assert len(deps) == 1
    assert deps[0].name == "base-skill"

    dependents = sg.get_dependents("base-skill")
    assert len(dependents) == 1
    assert dependents[0].name == "advanced-skill"

    sg.remove_dependency("advanced-skill", "base-skill")
    assert sg.get_dependencies("advanced-skill") == []
