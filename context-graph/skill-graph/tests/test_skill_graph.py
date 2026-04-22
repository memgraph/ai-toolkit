import pytest
from unittest.mock import MagicMock
from skill_graph import SkillGraph, Skill


@pytest.fixture
def mock_memgraph():
    return MagicMock()


@pytest.fixture
def sg(mock_memgraph):
    return SkillGraph(memgraph=mock_memgraph)


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------


def test_setup_creates_constraints_and_indexes(sg, mock_memgraph):
    sg.setup()
    calls = [c.args[0] for c in mock_memgraph.query.call_args_list]
    assert any("CONSTRAINT" in c and "Skill" in c for c in calls)
    assert any("INDEX" in c and "Skill" in c for c in calls)
    assert any("INDEX" in c and "Tag" in c for c in calls)


def test_drop_removes_constraints_and_indexes(sg, mock_memgraph):
    sg.drop()
    calls = [c.args[0] for c in mock_memgraph.query.call_args_list]
    assert any("DROP CONSTRAINT" in c for c in calls)
    assert any("DROP INDEX" in c and "Skill" in c for c in calls)
    assert any("DROP INDEX" in c and "Tag" in c for c in calls)


# ------------------------------------------------------------------
# Add
# ------------------------------------------------------------------


def test_add_skill_without_tags(sg, mock_memgraph):
    skill = Skill(name="s1", description="desc", content="body")
    result = sg.add_skill(skill)

    assert result.name == "s1"
    # Only the CREATE call, no UNWIND for tags
    assert mock_memgraph.query.call_count == 1


def test_add_skill_with_tags(sg, mock_memgraph):
    skill = Skill(name="s1", description="desc", content="body", tags=["a", "b"])
    sg.add_skill(skill)

    assert mock_memgraph.query.call_count == 2
    tag_call = mock_memgraph.query.call_args_list[1]
    assert "UNWIND" in tag_call.args[0]
    assert tag_call.kwargs["params"]["tags"] == ["a", "b"]


# ------------------------------------------------------------------
# Get
# ------------------------------------------------------------------


def test_get_skill_returns_none_when_missing(sg, mock_memgraph):
    mock_memgraph.query.return_value = []
    assert sg.get_skill("nope") is None


def test_get_skill_returns_skill(sg, mock_memgraph):
    mock_memgraph.query.return_value = [
        {
            "name": "s1",
            "description": "d",
            "content": "c",
            "created_at": "2025-01-01",
            "updated_at": "2025-01-01",
            "tags": ["x"],
        }
    ]
    skill = sg.get_skill("s1")
    assert skill is not None
    assert skill.name == "s1"
    assert skill.tags == ["x"]


# ------------------------------------------------------------------
# Update
# ------------------------------------------------------------------


def test_update_skill_sets_fields(sg, mock_memgraph):
    mock_memgraph.query.return_value = [
        {
            "name": "s1",
            "description": "new",
            "content": "new_c",
            "created_at": "2025-01-01",
            "updated_at": "2025-01-02",
            "tags": [],
        }
    ]
    sg.update_skill("s1", description="new", content="new_c")

    set_call = mock_memgraph.query.call_args_list[0]
    assert "SET" in set_call.args[0]
    assert "description" in set_call.args[0]
    assert "content" in set_call.args[0]


def test_update_skill_replaces_tags(sg, mock_memgraph):
    mock_memgraph.query.return_value = [
        {
            "name": "s1",
            "description": "d",
            "content": "c",
            "created_at": "2025-01-01",
            "updated_at": "2025-01-02",
            "tags": ["new_tag"],
        }
    ]
    sg.update_skill("s1", tags=["new_tag"])

    # SET call, DELETE old tags, MERGE new tags, GET
    assert mock_memgraph.query.call_count == 4


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------


def test_delete_skill_returns_true(sg, mock_memgraph):
    mock_memgraph.query.return_value = [{"deleted": 1}]
    assert sg.delete_skill("s1") is True


def test_delete_skill_returns_false_when_missing(sg, mock_memgraph):
    mock_memgraph.query.return_value = [{"deleted": 0}]
    assert sg.delete_skill("s1") is False


# ------------------------------------------------------------------
# List / Search
# ------------------------------------------------------------------


def test_list_skills(sg, mock_memgraph):
    mock_memgraph.query.return_value = [
        {
            "name": "a",
            "description": "d",
            "content": "c",
            "created_at": "t",
            "updated_at": "t",
            "tags": [],
        },
        {
            "name": "b",
            "description": "d2",
            "content": "c2",
            "created_at": "t",
            "updated_at": "t",
            "tags": ["x"],
        },
    ]
    skills = sg.list_skills()
    assert len(skills) == 2
    assert skills[0].name == "a"


def test_search_by_tags(sg, mock_memgraph):
    mock_memgraph.query.return_value = []
    result = sg.search_by_tags(["python"])
    assert result == []
    assert "ALL(tag IN" in mock_memgraph.query.call_args.args[0]


def test_search_by_name(sg, mock_memgraph):
    mock_memgraph.query.return_value = []
    sg.search_by_name("graph")
    assert "CONTAINS" in mock_memgraph.query.call_args.args[0]


# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------


def test_add_dependency(sg, mock_memgraph):
    sg.add_dependency("a", "b")
    call = mock_memgraph.query.call_args
    assert "DEPENDS_ON" in call.args[0]
    assert call.kwargs["params"] == {"skill_name": "a", "depends_on": "b"}


def test_remove_dependency(sg, mock_memgraph):
    sg.remove_dependency("a", "b")
    call = mock_memgraph.query.call_args
    assert "DELETE r" in call.args[0]


def test_get_dependencies(sg, mock_memgraph):
    mock_memgraph.query.return_value = [
        {
            "name": "dep",
            "description": "d",
            "content": "c",
            "created_at": "t",
            "updated_at": "t",
            "tags": [],
        }
    ]
    deps = sg.get_dependencies("a")
    assert len(deps) == 1
    assert deps[0].name == "dep"


def test_get_dependents(sg, mock_memgraph):
    mock_memgraph.query.return_value = []
    result = sg.get_dependents("a")
    assert result == []
