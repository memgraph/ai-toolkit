"""Memory files reconciled one by one when they change, against a real Memgraph."""

from __future__ import annotations

import pytest

pytest.importorskip("actions_graph", reason="actions-graph not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from sessions_graph.memory_store import is_memory_tool, memory_paths_written
from sessions_graph.reconciliation import extract_reconcilable_text

from actions_graph.models import ToolCall
from tests.test_reconciliation import _SurfaceEngine

ONTOLOGY = (
    "entity_types:\n"
    "  - {label: User, description: the user, identity: global}\n"
    "  - {label: Person, description: someone else, identity: global}\n"
    "  - {label: Tool, description: a tool, identity: global}\n"
    "relation_types:\n"
    "  - {label: uses, description: works with, start_labels: [User, Person], end_labels: [Tool]}\n"
)


@pytest.fixture
def backend(tmp_path):
    from unstructured2graph import load_ontology
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    path = tmp_path / "ontology.yaml"
    path.write_text(ONTOLOGY, encoding="utf-8")
    return GLiNER2Backend(
        ontology=load_ontology(path),
        model=_SurfaceEngine(
            {"Ante": "Person", "uv": "Tool", "pip": "Tool"}, [("uses", "Ante", "uv"), ("uses", "Ante", "pip")]
        ),
    )


def _tools_used(memgraph):
    rows = memgraph.query("MATCH ()-[r:uses]->(t:Tool) RETURN t.text AS tool, r.source_id AS source ORDER BY tool")
    return [(row["tool"], row["source"]) for row in rows]


async def test_a_write_marks_the_file_pending_and_reconciling_extracts_it(graph, memgraph, backend):
    store = graph.memory_store("alice")
    store.create("/memories/user/tools.md", "Ante uses uv for everything.")
    [memory_id] = graph.get_pending_memory_reconciliations()

    result = await graph.reconcile_memory(memory_id, extraction_backend=backend)

    assert result.status == "completed", result.error
    assert result.path == "/memories/user/tools.md"
    assert _tools_used(memgraph) == [("uv", memory_id)]
    assert graph.get_pending_memory_reconciliations() == []
    assert (
        memgraph.query(
            "MATCH (:Memory {memory_id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n", {"id": memory_id}
        )[0]["n"]
        == 1
    )


async def test_reconciling_an_edit_replaces_what_the_old_text_gave(graph, memgraph, backend):
    store = graph.memory_store("alice")
    store.create("/memories/user/tools.md", "Ante uses uv for everything.")
    [memory_id] = graph.get_pending_memory_reconciliations()
    await graph.reconcile_memory(memory_id, extraction_backend=backend)

    store.str_replace("/memories/user/tools.md", "uv", "pip")
    await graph.reconcile_memory(memory_id, extraction_backend=backend)

    assert _tools_used(memgraph) == [("pip", memory_id)]
    assert memgraph.query("MATCH (t:Tool {text: 'uv'}) RETURN count(t) AS n")[0]["n"] == 0


async def test_deleting_a_file_removes_what_it_gave(graph, memgraph, backend):
    store = graph.memory_store("alice")
    store.create("/memories/user/tools.md", "Ante uses uv for everything.")
    [memory_id] = graph.get_pending_memory_reconciliations()
    await graph.reconcile_memory(memory_id, extraction_backend=backend)

    store.delete("/memories/user/tools.md")

    assert _tools_used(memgraph) == []
    assert memgraph.query("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"] == 0


async def test_a_failure_is_recorded_on_the_file(graph, memgraph):
    class Broken:
        ontology = None

        async def extract(self, *args, **kwargs):
            raise RuntimeError("extractor down")

    graph.memory_store("alice").create("/memories/a.md", "text")
    [memory_id] = graph.get_pending_memory_reconciliations()

    result = await graph.reconcile_memory(memory_id, extraction_backend=Broken())

    assert result.status == "failed"
    row = memgraph.query("MATCH (m:Memory) RETURN m.extraction_status AS s, m.extraction_error AS e")[0]
    assert row["s"] == "failed" and row["e"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mcp__plugin_context-graph_context-graph__memory", True),
        ("mcp__context-graph__memory", True),
        ("memory", True),
        ("mcp__other-server__memory", False),
        ("mcp__context-graph__recall", False),
        ("Write", False),
    ],
)
def test_memory_tool_names_across_harnesses(name, expected):
    assert is_memory_tool(name) is expected


def test_memory_tool_calls_are_not_session_content():
    call = ToolCall(
        session_id="s",
        tool_name="mcp__context-graph__memory",
        tool_input={"command": "create", "path": "/memories/a.md", "file_text": "x"},
    )
    other = ToolCall(session_id="s", tool_name="Read", tool_input={"file_path": "a.py"})

    assert extract_reconcilable_text(call) is None
    assert extract_reconcilable_text(other)


def test_written_paths_from_tool_input():
    assert memory_paths_written({"command": "create", "path": "/memories//a.md"}) == ["/memories/a.md"]
    assert memory_paths_written({"command": "rename", "old_path": "/memories/a.md", "new_path": "/memories/b.md"}) == [
        "/memories/b.md"
    ]
    assert memory_paths_written({"command": "view", "path": "/memories/a.md"}) == []
    assert memory_paths_written({"command": "create", "path": "/etc/x"}) == []


def test_the_hook_records_which_session_wrote_a_file(graph, memgraph):
    from sessions_graph.connector import SessionsGraphConnector

    from agent_context_graph.events import SessionStartEvent, ToolEndEvent

    connector = SessionsGraphConnector(graph)
    connector.on_event(SessionStartEvent(session_id="s-9", user_id="alice"))
    graph.memory_store("alice").create("/memories/a.md", "x")  # as the MCP server would, without a session

    connector.on_event(
        ToolEndEvent(
            session_id="s-9",
            tool_name="mcp__plugin_context-graph_context-graph__memory",
            metadata={"tool_input": {"command": "create", "path": "/memories/a.md", "file_text": "x"}},
        )
    )

    rows = memgraph.query("MATCH (s:Session)-[:PRODUCED_MEMORY]->(m:Memory) RETURN s.session_id AS s, m.path AS p")
    assert rows == [{"s": "s-9", "p": "/memories/a.md"}]
