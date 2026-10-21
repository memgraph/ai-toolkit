"""Forgetting a session against a real Memgraph: its share goes, what another session backs stays."""

from __future__ import annotations

import pytest

pytest.importorskip("actions_graph", reason="actions-graph not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from actions_graph import MessageRole, Session
from memgraph_context_graph import session as session_node
from memgraph_context_graph.forget import forget_session
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


async def _reconciled_session(graph, actions_graph, backend, session_id, text):
    actions_graph.create_session(Session(session_id=session_id, working_directory="/repo"))
    actions_graph.record_message(session_id=session_id, role=MessageRole.USER, content=text)
    summary = await graph.reconciler.reconcile(session_id, extraction_backend=backend)
    assert summary.status == "completed", summary.error


def _count(memgraph, query, **params):
    return memgraph.query(query, params)[0]["n"]


def _tools(memgraph):
    # Without enforce_ontology, entities carry only the extraction workspace label.
    rows = memgraph.query("MATCH (:gliner2 {text: 'Ante'})-[:uses]->(t) RETURN t.text AS tool ORDER BY tool")
    return [row["tool"] for row in rows]


async def test_forgetting_one_session_keeps_what_the_other_backs(graph, memgraph, actions_graph, backend):
    await _reconciled_session(graph, actions_graph, backend, "forget-a", "Ante uses uv for everything.")
    await _reconciled_session(graph, actions_graph, backend, "forget-b", "Ante uses pip at work.")
    assert _tools(memgraph) == ["pip", "uv"]

    result = forget_session(memgraph, "forget-a")

    assert result.existed
    assert result.removed["actions"] == 1 and result.removed["chunks"] == 1 and result.removed["users"] == 1
    assert _tools(memgraph) == ["pip"], "the forgotten session's fact went, the other's stayed"
    assert _count(memgraph, "MATCH (t:gliner2 {text: 'uv'}) RETURN count(t) AS n") == 0
    assert _count(memgraph, "MATCH (:gliner2 {text: 'Ante'})-[m:MENTIONED_IN]->() RETURN count(m) AS n") == 1, (
        "the shared entity stays, with only the other session's mention"
    )
    assert _count(memgraph, "MATCH (s:Session {session_id: 'forget-a'}) RETURN count(s) AS n") == 0
    assert _count(memgraph, "MATCH (u:User {user_id: 'anon-forget-a'}) RETURN count(u) AS n") == 0
    assert _count(memgraph, "MATCH (c:Chunk) WHERE c.text CONTAINS 'uv' RETURN count(c) AS n") == 0
    assert _count(memgraph, "MATCH (s:Session {session_id: 'forget-b'})-[:HAS_ACTION]->(a) RETURN count(a) AS n") == 1
    assert _count(memgraph, "MATCH (c:Chunk) WHERE c.text CONTAINS 'pip' RETURN count(c) AS n") == 1


async def test_a_forgotten_session_is_never_captured_or_reconciled_again(graph, memgraph, actions_graph, backend):
    await _reconciled_session(graph, actions_graph, backend, "forget-a", "Ante uses uv for everything.")
    forget_session(memgraph, "forget-a")

    actions_graph.record_message(session_id="forget-a", role=MessageRole.USER, content="Ante uses uv again.")
    summary = await graph.reconciler.reconcile("forget-a", extraction_backend=backend)

    assert summary.status == "forgotten"
    assert _count(memgraph, "MATCH (s:Session {session_id: 'forget-a'}) RETURN count(s) AS n") == 0
    assert _count(memgraph, "MATCH (n) WHERE n:Action OR n:Chunk OR n:gliner2 RETURN count(n) AS n") == 0
    assert session_node.is_forgotten(memgraph, "forget-a")


async def test_a_session_forgotten_while_it_is_reconciled_leaves_nothing(
    graph, memgraph, actions_graph, backend, monkeypatch
):
    actions_graph.create_session(Session(session_id="forget-a"))
    actions_graph.record_message(session_id="forget-a", role=MessageRole.USER, content="Ante uses uv for everything.")
    link = graph.reconciler._link_chunks_to_sources

    def forgotten_midway(sources, chunks):
        forget_session(memgraph, "forget-a")  # Lands after extraction wrote, before the chunks are linked.
        link(sources, chunks)

    monkeypatch.setattr(graph.reconciler, "_link_chunks_to_sources", forgotten_midway)
    summary = await graph.reconciler.reconcile("forget-a", extraction_backend=backend)

    assert summary.status == "forgotten"
    assert _count(memgraph, "MATCH (n) WHERE n:Chunk OR n:gliner2 OR n:User RETURN count(n) AS n") == 0
    assert _count(memgraph, "MATCH ()-[r:uses]->() RETURN count(r) AS n") == 0


async def test_memory_files_the_session_wrote_stay_but_no_longer_name_it(graph, memgraph, actions_graph):
    actions_graph.create_session(Session(session_id="forget-a"))
    store = graph.memory_store("alice", session_id="forget-a")
    store.create("/memories/user/tools.md", "Ante uses uv.")
    store.str_replace("/memories/user/tools.md", "uv", "pip")

    result = forget_session(memgraph, "forget-a")

    assert result.removed["memory_versions"] == 1
    assert _count(memgraph, "MATCH (m:Memory {path: '/memories/user/tools.md'}) RETURN count(m) AS n") == 1
    assert _count(memgraph, "MATCH (v:MemoryVersion) WHERE v.session_id IS NOT NULL RETURN count(v) AS n") == 0


async def test_forgetting_an_unknown_session_only_marks_it(memgraph, graph):
    result = forget_session(memgraph, "never-seen")

    assert not result.existed
    assert session_node.is_forgotten(memgraph, "never-seen")


@pytest.fixture
def cli_config():
    """The config file the autouse fixture isolated, pointed at the test Memgraph, never the user's."""
    import os

    from memgraph_context_graph.adapters import _identity

    _identity.write_full_config(user_id="tester", memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"))
    _identity._reset_cache()


def test_forget_project_lists_first_refuses_without_a_terminal_and_deletes_with_yes(
    memgraph, actions_graph, cli_config, capsys, monkeypatch, tmp_path
):
    from memgraph_context_graph.cli import main

    project = tmp_path / "client-repo"
    for session_id, cwd in (("in-1", project), ("in-2", project / "src"), ("out", tmp_path / "elsewhere")):
        actions_graph.create_session(Session(session_id=session_id, working_directory=str(cwd)))
        actions_graph.record_message(session_id=session_id, role=MessageRole.USER, content="hello")

    def sessions():
        return sorted(row["id"] for row in memgraph.query("MATCH (s:Session) RETURN s.session_id AS id"))

    assert main(["forget", "project", str(project), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "in-1" in out and "in-2" in out and "out" not in out.replace("would be forgotten", "")
    assert sessions() == ["in-1", "in-2", "out"], "a dry run deletes nothing"

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["forget", "project", str(project)]) == 2
    assert sessions() == ["in-1", "in-2", "out"]

    assert main(["forget", "project", str(project), "--yes"]) == 0
    assert sessions() == ["out"]
    assert "Forgot in-1" in capsys.readouterr().out
