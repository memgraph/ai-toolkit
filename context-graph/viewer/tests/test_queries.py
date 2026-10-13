"""The viewer's queries against a real Memgraph, over graphs captured through the real hooks pipeline."""

from __future__ import annotations

import ast
import inspect
import re

import pytest
from conftest import OTHER_PROJECT, OTHER_USER, PROJECT, USER, extract, mark_reconciled

from context_graph_viewer import queries
from context_graph_viewer.queries import ALL, USER_AND_UNATTRIBUTED, Focus, Scope

pytest.importorskip("actions_graph")
pytest.importorskip("sessions_graph")


def _ids(rows):
    return {row["id"] for row in rows}


def test_scope_cycle_narrows_to_the_user_then_widens(capture, reader):
    """Your sessions by default; then plus unattributed ones; then everyone's."""
    capture.session("mine-1")
    capture.session("mine-2", cwd=OTHER_PROJECT)
    capture.session("theirs", user=OTHER_USER)
    capture.session("nobody", user=None)

    scope = Scope(user_id=USER)
    assert _ids(queries.dashboard(reader, scope).sessions) == {"mine-1", "mine-2"}
    assert _ids(queries.dashboard(reader, scope.widened()).sessions) == {"mine-1", "mine-2", "nobody"}
    assert _ids(queries.dashboard(reader, scope.widened().widened()).sessions) == {
        "mine-1",
        "mine-2",
        "nobody",
        "theirs",
    }
    assert queries.dashboard(reader, scope).unattributed == 1


def test_a_session_reconcile_gave_an_anon_owner_is_unattributed_but_a_real_owner_wins(capture, reader, memgraph):
    """Reconcile files ownerless sessions under `anon-<id>`; a later real owner takes precedence (#383)."""
    capture.session("anon-only", user=None)
    capture.session("both")
    memgraph.query(
        "MATCH (s:Session) WHERE s.session_id IN ['anon-only', 'both'] "
        "MERGE (u:User {user_id: 'anon-' + s.session_id}) MERGE (u)-[:HAD_SESSION]->(s)"
    )

    rows = {row["id"]: row for row in queries.dashboard(reader, Scope(ALL, USER)).sessions}
    assert rows["anon-only"]["unattributed"] is True
    assert (rows["both"]["owner"], rows["both"]["unattributed"]) == (USER, False)
    assert _ids(queries.dashboard(reader, Scope(user_id=USER)).sessions) == {"both"}


def test_project_filter_narrows_sessions_and_lists_every_project(capture, reader):
    """The project picker lists every working directory in the owner scope, whatever is picked."""
    capture.session("billing")
    capture.session("crawler", cwd=OTHER_PROJECT)

    data = queries.dashboard(reader, Scope(user_id=USER, project=PROJECT))
    assert _ids(data.sessions) == {"billing"}
    assert data.projects == [PROJECT, OTHER_PROJECT]
    assert data.total_sessions == 1


def test_session_rows_count_actions_errors_and_touches(capture, reader):
    """Actions exclude tool results; failed results are tool errors; a gh fetch leaves an open Touch."""
    capture.session("busy", commands=("ls", "make test", "gh issue view 2000 -R memgraph/memgraph"),
                    failing=("make test",))  # fmt: skip

    row = queries.dashboard(reader, Scope(user_id=USER)).sessions[0]
    assert row["actions"] == 4  # the prompt and three calls
    assert row["errors"] == 1
    assert (row["touches"], row["touches_open"]) == (1, 1)
    assert row["last_event"] is not None


def test_pipeline_counts_failures_growth_and_ontology(capture, reader, memgraph):
    """The pipeline panel's numbers come from Session properties and the entities reconciled sessions grounded."""
    capture.session("done")
    capture.session("broken")
    capture.session("open", end=False)
    mark_reconciled(memgraph, "done", total_input_tokens=100, total_output_tokens=20)
    mark_reconciled(memgraph, "broken", status="failed", reconciliation_error="extraction timed out")
    extract(memgraph, "done", [("Memgraph", "Organization"), ("tokio", "Artifact")])
    memgraph.query(
        "MATCH (u:User {user_id: $u}) CREATE (u)-[:ADOPTED]->(:OntologyVersion {user_id: $u, version: 3})",
        {"u": USER},
    )

    data = queries.dashboard(reader, Scope(user_id=USER))
    by_status = {row["rstatus"]: row["n"] for row in data.pipeline}
    assert by_status == {"completed": 1, "failed": 1, "pending": 1}
    assert data.tokens == 120
    assert [f["id"] for f in data.failures] == ["broken"]
    assert data.failures[0]["rerror"] == "extraction timed out"
    assert sum(data.growth.values()) == 2
    assert data.ontology == [{"user": USER, "version": 3, "milestone": None}]


def test_session_anatomy_folds_a_subagent_under_the_call_that_spawned_it(capture, reader):
    """The subagent's own actions hang off it, and it hangs off the Task call (SPAWNED)."""
    capture.subagent_session("parent")

    anatomy = queries.session_anatomy(reader, "parent")
    assert anatomy is not None
    task = next(a for a in anatomy.actions if a["tool"] == "Task")
    grep = next(a for a in anatomy.actions if a["tool"] == "Grep")
    assert grep["agent_id"] == "agent-1"
    assert task["agent_id"] is None
    assert [(g["id"], g["spawned_by"]) for g in anatomy.agents] == [("agent-1", task["id"])]
    assert grep["result_at"] is not None


def test_session_anatomy_joins_a_pending_touch_to_its_call_by_tool_use_id(capture, reader):
    """Before the Sweep resolves it a Touch has no CAUSED_BY; the harness's tool-use id still places it."""
    capture.session("fetching", commands=("gh issue view 2000 -R memgraph/memgraph",))

    anatomy = queries.session_anatomy(reader, "fetching")
    assert anatomy is not None
    call = next(a for a in anatomy.actions if a["type"] == "tool_call")
    assert [t["status"] for t in anatomy.touches[call["id"]]] == ["pending"]
    assert anatomy.loose_touches == []


def test_session_anatomy_reports_what_the_session_produced(capture, reader, memgraph):
    """Chunks, entities and the reconcile outcome, linked at session level."""
    capture.session("made")
    mark_reconciled(memgraph, "made")
    extract(memgraph, "made", [("Memgraph", "Organization"), ("tokio", "Artifact")])

    anatomy = queries.session_anatomy(reader, "made")
    assert anatomy is not None
    produced = anatomy.produced
    assert (produced["chunks"], produced["entity_count"], produced["rstatus"]) == (1, 2, "completed")
    assert sorted(produced["entities"]) == ["Memgraph", "tokio"]


def test_session_anatomy_of_an_unknown_session_is_none(memgraph, reader):
    assert queries.session_anatomy(reader, "missing") is None


def test_entities_merge_by_name_and_count_support_per_session(capture, reader, memgraph):
    """GLiNER2 gives one name a node per chunk; the list shows it once, supported by both sessions."""
    for session_id in ("s1", "s2"):
        capture.session(session_id)
        mark_reconciled(memgraph, session_id)
        extract(memgraph, session_id, [("Memgraph", "Organization"), (f"only-{session_id}", "Artifact")])

    rows = {row["label"]: row for row in queries.entities(reader, Scope(user_id=USER))}
    assert rows["Memgraph"]["support"] == 2
    assert len(rows["Memgraph"]["keys"]) == 2
    assert rows["only-s1"]["support"] == 1
    assert next(iter(rows)) == "Memgraph"


def test_entities_hide_value_types_unless_asked(capture, reader, memgraph):
    capture.session("dated")
    extract(memgraph, "dated", [("Memgraph", "Organization"), ("next Tuesday", "Date")])

    assert [r["label"] for r in queries.entities(reader, Scope(user_id=USER))] == ["Memgraph"]
    shown = queries.entities(reader, Scope(user_id=USER), hide_values=False)
    assert {r["label"] for r in shown} == {"Memgraph", "next Tuesday"}


def test_entities_of_one_session_and_out_of_scope_ones(capture, reader, memgraph):
    """`g` lists one session's entities; another user's entities stay out of your scope."""
    capture.session("mine")
    capture.session("theirs", user=OTHER_USER)
    extract(memgraph, "mine", [("Memgraph", "Organization")])
    extract(memgraph, "theirs", [("secret project", "Topic")])

    assert [r["label"] for r in queries.entities(reader, Scope(user_id=USER))] == ["Memgraph"]
    assert [r["label"] for r in queries.entities(reader, Scope(user_id=USER), session_id="theirs")] == [
        "secret project"
    ]


def test_neighbours_fold_parallel_edges_and_mark_off_ontology_relations(capture, reader, memgraph):
    """Two chunks stating one fact are one row with the strongest confidence; the User shows as you."""
    capture.session("s1")
    capture.session("s2")
    first = extract(memgraph, "s1", [("billing", "Project"), ("tokio", "Artifact")],
                    [("billing", "depends_on", "tokio", 0.6), ("billing", "owned_by", "@user", 0.9)])  # fmt: skip
    second = extract(memgraph, "s2", [("billing", "Project"), ("tokio", "Artifact")],
                     [("billing", "depends_on", "tokio", 0.8), ("billing", "mentions", "tokio", 0.7)],
                     nonconformant=("mentions",))  # fmt: skip
    focus = Focus((first["billing"], second["billing"]), "billing")

    rows = queries.neighbours(reader, focus, Scope(user_id=USER), queries.entity_keys(reader, Scope(user_id=USER)))
    by_relation = {row["rel_type"]: row for row in rows}
    assert set(by_relation) == {"depends_on", "owned_by", "mentions"}
    depends = by_relation["depends_on"]
    assert (depends["edges"], depends["confidence"], depends["label"], depends["outgoing"]) == (2, 0.8, "tokio", True)
    assert sorted(depends["keys"]) == sorted([first["tokio"], second["tokio"]])
    assert by_relation["owned_by"]["label"] == f"you ({USER})"
    assert by_relation["mentions"]["nonconformant"] is True
    assert by_relation["depends_on"]["nonconformant"] is False


def test_neighbours_leave_out_entities_outside_the_scope(capture, reader, memgraph):
    """A shared node can relate to another user's entity; your scope doesn't show it."""
    capture.session("mine")
    capture.session("theirs", user=OTHER_USER)
    mine = extract(memgraph, "mine", [("billing", "Project")])
    theirs = extract(memgraph, "theirs", [("payroll", "Project")])
    memgraph.query(
        "MATCH (a {entity_id: $a}), (b {entity_id: $b}) CREATE (a)-[:related_to {confidence: 0.9}]->(b)",
        {"a": mine["billing"], "b": theirs["payroll"]},
    )
    focus = Focus((mine["billing"],), "billing")

    scoped = queries.neighbours(reader, focus, Scope(user_id=USER), queries.entity_keys(reader, Scope(user_id=USER)))
    assert scoped == []
    everyone = Scope(ALL, USER)
    assert [r["label"] for r in queries.neighbours(reader, focus, everyone, queries.entity_keys(reader, everyone))] == [
        "payroll"
    ]


def test_neighbours_of_lightrag_entities_are_untyped(capture, reader, memgraph):
    """LightRAG keys entities by name and joins them with :DIRECTED edges carrying keywords and weights."""
    capture.session("lr")
    memgraph.query(
        "MATCH (s:Session {session_id: 'lr'})-[:HAS_ACTION]->(a:Action) WITH a LIMIT 1 "
        "CREATE (a)-[:HAS_CHUNK]->(c:Chunk {hash: 'lr-chunk'}) "
        "CREATE (m:base {entity_id: 'Memgraph', entity_type: 'organization', description: 'A graph database'}) "
        "CREATE (g:base {entity_id: 'Cypher', entity_type: 'concept'}) "
        "CREATE (m)-[:MENTIONED_IN]->(c), (g)-[:MENTIONED_IN]->(c) "
        "CREATE (m)-[:DIRECTED {keywords: 'query language,uses', weight: 2.0, "
        "description: 'Memgraph speaks Cypher<SEP>Cypher is its query language'}]->(g)"
    )

    rows = queries.neighbours(reader, Focus(("Memgraph",), "Memgraph"), Scope(user_id=USER), None)
    assert [(r["rel_type"], r["label"], r["weight"], r["confidence"]) for r in rows] == [
        ("DIRECTED", "Cypher", 2.0, None)
    ]
    assert [r["label"] for r in queries.entities(reader, Scope(user_id=USER))] == ["Cypher", "Memgraph"]


def test_provenance_lists_supporting_sessions_in_scope(capture, reader, memgraph):
    capture.session("s1")
    capture.session("s2", cwd=OTHER_PROJECT)
    mark_reconciled(memgraph, "s1")
    a = extract(memgraph, "s1", [("Memgraph", "Organization")])
    b = extract(memgraph, "s2", [("Memgraph", "Organization")])
    focus = Focus((a["Memgraph"], b["Memgraph"]), "Memgraph")

    prov = queries.provenance(reader, focus, Scope(user_id=USER))
    assert {s["session"] for s in prov["sessions"]} == {"s1", "s2"}
    assert prov["type"] == "Organization"
    narrowed = queries.provenance(reader, focus, Scope(user_id=USER, project=OTHER_PROJECT))
    assert [s["session"] for s in narrowed["sessions"]] == ["s2"]


def test_entities_ground_in_memory_files_too(capture, reader, memgraph):
    """Memory files are reconciled on their own and belong to the user, so they count as support."""
    capture.session("s1")
    extract(memgraph, "s1", [("Memgraph", "Organization")])
    memgraph.query(
        "MATCH (u:User {user_id: $u}) "
        "CREATE (u)-[:HAS_MEMORY]->(m:Memory {path: 'notes.md', extraction_status: 'completed'}) "
        "CREATE (m)-[:HAS_CHUNK]->(c:Chunk {hash: 'mem-chunk'}) "
        "CREATE (e:gliner2:Organization {entity_id: 'mem-e', text: 'Memgraph', entity_type: 'Organization'}) "
        "CREATE (e)-[:MENTIONED_IN]->(c)",
        {"u": USER},
    )

    row = queries.entities(reader, Scope(user_id=USER))[0]
    assert (row["label"], row["support"]) == ("Memgraph", 2)
    assert queries.provenance(reader, Focus(tuple(row["keys"]), "Memgraph"), Scope(user_id=USER))["memories"] == [
        "notes.md"
    ]
    assert [m["path"] for m in queries.dashboard(reader, Scope(user_id=USER)).memories] == ["notes.md"]


def test_queries_never_write():
    """The viewer is read-only: no Cypher it can send creates, merges, sets or deletes anything."""
    tree = ast.parse(inspect.getsource(queries))
    cypher = " ".join(
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert not re.findall(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", cypher)


@pytest.mark.parametrize("mode", [USER_AND_UNATTRIBUTED, ALL])
def test_every_scope_mode_runs(capture, reader, mode):
    """Each scope's queries are valid Cypher on real Memgraph, even on a graph with one empty session."""
    capture.session("empty", commands=(), end=False)
    scope = Scope(mode, USER, PROJECT)
    queries.dashboard(reader, scope)
    queries.entities(reader, scope)
    assert queries.entity_keys(reader, scope) == []
