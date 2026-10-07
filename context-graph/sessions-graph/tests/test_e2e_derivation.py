"""sessions-graph derive against a real Memgraph: schedule, claim, gate and adoption.

The LLM and the observer are stood in for (the gate's numbers are what these
tests choose), and so is the GLiNER2 checkpoint the re-extraction loads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("hygm", reason="hygm not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from sessions_graph import derivation
from sessions_graph.cli import _spawn_due_derivations

from actions_graph import MessageRole, Session
from hygm import Change, Derivation, default_model
from hygm.strategies.llm import CONSOLIDATE_SCHEMA, PROPOSE_SCHEMA, PRUNE_SCHEMA
from unstructured2graph.gliner2_backend import GLiNER2Backend
from unstructured2graph.gliner2_observer import Measurement

LONG = "I have been rewriting our ingestion service in Rust and benchmarking it against the Go one. " * 12


class _Engine:
    """Enough of a gliner2 engine for re-extraction: it finds nothing."""

    def create_schema(self):
        return SimpleNamespace(
            entity=lambda *a, **k: self.create_schema(), relation=lambda *a, **k: self.create_schema()
        )

    def compile_schema(self, schema):
        return schema

    def extract(self, text, schema, config=None):
        return SimpleNamespace(entities=[], relations=[], feasible=True)


@pytest.fixture(autouse=True)
def _no_gliner2_checkpoint(monkeypatch):
    monkeypatch.setattr(GLiNER2Backend, "_engine", staticmethod(lambda model, model_name: model or _Engine()))


class _Llm:
    """Adds a Library type and a maintains relation; records how many calls it took."""

    def __init__(self):
        self.calls = 0

    def __call__(self, system, prompt, schema):
        self.calls += 1
        if schema is PROPOSE_SCHEMA:
            return {"entity_types": [], "relations": []}
        if schema is CONSOLIDATE_SCHEMA:
            return {
                "entity_types": [{"label": "Library", "description": "a code library"}],
                "relations": [{"name": "maintains", "intended_head": ["User"], "intended_tail": ["Library"]}],
                "merges": [],
            }
        assert schema is PRUNE_SCHEMA
        return {
            "entity_types": [{"label": "Library", "identity": "global", "reason": ""}],
            "relations": [{"name": "maintains", "head": ["User"], "tail": ["Library"], "reason": ""}],
        }


class _Observer:
    """Observes Library and maintains firing; scores a model with Library as `learned`, any other as `base`."""

    def __init__(self, base=(0.30, 0.50), learned=(0.20, 0.60)):
        self.base, self.learned = base, learned
        self.measured = []

    def observe(self, model, sample):
        return {
            "types": {"Library": {"mentions": 9}, "User": {"mentions": 20}},
            "relations": {"maintains": {"edges": 4, "pairs": [{"head": "User", "tail": "Library", "count": 4}]}},
        }

    def measure(self, model, sample, catch_alls):
        self.measured.append(len(sample))
        share, coverage = self.learned if "Library" in model.node_labels() else self.base
        return Measurement(share, coverage, spans={(0, 0, 4): "User"}, mentions=10, user_turns=2)


def _sessions(memgraph, actions_graph, user_id, count, *, start=0, text=LONG):
    when = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for i in range(start, start + count):
        sid = f"{user_id}-s{i}"
        actions_graph.create_session(Session(session_id=sid))
        for turn in range(2):
            actions_graph.record_message(session_id=sid, role=MessageRole.USER, content=f"{turn}: {text}")
        actions_graph.record_message(session_id=sid, role=MessageRole.ASSISTANT, content="Sounds good.")
        memgraph.query(
            """
            MERGE (u:User {user_id: $user_id}) WITH u MATCH (s:Session {session_id: $sid})
            MERGE (u)-[:HAD_SESSION]->(s)
            SET s.reconciliation_status = 'completed', s.reconciled_at = $at
            """,
            params={"user_id": user_id, "sid": sid, "at": (when + timedelta(minutes=i)).isoformat()},
        )


def test_milestones_double_to_128_then_step_by_128():
    seen, after = [], 0
    for _ in range(10):
        after = derivation.next_milestone(after)
        seen.append(after)
    assert seen == [2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
    assert derivation.next_milestone(200) == 256


def test_only_reconciled_sessions_past_the_content_floor_count(graph, memgraph, actions_graph):
    _sessions(memgraph, actions_graph, "alice", 2)
    _sessions(memgraph, actions_graph, "alice", 1, start=2, text="hi")
    memgraph.query("MATCH (s:Session {session_id: 'alice-s1'}) SET s.reconciliation_status = 'pending'")

    assert [sid for sid, _ in derivation.qualifying_sessions(memgraph, "alice")] == ["alice-s0"]
    assert derivation.due(memgraph, "alice") is None


@pytest.mark.asyncio
async def test_a_passing_candidate_is_adopted_and_the_schedule_moves_on(graph, memgraph, actions_graph):
    _sessions(memgraph, actions_graph, "alice", 2)
    assert derivation.due(memgraph, "alice") == 2
    llm, observer = _Llm(), _Observer()

    report = await derivation.run(graph, "alice", llm=llm, observer=observer, actions_graph=actions_graph, seeds=2)

    assert (report.outcome, report.adopted_version, report.in_sample, report.delta) == ("adopted", 1, True, 2)
    assert [c.passed for c in report.candidates] == [True, True]
    version = graph.adopted_ontology("alice")
    assert (version.version, version.source) == (1, "derived")
    assert "Library" in version.model.node_labels()
    assert {"op": "add", "kind": "node", "label": "Library", "into": None, "reason": ""} in version.changelog
    assert version.report["base"] == {"catch_all_share": 0.3, "coverage": 0.5}
    assert version.report["outcome"] == "adopted"
    assert version.counts["nodes"]["Library"] == 9
    assert report.reextracted == 2  # the run added types, so its delta is re-read under them
    rows = memgraph.query("MATCH (s:Session) RETURN collect(DISTINCT s.ontology_version) AS versions")
    assert rows == [{"versions": [1]}]
    rejected = memgraph.query("MATCH (:User {user_id: 'alice'})-[:REJECTED]->(v) RETURN count(v) AS n")
    assert rejected == [{"n": 1}]  # the other passing seed, kept for the report
    assert derivation.due(memgraph, "alice") is None  # next milestone is 4
    assert memgraph.query("MATCH (u:User {user_id: 'alice'}) RETURN u.derive_claimed_until AS c") == [{"c": None}]


@pytest.mark.asyncio
async def test_when_no_candidate_passes_n_stays_and_the_delta_rolls_over(graph, memgraph, actions_graph):
    _sessions(memgraph, actions_graph, "alice", 2)
    observer = _Observer(learned=(0.40, 0.50))  # more lands in the catch-alls: worse

    report = await derivation.run(graph, "alice", llm=_Llm(), observer=observer, actions_graph=actions_graph, seeds=1)

    assert report.outcome == "rejected"
    assert graph.adopted_ontology("alice").version == 0
    rejected = memgraph.query(
        "MATCH (:User {user_id: 'alice'})-[:REJECTED]->(v) RETURN v.status AS status, v.version AS version"
    )
    assert rejected == [{"status": "rejected", "version": 1}]
    state = memgraph.query("MATCH (u:User {user_id: 'alice'}) RETURN u.derive_milestone AS m, u.derive_watermark AS w")
    assert state == [{"m": 2, "w": None}]


@pytest.mark.asyncio
async def test_from_eight_sessions_the_gate_runs_on_held_out_sessions(graph, memgraph, actions_graph):
    _sessions(memgraph, actions_graph, "alice", 8)
    observer = _Observer()

    report = await derivation.run(
        graph, "alice", llm=_Llm(), observer=observer, actions_graph=actions_graph, force=True, seeds=1
    )

    assert (report.in_sample, report.held_out, report.sampled) == (False, 2, 8)
    assert set(observer.measured) == {2}


@pytest.mark.asyncio
async def test_a_second_run_finds_the_claim_and_derive_off_never_runs(graph, memgraph, actions_graph, tmp_path):
    _sessions(memgraph, actions_graph, "alice", 2)
    assert derivation._claim(memgraph, "alice")

    claimed = await derivation.run(graph, "alice", llm=_Llm(), observer=_Observer(), actions_graph=actions_graph)
    assert claimed.outcome == "claimed elsewhere"

    derivation._release(memgraph, "alice")
    schema = tmp_path / "fixed.yaml"
    schema.write_text("entity_types:\n  - {label: User, description: me, identity: global}\n")
    graph.supply_ontology_file("alice", schema, derive="off")
    off = await derivation.run(graph, "alice", llm=_Llm(), observer=_Observer(), actions_graph=actions_graph)
    assert off.outcome == "derive off"
    assert derivation.due(memgraph, "alice") is None


def test_a_merge_relabels_entities_and_relationships_in_place(graph, memgraph):
    memgraph.query(
        "CREATE (a:gliner2:Framework {entity_type: 'Framework', text: 'Django'})"
        "-[:built_with {source_id: 't1'}]->(b:gliner2:Library {entity_type: 'Library', text: 'pytest'})"
    )
    merged = Derivation(
        model=default_model(),
        pool=default_model(),
        counts={},
        changelog=(
            Change("merge", "node", "Framework", "Library"),
            Change("merge", "relation", "built_with", "uses"),
        ),
    )

    derivation._relabel_merges(graph, merged)

    rows = memgraph.query("MATCH (n:gliner2) RETURN n.text AS text, n.entity_type AS type, labels(n) AS labels")
    assert {row["text"]: (row["type"], sorted(row["labels"])) for row in rows} == {
        "Django": ("Library", ["Library", "gliner2"]),
        "pytest": ("Library", ["Library", "gliner2"]),
    }
    edges = memgraph.query("MATCH ()-[r]->() RETURN type(r) AS type, r.source_id AS source")
    assert edges == [{"type": "uses", "source": "t1"}]


def test_reconcile_spawns_a_detached_derive_for_users_at_a_milestone(graph, memgraph, actions_graph, monkeypatch):
    _sessions(memgraph, actions_graph, "alice", 2)
    _sessions(memgraph, actions_graph, "bob", 1)
    spawned = []
    monkeypatch.setattr("sessions_graph.connector._spawn_detached", lambda args, env: spawned.append(args))

    _spawn_due_derivations(graph, ["alice-s0", "alice-s1", "bob-s0"])

    assert spawned == [["derive", "--user", "alice"]]
