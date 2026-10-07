"""A user's ontology versions, against a real Memgraph."""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("hygm", reason="hygm not installed")

from sessions_graph import ontology
from sessions_graph.cli import main

from hygm import CORE_LABELS, HygmModel, NodeType, RelationType, default_model

CODING = """
entity_types:
  - {label: User, description: the user, identity: global}
  - {label: Person, description: someone else, identity: global}
  - {label: Library, description: a code library, identity: global}
relation_types:
  - {label: maintains, description: maintains, start_labels: [User, Person], end_labels: [Library]}
"""


@pytest.fixture
def schema(tmp_path):
    path = tmp_path / "coding.yaml"
    path.write_text(textwrap.dedent(CODING))
    return path


def _chain(memgraph, user_id):
    return memgraph.query(
        """
        MATCH (v:OntologyVersion {user_id: $user_id})
        OPTIONAL MATCH (:User {user_id: $user_id})-[a:ADOPTED]->(v)
        OPTIONAL MATCH (v)-[:NEXT]->(next:OntologyVersion)
        RETURN v.version AS version, v.status AS status, a IS NOT NULL AS adopted, next.version AS next
        ORDER BY version
        """,
        params={"user_id": user_id},
    )


def test_a_user_without_a_version_is_on_the_default(graph):
    version = graph.adopted_ontology("alice")

    assert (version.version, version.source) == (0, "default")
    assert version.model == default_model()


def test_extend_adds_the_core_and_pins_the_supplied_types(graph, memgraph, schema):
    version = graph.supply_ontology_file("alice", schema)

    assert (version.version, version.source, version.derive) == (1, "supplied", "extend")
    assert version.pinned == ("User", "Person", "Library", "maintains")
    assert set(CORE_LABELS) <= set(version.model.node_labels())
    assert graph.adopted_ontology("alice") == version
    assert _chain(memgraph, "alice") == [{"version": 1, "status": "adopted", "adopted": True, "next": None}]


def test_off_uses_the_schema_exactly_as_given(graph, schema):
    version = graph.supply_ontology_file("alice", schema, derive="off")

    assert version.model.node_labels() == ("User", "Person", "Library")
    assert graph.adopted_ontology("alice").derive == "off"


def test_a_new_version_supersedes_the_last_and_chains_after_it(graph, memgraph, schema):
    graph.supply_ontology_file("alice", schema)
    graph.supply_ontology_file("alice", schema, derive="off")

    assert _chain(memgraph, "alice") == [
        {"version": 1, "status": "superseded", "adopted": False, "next": 2},
        {"version": 2, "status": "adopted", "adopted": True, "next": None},
    ]


def test_an_invalid_schema_writes_nothing(graph, memgraph, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(
        "entity_types:\n  - {label: User, description: me}\n"
        "relation_types:\n  - {label: uses, description: x, start_labels: [User], end_labels: [User]}\n"
    )

    with pytest.raises(ValueError, match="Person"):
        graph.supply_ontology_file("alice", path)
    assert _chain(memgraph, "alice") == []


def test_sync_adopts_a_file_only_when_its_content_or_mode_changes(graph, schema):
    assert graph.sync_ontology_file("alice", schema).version == 1
    assert graph.sync_ontology_file("alice", schema) is None

    assert graph.sync_ontology_file("alice", schema, derive="off").version == 2

    schema.write_text(schema.read_text().replace("a code library", "a package"))
    assert graph.sync_ontology_file("alice", schema, derive="off").version == 3


def test_extend_carries_learned_types_that_do_not_clash(graph, memgraph, schema):
    """Types derivation learned survive a newly supplied schema, unless the schema declares the same label."""
    learned = HygmModel(
        node_types=(
            *default_model().node_types,
            NodeType("Library", "learned: a dependency", "global"),
            NodeType("Bug", "a defect", "chunk"),
        ),
        relation_types=(
            *default_model().relation_types,
            RelationType("fixed", "", ("User", "Person"), ("Bug",)),
        ),
    )
    ontology._adopt(
        memgraph,
        ontology.OntologyVersion(user_id="alice", version=1, model=learned, source="derived", created_at="t"),
    )

    version = graph.supply_ontology_file("alice", schema)

    assert version.model.node_type("Library").description == "a code library"
    assert version.model.node_type("Bug") is not None
    assert "fixed" in version.model.relation_labels()
    assert "Organization" not in version.model.node_labels()


def test_cli_loads_and_shows_a_users_version(graph, schema, capsys):
    assert main(["ontology", "load", "--user", "alice", "--file", str(schema)]) == 0
    assert "Adopted version 1 for alice" in capsys.readouterr().out

    assert main(["ontology", "show", "--user", "alice"]) == 0
    shown = capsys.readouterr().out
    assert "alice: version 1 (supplied), derive=extend" in shown
    assert "Library (pinned) [global] a code library" in shown
    assert "maintains: User|Person -> Library" in shown


def test_cli_refuses_without_a_user(graph, capsys):
    assert main(["ontology", "show"]) == 2
    assert "No user" in capsys.readouterr().err


def test_reconcile_applies_the_configured_file_and_survives_a_broken_one(graph, schema, capsys):
    from sessions_graph.cli import _sync_configured_ontology

    from agent_context_graph.adapters import _identity

    _identity.write_config(user_id="alice", ontology_path=str(schema))
    _sync_configured_ontology(graph)
    assert "Adopted ontology version 1 for alice" in capsys.readouterr().out

    schema.write_text("entity_types: not-a-list\n")
    _sync_configured_ontology(graph)
    assert "not applied, keeping the adopted version" in capsys.readouterr().err
    assert graph.adopted_ontology("alice").version == 1


def test_a_version_whose_pool_names_types_it_does_not_hold_reads_back(graph, memgraph):
    """Retiring `paid` pools it with its User -> Money endpoints; the pool holds neither type."""
    pool = HygmModel(node_types=(), relation_types=(RelationType("paid", "", ("User", "Person"), ("Money",)),))
    ontology._adopt(
        memgraph,
        ontology.OntologyVersion(
            user_id="alice", version=1, model=default_model(), source="derived", created_at="t", pool=pool
        ),
    )

    assert graph.adopted_ontology("alice").pool == pool
