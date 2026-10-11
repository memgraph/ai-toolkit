"""The shared-space reconcile worker against a real Enterprise Memgraph."""

from __future__ import annotations

from types import SimpleNamespace

import neo4j
import pytest
from sessions_graph.tool import MemoryTool

from agent_context_graph.adapters import _identity
from context_graph_spaces.provision import SERVICE_ACCOUNT, apply
from context_graph_spaces.spec import parse_spec
from context_graph_spaces.worker import reconcile_spaces

pytest.importorskip("unstructured2graph", reason="the worker needs sessions-graph[reconciliation]")

DOC = "/memories/spaces/payments/tools.md"
ONTOLOGY = (
    "entity_types:\n"
    "  - {label: Person, description: someone, identity: global}\n"
    "  - {label: Tool, description: a tool, identity: global}\n"
    "relation_types:\n"
    "  - {label: uses, description: works with, start_labels: [Person], end_labels: [Tool]}\n"
)


class _SurfaceEngine:
    """Stands in for the GLiNER2 model at the extraction boundary: types known surfaces, relates pairs."""

    def __init__(self, surfaces, relations):
        self.surfaces, self.relations = surfaces, relations

    def create_schema(self):
        class _Schema:
            def entity(self, *args, **kwargs):
                return self

            def relation(self, *args, **kwargs):
                return self

        return _Schema()

    def compile_schema(self, schema):
        return schema

    def extract(self, text, schema, config=None):
        entities = [
            SimpleNamespace(
                id=surface,
                type=kind,
                text=surface,
                start=text.find(surface),
                end=text.find(surface) + len(surface),
                confidence=0.9,
            )
            for surface, kind in self.surfaces.items()
            if surface in text
        ]
        found = {entity.id for entity in entities}
        relations = [
            SimpleNamespace(type=kind, head=head, tail=tail, confidence=0.9)
            for kind, head, tail in self.relations
            if head in found and tail in found
        ]
        return SimpleNamespace(entities=entities, relations=relations)


@pytest.fixture
def backend(tmp_path):
    from unstructured2graph import load_ontology
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    path = tmp_path / "ontology.yaml"
    path.write_text(ONTOLOGY, encoding="utf-8")
    return GLiNER2Backend(
        ontology=load_ontology(path), model=_SurfaceEngine({"Ante": "Person", "uv": "Tool"}, [("uses", "Ante", "uv")])
    )


@pytest.fixture
def team(admin):
    spec = parse_spec({"members": ["alice", "bob"], "spaces": {"payments": {"members": ["alice", "bob"]}}})
    return spec, apply(spec, admin).passwords


def _write_shared(enterprise_url, join, passwords, login, arguments):
    join(enterprise_url, login, passwords[login])
    return MemoryTool().call(arguments, _identity.load_config())


def _derived(admin):
    return {
        "chunks": admin("MATCH (c:Chunk) RETURN count(c) AS n", "space_payments", None)[0]["n"],
        "uses": admin("MATCH ()-[r:uses]->() RETURN count(r) AS n", "space_payments", None)[0]["n"],
    }


def test_the_worker_extracts_what_members_share(enterprise_url, join, team, admin, backend):
    spec, passwords = team
    _write_shared(
        enterprise_url, join, passwords, "alice", {"command": "create", "path": DOC, "file_text": "Ante uses uv."}
    )

    (report,) = reconcile_spaces(
        spec, url=enterprise_url, password=passwords[SERVICE_ACCOUNT], extraction_backend=backend
    )

    assert (report.space, report.completed, report.failed, report.error) == ("payments", [DOC], [], None)
    assert _derived(admin) == {"chunks": 1, "uses": 1}
    again = reconcile_spaces(spec, url=enterprise_url, password=passwords[SERVICE_ACCOUNT], extraction_backend=backend)
    assert again[0].completed == []


def test_members_cant_write_derived_data(enterprise_url, team, connect):
    _, passwords = team

    with pytest.raises(neo4j.exceptions.ClientError, match="CREATE permission"):
        connect("alice", passwords["alice"], "CREATE (:Chunk {hash: 'forged'})", database="space_payments")


def test_a_deleted_files_extraction_is_cleaned_up_on_the_next_pass(enterprise_url, join, team, admin, backend):
    spec, passwords = team
    _write_shared(
        enterprise_url, join, passwords, "alice", {"command": "create", "path": DOC, "file_text": "Ante uses uv."}
    )
    reconcile_spaces(spec, url=enterprise_url, password=passwords[SERVICE_ACCOUNT], extraction_backend=backend)
    _write_shared(enterprise_url, join, passwords, "bob", {"command": "delete", "path": DOC})
    assert _derived(admin) == {"chunks": 1, "uses": 1}

    (report,) = reconcile_spaces(
        spec, url=enterprise_url, password=passwords[SERVICE_ACCOUNT], extraction_backend=backend
    )

    assert report.pruned == 1
    assert _derived(admin) == {"chunks": 0, "uses": 0}
    # Only the deleted file's kept version remains next to the space's owner: deletes are recoverable.
    assert admin("MATCH (n) WHERE NOT n:User RETURN labels(n) AS l", "space_payments", None) == [
        {"l": ["MemoryVersion"]}
    ]


def test_a_wrong_password_is_reported_per_space(enterprise_url, team):
    spec, _ = team

    (report,) = reconcile_spaces(spec, url=enterprise_url, password="wrong")

    assert report.error is not None and report.completed == []


def test_the_cli_needs_the_service_password(monkeypatch, tmp_path, capsys):
    from context_graph_spaces.cli import SERVICE_PASSWORD_ENV, main

    monkeypatch.delenv(SERVICE_PASSWORD_ENV, raising=False)
    spec = tmp_path / "spaces.toml"
    spec.write_text('members = ["alice"]\n')

    assert main(["reconcile", str(spec)]) == 2
    assert SERVICE_PASSWORD_ENV in capsys.readouterr().err
