"""Unit tests for GLiNER2Backend, run entirely against an injected fake model
-- no real `gliner2` install or model download needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from unstructured2graph import Chunk, EntityType, Ontology, RelationType
from unstructured2graph.gliner2_backend import GLiNER2Backend, _entity_id, _normalize_text

ENTITY_ONLY_ONTOLOGY = Ontology(entity_types=(EntityType(label="person", description="A human"),))

ENTITY_AND_RELATION_ONTOLOGY = Ontology(
    entity_types=(
        EntityType(label="person", description="A human"),
        EntityType(label="company", description="A business"),
    ),
    relation_types=(RelationType(label="works_for", description="Employment relationship"),),
)


class _FakeModel:
    """Records every call so tests can assert on what schema was passed, and
    returns whatever canned result the test configured."""

    def __init__(self, entities_result=None, relations_result=None):
        self.entities_result = entities_result or {"entities": {}}
        self.relations_result = relations_result or {"relation_extraction": {}}
        self.entity_calls: list[tuple[str, dict]] = []
        self.relation_calls: list[tuple[str, dict]] = []

    def extract_entities(self, text, schema, **kwargs):
        self.entity_calls.append((text, schema))
        return self.entities_result

    def extract_relations(self, text, schema, **kwargs):
        self.relation_calls.append((text, schema))
        return self.relations_result


def test_entity_schema_built_from_ontology():
    model = _FakeModel()
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model)

    assert backend._entity_schema == {"person": "A human"}
    assert backend._relation_schema == {}


def test_workspace_label_defaults_and_is_configurable():
    model = _FakeModel()
    assert GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model).workspace_label == "gliner2"
    assert GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model, workspace="custom").workspace_label == "custom"


def test_normalize_text_collapses_whitespace_and_case():
    assert _normalize_text("  Alice   Johnson\n") == "alice johnson"


def test_entity_id_is_deterministic_and_scoped_to_chunk_and_type():
    id_a = _entity_id("hash1", "person", "alice")
    id_b = _entity_id("hash1", "person", "alice")
    id_diff_chunk = _entity_id("hash2", "person", "alice")
    id_diff_type = _entity_id("hash1", "company", "alice")

    assert id_a == id_b
    assert id_a != id_diff_chunk
    assert id_a != id_diff_type


@pytest.mark.asyncio
async def test_entity_only_ontology_never_calls_extract_relations():
    model = _FakeModel(entities_result={"entities": {"person": [{"text": "Alice", "start": 0, "end": 5}]}})
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model)

    with patch("unstructured2graph.gliner2_backend.create_nodes_from_list") as mock_create_nodes:
        await backend.aingest_chunk(MagicMock(), Chunk(text="Alice works here.", hash="h1"))

    assert model.relation_calls == []
    mock_create_nodes.assert_called_once()


@pytest.mark.asyncio
async def test_entities_are_written_with_file_path_and_entity_type():
    model = _FakeModel(
        entities_result={
            "entities": {"person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.9}]},
        }
    )
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model)
    chunk = Chunk(text="Alice works here.", hash="h1")

    with patch("unstructured2graph.gliner2_backend.create_nodes_from_list") as mock_create_nodes:
        await backend.aingest_chunk(MagicMock(), chunk)

    mock_create_nodes.assert_called_once()
    _memgraph, node_dicts, workspace, _batch_size = mock_create_nodes.call_args.args
    kwargs = mock_create_nodes.call_args.kwargs
    assert workspace == "gliner2"
    assert kwargs["merge_key"] == "entity_id"
    assert len(node_dicts) == 1
    node = node_dicts[0]
    assert node["entity_type"] == "person"
    assert node["file_path"] == "h1"
    assert node["entity_id"] == _entity_id("h1", "person", "alice")


@pytest.mark.asyncio
async def test_entity_confidence_threshold_filters_low_confidence_entities():
    model = _FakeModel(
        entities_result={
            "entities": {
                "person": [
                    {"text": "Alice", "start": 0, "end": 5, "confidence": 0.95},
                    {"text": "Bob", "start": 10, "end": 13, "confidence": 0.2},
                ]
            }
        }
    )
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model, entity_confidence_threshold=0.5)
    chunk = Chunk(text="Alice and Bob work here.", hash="h1")

    with patch("unstructured2graph.gliner2_backend.create_nodes_from_list") as mock_create_nodes:
        await backend.aingest_chunk(MagicMock(), chunk)

    node_dicts = mock_create_nodes.call_args.args[1]
    assert [n["text"] for n in node_dicts] == ["Alice"]


@pytest.mark.asyncio
async def test_relations_are_matched_to_entities_by_span_and_written_as_typed_edges():
    model = _FakeModel(
        entities_result={
            "entities": {
                "person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.9}],
                "company": [{"text": "Acme Corp", "start": 15, "end": 24, "confidence": 0.9}],
            }
        },
        relations_result={
            "relation_extraction": {
                "works_for": [
                    {
                        "head": {"text": "Alice", "start": 0, "end": 5, "confidence": 0.9},
                        "tail": {"text": "Acme Corp", "start": 15, "end": 24, "confidence": 0.9},
                    }
                ]
            }
        },
    )
    backend = GLiNER2Backend(ontology=ENTITY_AND_RELATION_ONTOLOGY, model=model)
    chunk = Chunk(text="Alice works at Acme Corp.", hash="h1")

    with (
        patch("unstructured2graph.gliner2_backend.create_nodes_from_list"),
        patch("unstructured2graph.gliner2_backend.upsert_typed_relationships") as mock_upsert_rel,
    ):
        await backend.aingest_chunk(MagicMock(), chunk)

    mock_upsert_rel.assert_called_once()
    _memgraph, workspace, match_key, relationships_by_type = mock_upsert_rel.call_args.args
    assert workspace == "gliner2"
    assert match_key == "entity_id"
    assert list(relationships_by_type.keys()) == ["works_for"]
    [relationship] = relationships_by_type["works_for"]
    assert relationship["from"] == _entity_id("h1", "person", "alice")
    assert relationship["to"] == _entity_id("h1", "company", "acme corp")


@pytest.mark.asyncio
async def test_relation_with_unmatched_endpoint_is_skipped_and_logged(caplog):
    model = _FakeModel(
        entities_result={"entities": {"person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.9}]}},
        relations_result={
            "relation_extraction": {
                "works_for": [
                    {
                        "head": {"text": "Alice", "start": 0, "end": 5, "confidence": 0.9},
                        "tail": {"text": "Someone Else", "start": 99, "end": 111, "confidence": 0.9},
                    }
                ]
            }
        },
    )
    backend = GLiNER2Backend(ontology=ENTITY_AND_RELATION_ONTOLOGY, model=model)
    chunk = Chunk(text="Alice works at an unmentioned company.", hash="h1")

    with (
        patch("unstructured2graph.gliner2_backend.create_nodes_from_list"),
        patch("unstructured2graph.gliner2_backend.upsert_typed_relationships") as mock_upsert_rel,
    ):
        await backend.aingest_chunk(MagicMock(), chunk)

    mock_upsert_rel.assert_not_called()
    assert "Skipping 'works_for' relation" in caplog.text


@pytest.mark.asyncio
async def test_relation_confidence_threshold_filters_low_confidence_relations():
    model = _FakeModel(
        entities_result={
            "entities": {
                "person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.9}],
                "company": [{"text": "Acme Corp", "start": 15, "end": 24, "confidence": 0.9}],
            }
        },
        relations_result={
            "relation_extraction": {
                "works_for": [
                    {
                        "head": {"text": "Alice", "start": 0, "end": 5, "confidence": 0.9},
                        "tail": {"text": "Acme Corp", "start": 15, "end": 24, "confidence": 0.1},
                    }
                ]
            }
        },
    )
    backend = GLiNER2Backend(ontology=ENTITY_AND_RELATION_ONTOLOGY, model=model, relation_confidence_threshold=0.5)
    chunk = Chunk(text="Alice works at Acme Corp.", hash="h1")

    with (
        patch("unstructured2graph.gliner2_backend.create_nodes_from_list"),
        patch("unstructured2graph.gliner2_backend.upsert_typed_relationships") as mock_upsert_rel,
    ):
        await backend.aingest_chunk(MagicMock(), chunk)

    mock_upsert_rel.assert_not_called()
