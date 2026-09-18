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


class _FakeSchema:
    """Mirrors the real gliner2 schema builder's chainable .entities()/
    .relations(), recording what was passed so tests can assert on it."""

    def __init__(self):
        self.entity_schema = None
        self.relation_schema = None

    def entities(self, schema):
        self.entity_schema = schema
        return self

    def relations(self, schema):
        self.relation_schema = schema
        return self


class _FakeModel:
    """Records every call so tests can assert on what schema was passed, and
    returns whatever canned result the test configured -- merged into the
    single {"entities": ..., "relation_extraction": ...} shape the real
    model.extract_long() returns from one combined call (gliner2_backend.py's
    _extract_sync no longer calls extract_entities()/extract_relations()
    separately, and uses extract_long() rather than plain extract() -- see
    #336: extract() silently undercounts entities on text longer than
    GLiNER2's effective context)."""

    def __init__(self, entities_result=None, relations_result=None):
        self.result: dict = dict(entities_result or {"entities": {}})
        self.result.update(relations_result or {})
        self.extract_calls: list[tuple[str, _FakeSchema]] = []

    def create_schema(self):
        return _FakeSchema()

    def extract_long(self, text, schema, **kwargs):
        self.extract_calls.append((text, schema))
        return self.result


def test_entity_schema_built_from_ontology():
    model = _FakeModel()
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model)

    assert backend._entity_schema == {"person": "A human"}
    assert backend._relation_schema == {}


def test_chunk_size_and_overlap_default_and_are_configurable():
    default_backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=_FakeModel())
    assert default_backend._chunk_size == 384
    assert default_backend._chunk_overlap == 64

    custom_backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=_FakeModel(), chunk_size=128, chunk_overlap=32)
    assert custom_backend._chunk_size == 128
    assert custom_backend._chunk_overlap == 32


@pytest.mark.asyncio
async def test_extract_long_is_called_with_the_configured_chunk_size_and_overlap():
    """Regression test for #336: _extract_sync must call extract_long(),
    not extract() -- extract() silently undercounts entities on text longer
    than GLiNER2's effective context, with no error to catch the regression
    otherwise."""
    model = MagicMock()
    model.create_schema.return_value = _FakeSchema()
    model.extract_long.return_value = {"entities": {}}
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model, chunk_size=128, chunk_overlap=32)

    with patch("unstructured2graph.gliner2_backend.create_nodes_from_list"):
        await backend.aingest_chunk(MagicMock(), Chunk(text="Alice works here.", hash="h1"))

    model.extract_long.assert_called_once()
    model.extract.assert_not_called()
    _text, _schema = model.extract_long.call_args.args
    kwargs = model.extract_long.call_args.kwargs
    assert kwargs["chunk_size"] == 128
    assert kwargs["chunk_overlap"] == 32
    assert kwargs["include_spans"] is True
    assert kwargs["include_confidence"] is True


def test_workspace_label_defaults_and_is_configurable():
    model = _FakeModel()
    assert GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model).workspace_label == "gliner2"
    assert GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model, workspace="custom").workspace_label == "custom"


def test_invalid_workspace_raises():
    model = _FakeModel()
    with pytest.raises(ValueError, match="Invalid workspace"):
        GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model, workspace="not a valid label!")


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
async def test_entity_only_ontology_never_requests_relations_schema():
    model = _FakeModel(entities_result={"entities": {"person": [{"text": "Alice", "start": 0, "end": 5}]}})
    backend = GLiNER2Backend(ontology=ENTITY_ONLY_ONTOLOGY, model=model)

    with patch("unstructured2graph.gliner2_backend.create_nodes_from_list") as mock_create_nodes:
        await backend.aingest_chunk(MagicMock(), Chunk(text="Alice works here.", hash="h1"))

    assert len(model.extract_calls) == 1
    _text, schema = model.extract_calls[0]
    assert schema.entity_schema == {"person": "A human"}
    assert schema.relation_schema is None
    mock_create_nodes.assert_called_once()


@pytest.mark.asyncio
async def test_entity_and_relation_ontology_requests_combined_schema_in_one_call():
    """The whole point of the joint pass: one model.extract_long() call
    carries both schemas, not two separate extract_entities()/
    extract_relations() calls."""
    model = _FakeModel()
    backend = GLiNER2Backend(ontology=ENTITY_AND_RELATION_ONTOLOGY, model=model)

    with patch("unstructured2graph.gliner2_backend.create_nodes_from_list"):
        await backend.aingest_chunk(MagicMock(), Chunk(text="Alice works at Acme Corp.", hash="h1"))

    assert len(model.extract_calls) == 1
    _text, schema = model.extract_calls[0]
    assert schema.entity_schema == {"person": "A human", "company": "A business"}
    assert schema.relation_schema == {"works_for": "Employment relationship"}


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
