import pytest

from hygm import HygmModel, NodeType, RelationType, is_valid_identifier, require_valid_identifier


@pytest.mark.parametrize("value", ["Person", "_x", "works_for", "A1"])
def test_valid_identifiers(value):
    assert is_valid_identifier(value)
    assert require_valid_identifier(value, "label") == value


@pytest.mark.parametrize("value", ["", "1abc", "a b", "a`b", "a:b", "a-b"])
def test_invalid_identifiers_raise_naming_the_role(value):
    assert not is_valid_identifier(value)
    with pytest.raises(ValueError, match="workspace"):
        require_valid_identifier(value, "workspace")


def test_node_type_defaults_to_chunk_identity():
    assert NodeType("Topic").identity == "chunk"
    assert NodeType("Topic").description == ""


def test_node_type_rejects_unknown_identity_and_bad_label():
    with pytest.raises(ValueError, match="identity"):
        NodeType("Topic", identity="session")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="identifier"):
        NodeType("Bad Label")


def test_relation_type_coerces_endpoint_lists_to_tuples_and_rejects_a_bare_string():
    relation = RelationType("visited", start_labels=["User", "Person"])  # ty: ignore[invalid-argument-type]
    assert relation.start_labels == ("User", "Person")
    assert relation.end_labels == ()
    assert relation.constrained
    assert not RelationType("mentions").constrained
    with pytest.raises(TypeError, match="tuple"):
        RelationType("visited", end_labels="Location")  # ty: ignore[invalid-argument-type]


def test_endpoint_labels_resolves_unconstrained_to_every_declared_label():
    model = HygmModel(
        node_types=(NodeType("User"), NodeType("Person"), NodeType("Location")),
        relation_types=(RelationType("visited", start_labels=("User", "Person")),),
    )
    relation = model.relation_types[0]
    assert model.endpoint_labels(relation, "start") == ("User", "Person")
    assert model.endpoint_labels(relation, "end") == ("User", "Person", "Location")
    assert model.node_type("Location") == NodeType("Location")
    assert model.node_type("Missing") is None
