from sessions_graph.default_model import default_model, with_core
from sessions_graph.learning import CATCH_ALL_LABELS, CORE_LABELS, VERSION_1

from hygm import HygmModel, NodeType, RelationType, model_to_mapping, needs_version_2, validate_model
from unstructured2graph.profile import VALUE_LABELS, identity_of


def test_the_default_passes_the_gate_and_holds_the_core_and_catch_alls():
    model = default_model()

    assert validate_model(model).success
    labels = model.node_labels()
    assert set(CORE_LABELS) <= set(labels)
    assert set(CATCH_ALL_LABELS) <= set(labels)
    assert {"Organization", "Location", "Event"} <= set(labels)


def test_every_value_type_has_a_relation_into_it_and_is_one_node_per_mention():
    model = default_model()

    identity = {t.label: identity_of(t) for t in model.node_types}
    for label in VALUE_LABELS:
        assert identity[label] == "span"
        assert any(label in relation.end_labels for relation in model.relation_types), label


def test_every_relation_is_constrained_on_both_ends():
    assert all(relation.start_labels and relation.end_labels for relation in default_model().relation_types)


def test_with_core_adds_the_missing_core_after_the_supplied_types():
    supplied = HygmModel(
        node_types=(
            NodeType("User", "me, as the schema says", profiles={"extraction": {"identity": "global"}}),
            NodeType("Library", "a code library"),
        ),
        relation_types=(RelationType("maintains", "", ("User", "Person"), ("Library",)),),
    )

    model = with_core(supplied)

    assert model.node_types[:2] == supplied.node_types
    assert model.node_labels()[2:] == tuple(label for label in CORE_LABELS if label != "User")
    assert model.relation_types == supplied.relation_types


def test_artifacts_merge_across_sessions_and_topics_stay_per_chunk():
    identity = {t.label: identity_of(t) for t in default_model().node_types}

    assert identity["Artifact"] == "global"
    assert identity["Topic"] == "chunk"


def test_the_default_is_written_as_version_1_under_the_version_1_rule():
    """Every stored ontology version starts from the default, so it must stay byte-stable as version 1."""
    model = default_model()

    assert not needs_version_2(model, VERSION_1)
    assert "entity_types" in model_to_mapping(model, version_1_profiles=VERSION_1)
