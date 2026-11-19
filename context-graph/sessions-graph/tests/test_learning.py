"""The `learning` profile sessions-graph owns, and the combined rule for reading stored version-1 ontologies."""

from sessions_graph.learning import VERSION_1, catch_all_labels, protected_labels, with_implied_profiles

from hygm import HygmModel, NodeType, model_from_mapping, model_to_mapping, needs_version_2
from unstructured2graph.profile import conversation, value_labels


def test_a_version_1_file_implies_the_profiles_that_were_hardcoded():
    raw = {
        "entity_types": [
            {"label": "User", "description": ""},
            {"label": "Person", "description": ""},
            {"label": "Money", "description": ""},
            {"label": "Topic", "description": ""},
            {"label": "Recipe", "description": ""},
        ]
    }

    model = model_from_mapping(raw, "test", version_1_profiles=VERSION_1)

    assert conversation(model) == ("User", "Person")
    assert value_labels(model) == ("Money",)
    assert protected_labels(model) == ("User", "Person", "Money")
    assert catch_all_labels(model) == ("Topic",)
    assert not needs_version_2(model, VERSION_1)


def test_a_stored_version_1_ontology_writes_back_exactly_as_it_was_read():
    raw = {
        "entity_types": [
            {"label": "User", "description": "the user", "identity": "global"},
            {"label": "Person", "description": "someone else", "identity": "global"},
            {"label": "Duration", "description": "a length of time", "identity": "span"},
            {"label": "Topic", "description": "a subject", "identity": "chunk"},
        ],
        "relation_types": [
            {"label": "best", "description": "", "start_labels": ["User", "Person"], "end_labels": ["Duration"]},
        ],
    }

    model = model_from_mapping(raw, "stored", version_1_profiles=VERSION_1)

    assert model.node_types[2].profiles == {
        "extraction": {"identity": "span", "value": True},
        "learning": {"protected": True},
    }
    assert model_to_mapping(model, version_1_profiles=VERSION_1) == raw


def test_profiles_other_than_the_implied_ones_need_version_2():
    raw = {
        "version": 2,
        "node_types": [{"label": "Recipe", "description": "", "profiles": {"learning": {"catch_all": True}}}],
    }

    model = model_from_mapping(raw, "test", version_1_profiles=VERSION_1)

    assert catch_all_labels(model) == ("Recipe",)
    assert needs_version_2(model, VERSION_1)
    assert model_to_mapping(model, version_1_profiles=VERSION_1) == raw


def test_a_models_own_learning_settings_win_over_what_version_1_implies():
    model = HygmModel(node_types=(NodeType("Person", profiles={"learning": {"protected": False}}), NodeType("Topic")))

    implied = with_implied_profiles(model)

    assert protected_labels(implied) == ()
    assert catch_all_labels(implied) == ("Topic",)
