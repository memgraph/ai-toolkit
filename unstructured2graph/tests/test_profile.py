"""The `extraction` and `conversation` profiles unstructured2graph owns: what their settings mean."""

from hygm import (
    HygmModel,
    NodeType,
    RelationType,
    model_from_mapping,
    model_to_mapping,
    needs_version_2,
    validate_model,
)
from unstructured2graph.profile import (
    conversation,
    identity_of,
    implied_profiles,
    validate_for_conversation,
    value_labels,
    with_implied_profiles,
)

CONVERSATIONAL = {"conversation": {"speaker": "User", "other_person": "Person"}}


def _model(*relations, labels=("User", "Person", "Location"), profiles=CONVERSATIONAL):
    return HygmModel(node_types=tuple(NodeType(label) for label in labels), relation_types=relations, profiles=profiles)


def test_a_speaker_endpoint_without_the_other_person_breaks_the_conversation_rule():
    model = _model(RelationType("visited", start_labels=("User",), end_labels=("Location",)))

    result = validate_for_conversation(model)

    assert not result.success
    assert "'User' but not 'Person'" in result.critical_issues[0].message
    assert result.critical_issues[0].code == "profile_rule"


def test_an_unconstrained_endpoint_admits_every_label_for_the_conversation_rule():
    assert validate_for_conversation(_model(RelationType("mentions"))).success
    result = validate_for_conversation(_model(RelationType("mentions"), labels=("User", "Location")))
    assert len(result.critical_issues) == 2  # both sides resolve to {User, Location}


def test_the_rule_follows_the_profiles_roles_not_fixed_labels():
    profiles = {"conversation": {"speaker": "Caller", "other_person": "Contact"}}
    model = _model(RelationType("called", start_labels=("Caller",)), labels=("Caller", "Contact"), profiles=profiles)

    assert conversation(model) == ("Caller", "Contact")
    assert not validate_for_conversation(model).success


def test_without_a_conversation_profile_user_and_no_person_is_a_valid_model():
    model = _model(RelationType("visited", start_labels=("User",)), labels=("User", "Location"), profiles={})

    assert conversation(model) is None
    assert validate_model(model).success
    assert validate_for_conversation(model).success


def test_identity_is_the_extraction_profiles_setting_and_defaults_to_chunk():
    assert identity_of(NodeType("Topic")) == "chunk"
    assert identity_of(NodeType("Book", profiles={"extraction": {"identity": "global"}})) == "global"
    assert identity_of(NodeType("Year", profiles={"extraction": {"identity": "span", "value": True}})) == "span"


def test_a_version_1_file_implies_the_values_and_conversation_roles_that_were_hardcoded():
    raw = {
        "entity_types": [
            {"label": "User", "description": "", "identity": "global"},
            {"label": "Person", "description": ""},
            {"label": "Money", "description": "", "identity": "span"},
            {"label": "Topic", "description": ""},
        ]
    }

    model = model_from_mapping(raw, "test", version_1_profiles=implied_profiles)

    assert conversation(model) == ("User", "Person")
    assert value_labels(model) == ("Money",)
    assert identity_of(model.node_types[2]) == "span"
    assert not needs_version_2(model, implied_profiles)
    written = model_to_mapping(model, version_1_profiles=implied_profiles)
    assert model_from_mapping(written, "written", version_1_profiles=implied_profiles) == model


def test_without_a_user_type_a_version_1_file_implies_no_conversation():
    raw = {"entity_types": [{"label": "Topic", "description": ""}]}

    model = model_from_mapping(raw, "test", version_1_profiles=implied_profiles)

    assert conversation(model) is None
    assert value_labels(model) == ()


def test_a_models_own_settings_win_over_what_version_1_implies():
    model = HygmModel(
        node_types=(NodeType("User"), NodeType("Caller"), NodeType("Money", profiles={"extraction": {"value": False}})),
        profiles={"conversation": {"other_person": "Caller"}},
    )

    implied = with_implied_profiles(model)

    assert conversation(implied) == ("User", "Caller")
    assert value_labels(implied) == ()
