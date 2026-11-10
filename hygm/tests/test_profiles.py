import pytest

from hygm import (
    HygmModel,
    ModelError,
    NodeType,
    RelationType,
    catch_all_labels,
    conversation,
    model_from_mapping,
    model_to_mapping,
    needs_version_2,
    protected_labels,
    validate_for_conversation,
    validate_model,
    value_labels,
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

    assert validate_model(model).success
    assert validate_for_conversation(model).success


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

    model = model_from_mapping(raw, "test")

    assert conversation(model) == ("User", "Person")
    assert value_labels(model) == ("Money",)
    assert protected_labels(model) == ("User", "Person", "Money")
    assert catch_all_labels(model) == ("Topic",)
    assert not needs_version_2(model)


def test_profiles_other_than_the_implied_ones_need_version_2():
    raw = {
        "version": 2,
        "node_types": [{"label": "Recipe", "description": "", "profiles": {"learning": {"catch_all": True}}}],
    }

    model = model_from_mapping(raw, "test")

    assert catch_all_labels(model) == ("Recipe",)
    assert model_to_mapping(model) == raw


@pytest.mark.parametrize(
    ("profiles", "level"),
    [
        ({"learning": {"protected": "yes"}}, "node"),
        ({"extraction": {"value": 1}}, "node"),
        ({"conversation": {"speaker": "the user"}}, "model"),
    ],
)
def test_a_bad_profile_value_is_refused(profiles, level):
    node = {"label": "A", "description": ""}
    raw = {"version": 2, "node_types": [{**node, "profiles": profiles} if level == "node" else node]}
    if level == "model":
        raw["profiles"] = profiles

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "test")

    assert raised.value.code == "invalid_profile_value"


def test_a_profile_setting_at_the_wrong_level_is_an_unknown_key():
    raw = {"version": 2, "node_types": [{"label": "A", "description": "", "profiles": {"conversation": {}}}]}
    raw["node_types"][0]["profiles"]["conversation"]["speaker"] = "A"

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "test")

    assert raised.value.code == "unknown_key"
