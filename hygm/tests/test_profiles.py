"""The profile registry: hygm checks every registered profile's settings without its owner installed."""

import pytest

from hygm import OWNERS, REGISTERED_PROFILES, ModelError, model_from_mapping, model_to_mapping, needs_version_2


def test_every_registered_profile_has_an_owner():
    assert set(REGISTERED_PROFILES) == set(OWNERS) == {"extraction", "conversation", "learning"}


def test_a_profile_beyond_identity_needs_version_2_without_an_owners_rule():
    raw = {
        "version": 2,
        "node_types": [{"label": "Recipe", "description": "", "profiles": {"learning": {"catch_all": True}}}],
    }

    model = model_from_mapping(raw, "test")

    assert model.node_types[0].profiles == {"learning": {"catch_all": True}}
    assert needs_version_2(model)
    assert model_to_mapping(model) == raw


def test_a_version_1_file_implies_no_profile_without_an_owners_rule():
    raw = {"entity_types": [{"label": "User", "description": ""}, {"label": "Money", "description": ""}]}

    model = model_from_mapping(raw, "test")

    assert model.profiles == {}
    assert all(t.profiles == {} for t in model.node_types)
    assert not needs_version_2(model)


def test_an_unregistered_profile_is_refused():
    raw = {"version": 2, "node_types": [{"label": "A", "description": "", "profiles": {"extration": {}}}]}

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "test")

    assert raised.value.code == "unknown_profile"


def test_a_private_x_profile_is_kept_unchecked():
    raw = {
        "version": 2,
        "profiles": {"x_mytool": {"anything": [1, 2]}},
        "node_types": [{"label": "A", "description": "", "profiles": {"x_mytool": {"color": "blue"}}}],
    }

    model = model_from_mapping(raw, "test")

    assert model.profiles == {"x_mytool": {"anything": [1, 2]}}
    assert model.node_types[0].profiles == {"x_mytool": {"color": "blue"}}
    assert model_to_mapping(model) == raw


@pytest.mark.parametrize(
    ("profiles", "level"),
    [
        ({"learning": {"protected": "yes"}}, "node"),
        ({"extraction": {"value": 1}}, "node"),
        ({"extraction": {"identity": "session"}}, "node"),
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


def test_a_bad_version_1_identity_is_a_bad_profile_value():
    with pytest.raises(ModelError) as raised:
        model_from_mapping({"entity_types": [{"label": "A", "description": "", "identity": "session"}]}, "test")

    assert raised.value.code == "invalid_profile_value"


def test_a_profile_setting_at_the_wrong_level_is_an_unknown_key():
    raw = {"version": 2, "node_types": [{"label": "A", "description": "", "profiles": {"conversation": {}}}]}
    raw["node_types"][0]["profiles"]["conversation"]["speaker"] = "A"

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "test")

    assert raised.value.code == "unknown_key"
