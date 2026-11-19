import json
from dataclasses import replace
from pathlib import Path

import pytest

from hygm import (
    HygmModel,
    ImpliedProfiles,
    ModelError,
    NodeType,
    Property,
    RelationType,
    model_from_mapping,
    model_to_mapping,
    needs_version_2,
)

VALID_MODELS = Path(__file__).parent.parent / "conformance" / "model" / "valid"


def test_a_model_version_1_can_express_is_written_exactly_as_before_the_contract():
    """Stored ontology versions in users' graphs must serialize byte for byte as they always have."""
    raw = json.loads((VALID_MODELS / "v1-as.json").read_text(encoding="utf-8"))
    model = model_from_mapping(raw, "v1-as.json")

    assert not needs_version_2(model)
    assert model_to_mapping(model) == raw


def _protects_things(model: HygmModel) -> ImpliedProfiles:
    """An owner's version-1 rule: `Thing` is protected, and the model has a speaker."""
    nodes = {label: ({"learning": {"protected": True}} if label == "Thing" else {}) for label in model.node_labels()}
    return {"conversation": {"speaker": "Thing"}}, nodes


def test_the_version_1_hook_is_applied_to_version_1_files_only():
    raw = json.loads((VALID_MODELS / "v1-as.json").read_text(encoding="utf-8"))
    version_2 = {
        "version": 2,
        "node_types": [
            {"label": "Thing", "description": "anything", "profiles": {"extraction": {"identity": "global"}}}
        ],
        "relation_types": [{"label": "related", "description": "", "start_labels": ["Thing"]}],
    }

    hooked = model_from_mapping(raw, "v1", version_1_profiles=_protects_things)

    assert hooked.profiles == {"conversation": {"speaker": "Thing"}}
    assert hooked.node_types[0].profiles == {"extraction": {"identity": "global"}, "learning": {"protected": True}}
    assert model_from_mapping(version_2, "v2", version_1_profiles=_protects_things) == model_from_mapping(raw, "v1")


def test_the_writer_keeps_version_1_exactly_when_the_profiles_are_the_hooks():
    raw = json.loads((VALID_MODELS / "v1-as.json").read_text(encoding="utf-8"))
    hooked = model_from_mapping(raw, "v1", version_1_profiles=_protects_things)

    assert model_to_mapping(hooked, version_1_profiles=_protects_things) == raw
    assert model_to_mapping(hooked)["version"] == 2

    thing = hooked.node_types[0]
    more = replace(hooked, node_types=(replace(thing, profiles={**thing.profiles, "x_note": {"a": 1}}),))
    fewer = replace(hooked, profiles={})
    assert model_to_mapping(more, version_1_profiles=_protects_things)["version"] == 2
    assert model_to_mapping(fewer, version_1_profiles=_protects_things)["version"] == 2

    plain = model_from_mapping(raw, "v1")
    assert model_to_mapping(plain) == raw
    assert model_to_mapping(plain, version_1_profiles=_protects_things)["version"] == 2


def test_version_2_is_written_only_when_needed_and_leaves_out_defaults():
    model = HygmModel(
        node_types=(
            NodeType("Party", "a party", profiles={"extraction": {"identity": "global"}}),
            NodeType("Customer", "a customer", parents=("Party",), properties=(Property("tier"),)),
        ),
        relation_types=(RelationType("knows", ""),),
    )

    assert model_to_mapping(model) == {
        "version": 2,
        "node_types": [
            {"label": "Party", "description": "a party", "profiles": {"extraction": {"identity": "global"}}},
            {"label": "Customer", "description": "a customer", "parents": ["Party"], "properties": [{"name": "tier"}]},
        ],
        "relation_types": [{"label": "knows", "description": ""}],
    }
    assert model_from_mapping(model_to_mapping(model), "round trip") == model


def test_identity_in_version_2_lives_in_the_extraction_profile():
    model = model_from_mapping(
        {
            "version": 2,
            "node_types": [
                {
                    "label": "Author",
                    "description": "",
                    "profiles": {"extraction": {"identity": "span"}, "x_me": {"a": 1}},
                }
            ],
        },
        "test",
    )

    assert model.node_types[0].profiles == {"extraction": {"identity": "span"}, "x_me": {"a": 1}}


@pytest.mark.parametrize(
    "spelled",
    [
        {"entity_types": [{"label": "Topic", "description": "", "identity": "chunk"}]},
        {
            "version": 2,
            "node_types": [{"label": "Topic", "description": "", "profiles": {"extraction": {"identity": "chunk"}}}],
        },
    ],
)
def test_a_chunk_identity_spelled_out_reads_as_the_default(spelled):
    unspelled = model_from_mapping({"entity_types": [{"label": "Topic", "description": ""}]}, "default")

    assert model_from_mapping(spelled, "spelled") == unspelled
    assert unspelled.node_types[0].profiles == {}


def test_an_unknown_key_is_an_error_naming_it():
    with pytest.raises(ModelError, match="'parent'") as raised:
        model_from_mapping({"entity_types": [{"label": "A", "description": "", "parent": "B"}]}, "test")

    assert raised.value.code == "unknown_key"


def test_version_2_keys_in_a_version_1_file_point_at_the_version():
    with pytest.raises(ModelError, match="version: 2"):
        model_from_mapping({"node_types": []}, "test")


def test_a_gate_failure_reports_every_code():
    raw = {
        "version": 2,
        "node_types": [
            {"label": "A", "description": "", "parents": ["Missing"]},
            {"label": "A", "description": ""},
        ],
    }

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "test")

    assert set(raised.value.codes) == {"duplicate_label", "unknown_parent"}


def test_an_unvalidated_store_skips_the_gate_but_not_the_format():
    pool = {
        "version": 2,
        "node_types": [],
        "relation_types": [{"label": "paid", "description": "", "end_labels": ["X"]}],
    }

    assert model_from_mapping(pool, "pool", validate=False).relation_labels() == ("paid",)
    with pytest.raises(ModelError, match="unknown key"):
        model_from_mapping({**pool, "extra": 1}, "pool", validate=False)
