import pytest

from hygm import (
    HygmModel,
    ModelError,
    NodeType,
    Property,
    RelationType,
    default_model,
    model_from_mapping,
    model_to_mapping,
    needs_version_2,
)


def test_a_model_version_1_can_express_is_written_exactly_as_before_the_contract():
    """Stored ontology versions in users' graphs must serialize byte for byte as they always have."""
    model = default_model()

    written = model_to_mapping(model)

    assert not needs_version_2(model)
    assert written == {
        "entity_types": [
            {"label": t.label, "description": t.description, "identity": t.identity} for t in model.node_types
        ],
        "relation_types": [
            {
                "label": r.label,
                "description": r.description,
                "start_labels": list(r.start_labels),
                "end_labels": list(r.end_labels),
            }
            for r in model.relation_types
        ],
    }


def test_version_2_is_written_only_when_needed_and_leaves_out_defaults():
    model = HygmModel(
        node_types=(
            NodeType("Party", "a party", identity="global"),
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

    author = model.node_types[0]
    assert author.identity == "span"
    assert author.profiles == {"x_me": {"a": 1}}


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
