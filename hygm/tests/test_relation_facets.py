"""Relation facets: counts, inverses, symmetry and per-type restrictions (docs/contract.md, 4.16)."""

from dataclasses import replace
from pathlib import Path

import pytest

from hygm import (
    HygmModel,
    ManualStrategy,
    MergeRelationTypes,
    ModelError,
    NodeType,
    RelationRestriction,
    RelationType,
    RenameNodeType,
    RenameRelationType,
    apply,
    diff,
    model_from_mapping,
    model_to_mapping,
    validate_model,
)

WINES = Path(__file__).parent.parent / "conformance" / "model" / "valid" / "v2-relation-facets.yaml"


@pytest.fixture(scope="module")
def wines() -> HygmModel:
    return ManualStrategy().create_model(WINES)


def _codes(model: HygmModel) -> list[str | None]:
    return [issue.code for issue in validate_model(model).critical_issues]


def test_an_inverse_is_answered_from_either_side_and_a_symmetric_relation_is_its_own(wines):
    assert wines.inverse("produces") == "maker"
    assert wines.inverse("maker") == "produces"
    assert wines.inverse("pairs_with") == "pairs_with"
    assert wines.inverse("grape") is None
    assert wines.inverse("undeclared") is None


def test_restrictions_are_inherited_down_the_hierarchy(wines):
    maker = wines.relation_for("Bordeaux", "maker")
    grape = wines.relation_for("Bordeaux", "grape")

    assert (maker.end_labels, maker.min_count, maker.max_count) == (("FrenchWinery",), 1, 1)
    assert (grape.end_labels, grape.min_count, grape.max_count) == (("Grape",), 2, 3)
    assert wines.relation_for("FrenchWine", "grape").min_count == 1
    assert wines.relation_for("Wine", "maker") == wines.relation_type("maker")


def test_relation_for_is_none_for_a_type_that_cannot_start_the_relation(wines):
    assert wines.relation_for("Grape", "maker") is None
    assert wines.relation_for("Wine", "undeclared") is None


def test_restrictions_from_two_parents_combine_to_what_both_allow():
    model = HygmModel(
        node_types=(
            NodeType("Thing"),
            NodeType("Place", parents=("Thing",)),
            NodeType("City", parents=("Place",)),
            NodeType("Wine"),
            NodeType("Local", parents=("Wine",), relation_restrictions=(RelationRestriction("from_", ("Place",), 1),)),
            NodeType(
                "Urban",
                parents=("Wine",),
                relation_restrictions=(RelationRestriction("from_", ("City",), max_count=2),),
            ),
            NodeType("CityWine", parents=("Local", "Urban")),
        ),
        relation_types=(RelationType("from_", start_labels=("Wine",), end_labels=("Thing",)),),
    )

    combined = model.relation_for("CityWine", "from_")

    assert combined is not None
    assert (combined.end_labels, combined.min_count, combined.max_count) == (("City",), 1, 2)
    assert _codes(model) == []


def test_the_new_keys_round_trip_and_need_version_2(wines):
    written = model_to_mapping(wines)

    assert written["node_types"][4]["relation_restrictions"] == [
        {"relation": "maker", "end_labels": ["FrenchWinery"]},
        {"relation": "grape", "min_count": 1},
    ]
    assert written["relation_types"][0] == {
        "label": "maker",
        "description": "who made it",
        "start_labels": ["Wine"],
        "end_labels": ["Winery"],
        "min_count": 1,
        "max_count": 1,
    }
    assert model_from_mapping(written, "again") == wines
    for minimal in (
        RelationType("r", symmetric=True),
        RelationType("r", max_count=1),
        RelationType("r", inverse_of="r"),
    ):
        assert "version" in model_to_mapping(HygmModel(node_types=(NodeType("A"),), relation_types=(minimal,)))


def test_relation_counts_must_agree():
    with pytest.raises(ModelError) as raised:
        RelationType("r", min_count=2, max_count=1)
    assert raised.value.codes == ("count_conflict",)
    with pytest.raises(ModelError) as raised:
        RelationRestriction("r", min_count=-1)
    assert raised.value.codes == ("invalid_format",)


def test_an_inverse_naming_another_inverse_is_a_conflict():
    model = HygmModel(
        node_types=(NodeType("A"), NodeType("B")),
        relation_types=(
            RelationType("ab", start_labels=("A",), end_labels=("B",), inverse_of="ba"),
            RelationType("ba", start_labels=("B",), end_labels=("A",), inverse_of="ba2"),
            RelationType("ba2", start_labels=("B",), end_labels=("A",)),
        ),
    )

    assert set(_codes(model)) == {"inverse_conflict"}


def test_renames_and_merges_carry_inverses_and_restrictions(wines):
    renamed = apply(wines, [RenameRelationType("maker", "made_by"), RenameNodeType("FrenchWinery", "Domaine")])

    produces = renamed.relation_type("produces")
    french_wine = renamed.node_type("FrenchWine")
    assert produces is not None and produces.inverse_of == "made_by"
    assert french_wine is not None
    assert french_wine.relation_restrictions[0] == RelationRestriction("made_by", ("Domaine",))
    merged = apply(wines, [MergeRelationTypes("pairs_with", "grape")], validate=False)
    grape = merged.relation_type("grape")
    assert grape is not None and grape.symmetric is False


def test_diff_refuses_a_change_to_a_relation_facet_or_restriction(wines):
    no_inverse = replace(
        wines,
        relation_types=tuple(replace(r, inverse_of=None) if r.label == "produces" else r for r in wines.relation_types),
    )
    no_restriction = replace(
        wines,
        node_types=tuple(
            replace(t, relation_restrictions=()) if t.label == "Bordeaux" else t for t in wines.node_types
        ),
    )

    for changed, key in ((no_inverse, "inverse_of"), (no_restriction, "relation_restrictions")):
        with pytest.raises(ModelError, match=key) as raised:
            diff(wines, changed)
        assert raised.value.codes == ("unknown_operation",)
