"""Property facets and narrowing them in subtypes (docs/contract.md, 4.14 and 4.15)."""

from pathlib import Path

import pytest

from hygm import (
    HygmModel,
    ManualStrategy,
    Mapping,
    ModelError,
    NodeBinding,
    NodeType,
    Property,
    PropertyBinding,
    Source,
    Statement,
    TableRef,
    check_mapping,
    memgraph_plan,
    model_from_mapping,
    model_to_mapping,
    validate_model,
)

VALID = Path(__file__).parent.parent / "conformance" / "model" / "valid"


def _codes(model: HygmModel) -> list[str | None]:
    return [issue.code for issue in validate_model(model).critical_issues]


def _raises(code: str, **facets) -> None:
    with pytest.raises(ModelError) as raised:
        Property("p", **facets)
    assert raised.value.codes == (code,)


def test_required_and_min_count_are_one_fact_with_one_spelling():
    assert Property("p", required=True) == Property("p", min_count=1)
    assert Property("p", min_count=0) == Property("p")
    assert Property("p", min_count=3, max_count=None).required
    assert Property("p", min_count=3, max_count=None).least_count == 3
    assert Property("p", required=True).least_count == 1


def test_counts_must_agree():
    _raises("count_conflict", required=True, min_count=0)
    _raises("count_conflict", min_count=3, max_count=2)
    _raises("count_conflict", required=True, max_count=0)
    _raises("invalid_format", min_count=-1)
    _raises("invalid_format", max_count=True)


def test_a_property_is_multi_valued_above_one_or_without_a_limit():
    assert Property("p", max_count=None).multi_valued
    assert Property("p", max_count=2).multi_valued
    assert not Property("p").multi_valued
    assert not Property("p", max_count=0).multi_valued


@pytest.mark.parametrize(
    "facets",
    [
        {"values": ("a", "b"), "default": "c"},
        {"type": "integer", "value": "750"},
        {"type": "integer", "value": True},
        {"type": "float", "minimum": 5, "default": 4.5},
        {"type": "float", "maximum": 5, "value": 5.5},
        {"type": "integer", "minimum": 10, "maximum": 1},
        {"minimum": "a"},
        {"type": "boolean", "maximum": True},
        {"type": "date", "default": "30/06/2020"},
        {"type": "date", "minimum": "2020-01-01", "default": "2019-12-31"},
        {"type": "datetime", "minimum": "2020-01-01T00:00:00Z", "maximum": "2020-01-01T00:00:00"},
        {"max_count": None, "values": ("a",), "default": ("a", "b")},
        {"max_count": None, "default": "a"},
        {"default": ["a"]},
    ],
)
def test_a_default_value_or_bound_the_facets_do_not_allow_is_an_invalid_value(facets):
    _raises("invalid_value", **facets)


def test_valid_facets_and_iso_8601_dates():
    Property("at", type="datetime", minimum="2020-01-01T00:00:00Z", default="2024-05-01T12:30:00+02:00")
    Property("on", type="date", minimum="2020-01-01", maximum="2030-12-31", value="2024-05-01")
    Property("n", type="decimal", minimum=0, maximum=1.5, default=1, unit="EUR")
    Property("doc", type="json", default={"a": [1, 2]})


def test_a_multi_valued_default_is_a_list_within_the_counts():
    prop = Property("grapes", max_count=3, values=("merlot", "syrah"), default=["merlot", "syrah"])

    assert prop.default == ("merlot", "syrah")
    assert model_to_mapping(HygmModel(node_types=(NodeType("Wine", properties=(prop,)),)))["node_types"][0][
        "properties"
    ] == [{"name": "grapes", "max_count": 3, "values": ["merlot", "syrah"], "default": ["merlot", "syrah"]}]
    _raises("invalid_value", max_count=3, default=["a", "b", "c", "d"])


def _wine(*subtypes: NodeType, **facets) -> HygmModel:
    return HygmModel(node_types=(NodeType("Wine", properties=(Property("p", **facets),)), *subtypes))


def _sub(**facets) -> NodeType:
    return NodeType("Sub", parents=("Wine",), properties=(Property("p", **facets),))


@pytest.mark.parametrize(
    ("inherited", "declared"),
    [
        ({"values": ("a", "b", "c")}, {"values": ("a", "b")}),
        ({"values": ("a", "b")}, {"value": "a"}),
        ({}, {"required": True}),
        ({"max_count": None}, {"min_count": 2, "max_count": 4}),
        ({"max_count": None}, {"max_count": 0}),
        ({"type": "integer", "minimum": 0, "maximum": 10}, {"type": "integer", "minimum": 2, "maximum": 8}),
        ({"type": "integer", "default": 1}, {"type": "integer", "default": 2, "unit": "bottles"}),
        ({"required": True}, {}),
        ({"type": "date", "maximum": "2030-01-01"}, {"type": "date", "maximum": "2029-12-31"}),
    ],
)
def test_a_redeclaration_may_narrow(inherited, declared):
    assert _codes(_wine(_sub(**declared), **inherited)) == []


@pytest.mark.parametrize(
    ("inherited", "declared"),
    [
        ({"type": "integer"}, {}),
        ({"values": ("a", "b")}, {"values": ("a", "c")}),
        ({"min_count": 3, "max_count": None}, {"min_count": 2, "max_count": None}),
        ({"max_count": 3}, {"max_count": 4}),
        ({"max_count": 3}, {"max_count": None}),
        ({"max_count": None}, {}),
        ({"type": "integer", "minimum": 5}, {"type": "integer", "minimum": 4}),
        ({"type": "integer", "maximum": 5}, {"type": "integer", "maximum": 6}),
        ({"value": "a"}, {"value": "b"}),
        ({"required": True}, {"max_count": 0}),
        ({"value": "a"}, {"values": ("b",)}),
    ],
)
def test_a_redeclaration_that_widens_or_changes_the_type_is_a_property_conflict(inherited, declared):
    assert _codes(_wine(_sub(**declared), **inherited)) == ["property_conflict"]


def test_a_redeclaration_takes_what_it_leaves_unstated_from_what_it_inherits():
    model = _wine(_sub(type="integer", max_count=0), type="integer", minimum=0, description="tannin, 0 to 10")

    (tannin,) = model.properties_of("Sub")

    assert (tannin.type, tannin.minimum, tannin.max_count, tannin.description) == ("integer", 0, 0, "tannin, 0 to 10")
    assert model.inherited("Sub", "p") == model.node_types[0].properties[0]
    assert model.inherited("Wine", "p") is None


def test_doubly_inherited_facets_merge_to_the_narrowest_both_allow():
    model = HygmModel(
        node_types=(
            NodeType("A", properties=(Property("tags", max_count=None, values=("x", "y", "z")),)),
            NodeType("B", properties=(Property("tags", max_count=4, values=("y", "z", "w"), required=True),)),
            NodeType("C", parents=("A", "B")),
        )
    )

    (tags,) = model.properties_of("C")

    assert (tags.values, tags.max_count, tags.required) == (("y", "z"), 4, True)
    assert _codes(model) == []


def test_doubly_inherited_facets_with_no_common_value_are_a_property_conflict():
    model = HygmModel(
        node_types=(
            NodeType("A", properties=(Property("n", type="integer", maximum=3),)),
            NodeType("B", properties=(Property("n", type="integer", minimum=5),)),
            NodeType("C", parents=("A", "B")),
        )
    )

    assert _codes(model) == ["property_conflict"]


def test_a_multi_valued_key_is_a_count_conflict():
    model = HygmModel(
        node_types=(NodeType("A", key=("ids",), properties=(Property("ids", required=True, max_count=None),)),)
    )

    assert _codes(model) == ["count_conflict"]


def test_a_file_redeclaration_inherits_an_unstated_type_and_max_count_and_is_written_back_the_same_way():
    raw = {
        "version": 2,
        "node_types": [
            {
                "label": "Sub",
                "description": "",
                "parents": ["Wine"],
                "properties": [{"name": "tannin", "max_count": 0}],
            },
            {"label": "Wine", "description": "", "properties": [{"name": "tannin", "type": "integer"}]},
        ],
    }

    model = model_from_mapping(raw, "test")

    assert model.node_type("Sub").properties[0] == Property("tannin", type="integer", max_count=0)
    assert model_to_mapping(model) == raw


def test_a_file_cannot_make_an_inherited_required_property_optional():
    raw = {
        "version": 2,
        "node_types": [
            {"label": "Wine", "description": "", "properties": [{"name": "name", "required": True}]},
            {"label": "Sub", "description": "", "parents": ["Wine"], "properties": [{"name": "name", "min_count": 0}]},
        ],
    }

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "test")

    assert raised.value.codes == ("property_conflict",)


def test_every_valid_fixture_writes_back_unchanged_in_meaning():
    model = ManualStrategy().create_model(VALID / "v2-narrowing.yaml")

    assert model_from_mapping(model_to_mapping(model), "again") == model
    assert model.properties_of("DessertWine")[0].value == "sweet"
    assert model.properties_of("DessertWine")[0].description == "How sweet it tastes"


def test_the_plan_types_a_multi_valued_property_as_a_list_and_nothing_else():
    model = HygmModel(
        node_types=(
            NodeType(
                "Wine",
                properties=(
                    Property("grapes", max_count=None, values=("merlot", "syrah"), min_count=2),
                    Property("awards", type="integer", max_count=3),
                    Property("alcohol", type="float", minimum=5, maximum=22, value=13.5),
                ),
            ),
            NodeType("Blend", parents=("Wine",), properties=(Property("grapes", max_count=4),)),
        )
    )

    assert memgraph_plan(model).statements == (
        Statement("exists", "Wine", ("grapes",)),
        Statement("enum", "Wine_grapes", values=("merlot", "syrah")),
        Statement("typed", "Wine", ("grapes",), data_type="LIST"),
        Statement("typed", "Wine", ("awards",), data_type="LIST"),
        Statement("typed", "Wine", ("alcohol",), data_type="FLOAT"),
    )


def test_a_mapping_may_bind_a_multi_valued_property():
    model = HygmModel(node_types=(NodeType("Wine", properties=(Property("grapes", max_count=None),)),))
    mapping = Mapping(
        model="wine.model.yaml",
        sources=(Source("cellar", "table"),),
        nodes=(
            NodeBinding(
                type="Wine",
                source="cellar",
                table=TableRef("wines"),
                id=("id",),
                properties=(PropertyBinding("grapes", "grapes"),),
            ),
        ),
    )

    check_mapping(mapping, model)
