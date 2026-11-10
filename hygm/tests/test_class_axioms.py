"""Abstract and disjoint types, synonyms and display names (docs/contract.md, 4.13)."""

from dataclasses import replace

import pytest

from hygm import (
    AddNodeType,
    HygmModel,
    MergeNodeTypes,
    ModelError,
    NodeType,
    Property,
    RelationType,
    RenameNodeType,
    apply,
    diff,
    model_from_mapping,
    model_to_mapping,
    needs_version_2,
    validate_model,
)


def _wines(*extra: NodeType) -> HygmModel:
    return HygmModel(
        node_types=(
            NodeType("Wine", abstract=True),
            NodeType("RedWine", parents=("Wine",), disjoint_with=("WhiteWine",)),
            NodeType("WhiteWine", parents=("Wine",)),
            NodeType("Fortified"),
            NodeType("Port", parents=("RedWine", "Fortified")),
            NodeType("Chablis", parents=("WhiteWine",)),
            NodeType("Rose", parents=("Wine",)),
            *extra,
        )
    )


def _codes(model: HygmModel) -> list[str | None]:
    return [issue.code for issue in validate_model(model).critical_issues]


def test_disjointness_is_symmetric_and_reaches_descendants_through_multiple_inheritance():
    model = _wines()

    assert model.disjoint("RedWine", "WhiteWine")
    assert model.disjoint("WhiteWine", "RedWine")
    assert model.disjoint("Port", "Chablis")
    assert model.disjoint("Chablis", "Port")
    assert not model.disjoint("Rose", "RedWine")
    assert not model.disjoint("Port", "Fortified")
    assert _codes(model) == []


def test_a_type_inheriting_from_two_disjoint_types_is_a_conflict():
    model = _wines(NodeType("Blush", parents=("Port", "Chablis")))

    assert _codes(model) == ["disjoint_conflict"]


@pytest.mark.parametrize("other", ["Wine", "RedWine"])
def test_a_type_disjoint_with_its_ancestor_or_itself_is_a_conflict(other):
    model = replace(
        _wines(), node_types=(*_wines().node_types[:1], NodeType("RedWine", parents=("Wine",), disjoint_with=(other,)))
    )

    assert _codes(model) == ["disjoint_conflict"]


def test_disjoint_with_an_undeclared_type_is_unknown_type():
    model = HygmModel(node_types=(NodeType("RedWine", disjoint_with=("WhiteWine",)),))

    assert _codes(model) == ["unknown_type"]


@pytest.mark.parametrize(
    "model",
    [
        HygmModel(node_types=(NodeType("Wine"), NodeType("Grape", synonyms=("WINE",)))),
        HygmModel(node_types=(NodeType("Winery", synonyms=("chateau",)), NodeType("Castle", synonyms=("Chateau",)))),
        HygmModel(
            node_types=(NodeType("Wine"),),
            relation_types=(RelationType("made_by"), RelationType("produced_by", synonyms=("Made_By",))),
        ),
        HygmModel(
            node_types=(NodeType("Wine", properties=(Property("colour"), Property("hue", synonyms=("colour",)))),)
        ),
    ],
    ids=["synonym-is-a-label", "shared-synonym", "relation", "property"],
)
def test_a_synonym_naming_something_else_is_a_duplicate_name(model):
    assert _codes(model) == ["duplicate_name"]


def test_a_synonym_may_repeat_its_own_label_and_labels_differing_in_case_are_not_synonyms():
    own = HygmModel(node_types=(NodeType("Wine", synonyms=("wine", "vino")), NodeType("Grape")))
    cased = HygmModel(node_types=(NodeType("Wine"), NodeType("WINE")))

    assert _codes(own) == []
    assert _codes(cased) == []


def test_names_take_language_tags_only():
    NodeType("Wine", names={"en": "Wine", "pt-BR": "Vinho", "zh-Hant-TW": "葡萄酒"})

    for tag in ("en_US", "e", "english language"):
        with pytest.raises(ModelError) as raised:
            NodeType("Wine", names={tag: "Wine"})
        assert raised.value.codes == ("invalid_format",)


def test_the_new_keys_round_trip_and_need_version_2():
    model = HygmModel(
        node_types=(
            NodeType("Wine", abstract=True, synonyms=("vino",), names={"fr": "Vin"}),
            NodeType(
                "RedWine",
                parents=("Wine",),
                disjoint_with=("WhiteWine",),
                properties=(Property("grape", synonyms=("varietal",), names={"fr": "Cépage"}),),
            ),
            NodeType("WhiteWine", parents=("Wine",)),
        ),
        relation_types=(RelationType("paired_with", synonyms=("goes_with",), names={"en": "pairs with"}),),
    )

    written = model_to_mapping(model)

    assert written["node_types"][0] == {
        "label": "Wine",
        "description": "",
        "abstract": True,
        "synonyms": ["vino"],
        "names": {"fr": "Vin"},
    }
    assert model_from_mapping(written, "again") == model
    for minimal in (NodeType("A", abstract=True), NodeType("A", names={"en": "a"}), NodeType("A", synonyms=("b",))):
        assert needs_version_2(HygmModel(node_types=(minimal,)))


def test_diff_refuses_a_changed_axiom_but_an_added_type_carries_its_axioms():
    a = _wines()
    b = replace(a, node_types=tuple(replace(t, abstract=False) if t.label == "Wine" else t for t in a.node_types))
    c = replace(a, node_types=(*a.node_types, NodeType("Tawny", parents=("Port",), synonyms=("aged_port",))))

    with pytest.raises(ModelError, match="abstract") as raised:
        diff(a, b)

    assert raised.value.codes == ("unknown_operation",)
    assert apply(a, diff(a, c)) == c


def test_renames_and_merges_carry_disjointness():
    model = _wines()

    renamed = apply(model, [RenameNodeType("WhiteWine", "BlancWine")])
    merged = apply(
        model, [MergeNodeTypes("Chablis", "WhiteWine"), MergeNodeTypes("WhiteWine", "RedWine")], validate=False
    )

    assert renamed.node_type("RedWine").disjoint_with == ("BlancWine",)
    assert merged.node_type("RedWine").disjoint_with == ()
    assert apply(model, [AddNodeType(NodeType("Sherry", abstract=True))]).node_type("Sherry").abstract
