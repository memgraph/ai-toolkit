"""Lint: the design guidelines (docs/contract.md, 9.1), one test per code on the guide's wine examples."""

from pathlib import Path

import pytest

from hygm import (
    LINT_CODES,
    Conventions,
    HygmModel,
    ManualStrategy,
    NodeType,
    Ontology,
    OwlImportStrategy,
    Property,
    RelationType,
    lint_model,
)

VALID = Path(__file__).parent.parent / "conformance" / "model" / "valid"
FIXTURES = Path(__file__).parent / "fixtures"
TESTED: set[str] = set()


def _type(label: str, *parents: str, **fields) -> NodeType:
    return NodeType(label, f"a {label}", parents=parents, **fields)


def _found(code: str, model: HygmModel) -> list[str]:
    """The elements `model` has a `code` finding on."""
    TESTED.add(code)
    return [f.element for f in lint_model(model) if f.code == code]


def test_single_subtype():
    model = HygmModel(node_types=(_type("RedBurgundy"), _type("CotesDOr", "RedBurgundy")))

    assert _found("single_subtype", model) == ["node:RedBurgundy"]


def test_many_subtypes():
    subtypes = tuple(_type(f"Wine{n}", "Wine") for n in range(13))

    assert _found("many_subtypes", HygmModel(node_types=(_type("Wine"), *subtypes))) == ["node:Wine"]
    assert _found("many_subtypes", HygmModel(node_types=(_type("Wine"), *subtypes[:12]))) == []


def test_redundant_endpoint():
    model = HygmModel(
        node_types=(_type("Wine"), _type("RedWine", "Wine"), _type("Winery")),
        relation_types=(RelationType("maker", "who made it", ("Wine", "RedWine"), ("Winery",)),),
    )

    assert _found("redundant_endpoint", model) == ["relation:maker"]


def test_endpoint_covers_all_subtypes():
    model = HygmModel(
        node_types=(_type("Wine"), _type("RedWine", "Wine"), _type("WhiteWine", "Wine"), _type("Winery")),
        relation_types=(RelationType("maker", "who made it", ("RedWine", "WhiteWine"), ("Winery",)),),
    )

    assert _found("endpoint_covers_all_subtypes", model) == ["relation:maker"]


def test_property_on_every_subtype():
    body = Property("body", description="how full it feels")
    model = HygmModel(
        node_types=(
            _type("Wine"),
            _type("RedWine", "Wine", properties=(body,)),
            _type("WhiteWine", "Wine", properties=(body,)),
        )
    )

    assert _found("property_on_every_subtype", model) == ["node:Wine"]


def test_empty_subtype():
    model = HygmModel(
        node_types=(
            _type("Wine"),
            _type("RedWine", "Wine"),
            _type("WhiteWine", "Wine", properties=(Property("sweetness", description="how sweet"),)),
        )
    )

    assert _found("empty_subtype", model) == ["node:RedWine"]


def test_plural_of_parent():
    model = HygmModel(node_types=(_type("Wines"), _type("Wine", "Wines")))

    assert _found("plural_of_parent", model) == ["node:Wine"]


def test_naming_case_against_the_conventions_or_else_the_majority():
    names = (_type("RedWine"), _type("white_wine"), _type("Rose"))
    declared = HygmModel(
        node_types=names, ontology=Ontology(conventions=Conventions(node_case="snake", property_case="snake"))
    )

    assert _found("naming_case", HygmModel(node_types=names)) == ["node:white_wine"]
    assert _found("naming_case", declared) == ["node:RedWine", "node:Rose"]


def test_naming_number():
    model = HygmModel(node_types=(_type("Wine"), _type("Grapes"), _type("Winery")))

    assert _found("naming_number", model) == ["node:Grapes"]


def test_meta_word_in_name():
    model = HygmModel(
        node_types=(_type("WineClass", properties=(Property("wine_type", description="its kind"),)), _type("Winery")),
        relation_types=(RelationType("maker_relation", "who made it", ("WineClass",), ("Winery",)),),
    )

    assert _found("meta_word_in_name", model) == [
        "node:WineClass",
        "relation:maker_relation",
        "property:WineClass.wine_type",
    ]


def test_sibling_naming():
    model = HygmModel(node_types=(_type("Wine"), _type("RedWine", "Wine"), _type("White", "Wine")))

    assert _found("sibling_naming", model) == ["node:Wine"]


def test_missing_description_but_not_on_a_redeclaration_that_inherits_one():
    model = HygmModel(
        node_types=(
            NodeType("Wine", properties=(Property("sugar", description="how sweet"),)),
            _type("DessertWine", "Wine", properties=(Property("sugar", values=None, required=True),)),
        ),
        relation_types=(RelationType("pairs_with", properties=(Property("since"),)),),
    )

    assert _found("missing_description", model) == [
        "node:Wine",
        "relation:pairs_with",
        "relation_property:pairs_with.since",
    ]


def test_no_competency_questions():
    asked = Ontology(competency_questions=("Does Cabernet Sauvignon go well with seafood?",))

    assert _found("no_competency_questions", HygmModel(node_types=())) == ["ontology"]
    assert _found("no_competency_questions", HygmModel(node_types=(), ontology=asked)) == []


def test_unused_type_but_not_one_a_relation_admits_through_an_ancestor():
    model = HygmModel(
        node_types=(_type("Wine"), _type("RedWine", "Wine"), _type("Winery"), _type("Cellar")),
        relation_types=(RelationType("maker", "who made it", ("Wine",), ("Winery",)),),
    )

    assert _found("unused_type", model) == ["node:Cellar"]


def test_inverse_not_declared():
    maker = RelationType("maker", "who made it", ("Wine",), ("Winery",))
    produces = RelationType("produces", "what it made", ("Winery",), ("Wine",))
    declared = RelationType("produces", "what it made", ("Winery",), ("Wine",), inverse_of="maker")
    nodes = (_type("Wine"), _type("Winery"))

    assert _found("inverse_not_declared", HygmModel(node_types=nodes, relation_types=(maker, produces))) == [
        "relation:maker"
    ]
    assert _found("inverse_not_declared", HygmModel(node_types=nodes, relation_types=(maker, declared))) == []


def test_every_code_has_a_test():
    assert set(LINT_CODES) == TESTED


@pytest.mark.parametrize("path", sorted(VALID.iterdir()), ids=lambda p: p.name)
def test_every_valid_fixture_lints_without_a_crash(path):
    findings = lint_model(ManualStrategy().create_model(path))

    assert {f.code for f in findings} <= set(LINT_CODES)
    assert all(f.severity in ("warning", "info") for f in findings)


def test_the_wine_ontology_lints_with_only_what_the_guide_would_flag():
    findings = lint_model(OwlImportStrategy().create_model(FIXTURES / "wine.ttl"))

    assert {(f.code, f.element) for f in findings} == {
        ("single_subtype", "node:RedWine"),
        ("single_subtype", "node:Winery"),
        ("empty_subtype", "node:FrenchWinery"),
        ("empty_subtype", "node:RoseWine"),
        ("no_competency_questions", "ontology"),
    }


def test_a_model_that_fails_validation_still_lints():
    cycle = HygmModel(node_types=(_type("Wine", "Drink"), _type("Drink", "Wine")))

    assert isinstance(lint_model(cycle), list)
