"""Individuals and seeding them (docs/contract.md, 4.17 and 7.1). tests/test_plan_e2e.py seeds a real Memgraph."""

from pathlib import Path

import pytest

from hygm import (
    HygmModel,
    Individual,
    ManualStrategy,
    ModelError,
    NodeType,
    Property,
    RelationType,
    RenameNodeType,
    RenameProperty,
    RenameRelationType,
    SeedStatement,
    apply,
    model_to_mapping,
    seed_statements,
    validate_model,
)

VALID = Path(__file__).parent.parent / "conformance" / "model" / "valid"


def _codes(model: HygmModel) -> list[str | None]:
    return [issue.code for issue in validate_model(model).critical_issues]


def _bottles(*individuals: Individual, **facets) -> HygmModel:
    return HygmModel(
        node_types=(NodeType("Bottle", properties=(Property("ml", type="integer", **facets),)),),
        individuals=individuals,
    )


def test_a_fixed_value_or_a_default_fills_a_required_property():
    assert _codes(_bottles(Individual("b", "Bottle"), required=True, value=750)) == []
    assert _codes(_bottles(Individual("b", "Bottle"), required=True, default=750)) == []
    assert _codes(_bottles(Individual("b", "Bottle"), required=True)) == ["missing_value"]


def test_a_null_value_and_a_forbidden_property_are_refused():
    assert _codes(_bottles(Individual("b", "Bottle", properties={"ml": None}))) == ["invalid_value"]
    assert _codes(_bottles(Individual("b", "Bottle", properties={"ml": 750}), max_count=0)) == ["count_conflict"]


def test_inverse_and_symmetric_relationships_count_without_being_repeated():
    model = ManualStrategy().create_model(VALID / "v2-individuals.yaml")

    assert _codes(model) == []
    assert model.individual("margaux_2015").relations == {"pairs_with": ("pavillon_2015",)}


def test_imported_individuals_are_targets_and_are_written_back_as_imports():
    model = ManualStrategy().create_model(VALID / "v2-individuals-imported.yaml")

    written = model_to_mapping(model)

    assert [i.id for i in model.individuals] == ["bordeaux", "burgundy", "chateau_margaux"]
    assert [i["id"] for i in written["individuals"]] == ["chateau_margaux"]


def test_renames_carry_through_individuals():
    model = ManualStrategy().create_model(VALID / "v2-individuals.yaml")

    renamed = apply(
        model,
        [
            RenameNodeType("Winery", "Estate"),
            RenameRelationType("located_in", "lies_in"),
            RenameProperty("node", "Wine", "alcohol", "abv"),
        ],
    )

    assert renamed.individual("chateau_margaux").type == "Estate"
    assert "lies_in" in renamed.individual("chateau_margaux").relations
    assert renamed.individual("margaux_2015").properties["abv"] == 13.5


def test_an_id_and_a_relation_must_be_identifiers():
    with pytest.raises(ModelError) as raised:
        Individual("chateau margaux", "Winery")
    assert raised.value.codes == ("invalid_identifier",)
    with pytest.raises(ModelError):
        Individual("m", "Winery", relations={"made by": ("x",)})


def test_the_seed_merges_by_key_or_id_and_passes_every_value_as_a_parameter():
    model = ManualStrategy().create_model(VALID / "v2-individuals.yaml")

    seed = seed_statements(model)

    assert seed[0] == SeedStatement(
        "MERGE (n:`WineRegion` {`name`: $k0}) SET n.`name` = $v0", {"k0": "Bordeaux", "v0": "Bordeaux"}
    )
    margaux = seed[2]
    assert margaux.cypher.startswith("MERGE (n:`RedWine` {`id`: $k0}) SET n:`Wine`, ")
    assert margaux.parameters["k0"] == "margaux_2015"
    assert "Château" not in margaux.cypher and "13.5" not in margaux.cypher
    assert "n.`colour` = ToEnum($e" in margaux.cypher
    assert "n.`bottle_ml` = $v" in margaux.cypher
    assert seed[-1] == SeedStatement(
        "MATCH (a:`RedWine` {`id`: $a0}), (b:`Winery` {`id`: $b0}) MERGE (a)-[:`maker`]->(b)",
        {"a0": "pavillon_2015", "b0": "chateau_margaux"},
    )


def test_the_seed_refuses_an_invalid_model_and_an_id_property_that_disagrees():
    with pytest.raises(ModelError):
        seed_statements(_bottles(Individual("b", "Bottle"), required=True))
    model = HygmModel(
        node_types=(NodeType("A", properties=(Property("id"),)),),
        individuals=(Individual("a", "A", properties={"id": "b"}),),
    )
    with pytest.raises(ModelError) as raised:
        seed_statements(model)
    assert raised.value.codes == ("invalid_value",)


def test_an_individual_may_relate_to_itself():
    model = HygmModel(
        node_types=(NodeType("A"),),
        relation_types=(RelationType("r"),),
        individuals=(Individual("a", "A", relations={"r": ("a",)}),),
    )

    assert _codes(model) == []
