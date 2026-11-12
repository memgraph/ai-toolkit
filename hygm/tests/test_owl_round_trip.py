"""OWL export, and the import of the constructs of contract 4.11-4.17 (docs/contract.md, section 10)."""

import json
import textwrap
from pathlib import Path

import pytest

from hygm import (
    HygmModel,
    Import,
    ManualStrategy,
    ModelError,
    NodeType,
    OwlImportStrategy,
    Property,
    RelationType,
    model_from_owl,
    model_to_mapping,
    model_to_owl,
    validate_model,
)

VALID = Path(__file__).parent.parent / "conformance" / "model" / "valid"
FIXTURES = Path(__file__).parent / "fixtures"

PREFIXES = """
@prefix : <http://example.org/wine#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

#: Lists whose order OWL keeps (`owl:hasKey`, `owl:oneOf`); every other one it doesn't.
_ORDERED = {"key", "values"}


def _canonical(value, key=None):
    """A model file mapping with every list OWL keeps no order of sorted."""
    if isinstance(value, dict):
        return {k: _canonical(v, k) for k, v in value.items()}
    if isinstance(value, list):
        items = [_canonical(v) for v in value]
        return items if key in _ORDERED else sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    return value


def _same_up_to_order(a: HygmModel, b: HygmModel) -> bool:
    return _canonical(model_to_mapping(a)) == _canonical(model_to_mapping(b))


def _import(body: str) -> HygmModel:
    return model_from_owl(PREFIXES + textwrap.dedent(body))


@pytest.mark.parametrize("path", sorted(VALID.iterdir()), ids=lambda p: p.name)
def test_every_valid_fixture_round_trips_through_owl(path):
    model = ManualStrategy().create_model(path)

    back = model_from_owl(model_to_owl(model), location=path)

    assert _same_up_to_order(back, model)


@pytest.mark.parametrize("name", ["wine.ttl", "agents.ttl"])
def test_an_ontology_written_by_another_tool_round_trips(name):
    model = OwlImportStrategy().create_model(FIXTURES / name)

    for format in ("turtle", "xml", "json-ld"):
        assert model_from_owl(model_to_owl(model, format=format), format=format) == model


def test_the_wine_ontology_imports_every_construct():
    model = OwlImportStrategy().create_model(FIXTURES / "wine.ttl")

    assert validate_model(model).success
    assert model.ontology is not None
    assert (model.ontology.name, model.ontology.version_info) == ("wine", "1.0")
    winery = model.node_type("Winery")
    assert winery is not None
    assert (winery.synonyms, winery.names) == (("chateau", "estate"), {"en": "Winery", "fr": "Domaine"})
    assert model.disjoint("RedWine", "WhiteWine") and model.disjoint("RoseWine", "RedWine")
    assert model.properties_of("RedWine")[0].value == "red"
    vintage = next(p for p in model.properties_of("Wine") if p.name == "vintage")
    assert (vintage.type, vintage.minimum, vintage.maximum) == ("integer", 1900, 2100)
    assert model.inverse("produces") == "hasMaker"
    pairs_with, has_maker = model.relation_type("pairsWith"), model.relation_type("hasMaker")
    assert pairs_with is not None and pairs_with.symmetric
    assert has_maker is not None and has_maker.max_count == 1
    bordeaux_maker = model.relation_for("Bordeaux", "hasMaker")
    grapes = model.relation_for("Bordeaux", "madeFromGrape")
    assert bordeaux_maker is not None and bordeaux_maker.end_labels == ("FrenchWinery",)
    assert grapes is not None and (grapes.min_count, grapes.max_count) == (2, 5)
    latour = model.individual("ChateauLatour")
    assert latour is not None
    assert (latour.type, latour.properties, latour.relations) == (
        "FrenchWinery",
        {"wineryName": "Chateau Latour"},
        {"locatedIn": ("Pauillac",)},
    )


def test_an_exported_model_uses_owl_constructs_and_hygm_annotations_only_for_the_rest():
    model = HygmModel(
        node_types=(
            NodeType(
                "Wine", abstract=True, properties=(Property("bottle_ml", type="integer", default=750, unit="ml"),)
            ),
            NodeType("RedWine", parents=("Wine",), disjoint_with=("WhiteWine",)),
            NodeType("WhiteWine", parents=("Wine",)),
        ),
        relation_types=(RelationType("pairs_with", start_labels=("Wine",), end_labels=("Wine",), symmetric=True),),
    )

    turtle = model_to_owl(model)

    for construct in ("owl:disjointWith", "owl:SymmetricProperty", "owl:FunctionalProperty", "rdfs:subClassOf"):
        assert construct in turtle
    for annotation in ("hygm:abstract", "hygm:default", "hygm:unit", "hygm:base"):
        assert annotation in turtle
    assert "<urn:hygm:model#Wine>" in turtle
    assert _same_up_to_order(model_from_owl(turtle), model)


def test_a_minted_iri_reads_back_as_none_and_an_own_one_is_kept():
    model = HygmModel(node_types=(NodeType("Wine"), NodeType("Grape", iri="http://example.org/wine#Grape")))

    back = model_from_owl(model_to_owl(model, base="https://example.org/minted/"))

    assert [(t.label, t.iri) for t in back.node_types] == [("Grape", "http://example.org/wine#Grape"), ("Wine", None)]


def test_a_label_a_minted_iri_would_collide_on_gets_another_and_keeps_its_name():
    model = HygmModel(node_types=(NodeType("Wine"),), relation_types=(RelationType("Wine"),))

    turtle = model_to_owl(model)

    assert "<urn:hygm:model#Wine_2>" in turtle
    assert model_from_owl(turtle) == model


def test_a_base_must_end_in_a_separator():
    with pytest.raises(ValueError, match="base"):
        model_to_owl(HygmModel(node_types=()), base="urn:hygm:model")


def test_a_model_that_changed_what_it_imports_is_refused_as_the_model_file_refuses_it():
    imported = ManualStrategy().create_model(VALID / "v2-imports.yaml")
    changed = HygmModel(
        node_types=tuple(NodeType(t.label, "changed") if t.label == "Food" else t for t in imported.node_types),
        relation_types=imported.relation_types,
        imports=imported.imports,
    )

    with pytest.raises(ModelError) as raised:
        model_to_owl(changed)

    assert raised.value.codes == ("duplicate_label",)


def test_disjointness_from_pairs_and_lists_is_read_once_per_pair():
    model = _import(
        """
        :Red a owl:Class ; owl:disjointWith :White .
        :White a owl:Class .
        :Rose a owl:Class .
        [] a owl:AllDisjointClasses ; owl:members ( :Red :Rose :White ) .
        """
    )

    assert {t.label: t.disjoint_with for t in model.node_types} == {
        "Red": ("Rose", "White"),
        "Rose": ("White",),
        "White": (),
    }


def test_restrictions_on_an_object_property_are_relation_restrictions():
    model = _import(
        """
        :Wine a owl:Class .
        :Winery a owl:Class .
        :FrenchWinery a owl:Class ; rdfs:subClassOf :Winery .
        :FrenchWine a owl:Class ; rdfs:subClassOf :Wine ,
            [ a owl:Restriction ; owl:onProperty :maker ; owl:allValuesFrom :FrenchWinery ] ,
            [ a owl:Restriction ; owl:onProperty :maker ; owl:qualifiedCardinality 1 ; owl:onClass :FrenchWinery ] .
        :maker a owl:ObjectProperty ; rdfs:domain :Wine ; rdfs:range :Winery .
        """
    )

    restricted = model.relation_for("FrenchWine", "maker")

    assert restricted is not None
    assert (restricted.end_labels, restricted.min_count, restricted.max_count) == (("FrenchWinery",), 1, 1)


def test_a_restriction_on_a_subclass_narrows_the_inherited_values_and_bounds_there():
    model = _import(
        """
        :Wine a owl:Class .
        :Dessert a owl:Class ; rdfs:subClassOf :Wine ,
            [ a owl:Restriction ; owl:onProperty :sugar ;
              owl:allValuesFrom [ a rdfs:Datatype ; owl:oneOf ( "sweet" ) ] ] ,
            [ a owl:Restriction ; owl:onProperty :alcohol ;
              owl:allValuesFrom [ a rdfs:Datatype ; owl:onDatatype xsd:decimal ;
                                  owl:withRestrictions ( [ xsd:minInclusive 15 ] ) ] ] .
        :sugar a owl:DatatypeProperty ; rdfs:domain :Wine ;
            rdfs:range [ a rdfs:Datatype ; owl:oneOf ( "dry" "sweet" ) ] .
        :alcohol a owl:DatatypeProperty ; rdfs:domain :Wine ; rdfs:range xsd:decimal .
        """
    )

    effective = {p.name: p for p in model.properties_of("Dessert")}

    assert effective["sugar"].values == ("sweet",)
    assert effective["alcohol"].minimum == 15


def test_language_tagged_labels_are_names_and_an_untagged_one_still_describes():
    model = _import(
        """
        :Wine a owl:Class ; rdfs:label "a drink" , "Wine"@en ; skos:prefLabel "Vin"@fr ; skos:altLabel "vino" .
        """
    )

    wine = model.node_type("Wine")

    assert wine is not None
    assert (wine.description, wine.names, wine.synonyms) == ("a drink", {"en": "Wine", "fr": "Vin"}, ("vino",))


def test_an_import_of_a_local_file_is_converted_and_a_web_one_is_left_out(tmp_path):
    (tmp_path / "food.ttl").write_text(PREFIXES + ":Food a owl:Class .\n")
    path = tmp_path / "wine.ttl"
    path.write_text(
        PREFIXES
        + textwrap.dedent(
            """
            <http://example.org/wine> a owl:Ontology ;
                owl:imports <food.ttl> , <http://www.w3.org/2004/02/skos/core> .
            :Wine a owl:Class ; rdfs:subClassOf :Food .
            """
        )
    )

    model = OwlImportStrategy().create_model(path)

    assert [i.path for i in model.imports] == ["food.ttl"]
    assert isinstance(model.imports[0], Import)
    assert model.node_labels() == ("Food", "Wine")
    assert model.ancestors("Wine") == ("Food",)


def test_a_missing_import_is_an_unknown_model_and_a_cycle_an_import_cycle(tmp_path):
    header = "<http://example.org/{name}> a owl:Ontology ; owl:imports <{target}> .\n"
    (tmp_path / "a.ttl").write_text(PREFIXES + header.format(name="a", target="b.ttl"))
    (tmp_path / "b.ttl").write_text(PREFIXES + header.format(name="b", target="a.ttl"))
    (tmp_path / "c.ttl").write_text(PREFIXES + header.format(name="c", target="missing.ttl"))

    for name, code in (("a.ttl", "import_cycle"), ("c.ttl", "unknown_model")):
        with pytest.raises(ModelError) as raised:
            OwlImportStrategy().create_model(tmp_path / name)
        assert raised.value.codes == (code,)
