import textwrap
from pathlib import Path

import pytest

from hygm import (
    ModelError,
    NodeType,
    OwlImportStrategy,
    Property,
    RelationType,
    model_from_mapping,
    model_to_mapping,
    validate_model,
)

NS = "http://example.org/memory#"
FIXTURES = Path(__file__).parent / "fixtures"

TURTLE = """
@prefix : <http://example.org/memory#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Person a owl:Class ; rdfs:comment "another named individual" .
:User a owl:Class ; rdfs:label "the user" .
:Location a owl:Class .

:visited a owl:ObjectProperty ;
    rdfs:comment "travelled to" ;
    rdfs:domain [ a owl:Class ; owl:unionOf ( :User :Person ) ] ;
    rdfs:range :Location .

:worksFor a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Organization .

:mentions a owl:ObjectProperty .
"""

PREFIXES = """
@prefix : <http://example.org/memory#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _write(tmp_path, body):
    path = tmp_path / "ontology.ttl"
    path.write_text(textwrap.dedent(body))
    return path


def _import(tmp_path, body):
    return OwlImportStrategy().create_model(_write(tmp_path, PREFIXES + textwrap.dedent(body)))


def _properties(model, label):
    node = model.node_type(label)
    assert node is not None
    return {p.name: p for p in node.properties}


def test_owl_classes_and_object_properties_map_onto_the_model(tmp_path):
    model = OwlImportStrategy().create_model(_write(tmp_path, TURTLE))
    assert model.node_types == (
        NodeType("Location", "", iri=NS + "Location"),
        NodeType("Organization", "", iri=NS + "Organization"),  # named only by a range, still a node type
        NodeType("Person", "another named individual", iri=NS + "Person"),
        NodeType("User", "the user", iri=NS + "User"),
    )
    assert model.relation_types == (
        RelationType("mentions", "", iri=NS + "mentions"),
        RelationType("visited", "travelled to", ("User", "Person"), ("Location",), iri=NS + "visited"),
        RelationType("worksFor", "", ("Person",), ("Organization",), iri=NS + "worksFor"),
    )


def test_an_owl_model_is_not_held_to_the_conversation_rule(tmp_path):
    """The User/Person rule belongs to the conversation profile, which an imported ontology doesn't declare."""
    from hygm import conversation, validate_for_conversation

    body = TURTLE.replace("owl:unionOf ( :User :Person )", "owl:unionOf ( :User )")

    model = OwlImportStrategy().create_model(_write(tmp_path, body))

    assert conversation(model) is None
    assert validate_for_conversation(model).success


def test_owl_local_name_that_is_not_an_identifier_is_rejected(tmp_path):
    body = TURTLE.replace(":Location a owl:Class .", "<http://example.org/memory#Place-Name> a owl:Class .")
    with pytest.raises(ValueError, match="Place-Name"):
        OwlImportStrategy().create_model(_write(tmp_path, body))


def test_unparseable_owl_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not parse"):
        OwlImportStrategy().create_model(_write(tmp_path, "this is not turtle ::"), format="turtle")


def test_subclass_of_a_named_class_is_a_parent_and_restrictions_and_thing_are_not(tmp_path):
    model = _import(
        tmp_path,
        """
        :Party a owl:Class .
        :Customer a owl:Class ;
            rdfs:subClassOf :Party , :Account , :Customer , owl:Thing ,
                [ a owl:Restriction ; owl:onProperty :placed ; owl:someValuesFrom :Order ] .
        """,
    )

    assert model.node_labels() == ("Account", "Customer", "Party")  # a parent is a node type even undeclared
    customer = model.node_type("Customer")
    assert customer is not None and customer.parents == ("Account", "Party")
    assert model.ancestors("Customer") == ("Account", "Party")


def test_datatype_property_is_a_property_on_its_domain_and_each_class_of_a_union(tmp_path):
    model = _import(
        tmp_path,
        """
        :Person a owl:Class .
        :Organization a owl:Class .
        :birthDate a owl:DatatypeProperty ; rdfs:comment "Day of birth" ; rdfs:domain :Person ; rdfs:range xsd:date .
        :name a owl:DatatypeProperty ;
            rdfs:domain [ a owl:Class ; owl:unionOf ( :Person :Organization ) ] .
        :orphan a owl:DatatypeProperty ; rdfs:range xsd:string .
        """,
    )

    assert _properties(model, "Person") == {
        "birthDate": Property("birthDate", "date", "Day of birth", iri=NS + "birthDate"),
        "name": Property("name", "string", iri=NS + "name"),
    }
    assert _properties(model, "Organization") == {"name": Property("name", "string", iri=NS + "name")}
    assert all("orphan" not in _properties(model, label) for label in model.node_labels())


@pytest.mark.parametrize(
    ("range_", "expected"),
    [
        ("xsd:string", "string"),
        ("xsd:anyURI", "string"),
        ("rdf:langString", "string"),
        ("xsd:integer", "integer"),
        ("xsd:int", "integer"),
        ("xsd:long", "integer"),
        ("xsd:short", "integer"),
        ("xsd:byte", "integer"),
        ("xsd:nonNegativeInteger", "integer"),
        ("xsd:float", "float"),
        ("xsd:double", "float"),
        ("xsd:decimal", "decimal"),
        ("xsd:boolean", "boolean"),
        ("xsd:date", "date"),
        ("xsd:dateTime", "datetime"),
        ("xsd:dateTimeStamp", "datetime"),
        ("rdf:JSON", "json"),
        ("[ a rdfs:Datatype ; owl:onDatatype xsd:int ; owl:withRestrictions ( [ xsd:minInclusive 0 ] ) ]", "integer"),
    ],
)
def test_datatype_property_range_is_its_type_by_the_xsd_table(tmp_path, range_, expected):
    model = _import(tmp_path, f":C a owl:Class .\n:p a owl:DatatypeProperty ; rdfs:domain :C ; rdfs:range {range_} .")

    assert _properties(model, "C")["p"].type == expected


@pytest.mark.parametrize("range_", ["xsd:gYear", "xsd:duration", "xsd:hexBinary", "<http://example.org/custom#Type>"])
def test_an_unlisted_datatype_maps_to_string(tmp_path, range_):
    model = _import(tmp_path, f":C a owl:Class .\n:p a owl:DatatypeProperty ; rdfs:domain :C ; rdfs:range {range_} .")

    assert _properties(model, "C")["p"].type == "string"


def test_no_range_or_disagreeing_ranges_map_to_string(tmp_path):
    model = _import(
        tmp_path,
        """
        :C a owl:Class .
        :bare a owl:DatatypeProperty ; rdfs:domain :C .
        :mixed a owl:DatatypeProperty ; rdfs:domain :C ; rdfs:range xsd:int , xsd:date .
        :same a owl:DatatypeProperty ; rdfs:domain :C ; rdfs:range xsd:int , xsd:long .
        """,
    )

    assert {name: p.type for name, p in _properties(model, "C").items()} == {
        "bare": "string",
        "mixed": "string",
        "same": "integer",
    }


def test_a_one_of_literals_range_is_the_values(tmp_path):
    model = _import(
        tmp_path,
        """
        :Customer a owl:Class .
        :tier a owl:DatatypeProperty ; rdfs:domain :Customer ;
            rdfs:range [ a rdfs:Datatype ; owl:oneOf ( "standard" "gold" "platinum" ) ] .
        :size a owl:DatatypeProperty ; rdfs:domain :Customer ;
            rdfs:range [ a rdfs:Datatype ; owl:oneOf ( 1 2 3 ) ] .
        """,
    )

    properties = _properties(model, "Customer")
    assert properties["tier"] == Property("tier", "string", values=("standard", "gold", "platinum"), iri=NS + "tier")
    assert properties["size"].type == "string" and properties["size"].values == ("1", "2", "3")


def test_has_key_is_the_key_and_each_key_property_is_required(tmp_path):
    model = _import(
        tmp_path,
        """
        :Order a owl:Class ; owl:hasKey ( :shop :orderId ) .
        :orderId a owl:DatatypeProperty ; rdfs:domain :Order .
        :shop a owl:DatatypeProperty ; rdfs:domain :Order .
        :total a owl:DatatypeProperty ; rdfs:domain :Order ; rdfs:range xsd:decimal .
        """,
    )

    order = model.node_type("Order")
    assert order is not None and order.key == ("shop", "orderId")
    assert {name: p.required for name, p in _properties(model, "Order").items()} == {
        "orderId": True,
        "shop": True,
        "total": False,
    }


def test_a_key_on_an_inherited_property_redeclares_it_required(tmp_path):
    model = _import(
        tmp_path,
        """
        :Party a owl:Class .
        :Customer a owl:Class ; rdfs:subClassOf :Party ; owl:hasKey ( :partyId ) .
        :partyId a owl:DatatypeProperty ; rdfs:domain :Party ; rdfs:range xsd:string .
        """,
    )

    assert _properties(model, "Party")["partyId"].required is False
    assert _properties(model, "Customer")["partyId"] == Property("partyId", required=True, iri=NS + "partyId")
    assert model.key_of("Customer") == ("partyId",)


def test_a_key_on_something_not_a_datatype_property_of_the_class_is_refused(tmp_path):
    with pytest.raises(ModelError) as e:
        _import(
            tmp_path,
            """
            :Order a owl:Class ; owl:hasKey ( :placedBy ) .
            :placedBy a owl:ObjectProperty ; rdfs:domain :Order .
            """,
        )
    assert e.value.codes == ("unknown_key_property",)


def test_two_has_key_axioms_are_refused(tmp_path):
    with pytest.raises(ModelError) as e:
        _import(
            tmp_path,
            """
            :Order a owl:Class ; owl:hasKey ( :a ) ; owl:hasKey ( :b ) .
            :a a owl:DatatypeProperty ; rdfs:domain :Order .
            :b a owl:DatatypeProperty ; rdfs:domain :Order .
            """,
        )
    assert e.value.code == "two_keys"


@pytest.mark.parametrize(
    "cardinality",
    [
        "owl:minCardinality 1",
        "owl:cardinality 1",
        "owl:qualifiedCardinality 1 ; owl:onDataRange xsd:string",
        "owl:minQualifiedCardinality 2 ; owl:onDataRange xsd:string",
    ],
)
def test_a_functional_property_with_an_at_least_one_restriction_is_required(tmp_path, cardinality):
    model = _import(
        tmp_path,
        f"""
        :Person a owl:Class ;
            rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :name ; {cardinality} ] .
        :name a owl:DatatypeProperty , owl:FunctionalProperty ; rdfs:domain :Person .
        """,
    )

    assert _properties(model, "Person")["name"].required is True


@pytest.mark.parametrize(
    ("types", "cardinality"),
    [
        ("owl:DatatypeProperty", "owl:minCardinality 1"),  # not functional
        ("owl:DatatypeProperty , owl:FunctionalProperty", "owl:minCardinality 0"),
        ("owl:DatatypeProperty , owl:FunctionalProperty", "owl:maxCardinality 1"),
    ],
)
def test_without_both_functional_and_at_least_one_a_property_is_optional(tmp_path, types, cardinality):
    model = _import(
        tmp_path,
        f"""
        :Person a owl:Class ;
            rdfs:subClassOf [ a owl:Restriction ; owl:onProperty :name ; {cardinality} ] .
        :name a {types} ; rdfs:domain :Person .
        """,
    )

    assert _properties(model, "Person")["name"].required is False


def test_a_restriction_on_a_subclass_tightens_the_inherited_property_there(tmp_path):
    model = _import(
        tmp_path,
        """
        :Agent a owl:Class .
        :Person a owl:Class ; rdfs:subClassOf :Agent ,
            [ a owl:Restriction ; owl:onProperty :name ; owl:minCardinality 1 ] .
        :name a owl:DatatypeProperty , owl:FunctionalProperty ; rdfs:domain :Agent .
        """,
    )

    assert _properties(model, "Agent")["name"].required is False
    assert _properties(model, "Person")["name"].required is True


def test_every_class_and_property_keeps_its_iri(tmp_path):
    model = _import(
        tmp_path,
        """
        :Person a owl:Class .
        <https://other.example/vocab/Company> a owl:Class .
        :worksFor a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range <https://other.example/vocab/Company> .
        <https://other.example/vocab/legalName> a owl:DatatypeProperty ;
            rdfs:domain <https://other.example/vocab/Company> .
        """,
    )

    company = model.node_type("Company")
    assert company is not None and company.iri == "https://other.example/vocab/Company"
    assert _properties(model, "Company")["legalName"].iri == "https://other.example/vocab/legalName"
    assert model.relation_types[0].iri == NS + "worksFor"


def test_a_subclass_cycle_is_refused(tmp_path):
    with pytest.raises(ModelError) as e:
        _import(
            tmp_path,
            """
            :A a owl:Class ; rdfs:subClassOf :B .
            :B a owl:Class ; rdfs:subClassOf :C .
            :C a owl:Class ; rdfs:subClassOf :A .
            """,
        )
    assert e.value.codes == ("parent_cycle",)


def test_a_property_name_that_is_not_an_identifier_is_rejected(tmp_path):
    body = ":C a owl:Class .\n<http://example.org/memory#birth-date> a owl:DatatypeProperty ; rdfs:domain :C ."
    with pytest.raises(ModelError, match="birth-date") as e:
        _import(tmp_path, body)
    assert e.value.code == "invalid_identifier"


def test_equivalence_and_disjointness_are_ignored(tmp_path):
    model = _import(
        tmp_path,
        """
        :Person a owl:Class ; owl:equivalentClass :Human ; owl:disjointWith :Robot .
        :Robot a owl:Class .
        """,
    )

    assert model.node_labels() == ("Person", "Robot")


def test_an_owl_thing_range_or_domain_leaves_the_relation_unconstrained_there(tmp_path):
    model = _import(
        tmp_path,
        """
        :Person a owl:Class .
        :likes a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range owl:Thing .
        :about a owl:ObjectProperty ; rdfs:domain owl:Thing ; rdfs:range :Person .
        """,
    )

    assert model.node_labels() == ("Person",)
    assert {r.label: (r.start_labels, r.end_labels) for r in model.relation_types} == {
        "about": ((), ("Person",)),
        "likes": (("Person",), ()),
    }


def test_a_datatype_property_whose_domain_is_owl_thing_is_skipped(tmp_path):
    model = _import(
        tmp_path,
        """
        :Person a owl:Class .
        :note a owl:DatatypeProperty ; rdfs:domain owl:Thing ; rdfs:range xsd:string .
        """,
    )

    assert model.node_labels() == ("Person",)
    assert _properties(model, "Person") == {}


def test_owl_thing_in_a_union_leaves_just_the_named_classes(tmp_path):
    model = _import(
        tmp_path,
        """
        :Person a owl:Class .
        :Place a owl:Class .
        :visited a owl:ObjectProperty ; rdfs:domain :Person ;
            rdfs:range [ a owl:Class ; owl:unionOf ( owl:Thing :Place ) ] .
        :label a owl:DatatypeProperty ; rdfs:domain [ a owl:Class ; owl:unionOf ( :Place owl:Thing ) ] .
        """,
    )

    assert model.relation_types[0].end_labels == ("Place",)
    assert list(_properties(model, "Place")) == ["label"]
    assert _properties(model, "Person") == {}


def test_an_imported_version_2_model_round_trips(tmp_path):
    model = _import(
        tmp_path,
        """
        :Party a owl:Class .
        :Customer a owl:Class ; rdfs:subClassOf :Party ; owl:hasKey ( :customerId ) .
        :customerId a owl:DatatypeProperty ; rdfs:domain :Customer .
        :tier a owl:DatatypeProperty ; rdfs:domain :Customer ;
            rdfs:range [ a rdfs:Datatype ; owl:oneOf ( "standard" "gold" ) ] .
        :placed a owl:ObjectProperty ; rdfs:domain :Customer .
        """,
    )

    assert model_to_mapping(model)["version"] == 2
    assert model_from_mapping(model_to_mapping(model), "round trip") == model


def test_a_public_style_ontology_imports_validates_and_round_trips():
    model = OwlImportStrategy().create_model(FIXTURES / "agents.ttl")

    assert validate_model(model).success
    assert model.node_labels() == ("Agent", "Corporation", "Organization", "Person", "Place")
    assert model.ancestors("Corporation") == ("Organization", "Agent")
    assert model.key_of("Person") == ("identifier",)
    assert {p.name: (p.type, p.required) for p in model.properties_of("Person")} == {
        "email": ("string", False),
        "identifier": ("string", True),
        "name": ("string", True),
        "birthDate": ("date", False),
        "gender": ("string", False),
    }
    assert {p.name: p.type for p in model.properties_of("Corporation")}["numberOfEmployees"] == "integer"
    member = next(r for r in model.relation_types if r.label == "member")
    assert model.endpoint_labels(member, "end") == ("Agent", "Corporation", "Organization", "Person")
    assert model_from_mapping(model_to_mapping(model), "round trip") == model
