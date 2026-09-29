import textwrap

import pytest

from hygm import NodeType, OwlImportStrategy, RelationType

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


def _write(tmp_path, body):
    path = tmp_path / "ontology.ttl"
    path.write_text(textwrap.dedent(body))
    return path


def test_owl_classes_and_object_properties_map_onto_the_model(tmp_path):
    model = OwlImportStrategy().create_model(_write(tmp_path, TURTLE))
    assert model.node_types == (
        NodeType("Location", ""),
        NodeType("Organization", ""),  # named only by a range, still a node type
        NodeType("Person", "another named individual"),
        NodeType("User", "the user"),
    )
    assert model.relation_types == (
        RelationType("mentions", ""),
        RelationType("visited", "travelled to", ("User", "Person"), ("Location",)),
        RelationType("worksFor", "", ("Person",), ("Organization",)),
    )


def test_owl_model_still_passes_the_user_requires_person_gate(tmp_path):
    body = TURTLE.replace("owl:unionOf ( :User :Person )", "owl:unionOf ( :User )")
    with pytest.raises(ValueError, match="'User' but not 'Person'"):
        OwlImportStrategy().create_model(_write(tmp_path, body))


def test_owl_local_name_that_is_not_an_identifier_is_rejected(tmp_path):
    body = TURTLE.replace(":Location a owl:Class .", "<http://example.org/memory#Place-Name> a owl:Class .")
    with pytest.raises(ValueError, match="Place-Name"):
        OwlImportStrategy().create_model(_write(tmp_path, body))


def test_unparseable_owl_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not parse"):
        OwlImportStrategy().create_model(_write(tmp_path, "this is not turtle ::"), format="turtle")
