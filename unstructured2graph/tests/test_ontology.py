"""Unit tests for unstructured2graph.ontology."""

import pytest

from unstructured2graph.ontology import DEFAULT_ONTOLOGY, DEFAULT_ONTOLOGY_PATH, EntityType, Ontology, load_ontology


def test_entity_types_guidance_lists_every_type_with_its_description():
    ontology = Ontology(
        entity_types=(
            EntityType("Person", "Human individuals"),
            EntityType("Location", "Geographic places"),
        )
    )

    guidance = ontology.entity_types_guidance()

    assert "- Person: Human individuals" in guidance
    assert "- Location: Geographic places" in guidance
    assert "If no type fits, use `Other`" in guidance


def test_addon_params_wraps_entity_types_guidance():
    ontology = Ontology(entity_types=(EntityType("Person", "Human individuals"),))

    params = ontology.addon_params()

    assert params == {"entity_types_guidance": ontology.entity_types_guidance()}


def test_allowed_labels_returns_label_names_in_declared_order():
    ontology = Ontology(
        entity_types=(
            EntityType("Person", "..."),
            EntityType("Organization", "..."),
        )
    )

    assert ontology.allowed_labels() == ("Person", "Organization")


def test_default_ontology_mirrors_lightrags_builtin_vocabulary():
    """Kept in sync with lightrag.prompt's default_entity_types_guidance so
    label promotion matches what LightRAG extracts by default, even for
    callers who never pass an ontology_path at all."""
    assert DEFAULT_ONTOLOGY.allowed_labels() == (
        "Person",
        "Creature",
        "Organization",
        "Location",
        "Event",
        "Concept",
        "Method",
        "Content",
        "Data",
        "Artifact",
        "NaturalObject",
    )


def test_default_ontology_path_points_at_a_real_bundled_file():
    assert DEFAULT_ONTOLOGY_PATH.is_file()


def test_load_ontology_parses_label_and_description(tmp_path):
    ontology_file = tmp_path / "ontology.yaml"
    ontology_file.write_text(
        """
        entity_types:
          - label: Person
            description: Human individuals
          - label: Location
            description: Geographic places
        """
    )

    ontology = load_ontology(ontology_file)

    assert ontology.allowed_labels() == ("Person", "Location")
    assert ontology.entity_types[0].description == "Human individuals"


def test_load_ontology_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not read ontology file"):
        load_ontology(tmp_path / "does_not_exist.yaml")


def test_load_ontology_invalid_yaml_raises_value_error(tmp_path):
    ontology_file = tmp_path / "broken.yaml"
    ontology_file.write_text("entity_types: [this is not: valid: yaml")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_ontology(ontology_file)


def test_load_ontology_missing_entity_types_key_raises_value_error(tmp_path):
    ontology_file = tmp_path / "no_types.yaml"
    ontology_file.write_text("something_else: []")

    with pytest.raises(ValueError, match="entity_types"):
        load_ontology(ontology_file)


def test_load_ontology_entry_missing_description_raises_value_error(tmp_path):
    ontology_file = tmp_path / "incomplete.yaml"
    ontology_file.write_text(
        """
        entity_types:
          - label: Person
        """
    )

    with pytest.raises(ValueError, match=r"label.*description"):
        load_ontology(ontology_file)


def test_load_ontology_two_calls_on_same_path_produce_equal_ontologies(tmp_path):
    """Coordination between LightRAG's addon_params and Memgraph's label
    gating relies on independent load_ontology() calls against the same path
    always agreeing -- not on passing one Ontology object between them."""
    ontology_file = tmp_path / "ontology.yaml"
    ontology_file.write_text(
        """
        entity_types:
          - label: Person
            description: Human individuals
        """
    )

    assert load_ontology(ontology_file) == load_ontology(ontology_file)
