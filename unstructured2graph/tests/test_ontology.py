"""Unit tests for unstructured2graph.ontology."""

from unstructured2graph.ontology import DEFAULT_ONTOLOGY, EntityType, Ontology


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
    callers who never wire addon_params()."""
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
