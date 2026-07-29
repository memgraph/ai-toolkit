from dataclasses import dataclass


@dataclass(frozen=True)
class EntityType:
    label: str
    description: str


@dataclass(frozen=True)
class Ontology:
    """
    A vocabulary of entity types, used both to steer LightRAG's own
    extraction prompt (via addon_params()) and to gate which entity_type
    values get promoted to real Memgraph labels (via allowed_labels()).
    Passing the same Ontology to both call sites keeps extraction and
    label-promotion in sync; nothing enforces that automatically since the
    two happen in different code (LightRAG init vs. ingestion).
    """

    entity_types: tuple[EntityType, ...]

    def entity_types_guidance(self) -> str:
        bullets = "\n".join(f"- {t.label}: {t.description}" for t in self.entity_types)
        return f"Classify each entity using one of the following types. If no type fits, use `Other`.\n\n{bullets}"

    def addon_params(self) -> dict[str, str]:
        return {"entity_types_guidance": self.entity_types_guidance()}

    def allowed_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.entity_types)


# Mirrors LightRAG's own built-in entity type vocabulary (lightrag.prompt's
# default_entity_types_guidance), so label promotion matches what LightRAG
# already extracts by default even before any caller wires addon_params().
DEFAULT_ONTOLOGY = Ontology(
    entity_types=(
        EntityType("Person", "Human individuals, real or fictional"),
        EntityType("Creature", "Non-human living beings (animals, mythical beings, etc.)"),
        EntityType("Organization", "Companies, institutions, government bodies, groups"),
        EntityType("Location", "Geographic places (cities, countries, buildings, regions)"),
        EntityType("Event", "Occurrences, incidents, ceremonies, meetings"),
        EntityType("Concept", "Abstract ideas, theories, principles, beliefs"),
        EntityType("Method", "Procedures, techniques, algorithms, workflows"),
        EntityType("Content", "Creative or informational works (books, articles, films, reports)"),
        EntityType("Data", "Quantitative or structured information (statistics, datasets, measurements)"),
        EntityType("Artifact", "Physical or digital objects created by humans (tools, software, devices)"),
        EntityType("NaturalObject", "Natural non-living objects (minerals, celestial bodies, chemical compounds)"),
    )
)
