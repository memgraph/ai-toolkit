from dataclasses import dataclass
from pathlib import Path

from hygm import HygmModel, ManualStrategy, NodeType, RelationType

SCRIPT_DIR = Path(__file__).parent
DEFAULT_ONTOLOGY_PATH = SCRIPT_DIR / "default_ontology.yaml"

#: One entry in an Ontology's entity vocabulary: hygm's NodeType (label,
#: description, identity). The label is the Memgraph label and GLiNER2 schema
#: key; the description steers LightRAG's prompt and GLiNER2's entity specs.
EntityType = NodeType

__all__ = ["DEFAULT_ONTOLOGY", "DEFAULT_ONTOLOGY_PATH", "EntityType", "Ontology", "RelationType", "load_ontology"]


@dataclass(frozen=True)
class Ontology:
    """
    A vocabulary of entity types (and, optionally, relation types with the
    entity types they may connect), used to steer LightRAG's own extraction
    prompt (via addon_params(), entity types only -- LightRAG has no
    relation-type steering hook) and GLiNER2's typed joint schema, and to gate
    which entity_type values get promoted to real Memgraph labels (via
    allowed_labels()) and which relationships conform (via
    memgraph.enforce_relation_domain_range). Load one from a config file with
    load_ontology() rather than constructing directly, so every consumer of a
    given ontology_path sees the same vocabulary.

    Note on naming: "enforce_ontology" (the flag callers pass to gate label
    promotion against this vocabulary) never rejects or filters an entity or
    relationship -- see ADR 0004 (never-reject-entities-for-ontology-non-conformance).
    A non-conforming one is always kept, stamped ontology_conformant=false.
    """

    entity_types: tuple[EntityType, ...]
    relation_types: tuple[RelationType, ...] = ()

    @classmethod
    def from_model(cls, model: HygmModel) -> "Ontology":
        return cls(entity_types=model.node_types, relation_types=model.relation_types)

    @property
    def model(self) -> HygmModel:
        return HygmModel(node_types=self.entity_types, relation_types=self.relation_types)

    def entity_types_guidance(self) -> str:
        bullets = "\n".join(f"- {t.label}: {t.description}" for t in self.entity_types)
        return f"Classify each entity using one of the following types. If no type fits, use `Other`.\n\n{bullets}"

    def addon_params(self) -> dict[str, str]:
        return {"entity_types_guidance": self.entity_types_guidance()}

    def allowed_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.entity_types)

    def allowed_relation_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.relation_types)


def load_ontology(path: str | Path) -> Ontology:
    """
    Load an Ontology from a YAML config file:

        entity_types:
          - label: Person
            description: Human individuals, real or fictional
            identity: global          # optional: global | chunk (default) | span
          - label: Organization
            description: Companies, institutions, government bodies, groups
        relation_types:               # optional
          - label: works_for
            description: Employment relationship between a person and an organization
            start_labels: [Person]    # optional; omitted = any declared entity type
            end_labels: [Organization]

    Parsed and validated by hygm's ManualStrategy (which rejects, among other
    things, an endpoint naming an undeclared type). This is the single source
    of truth an ontology_path is meant to name -- call it once per path at
    each call site (e.g. once when configuring MemgraphLightRAGWrapper's
    addon_params, once when gating label promotion) rather than passing a
    pre-built Ontology object between them, so both sides always reflect the
    same file on disk.

    Raises:
        ValueError: if the file cannot be read, parsed or validated.
    """
    return Ontology.from_model(ManualStrategy().create_model(path))


# Mirrors LightRAG's own built-in entity type vocabulary, so label promotion
# matches what LightRAG extracts by default even for callers who pass no
# ontology_path at all. See default_ontology.yaml for the actual vocabulary.
DEFAULT_ONTOLOGY = load_ontology(DEFAULT_ONTOLOGY_PATH)
