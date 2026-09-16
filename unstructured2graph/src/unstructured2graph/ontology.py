import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).parent
DEFAULT_ONTOLOGY_PATH = SCRIPT_DIR / "default_ontology.yaml"

# A label gets f-string-interpolated directly into Cypher (SET n:{label}) in
# promote_entity_types_to_labels(), so it's restricted to safe identifier
# characters -- anything else (backticks, colons, whitespace, quotes) could
# otherwise break or inject into the generated query.
_VALID_LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class EntityType:
    """One entry in an Ontology's entity vocabulary.

    Attributes:
        label: The Memgraph label / extraction-schema key this entity type
            promotes to. Must be a valid Cypher identifier -- load_ontology()
            validates this against _VALID_LABEL_PATTERN before constructing
            one; don't build EntityType directly with an unvalidated label.
        description: Human-readable guidance for what this type covers,
            surfaced to callers via Ontology.entity_types_guidance() (steers
            LightRAG's extraction prompt) and used directly as a GLiNER2Backend
            schema description.
    """

    label: str
    description: str


@dataclass(frozen=True)
class RelationType:
    """One entry in an Ontology's relation vocabulary -- the Entity Type
    counterpart for relationships. See EntityType for the field contract;
    the only difference is that `label` becomes a Cypher relationship type
    (e.g. `:works_for`) rather than a node label, and there is no LightRAG
    consumer for relation types at all (see Ontology's own docstring).
    """

    label: str
    description: str


@dataclass(frozen=True)
class Ontology:
    """
    A vocabulary of entity types (and, optionally, relation types), used to
    steer LightRAG's own extraction prompt (via addon_params(), entity types
    only -- LightRAG has no relation-type steering hook) and to gate which
    entity_type values get promoted to real Memgraph labels (via
    allowed_labels()). relation_types has no LightRAG consumer; it's read
    directly by backends that accept a relation schema natively (e.g. a
    local-model backend), as a {label: description} mapping.
    Load one from a config file with load_ontology() rather than
    constructing directly, so every consumer of a given ontology_path sees
    the same vocabulary.
    """

    entity_types: tuple[EntityType, ...]
    relation_types: tuple[RelationType, ...] = ()

    def entity_types_guidance(self) -> str:
        bullets = "\n".join(f"- {t.label}: {t.description}" for t in self.entity_types)
        return f"Classify each entity using one of the following types. If no type fits, use `Other`.\n\n{bullets}"

    def addon_params(self) -> dict[str, str]:
        return {"entity_types_guidance": self.entity_types_guidance()}

    def allowed_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.entity_types)

    def allowed_relation_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.relation_types)


def _parse_labeled_entries(
    items: list[Any], resolved_path: Path, section: str, cypher_kind: str
) -> list[tuple[str, str]]:
    """
    Validate a YAML list of {label, description} mappings under `section`
    (e.g. "entity_types") into (label, description) pairs. Shared by
    entity_types and relation_types parsing, since both need identical
    shape/identifier validation -- `cypher_kind` only varies the error
    message ("Memgraph label" vs. "Memgraph relationship type").
    """
    parsed = []
    for index, item in enumerate(items):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not isinstance(item.get("description"), str)
        ):
            raise ValueError(
                f"Ontology file {resolved_path}: {section}[{index}] must be a mapping with "
                "string 'label' and 'description'"
            )
        label = item["label"]
        if not _VALID_LABEL_PATTERN.match(label):
            raise ValueError(
                f"Ontology file {resolved_path}: {section}[{index}] label {label!r} must be a valid "
                f"identifier (letters, digits, underscore, not starting with a digit) since it's used "
                f"directly as a {cypher_kind}"
            )
        parsed.append((label, item["description"]))
    return parsed


def load_ontology(path: str | Path) -> Ontology:
    """
    Load an Ontology from a YAML config file:

        entity_types:
          - label: Person
            description: Human individuals, real or fictional
          - label: Organization
            description: Companies, institutions, government bodies, groups
        relation_types:  # optional -- no LightRAG-equivalent default vocabulary to mirror
          - label: works_for
            description: Employment relationship between a person and an organization

    This is the single source of truth an ontology_path is meant to name --
    call it once per path at each call site (e.g. once when configuring
    MemgraphLightRAGWrapper's addon_params, once when gating label
    promotion) rather than passing a pre-built Ontology object between them,
    so both sides always reflect the same file on disk.
    """
    resolved_path = Path(path)
    try:
        raw: Any = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"Could not read ontology file {resolved_path}: {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in ontology file {resolved_path}: {e}") from e

    if not isinstance(raw, dict) or not isinstance(raw.get("entity_types"), list):
        raise ValueError(f"Ontology file {resolved_path} must be a YAML mapping with an 'entity_types' list")

    entity_types = [
        EntityType(label=label, description=description)
        for label, description in _parse_labeled_entries(
            raw["entity_types"], resolved_path, "entity_types", "Memgraph label"
        )
    ]

    raw_relation_types = raw.get("relation_types", [])
    if not isinstance(raw_relation_types, list):
        raise ValueError(f"Ontology file {resolved_path}: 'relation_types' must be a list")
    relation_types = [
        RelationType(label=label, description=description)
        for label, description in _parse_labeled_entries(
            raw_relation_types, resolved_path, "relation_types", "Memgraph relationship type"
        )
    ]

    return Ontology(entity_types=tuple(entity_types), relation_types=tuple(relation_types))


# Mirrors LightRAG's own built-in entity type vocabulary, so label promotion
# matches what LightRAG extracts by default even for callers who pass no
# ontology_path at all. See default_ontology.yaml for the actual vocabulary.
DEFAULT_ONTOLOGY = load_ontology(DEFAULT_ONTOLOGY_PATH)
