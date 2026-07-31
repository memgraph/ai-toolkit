from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).parent
DEFAULT_ONTOLOGY_PATH = SCRIPT_DIR / "default_ontology.yaml"


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
    Load one from a config file with load_ontology() rather than
    constructing directly, so every consumer of a given ontology_path sees
    the same vocabulary.
    """

    entity_types: tuple[EntityType, ...]

    def entity_types_guidance(self) -> str:
        bullets = "\n".join(f"- {t.label}: {t.description}" for t in self.entity_types)
        return f"Classify each entity using one of the following types. If no type fits, use `Other`.\n\n{bullets}"

    def addon_params(self) -> dict[str, str]:
        return {"entity_types_guidance": self.entity_types_guidance()}

    def allowed_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.entity_types)


def load_ontology(path: str | Path) -> Ontology:
    """
    Load an Ontology from a YAML config file:

        entity_types:
          - label: Person
            description: Human individuals, real or fictional
          - label: Organization
            description: Companies, institutions, government bodies, groups

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

    if not isinstance(raw, dict) or "entity_types" not in raw:
        raise ValueError(f"Ontology file {resolved_path} must be a YAML mapping with an 'entity_types' key")

    entity_types = []
    for index, item in enumerate(raw["entity_types"]):
        if not isinstance(item, dict) or "label" not in item or "description" not in item:
            raise ValueError(
                f"Ontology file {resolved_path}: entity_types[{index}] must be a mapping with 'label' and 'description'"
            )
        entity_types.append(EntityType(label=item["label"], description=item["description"]))

    return Ontology(entity_types=tuple(entity_types))


# Mirrors LightRAG's own built-in entity type vocabulary, so label promotion
# matches what LightRAG extracts by default even for callers who pass no
# ontology_path at all. See default_ontology.yaml for the actual vocabulary.
DEFAULT_ONTOLOGY = load_ontology(DEFAULT_ONTOLOGY_PATH)
