"""ManualStrategy: a hand-authored YAML model."""

from pathlib import Path
from typing import Any

import yaml

from ..identifiers import is_valid_identifier
from ..types import IDENTITIES, HygmModel, NodeType, RelationType
from ..validation import validate_model


class ManualStrategy:
    """Loads a model from a YAML file shaped like::

    entity_types:
      - label: Person
        description: another named individual
        identity: global            # optional: global | chunk (default) | span
    relation_types:                 # optional
      - label: works_for
        description: is employed by  # documentation only for GLiNER2
        start_labels: [User, Person] # optional; omitted = any declared label
        end_labels: [Organization]
    """

    def create_model(self, path: str | Path) -> HygmModel:
        """Parse and validate the model at `path`.

        Raises:
            ValueError: if the file cannot be read or parsed, an entry is
                malformed, or the model fails validate_model()'s hard gate.
        """
        resolved = Path(path)
        try:
            raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except OSError as e:
            raise ValueError(f"Could not read ontology file {resolved}: {e}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in ontology file {resolved}: {e}") from e

        return model_from_mapping(raw, f"file {resolved}")


def model_from_mapping(raw: Any, source: str) -> HygmModel:
    """A validated model from the ``entity_types``/``relation_types`` mapping ManualStrategy's YAML holds.

    Args:
        raw: The parsed mapping.
        source: What it was read from, for error messages ("file schema.yaml", "version 3").

    Raises:
        ValueError: if an entry is malformed or the model fails validate_model()'s hard gate.
    """
    where_from = f"Ontology {source}"
    if not isinstance(raw, dict) or not isinstance(raw.get("entity_types"), list):
        raise ValueError(f"{where_from} must be a mapping with an 'entity_types' list")
    raw_relations = raw.get("relation_types", [])
    if not isinstance(raw_relations, list):
        raise ValueError(f"{where_from}: 'relation_types' must be a list")

    node_types = []
    for index, item in enumerate(raw["entity_types"]):
        label, description = _labeled(item, where_from, f"entity_types[{index}]", "Memgraph label")
        identity = item.get("identity", "chunk")
        if identity not in IDENTITIES:
            raise ValueError(
                f"{where_from}: entity_types[{index}] identity must be one of {IDENTITIES}, got {identity!r}"
            )
        node_types.append(NodeType(label=label, description=description, identity=identity))

    relation_types = []
    for index, item in enumerate(raw_relations):
        where = f"relation_types[{index}]"
        label, description = _labeled(item, where_from, where, "Memgraph relationship type")
        relation_types.append(
            RelationType(
                label=label,
                description=description,
                start_labels=_labels(item, "start_labels", where_from, where),
                end_labels=_labels(item, "end_labels", where_from, where),
            )
        )

    model = HygmModel(node_types=tuple(node_types), relation_types=tuple(relation_types))
    result = validate_model(model)
    if not result.success:
        problems = "; ".join(issue.message for issue in result.critical_issues)
        raise ValueError(f"{where_from} is not a valid model: {problems}")
    return model


def model_to_mapping(model: HygmModel) -> dict[str, Any]:
    """`model` as the mapping model_from_mapping() reads back: JSON- and YAML-safe."""
    return {
        "entity_types": [
            {"label": t.label, "description": t.description, "identity": t.identity} for t in model.node_types
        ],
        "relation_types": [
            {
                "label": r.label,
                "description": r.description,
                "start_labels": list(r.start_labels),
                "end_labels": list(r.end_labels),
            }
            for r in model.relation_types
        ],
    }


def _labeled(item: Any, where_from: str, where: str, cypher_kind: str) -> tuple[str, str]:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("label"), str)
        or not isinstance(item.get("description"), str)
    ):
        raise ValueError(f"{where_from}: {where} must be a mapping with string 'label' and 'description'")
    label = item["label"]
    if not is_valid_identifier(label):
        raise ValueError(
            f"{where_from}: {where} label {label!r} must be a valid identifier (letters, digits, "
            f"underscore, not starting with a digit) since it's used directly as a {cypher_kind}"
        )
    return label, item["description"]


def _labels(item: dict, key: str, where_from: str, where: str) -> tuple[str, ...]:
    value = item.get(key, [])
    if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
        raise ValueError(f"{where_from}: {where} {key} must be a list of labels")
    return tuple(value)
