"""ManualStrategy: a hand-authored model file."""

from pathlib import Path
from typing import Any

import yaml

from ..errors import ModelError
from ..format import model_from_mapping, model_to_mapping
from ..profiles import Version1Profiles
from ..types import HygmModel

__all__ = ["ManualStrategy", "model_from_mapping", "model_to_mapping"]


class ManualStrategy:
    """Loads a model from a YAML or JSON file in either format version (docs/contract.md, section 4).

    Version 2::

        version: 2
        node_types:
          - label: Customer
            description: a party that has placed an order
            parents: [Party]                      # optional
            key: [customer_id]                    # optional
            properties:                           # optional
              - {name: customer_id, type: string, required: true}
            profiles: {extraction: {identity: global}}
        relation_types:                           # optional
          - label: placed
            description: the customer placed the order
            start_labels: [Customer]              # optional; omitted = any declared label
            end_labels: [Order]

    Version 1, every model written before version 2, has no `version` key,
    `entity_types` instead of `node_types`, and `identity` on the node type.
    """

    def __init__(self, version_1_profiles: Version1Profiles | None = None) -> None:
        """
        Args:
            version_1_profiles: What a version-1 file implies, its owner's rule
                (docs/contract.md, 4.9); see model_from_mapping().
        """
        self.version_1_profiles = version_1_profiles

    def create_model(self, path: str | Path) -> HygmModel:
        """Parse and validate the model at `path`.

        Raises:
            ModelError: (a ValueError) if the file cannot be read or parsed,
                breaks the format, or fails validate_model()'s hard gate.
        """
        resolved = Path(path)
        try:
            raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except OSError as e:
            raise ModelError("invalid_format", f"Could not read ontology file {resolved}: {e}") from e
        except yaml.YAMLError as e:
            raise ModelError("invalid_format", f"Invalid YAML in ontology file {resolved}: {e}") from e

        return model_from_mapping(raw, f"file {resolved}", version_1_profiles=self.version_1_profiles)
