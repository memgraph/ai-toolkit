"""ManualStrategy: a hand-authored model file."""

from pathlib import Path
from typing import Any

import yaml

from ..errors import ModelError
from ..format import ImportResolver, model_from_mapping, model_to_mapping
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

    `imports` paths are relative to the importing file and resolved
    transitively (docs/contract.md, 4.12). An imported file that can't be read
    is `unknown_model`; one that is itself invalid is refused with its own
    codes; a file that imports itself, directly or through others, is
    `import_cycle`. Each file is read once per create_model(), so a diamond
    of imports resolves to one declaration of each type.
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
        return self._load(Path(path), (), {})

    def resolver(self, path: str | Path) -> ImportResolver:
        """What resolves the `imports` of the model file at `path`, for model_from_mapping().

        For a caller that holds the file's contents already, such as a
        round trip through model_to_mapping().
        """
        return self._resolver(Path(path), (), {})

    def _resolver(self, path: Path, stack: tuple[Path, ...], cache: dict[Path, HygmModel]) -> ImportResolver:
        importing = (*stack, path.resolve())

        def resolve(relative: str) -> HygmModel:
            target = (path.parent / relative).resolve()
            if target in importing:
                chain = " -> ".join(str(p) for p in (*importing[importing.index(target) :], target))
                raise ModelError("import_cycle", f"Ontology file {path} imports form a cycle: {chain}")
            if target not in cache:
                if not target.is_file():
                    raise ModelError("unknown_model", f"Ontology file {path} imports {relative!r}: no file {target}")
                cache[target] = self._load(target, importing, cache)
            return cache[target]

        return resolve

    def _load(self, path: Path, stack: tuple[Path, ...], cache: dict[Path, HygmModel]) -> HygmModel:
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as e:
            code = "unknown_model" if stack else "invalid_format"
            raise ModelError(code, f"Could not read ontology file {path}: {e}") from e
        except yaml.YAMLError as e:
            raise ModelError("invalid_format", f"Invalid YAML in ontology file {path}: {e}") from e

        return model_from_mapping(
            raw,
            f"file {path}",
            version_1_profiles=self.version_1_profiles,
            resolver=self._resolver(path, stack, cache),
        )
