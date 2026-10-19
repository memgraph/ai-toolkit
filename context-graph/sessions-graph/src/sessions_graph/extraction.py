"""How sessions and memory files are extracted: the GLiNER2 checkpoint, its thresholds and its chunking."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

#: ExtractionConfig field -> the GLiNER2Backend argument it sets.
_BACKEND_ARGUMENTS = {
    "model": "model_name",
    "entity_threshold": "entity_confidence_threshold",
    "relation_threshold": "relation_confidence_threshold",
    "chunk_size": "chunk_size",
    "chunk_overlap": "chunk_overlap",
}


@dataclass(frozen=True)
class ExtractionConfig:
    """The GLiNER2 settings reconciliation, re-extraction and derivation all extract with.

    One config for all three, so a session derivation re-extracts is read the
    same way reconciliation first read it. Every field left None keeps
    unstructured2graph's ``GLiNER2Backend`` default.
    """

    #: A GLiNER2 checkpoint.
    model: str | None = None
    #: Mentions and relations below these confidences are dropped.
    entity_threshold: float | None = None
    relation_threshold: float | None = None
    #: Words per window the model reads, and the words consecutive windows share.
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ExtractionConfig:
        """Defaults overridden by ``values`` (e.g. a config file's ``[extraction]``); unknown keys are ignored.

        Raises:
            ValueError: a threshold outside [0, 1], a chunk size that isn't positive, or a negative overlap.
        """
        known = {f.name for f in fields(cls)}
        overrides: dict[str, Any] = {}
        for key, value in values.items():
            if key not in known or value in (None, ""):
                continue
            if key == "model":
                overrides[key] = str(value)
            elif key.endswith("_threshold"):
                threshold = float(value)
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"extraction {key} must be between 0 and 1, got {threshold}")
                overrides[key] = threshold
            else:
                number = int(value)
                if number < (1 if key == "chunk_size" else 0):
                    raise ValueError(
                        f"extraction {key} must be {'> 0' if key == 'chunk_size' else '>= 0'}, got {number}"
                    )
                overrides[key] = number
        return cls(**overrides)

    def backend_arguments(self) -> dict[str, Any]:
        """The ``GLiNER2Backend`` keyword arguments this config sets; the backend defaults the rest."""
        return {
            argument: getattr(self, name)
            for name, argument in _BACKEND_ARGUMENTS.items()
            if getattr(self, name) is not None
        }
