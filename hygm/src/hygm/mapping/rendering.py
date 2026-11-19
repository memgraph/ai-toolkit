"""Rendering a model and its mapping into another tool's own format (docs/contract.md, 6.4)."""

from typing import Any, Protocol

from ..types import HygmModel
from .types import Mapping


class Renderer(Protocol):
    """Turns a model and a mapping checked against it into one target tool's format.

    Each consumer that keeps its own format supplies its renderer; hygm
    ships the interface, not the targets. A renderer is lossless for
    everything its target can express, and refuses everything else: when the
    model or the mapping uses something the target has no way to say (a
    `join`, a JSON `path`, a type hierarchy, a source kind), it raises a
    ValueError naming what it can't render, rather than dropping it.
    """

    def render(self, model: HygmModel, mapping: Mapping) -> Any:
        """The target's representation of `model` bound by `mapping`.

        Args:
            model: The model the mapping was checked against.
            mapping: A mapping that passed check_mapping() against `model`.

        Raises:
            ValueError: if either uses something the target can't express.
        """
        ...
