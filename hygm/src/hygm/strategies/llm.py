"""LlmRecommendationStrategy: a model derived from a sample of the corpus itself.

Interface only. The derivation contract (#353, revised by #366) is a fixed
core -- User, Person and the value types -- plus a derived domain through
batched propose, one consolidate call, one permissive observe pass and prune,
gated by validate_model(). It lands once its evidence run (#372) passes; until
then models come from ManualStrategy.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..types import HygmModel


class Observer(Protocol):
    """Runs an extractor permissively over texts and reports what it found.

    Defined here and implemented by the extracting package (unstructured2graph
    with GLiNER2), so hygm never depends on an extractor.
    """

    def observe(self, model: HygmModel, texts: Sequence[str]) -> Mapping[str, Any]:
        """Observation tables: per relation type, the endpoint-label pairs it
        fired on with counts and examples; per node type, mention statistics."""
        ...


class LlmRecommendationStrategy:
    """Derives a model from a sample of the corpus.

    Args:
        llm: Caller-supplied client; hygm takes no LLM SDK dependency.
        observer: The permissive extraction pass the derived ranges come from.
    """

    def __init__(self, llm: Any, observer: Observer) -> None:
        self.llm = llm
        self.observer = observer

    def create_model(self, data_sample: Sequence[str], domain_context: str | None = None) -> HygmModel:
        """Not implemented yet; see the module docstring.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("LlmRecommendationStrategy lands after its evidence run (#372); use ManualStrategy")
