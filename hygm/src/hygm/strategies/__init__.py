"""Strategies that produce a HygmModel. Each takes a different input, so they share no common interface yet."""

from .llm import LlmRecommendationStrategy, Observer
from .manual import ManualStrategy, model_from_mapping, model_to_mapping
from .owl import OwlImportStrategy

__all__ = [
    "LlmRecommendationStrategy",
    "ManualStrategy",
    "Observer",
    "OwlImportStrategy",
    "model_from_mapping",
    "model_to_mapping",
]
