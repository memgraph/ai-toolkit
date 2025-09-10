"""
LLM Model Resolver Library

A library for resolving model names and configurations across different AI frameworks.
"""

from .base import BaseModelResolver
from .deepeval_resolver import DeepEvalModelResolver
from .litellm_resolver import LiteLLMModelResolver

__version__ = "0.1.0"
__all__ = [
    "BaseModelResolver",
    "DeepEvalModelResolver",
    "LiteLLMModelResolver",
]
