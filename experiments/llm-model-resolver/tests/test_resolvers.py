"""
Tests for model resolvers.
"""

import pytest
from model_resolver import DeepEvalModelResolver, LiteLLMModelResolver


def test_deepeval_resolver_ollama():
    """Test DeepEval resolver with Ollama model."""
    resolver = DeepEvalModelResolver("ollama/llama3.2:latest")

    assert resolver.get_model_name() == "llama3.2:latest"
    assert resolver.get_model_config() == {"base_url": "http://localhost:11434"}
    assert resolver.get_model() is not None


def test_deepeval_resolver_openai():
    """Test DeepEval resolver with OpenAI model."""
    resolver = DeepEvalModelResolver("gpt-4")

    assert resolver.get_model_name() == "gpt-4"
    assert resolver.get_model_config() == {}
    assert resolver.get_model() is None


def test_litellm_resolver_ollama():
    """Test LiteLLM resolver with Ollama model."""
    resolver = LiteLLMModelResolver("ollama/llama3.2:latest")

    assert resolver.get_model_name() == "ollama/llama3.2:latest"
    assert resolver.get_model_config() == {"api_base": "http://localhost:11434"}
    assert resolver.get_model() is None


def test_litellm_resolver_openai():
    """Test LiteLLM resolver with OpenAI model."""
    resolver = LiteLLMModelResolver("gpt-4")

    assert resolver.get_model_name() == "gpt-4"
    assert resolver.get_model_config() == {}
    assert resolver.get_model() is None


def test_custom_model():
    """Test resolvers with custom model names."""
    deepeval_resolver = DeepEvalModelResolver("custom-model")
    litellm_resolver = LiteLLMModelResolver("custom-model")

    assert deepeval_resolver.get_model_name() == "custom-model"
    assert litellm_resolver.get_model_name() == "custom-model"
    assert deepeval_resolver.get_model_config() == {}
    assert litellm_resolver.get_model_config() == {}
