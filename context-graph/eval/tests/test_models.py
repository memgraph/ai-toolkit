"""Tests for resolving the answering model and the rubric judge from 'provider:model_id' specs."""

import pytest
from context_graph_eval.models import (
    DEFAULT_JUDGE_MODEL,
    build_model,
    minimal_effort_kwargs,
    parse_model_spec,
    resolved_spec,
)


def test_a_provider_qualified_spec_overrides_the_default_provider():
    assert parse_model_spec("openai:gpt-4o", default_provider="anthropic") == ("openai", "gpt-4o")


def test_a_bare_model_id_keeps_the_default_provider():
    assert parse_model_spec("claude-opus-4-1", default_provider="anthropic") == ("anthropic", "claude-opus-4-1")


def test_omitting_the_flag_keeps_the_default_provider_with_no_model_id():
    assert parse_model_spec(None, default_provider="openai") == ("openai", None)


def test_resolved_spec_reports_the_actual_fallback_model():
    """A run records what ran, not what was typed: omitting a model id still resolves per provider."""
    assert resolved_spec("anthropic", None) == f"anthropic:{DEFAULT_JUDGE_MODEL}"
    assert resolved_spec("openai", None) == "openai:default"
    assert resolved_spec("openai", "gpt-4.1") == "openai:gpt-4.1"


def test_an_unknown_provider_is_refused_not_guessed(capsys):
    assert build_model("azure", "gpt-4o") is None
    assert "unknown model provider" in capsys.readouterr().err


def test_no_model_is_built_without_a_matching_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert build_model("anthropic", None) is None
    assert build_model("openai", None) is None


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-sonnet-4-5-20250929", {}),  # no effort knob; no thinking unless asked
        ("anthropic", "claude-haiku-4-5", {}),
        ("anthropic", "claude-sonnet-4-6", {"output_config": {"effort": "low"}}),
        ("anthropic", "claude-sonnet-5-5", {"output_config": {"effort": "low"}}),
        ("anthropic", "claude-opus-5-5", {"output_config": {"effort": "low"}}),
        ("openai", "gpt-4o", {}),
        ("openai", "gpt-5-mini", {"reasoning_effort": "minimal"}),
        ("openai", "o4-mini", {"reasoning_effort": "low"}),
    ],
)
def test_judges_run_at_their_lowest_effort(provider, model, expected):
    assert minimal_effort_kwargs(provider, model) == expected
