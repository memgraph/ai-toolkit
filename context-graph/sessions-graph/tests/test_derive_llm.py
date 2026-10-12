"""The derivation LLM: which provider it picks, and what it reads back. The SDK clients are stood in for."""

from types import SimpleNamespace

import pytest
from sessions_graph.derive_llm import DerivationLlm, NoLlmConfiguredError, closed

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class _Anthropic:
    def __init__(self, content):
        self.sent: dict = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._content = content

    def _create(self, **kwargs):
        self.sent = kwargs
        return SimpleNamespace(
            content=self._content, usage=SimpleNamespace(input_tokens=100, output_tokens=7), stop_reason="end_turn"
        )


def test_anthropic_is_preferred_and_answers_through_structured_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    llm = DerivationLlm()
    client = _Anthropic([SimpleNamespace(type="text", text='{"x": 3}')])
    llm._client = client

    assert llm("system", "prompt", SCHEMA) == {"x": 3}
    assert client.sent["output_format"] == {"type": "json_schema", "schema": closed(SCHEMA)}
    assert (llm.usage.provider, llm.usage.model, llm.usage.calls) == ("anthropic", "claude-sonnet-5-5", 1)
    assert (llm.usage.input_tokens, llm.usage.output_tokens) == (100, 7)


def test_a_reply_without_text_is_an_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    llm = DerivationLlm()
    llm._client = _Anthropic([])

    with pytest.raises(ValueError, match="no structured answer"):
        llm("system", "prompt", SCHEMA)


def test_openai_answers_through_a_json_schema_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    llm = DerivationLlm(model="gpt-test")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content='{"x": 5}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=50, completion_tokens=3)
        )

    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    assert llm("system", "prompt", SCHEMA) == {"x": 5}
    assert sent["response_format"]["json_schema"]["schema"] == closed(SCHEMA)
    assert (llm.usage.provider, llm.usage.model, llm.usage.input_tokens) == ("openai", "gpt-test", 50)


def test_no_key_is_refused_with_where_to_set_one():
    with pytest.raises(NoLlmConfiguredError, match=r"llm\.anthropic_api_key"):
        DerivationLlm()


def test_closed_shuts_every_object_but_leaves_the_rest():
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}}}

    assert closed(schema) == {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "object", "additionalProperties": False}}},
        "additionalProperties": False,
    }
