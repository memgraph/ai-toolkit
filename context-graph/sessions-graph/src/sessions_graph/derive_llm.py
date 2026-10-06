"""The LLM derivation calls: ``hygm.Llm`` over whichever provider's key is configured.

Anthropic when ``ANTHROPIC_API_KEY`` is set, else OpenAI with
``OPENAI_API_KEY``. Both are asked for a reply matching the stage's JSON
schema through the provider's structured-output mode. (Current Claude models
refuse a forced tool call, so that route is out.) Structured outputs need
every object closed with ``additionalProperties: false``; hygm's schemas
leave that open, so it's added here. The keys reach the environment from the config file (see
``cli._fill_env_from_context_graph_config``), never from the hook's own.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

#: The default model per provider; ``sessions-graph derive --model`` overrides it.
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-5"}

# A pruned or consolidated vocabulary is a few thousand tokens; this leaves room.
_MAX_OUTPUT_TOKENS = 16_000
_STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class NoLlmConfiguredError(RuntimeError):
    """Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set."""


@dataclass
class LlmUsage:
    """What a derivation run spent, for its report."""

    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class DerivationLlm:
    """``hygm.Llm``: ``llm(system, prompt, schema) -> reply``, tallying usage.

    Args:
        model: Overrides the provider's default model.

    Raises:
        NoLlmConfiguredError: if no provider key is set.
    """

    def __init__(self, model: str | None = None) -> None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise NoLlmConfiguredError(
                "derivation needs an LLM: set llm.anthropic_api_key or llm.openai_api_key in the config file"
            )
        self.usage = LlmUsage(provider=provider, model=model or DEFAULT_MODELS[provider])
        self._client: Any = None

    def __call__(self, system: str, prompt: str, schema: Any) -> dict[str, Any]:
        """One call; the reply as a mapping matching `schema`.

        Raises:
            ValueError: if the reply carries no structured answer.
        """
        reply = (
            self._anthropic(system, prompt, schema)
            if self.usage.provider == "anthropic"
            else self._openai(system, prompt, schema)
        )
        self.usage.calls += 1
        return reply

    def _anthropic(self, system: str, prompt: str, schema: Any) -> dict[str, Any]:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        response = self._client.beta.messages.create(
            model=self.usage.model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            betas=[_STRUCTURED_OUTPUTS_BETA],
            output_format={"type": "json_schema", "schema": closed(schema)},
        )
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ValueError(f"{self.usage.model} gave no structured answer (stop reason {response.stop_reason})")
        return json.loads(text)

    def _openai(self, system: str, prompt: str, schema: Any) -> dict[str, Any]:
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        response = self._client.chat.completions.create(
            model=self.usage.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": closed(schema)}},
        )
        if response.usage is not None:
            self.usage.input_tokens += response.usage.prompt_tokens
            self.usage.output_tokens += response.usage.completion_tokens
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"{self.usage.model} gave no structured answer")
        return json.loads(content)


def closed(schema: Any) -> Any:
    """`schema` with every object closed to properties it doesn't declare, as structured outputs require."""
    if isinstance(schema, dict):
        out = {key: closed(value) for key, value in schema.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [closed(value) for value in schema]
    return schema
