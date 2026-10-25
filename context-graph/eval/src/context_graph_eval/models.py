"""The models a run calls: the answering model, and the deepeval rubric judge.

Each takes a ``provider:model_id`` spec. A bare model id keeps the role's
default provider. The benchmarks' own judges (``official_judge``,
``beam_judge``) call OpenAI directly with their upstream's pinned settings
and are not built here.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

#: The answering model every official number was measured with. Pinned rather
#: than left to deepeval's default, which moves with deepeval releases.
DEFAULT_AGENT_MODEL = "openai:gpt-4.1"

#: Dated rather than a moving alias (#304): a judge that changes underneath
#: you silently invalidates every prior baseline.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

#: Each role's provider for a bare model id. #304 asks only that the judge's
#: provider differ from the pipeline's; these defaults satisfy that today.
DEFAULT_JUDGE_PROVIDER = "anthropic"
DEFAULT_AGENT_PROVIDER = "openai"

#: provider -> (API key env var, fallback model id used when none is given).
_PROVIDER_KEYS: dict[str, tuple[str, str | None]] = {
    "anthropic": ("ANTHROPIC_API_KEY", DEFAULT_JUDGE_MODEL),
    # GPTModel has its own built-in default when given no model id.
    "openai": ("OPENAI_API_KEY", None),
}

#: Anthropic model families that accept output_config.effort. Older ones
#: (Sonnet 4.5, Haiku 4.5) reject it -- and run without extended thinking
#: unless it is requested, which is already their minimum.
_ANTHROPIC_EFFORT_MODELS = re.compile(r"^claude-(opus|sonnet|fable|mythos)-(4-[6-9]|[5-9])")
#: OpenAI reasoning models, and the lowest reasoning_effort each accepts.
_OPENAI_REASONING_MODELS = ((re.compile(r"^gpt-5"), "minimal"), (re.compile(r"^o[1-9]"), "low"))


def resolve_llm_credentials() -> None:
    """Fill LLM key env vars from context-graph's config file where unset (ADR 0002, ADR 0007).

    Eval runs standalone, so nothing has overlaid the config for it. Existing
    env always wins, so an explicit key stays in charge. Only the LLM keys:
    eval picks its own Memgraph. memgraph-context-graph is an optional
    extra -- skip quietly without it.
    """
    try:
        from memgraph_context_graph.adapters._identity import fill_env_from_config
    except ImportError:
        return
    fill_env_from_config(memgraph=False)


def parse_model_spec(spec: str | None, *, default_provider: str) -> tuple[str, str | None]:
    """Split a 'provider:model_id' value into (provider, model_id).

    A bare model id, or None, keeps ``default_provider``; None also leaves the
    model id None, meaning that provider's own default.
    """
    if spec and ":" in spec:
        provider, _, model_id = spec.partition(":")
        return provider, (model_id or None)
    return default_provider, spec


def resolved_spec(provider: str, model_id: str | None) -> str:
    """The 'provider:model_id' actually used, after that provider's fallback, for recording on a run."""
    _, fallback = _PROVIDER_KEYS.get(provider, (None, None))
    return f"{provider}:{model_id or fallback or 'default'}"


def minimal_effort_kwargs(provider: str, model_id: str) -> dict:
    """Generation kwargs that run ``model_id`` at its lowest reasoning effort, or {} when it has no such knob.

    A judge grades a short answer against a key, and the eval's cost is
    dominated by judge calls. Effort is set only where the model accepts it,
    since passing it elsewhere is a 400.
    """
    if provider == "anthropic" and _ANTHROPIC_EFFORT_MODELS.match(model_id):
        return {"output_config": {"effort": "low"}}
    if provider == "openai":
        for pattern, effort in _OPENAI_REASONING_MODELS:
            if pattern.match(model_id):
                return {"reasoning_effort": effort}
    return {}


def build_model(provider: str, model_id: str | None, *, minimal_effort: bool = False) -> Any | None:
    """A deepeval model for ``provider``, or None when it is unknown, has no API key, or fails to build."""
    if provider not in _PROVIDER_KEYS:
        print(f"unknown model provider {provider!r}; supported: {', '.join(_PROVIDER_KEYS)}", file=sys.stderr)
        return None
    env_var, _ = _PROVIDER_KEYS[provider]
    key = os.environ.get(env_var)
    if not key:
        return None
    try:
        if provider == "anthropic":
            from deepeval.models import AnthropicModel

            _clear_deepeval_anthropic_secret()
            model = model_id or DEFAULT_JUDGE_MODEL
            kwargs = minimal_effort_kwargs(provider, model) if minimal_effort else {}
            return AnthropicModel(model=model, _anthropic_api_key=key, generation_kwargs=kwargs)
        from deepeval.models import GPTModel

        kwargs = minimal_effort_kwargs(provider, model_id or "") if minimal_effort else {}
        return GPTModel(model=model_id, generation_kwargs=kwargs) if model_id else GPTModel(generation_kwargs=kwargs)
    except Exception as exc:
        print(f"could not build {provider} model {model_id!r}: {exc}", file=sys.stderr)
        return None


def _clear_deepeval_anthropic_secret() -> None:
    """Make deepeval's Anthropic client usable at all.

    ``AnthropicModel._build_client`` resolves
    ``settings.ANTHROPIC_API_KEY or self._anthropic_api_key``. The settings
    value is a pydantic ``SecretStr``, which is truthy and therefore always
    wins that ``or`` -- and httpx then refuses it outright ("Header value must
    be str or bytes, not SecretStr") before a single request is made. Passing
    the key explicitly cannot help while settings still holds one.

    Settings are cached at first access, so the cached object is corrected
    directly. The environment variable is deliberately left alone: clearing it
    would make a later ``build_model`` return None, silently dropping the judge.
    """
    try:
        from deepeval.config.settings import get_settings

        get_settings().ANTHROPIC_API_KEY = None
    except Exception:
        pass
