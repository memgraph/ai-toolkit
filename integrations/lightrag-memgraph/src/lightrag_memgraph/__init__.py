"""
LightRAG integration with Memgraph.

This package provides a wrapper around LightRAG that uses Memgraph as the graph storage backend.
"""

from .core import MemgraphLightRAGWrapper


# Patch lightrag.llm.anthropic for current Anthropic SDK:
# - require max_tokens; use top-level system= (not "system" role in messages).
def _patch_anthropic() -> None:
    try:
        import os
        import logging
        from typing import Any, Union
        from collections.abc import AsyncIterator

        import lightrag.llm.anthropic as _mod
        from anthropic import (
            AsyncAnthropic,
            APIConnectionError,
            RateLimitError,
            APITimeoutError,
        )
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )
        from lightrag.utils import safe_unicode_decode, logger, VERBOSE_DEBUG
        from lightrag.api import __api_version__

        _orig = _mod.anthropic_complete_if_cache
        if getattr(_orig, "_lightrag_memgraph_patched", False):
            return

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, APITimeoutError)
            ),
        )
        async def _wrapped(
            model: str,
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            enable_cot: bool = False,
            base_url: str | None = None,
            api_key: str | None = None,
            **kwargs: Any,
        ) -> Union[str, AsyncIterator[str]]:
            if history_messages is None:
                history_messages = []
            kwargs.setdefault("max_tokens", 4096)
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            default_headers = {
                "User-Agent": f"Mozilla/5.0 LightRAG/{__api_version__}",
                "Content-Type": "application/json",
            }
            kwargs.pop("hashing_kv", None)
            kwargs.pop("keyword_extraction", None)
            timeout = kwargs.pop("timeout", None)

            client = (
                AsyncAnthropic(
                    default_headers=default_headers, api_key=api_key, timeout=timeout
                )
                if base_url is None
                else AsyncAnthropic(
                    base_url=base_url,
                    default_headers=default_headers,
                    api_key=api_key,
                    timeout=timeout,
                )
            )

            # API expects top-level system=, not a message with role "system"
            messages: list[dict[str, Any]] = list(history_messages)
            messages.append({"role": "user", "content": prompt})

            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs,
            }
            if system_prompt:
                create_kwargs["system"] = system_prompt

            if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
                logging.getLogger("anthropic").setLevel(logging.INFO)

            response = await client.messages.create(**create_kwargs)

            # Consume stream and return a single string (caller expects str, not AsyncIterator)
            # Only content_block_delta events have delta.text; message_delta etc. have no .text
            parts: list[str] = []
            async for event in response:
                content = (
                    getattr(getattr(event, "delta", None), "text", None)
                    if hasattr(event, "delta")
                    else None
                )
                if not content:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                parts.append(content)
            return "".join(parts)

        _wrapped._lightrag_memgraph_patched = True  # type: ignore[attr-defined]
        _mod.anthropic_complete_if_cache = _wrapped
    except Exception:
        pass


_patch_anthropic()

__all__ = ["MemgraphLightRAGWrapper"]
