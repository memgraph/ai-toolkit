"""Unit tests for lightrag_memgraph.core's env-name bridging and LLM/embedding defaults.

These don't need a live Memgraph: _bridge_lightrag_env_names only checks what it
writes into os.environ, and _apply_lightrag_defaults only mutates a plain dict
(the embedding/LLM functions it defaults to aren't called until LightRAG actually
runs an insert/query).
"""

from __future__ import annotations

import os

import pytest
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import logger as lightrag_logger

from lightrag_memgraph.core import _apply_lightrag_defaults, _bridge_lightrag_env_names
from lightrag_memgraph.embeddings import memgraph_sentence_embed

_MEMGRAPH_ENV_NAMES = ("MEMGRAPH_URL", "MEMGRAPH_URI", "MEMGRAPH_USER", "MEMGRAPH_USERNAME")


@pytest.fixture(autouse=True)
def _clear_memgraph_env(monkeypatch):
    """Isolate every test from both the ambient environment and each other.

    _bridge_lightrag_env_names writes to os.environ directly (not through
    monkeypatch), so teardown must scrub it explicitly or a value set by one
    test would leak into the next.
    """
    for name in _MEMGRAPH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in _MEMGRAPH_ENV_NAMES:
        os.environ.pop(name, None)


def test_bridge_defaults_uri_to_local_memgraph_when_nothing_set():
    """Regression test for #217: with zero Memgraph env vars set, MEMGRAPH_URI
    must still end up pointing at the local default Memgraph, matching the
    behaviour on main (which defaulted it unconditionally at import time).
    """
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_URI"] == "bolt://localhost:7687"


def test_bridge_mirrors_url_onto_uri(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_URL", "bolt://example:1234")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_URI"] == "bolt://example:1234"


def test_bridge_never_overwrites_explicit_uri(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_URL", "bolt://example:1234")
    monkeypatch.setenv("MEMGRAPH_URI", "bolt://explicit:9999")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_URI"] == "bolt://explicit:9999"


def test_bridge_mirrors_user_onto_username(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_USER", "alice")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_USERNAME"] == "alice"


def test_bridge_never_overwrites_explicit_username(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_USER", "alice")
    monkeypatch.setenv("MEMGRAPH_USERNAME", "bob")
    _bridge_lightrag_env_names()
    assert os.environ["MEMGRAPH_USERNAME"] == "bob"


# --- _apply_lightrag_defaults -------------------------------------------------


def test_embedding_func_defaults_to_memgraph_sentence_embed_not_openai(monkeypatch):
    """Regression test for #222: a caller who omits embedding_func must not
    silently get billed OpenAI calls. The default is Memgraph's own local
    sentence-transformer instead, which needs no OPENAI_API_KEY.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    kwargs = {"llm_model_func": lambda *a, **kw: None}  # non-default LLM, so no key is required
    _apply_lightrag_defaults(kwargs)
    assert kwargs["embedding_func"] is memgraph_sentence_embed


def test_defaulting_embedding_func_logs_a_warning(monkeypatch):
    """lightrag's shared logger has propagate=False (set at import time), so
    caplog can't see it via the root logger; patch .warning directly instead.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    warnings: list[str] = []
    monkeypatch.setattr(lightrag_logger, "warning", lambda msg, *a, **kw: warnings.append(msg))
    kwargs = {"llm_model_func": lambda *a, **kw: None}
    _apply_lightrag_defaults(kwargs)
    assert any("embedding_func" in w for w in warnings)


def test_explicit_embedding_func_is_not_overridden(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def custom_embed(texts):
        return texts

    kwargs = {"llm_model_func": lambda *a, **kw: None, "embedding_func": custom_embed}
    _apply_lightrag_defaults(kwargs)
    assert kwargs["embedding_func"] is custom_embed


def test_llm_model_func_defaults_to_gpt_4o_mini_complete(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    kwargs = {"embedding_func": memgraph_sentence_embed}
    _apply_lightrag_defaults(kwargs)
    assert kwargs["llm_model_func"] is gpt_4o_mini_complete


def test_requires_openai_api_key_when_llm_defaults_to_gpt_4o_mini(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    kwargs = {"embedding_func": memgraph_sentence_embed}
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        _apply_lightrag_defaults(kwargs)


def test_requires_openai_api_key_when_embedding_func_is_explicitly_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    kwargs = {"llm_model_func": lambda *a, **kw: None, "embedding_func": openai_embed}
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        _apply_lightrag_defaults(kwargs)


def test_no_openai_key_required_with_non_openai_llm_and_embedding_default(monkeypatch):
    """The whole point of #222's fix: a Claude-only setup (no OPENAI_API_KEY at
    all) must be able to omit embedding_func and still initialize.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    kwargs = {"llm_model_func": lambda *a, **kw: None}
    _apply_lightrag_defaults(kwargs)  # must not raise
    assert kwargs["embedding_func"] is memgraph_sentence_embed
