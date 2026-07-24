"""Simple tests to verify lightrag-memgraph installation."""


def test_import_lightrag_memgraph():
    """Test that lightrag_memgraph can be imported."""
    from lightrag_memgraph import MemgraphLightRAGWrapper

    assert MemgraphLightRAGWrapper is not None


def test_wrapper_instantiation():
    """Test that MemgraphLightRAGWrapper can be instantiated."""
    from lightrag_memgraph import MemgraphLightRAGWrapper

    wrapper = MemgraphLightRAGWrapper()
    assert wrapper is not None
    assert wrapper.rag is None
    assert wrapper.log_level == "INFO"
    assert wrapper.full_memgraph_persistence is True


def test_wrapper_with_custom_params():
    """Test that MemgraphLightRAGWrapper accepts custom parameters."""
    from lightrag_memgraph import MemgraphLightRAGWrapper

    wrapper = MemgraphLightRAGWrapper(
        log_level="DEBUG",
        full_memgraph_persistence=False,
    )
    assert wrapper.log_level == "DEBUG"
    assert wrapper.full_memgraph_persistence is False


def test_embedding_defaults_are_exported_from_top_level():
    """DEFAULT_EMBEDDING_DIM/DEFAULT_MODEL_NAME must be reachable without
    digging into the embeddings submodule, so downstream packages can treat
    them as the single source of truth for the default embedding dimension."""
    from lightrag_memgraph import DEFAULT_EMBEDDING_DIM, DEFAULT_MODEL_NAME, embeddings

    assert DEFAULT_EMBEDDING_DIM == embeddings.DEFAULT_EMBEDDING_DIM
    assert DEFAULT_MODEL_NAME == embeddings.DEFAULT_MODEL_NAME
