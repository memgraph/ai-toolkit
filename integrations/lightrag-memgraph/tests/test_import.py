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
    assert wrapper.disable_embeddings is False


def test_wrapper_with_custom_params():
    """Test that MemgraphLightRAGWrapper accepts custom parameters."""
    from lightrag_memgraph import MemgraphLightRAGWrapper

    wrapper = MemgraphLightRAGWrapper(
        log_level="DEBUG",
        disable_embeddings=True,
    )
    assert wrapper.log_level == "DEBUG"
    assert wrapper.disable_embeddings is True
