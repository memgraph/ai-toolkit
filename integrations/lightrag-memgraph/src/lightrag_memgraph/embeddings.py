"""Default embedding function backed by Memgraph's own `embeddings` MAGE module.

https://memgraph.com/docs/advanced-algorithms/available-algorithms/embeddings

`embeddings.text()` runs a local Hugging Face SentenceTransformer model
*inside* Memgraph (the default model is `all-MiniLM-L6-v2`, 384 dims), so
using it as the default `embedding_func` means a zero-config
`MemgraphLightRAGWrapper` needs no external API key and incurs no per-call
embedding cost -- unlike `openai_embed`, which was the previous silent
default (see issue #222). It also avoids pulling `torch` /
`sentence-transformers` into this package: the model runs on the Memgraph
server, reusing the same driver already used for storage.
"""

from __future__ import annotations

import numpy as np
from lightrag.utils import EmbeddingFunc

from ._connection import get_database, get_driver

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_MAX_TOKEN_SIZE = 256  # all-MiniLM-L6-v2's trained max sequence length


def _build_embed_func(model_name: str):
    async def _embed(texts: list[str]) -> np.ndarray:
        driver = await get_driver()
        async with driver.session(database=get_database(), default_access_mode="READ") as session:
            result = await session.run(
                "CALL embeddings.text($texts, $config) YIELD success, embeddings, dimension "
                "RETURN success, embeddings, dimension",
                texts=texts,
                config={"model_name": model_name},
            )
            record = await result.single()
            await result.consume()
        if record is None or not record["success"]:
            raise RuntimeError(
                f"Memgraph's `embeddings.text()` procedure failed for model '{model_name}'. "
                "It requires Memgraph MAGE with the `embeddings` module loaded -- see "
                "https://memgraph.com/docs/advanced-algorithms/available-algorithms/embeddings. "
                "Pass an explicit embedding_func to MemgraphLightRAGWrapper.initialize() to use "
                "a different embedding provider instead."
            )
        return np.array(record["embeddings"], dtype=np.float32)

    return _embed


def build_memgraph_sentence_embed(
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    max_token_size: int = DEFAULT_MAX_TOKEN_SIZE,
) -> EmbeddingFunc:
    """Build an `EmbeddingFunc` that calls Memgraph's `embeddings.text()` for a given model.

    `all-MiniLM-L6-v2` (384 dims) is the module's own default and needs no
    `embedding_dim` override. For any other `model_name`, pass the matching
    `embedding_dim` -- the module doesn't report it up front, and a mismatch
    will raise (via `EmbeddingFunc`'s own dimension check) on first use.
    """
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        model_name=model_name,
        func=_build_embed_func(model_name),
    )


memgraph_sentence_embed = build_memgraph_sentence_embed()
