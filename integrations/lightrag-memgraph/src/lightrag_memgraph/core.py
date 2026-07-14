import logging
import os

import numpy as np
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger

from .registry import register_memgraph_storages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read from MEMGRAPH_URL (consistent with memgraph-toolbox) and set MEMGRAPH_URI for LightRAG
MEMGRAPH_URL = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
os.environ["MEMGRAPH_URI"] = MEMGRAPH_URL


def _dummy_embedding_func(dim: int = 1) -> EmbeddingFunc:
    """Build an EmbeddingFunc that returns constant embeddings (for disable_embeddings=True)."""

    async def _dummy_embed_func(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), dim), dtype=float)

    return EmbeddingFunc(embedding_dim=dim, func=_dummy_embed_func)


class MemgraphLightRAGWrapper:
    def __init__(
        self,
        log_level: str = "INFO",
        disable_embeddings: bool = False,
        full_memgraph_persistence: bool = True,
    ):
        """Wrap LightRAG configured to use Memgraph as its storage backend.

        Args:
            log_level: Logging level for LightRAG's loggers.
            disable_embeddings: If True, embeddings are stubbed out and the
                vector store falls back to LightRAG's local NanoVectorDB (a
                real vector index needs a real embedding dimension). KV and
                doc-status still persist to Memgraph when
                ``full_memgraph_persistence`` is enabled.
            full_memgraph_persistence: If True (default), LightRAG's KV,
                doc-status and (when embeddings are enabled) vector stores are
                persisted to Memgraph in addition to the graph. If False, only
                the graph is stored in Memgraph and the other stores use
                LightRAG's file-based defaults in ``working_dir``.
        """
        self.log_level = log_level
        self.disable_embeddings = disable_embeddings
        self.full_memgraph_persistence = full_memgraph_persistence
        self.rag: LightRAG | None = None

    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py
    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/llm
    async def initialize(self, **lightrag_kwargs) -> None:
        setup_logger("lightrag", level=self.log_level)
        logging.getLogger("nano-vectordb").setLevel(self.log_level)
        logging.getLogger("pikepdf").setLevel(self.log_level)
        if self.disable_embeddings:
            lightrag_kwargs["embedding_func"] = _dummy_embedding_func(dim=1)
            lightrag_kwargs["vector_storage"] = "NanoVectorDBStorage"
        if self.full_memgraph_persistence:
            # Register the Memgraph KV/vector/doc-status backends so LightRAG
            # accepts them by name, then route each store to Memgraph. KV and
            # doc-status always go to Memgraph; the vector store only goes to
            # Memgraph when embeddings are enabled (a real vector index needs a
            # real embedding dimension), otherwise it keeps the local
            # NanoVectorDB fallback set above. Explicit caller overrides win.
            register_memgraph_storages()
            lightrag_kwargs.setdefault("kv_storage", "MemgraphKVStorage")
            lightrag_kwargs.setdefault("doc_status_storage", "MemgraphDocStatusStorage")
            if not self.disable_embeddings:
                lightrag_kwargs.setdefault("vector_storage", "MemgraphVectorStorage")
        if "working_dir" in lightrag_kwargs:
            working_dir = lightrag_kwargs["working_dir"]
            if not os.path.exists(working_dir):
                os.mkdir(working_dir)
        if "llm_model_func" not in lightrag_kwargs:
            lightrag_kwargs["llm_model_func"] = gpt_4o_mini_complete
        if "embedding_func" not in lightrag_kwargs:
            lightrag_kwargs["embedding_func"] = openai_embed
        if (
            lightrag_kwargs["llm_model_func"] == gpt_4o_mini_complete
            or lightrag_kwargs["embedding_func"] == openai_embed
        ):
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise OSError("OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key.")
        self.rag = LightRAG(graph_storage="MemgraphStorage", **lightrag_kwargs)
        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    def get_lightrag(self) -> LightRAG:
        if self.rag is None:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        return self.rag

    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py
    async def ainsert(self, **kwargs) -> None:
        """
        Example call: await lightrag_wrapper.ainsert(input=text, file_paths=[id])

        If you want to inject info under each entity about the source input,
        pass file_paths as a list of strings (ids don't work, not written under each entity).
        """
        if self.rag is None:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        await self.rag.ainsert(**kwargs)

    async def afinalize(self) -> None:
        if self.rag is None:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        await self.rag.finalize_storages()
