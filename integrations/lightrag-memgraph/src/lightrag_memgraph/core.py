import logging
import os

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

from .registry import register_memgraph_storage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _bridge_lightrag_env_names() -> None:
    """Bridge the canonical toolbox env names to LightRAG's graph-backend names.

    The toolbox -- and this integration's KV/vector/doc-status storages via
    ``AsyncMemgraph``/``memgraph_env`` -- read the canonical ``MEMGRAPH_URL`` /
    ``MEMGRAPH_USER``. LightRAG's bundled graph backend
    (``lightrag.kg.memgraph_impl.MemgraphStorage``) instead reads
    ``MEMGRAPH_URI`` / ``MEMGRAPH_USERNAME``. Mirror the canonical names onto the
    LightRAG names (only when the LightRAG name is unset, so an explicit override
    is never clobbered) so the graph backend and our storages resolve to the
    SAME instance from a single env set. ``MEMGRAPH_PASSWORD`` /
    ``MEMGRAPH_DATABASE`` already share names, so no bridge is needed for them.
    """
    if "MEMGRAPH_URL" in os.environ and "MEMGRAPH_URI" not in os.environ:
        os.environ["MEMGRAPH_URI"] = os.environ["MEMGRAPH_URL"]
    if "MEMGRAPH_USER" in os.environ and "MEMGRAPH_USERNAME" not in os.environ:
        os.environ["MEMGRAPH_USERNAME"] = os.environ["MEMGRAPH_USER"]


class MemgraphLightRAGWrapper:
    def __init__(
        self,
        log_level: str = "INFO",
        full_memgraph_persistence: bool = True,
    ):
        """Wrap LightRAG configured to use Memgraph as its storage backend.

        LightRAG's entire working state -- graph, key/value, vector and
        doc-status stores -- is persisted in Memgraph. Vectors always go to
        Memgraph's native vector index, so a real ``embedding_func`` is required
        (there is no local vector-database fallback and no
        embedding-disabled mode).

        Args:
            log_level: Logging level for LightRAG's loggers.
            full_memgraph_persistence: If True (default), LightRAG's KV, vector
                and doc-status stores are persisted to Memgraph in addition to
                the graph. If False, only the graph is stored in Memgraph and
                the other stores use LightRAG's file-based defaults in
                ``working_dir``.
        """
        self.log_level = log_level
        self.full_memgraph_persistence = full_memgraph_persistence
        self.rag: LightRAG | None = None

    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py
    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/llm
    async def initialize(self, **lightrag_kwargs) -> None:
        setup_logger("lightrag", level=self.log_level)
        logging.getLogger("pikepdf").setLevel(self.log_level)
        # Bridge the canonical toolbox env names to LightRAG's graph-backend
        # names so the built-in MemgraphStorage and our storages hit the same
        # instance from a single env set.
        _bridge_lightrag_env_names()
        if self.full_memgraph_persistence:
            # Register the Memgraph KV/vector/doc-status backends so LightRAG
            # accepts them by name, then route every store to Memgraph. Vectors
            # always persist to Memgraph's native vector index (a real embedding
            # function is required). Explicit caller overrides win.
            register_memgraph_storage()
            lightrag_kwargs.setdefault("kv_storage", "MemgraphKVStorage")
            lightrag_kwargs.setdefault("doc_status_storage", "MemgraphDocStatusStorage")
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
