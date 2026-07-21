import logging
import os

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import logger, setup_logger

from .embeddings import DEFAULT_EMBEDDING_DIM, DEFAULT_MODEL_NAME, memgraph_sentence_embed
from .registry import register_memgraph_storage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _bridge_lightrag_env_names() -> None:
    """Mirror MEMGRAPH_URL/USER onto LightRAG's graph-backend names (MEMGRAPH_URI/USERNAME) so
    both resolve to the same instance from one env set. Never overwrites an explicit override.

    If neither name is set, default MEMGRAPH_URI to the local default Memgraph instance, matching
    the zero-config behaviour of a locally-running default Memgraph.
    """
    if "MEMGRAPH_URL" in os.environ and "MEMGRAPH_URI" not in os.environ:
        os.environ["MEMGRAPH_URI"] = os.environ["MEMGRAPH_URL"]
    os.environ.setdefault("MEMGRAPH_URI", "bolt://localhost:7687")
    if "MEMGRAPH_USER" in os.environ and "MEMGRAPH_USERNAME" not in os.environ:
        os.environ["MEMGRAPH_USERNAME"] = os.environ["MEMGRAPH_USER"]


def _apply_lightrag_defaults(lightrag_kwargs: dict) -> None:
    """Fill in `llm_model_func`/`embedding_func` defaults in place.

    `embedding_func` defaults to Memgraph's own local sentence-transformer
    (via the `embeddings` MAGE module, see `embeddings.py`), not `openai_embed`
    -- a zero-config wrapper should not silently make billed OpenAI calls.
    Defaulting is logged since it changes what network calls `ainsert`/`aquery`
    make.
    """
    if "llm_model_func" not in lightrag_kwargs:
        lightrag_kwargs["llm_model_func"] = gpt_4o_mini_complete
    if "embedding_func" not in lightrag_kwargs:
        logger.warning(
            "embedding_func not provided to MemgraphLightRAGWrapper.initialize(); defaulting to "
            f"Memgraph's local sentence-transformer embeddings ({DEFAULT_MODEL_NAME}, "
            f"{DEFAULT_EMBEDDING_DIM} dims) via the `embeddings` MAGE module -- no API key or "
            "external cost involved. Pass an explicit embedding_func (e.g. openai_embed) to use "
            "a different provider."
        )
        lightrag_kwargs["embedding_func"] = memgraph_sentence_embed
    if lightrag_kwargs["llm_model_func"] == gpt_4o_mini_complete or lightrag_kwargs["embedding_func"] == openai_embed:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise OSError("OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key.")


class MemgraphLightRAGWrapper:
    def __init__(
        self,
        log_level: str = "INFO",
        full_memgraph_persistence: bool = True,
    ):
        """Wrap LightRAG configured to use Memgraph as its storage backend.

        Vectors always go to Memgraph's native vector index, so a real
        ``embedding_func`` is required; if omitted from ``initialize()``, it
        defaults to Memgraph's own local sentence-transformer (see
        ``embeddings.py``), not a billed external API.

        Args:
            log_level: Logging level for LightRAG's loggers.
            full_memgraph_persistence: If True (default), KV/vector/doc-status
                also persist to Memgraph. If False, only the graph does, and
                the rest use LightRAG's file-based defaults in `working_dir`.
        """
        self.log_level = log_level
        self.full_memgraph_persistence = full_memgraph_persistence
        self.rag: LightRAG | None = None

    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py
    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/llm
    async def initialize(self, **lightrag_kwargs) -> None:
        setup_logger("lightrag", level=self.log_level)
        logging.getLogger("pikepdf").setLevel(self.log_level)
        _bridge_lightrag_env_names()
        if self.full_memgraph_persistence:
            # Route every store to Memgraph; explicit caller overrides still win.
            register_memgraph_storage()
            lightrag_kwargs.setdefault("kv_storage", "MemgraphKVStorage")
            lightrag_kwargs.setdefault("doc_status_storage", "MemgraphDocStatusStorage")
            lightrag_kwargs.setdefault("vector_storage", "MemgraphVectorStorage")
        if "working_dir" in lightrag_kwargs:
            working_dir = lightrag_kwargs["working_dir"]
            if not os.path.exists(working_dir):
                os.mkdir(working_dir)
        _apply_lightrag_defaults(lightrag_kwargs)
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
