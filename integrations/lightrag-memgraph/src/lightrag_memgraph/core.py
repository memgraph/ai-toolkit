import os
from typing import Optional
import logging

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
os.environ["MEMGRAPH_URI"] = MEMGRAPH_URI


class DummyEmbed:
    def __init__(self, dim: int = 1):
        self.embedding_dim = dim

    async def __call__(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.embedding_dim), dtype=float)


class MemgraphLightRAGWrapper:
    def __init__(
        self,
        log_level: str = "INFO",
        disable_embeddings: bool = False,
    ):
        self.log_level = log_level
        self.disable_embeddings = disable_embeddings
        self.rag: Optional[LightRAG] = None

    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py
    # https://github.com/HKUDS/LightRAG/blob/main/lightrag/llm
    async def initialize(self, **lightrag_kwargs) -> None:
        setup_logger("lightrag", level=self.log_level)
        logging.getLogger("nano-vectordb").setLevel(self.log_level)
        logging.getLogger("pikepdf").setLevel(self.log_level)
        if self.disable_embeddings:
            lightrag_kwargs["embedding_func"] = DummyEmbed(dim=1)
            lightrag_kwargs["vector_storage"] = "NanoVectorDBStorage"
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
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key."
                )
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
