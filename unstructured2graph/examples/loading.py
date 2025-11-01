import os
import logging

import asyncio
import shutil
from lightrag_memgraph import MemgraphLightRAGWrapper
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import from_unstructured
from sources import SOURCES

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIGHTRAG_DIR = os.path.join(SCRIPT_DIR, "..", "lightrag_storage.out")


async def from_unstructured_with_prep():
    # Delete & create LightRAG working directory.
    lightrag_log_file = os.path.join(SCRIPT_DIR, "..", "lightrag.log")
    if os.path.exists(lightrag_log_file):
        os.remove(lightrag_log_file)
    if os.path.exists(LIGHTRAG_DIR):
        shutil.rmtree(LIGHTRAG_DIR)
    if not os.path.exists(LIGHTRAG_DIR):
        os.mkdir(LIGHTRAG_DIR)

    # Cleanup Memgraph database.
    memgraph = Memgraph()
    memgraph.query("MATCH (n) DETACH DELETE n;")

    lightrag_wrapper = MemgraphLightRAGWrapper(disable_embeddings=True)
    await lightrag_wrapper.initialize(
        working_dir=LIGHTRAG_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await from_unstructured(SOURCES, memgraph, lightrag_wrapper)
    await lightrag_wrapper.afinalize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(from_unstructured_with_prep())
