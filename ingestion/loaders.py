import os
import uuid

import asyncio
import shutil
from lightrag_memgraph import MemgraphLightRAGWrapper
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from parsers import parse_pdf, parse_docx, parse_xls, parse_url
from memgraph_toolbox.api.memgraph import Memgraph

from memgraph import create_nodes_from_list, connect_chunks_to_entities

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKING_DIR = os.path.join(SCRIPT_DIR, "lightrag_storage.out")


async def from_unstructured(sources):
    documents = []

    ## PARSE
    # NOTE: Each element is a list of chunks from a Document (each source is an abstract Document)
    # TODO: Add chunking support + the way of referencting original sources and chunks once they end up in the database.
    #     * NOTE: LightRAG uses { source_id: "chunk-ID..." } to reference its chunks.
    for source in sources:
        if source.endswith(".pdf"):
            documents.append(parse_pdf(source))
        elif source.endswith(".docx") or source.endswith(".doc"):
            documents.append(parse_docx(source))
        elif source.endswith(".xlsx") or source.endswith(".xls"):
            documents.append(parse_xls(source))
        elif source.startswith("http"):
            documents.append(parse_url(source))
        else:
            raise ValueError(f"Unsupported source type: {source}")

    ## INGEST
    # TODO: Seems like all chunks are processed without the LLM KEY -> imporove.
    lightrag_wrapper = MemgraphLightRAGWrapper(disable_embeddings=True)
    await lightrag_wrapper.initialize(
        working_dir="./lightrag_storage.out",
        # set those two as defaults under the wrapper
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    for document in documents:
        memgraph_node_props = []
        for chunk in document:
            chunk_id = f"chunk-{uuid.uuid4()}"
            await lightrag_wrapper.ainsert(input=chunk, file_paths=[chunk_id])
            memgraph_node_props.append({"id": chunk_id, "text": chunk})
        create_nodes_from_list(memgraph, memgraph_node_props, "Chunk", 100)
    await lightrag_wrapper.afinalize()
    connect_chunks_to_entities(memgraph, "Chunk", "base")


if __name__ == "__main__":
    # Delete & create LightRAG working directory.
    lightrag_log_file = os.path.join(SCRIPT_DIR, "lightrag.log")
    if os.path.exists(lightrag_log_file):
        os.remove(lightrag_log_file)
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    # Cleanup Memgraph database.
    memgraph = Memgraph()
    memgraph.query("MATCH (n) DETACH DELETE n;")

    # Define all example sources. NOTE: only paths/urls are needed here.
    pypdf_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "pdf", "sample-files")
    docx_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "doc")
    sources = [
        os.path.join(
            pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
        ),
        os.path.join(docx_samples_dir, "sample3.docx"),
    ]

    asyncio.run(from_unstructured(sources))
