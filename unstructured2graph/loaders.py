import os
import uuid
from pathlib import Path
from typing import List, Union
import logging

import asyncio
import shutil
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from lightrag_memgraph import MemgraphLightRAGWrapper
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from memgraph_toolbox.api.memgraph import Memgraph

from memgraph import create_nodes_from_list, connect_chunks_to_entities

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKING_DIR = os.path.join(SCRIPT_DIR, "lightrag_storage.out")
logger = logging.getLogger(__name__)


def parse_source(source: Union[str, Path]) -> List[str]:
    """
    Parse a source file or URL using the unstructured library.

    The unstructured library supports many data sources including:
    PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, HTML, XML, JSON, CSV, RTF,
    ODT, EPUB, MSG, EML, TXT, MD, and more.

    Args:
        source: Path to file or URL string
    Returns:
        List of text chunks extracted from the source
    """
    source_str = str(source)
    try:
        # Use unstructured's partition function which auto-detects file type
        if source_str.startswith(("http://", "https://")):
            # For URLs, unstructured can handle them directly
            elements = partition(url=source_str)
        else:
            # For local files
            elements = partition(filename=source_str)
        # Chunk the elements by title for better semantic grouping
        chunks = chunk_by_title(elements)
        # Extract text from each chunk
        text_chunks = [
            str(chunk) for chunk in chunks if chunk.text and chunk.text.strip()
        ]
        return text_chunks
    except Exception as e:
        raise ValueError(f"Error parsing source {source_str}: {str(e)}")


async def from_unstructured(sources: List[Union[str, Path]], memgraph: Memgraph):
    """
    Process unstructured sources and ingest them into Memgraph using LightRAG.

    Args:
        sources: List of file paths or URLs to process
        memgraph: Memgraph instance for database operations
    """
    documents = []

    ## PARSE
    # NOTE: Using unstructured library which supports many data sources
    # NOTE: Each element is a list of chunks from a Document (each source is an abstract Document)
    # TODO: Add chunking support + the way of referencting original sources and chunks once they end up in the database.
    #     * NOTE: LightRAG uses { source_id: "chunk-ID..." } to reference its chunks.
    for source in sources:
        try:
            chunks = parse_source(source)
            logger.info(
                f"Source: {source}; No Chunks: {len(chunks)}; Chunks: {chunks};"
            )
            documents.append(chunks)
        except Exception as e:
            print(f"Warning: Failed to parse {source}: {e}")
            continue

    # Calculate statistics about chunk sizes (number of characters per chunk) across all documents
    all_chunk_lengths = [len(chunk) for doc in documents for chunk in doc]
    if all_chunk_lengths:
        min_chunk = min(all_chunk_lengths)
        max_chunk = max(all_chunk_lengths)
        avg_chunk = sum(all_chunk_lengths) / len(all_chunk_lengths)
        import statistics

        mean_chunk = statistics.mean(all_chunk_lengths)
        logger.info(
            f"Chunk size statistics - min: {min_chunk}, max: {max_chunk}, avg: {avg_chunk:.2f}, mean: {mean_chunk:.2f}"
        )
        print(
            f"Chunk size statistics - min: {min_chunk}, max: {max_chunk}, avg: {avg_chunk:.2f}, mean: {mean_chunk:.2f}"
        )
    else:
        logger.info("No chunks found, statistics unavailable.")
        print("No chunks found, statistics unavailable.")

    ## INGEST
    # TODO(gitbuda): Print progress bar and estimete how long it will take to process all chunks.
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
    logging.basicConfig(level=logging.INFO)

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
    xls_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "xls")
    sources = [
        os.path.join(
            pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
        ),
        os.path.join(docx_samples_dir, "sample3.docx"),
        # os.path.join(xls_samples_dir, "financial-sample.xlsx"),
        # "https://memgraph.com/docs/ai-ecosystem/graph-rag",
    ]
    asyncio.run(from_unstructured(sources, memgraph))
