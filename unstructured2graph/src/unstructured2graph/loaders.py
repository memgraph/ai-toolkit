from pathlib import Path
from typing import Any, List, Union
import logging
from dataclasses import dataclass
import hashlib
import statistics
import os
import time

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from memgraph_toolbox.api.memgraph import Memgraph
from lightrag_memgraph import MemgraphLightRAGWrapper

from .memgraph import (
    create_nodes_from_list,
    connect_chunks_to_entities,
    link_nodes_in_order,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    hash: str


@dataclass
class ChunkedDocument:
    chunks: List[Chunk]
    source: Union[str, Path]


def parse_source(
    source: Union[str, Path],
    partition_kwargs: dict[str, Any] | None = None,
) -> List[str]:
    """
    Parse a source file or URL using the unstructured library. The unstructured
    library supports many types of data sources and various parsing options.
    Args:
        source: Path to file or URL string
        partition_kwargs: Additional keyword arguments to pass to unstructured's
            partition function (e.g., strategy, languages, pdf_infer_table_structure,
            ocr_languages, headers, ssl_verify, etc.)
    Returns:
        List of text chunks extracted from the source
    """
    partition_kwargs = partition_kwargs or {}
    source_str = str(source)
    try:
        if source_str.startswith(("http://", "https://")):
            elements = partition(url=source_str, **partition_kwargs)
        else:
            elements = partition(filename=source_str, **partition_kwargs)
        chunks = chunk_by_title(elements)
        text_chunks = [
            Chunk(text=str(chunk), hash=hashlib.sha256(str(chunk).encode()).hexdigest())
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]
        return text_chunks
    except Exception as e:
        raise ValueError(f"Error parsing source {source_str}: {str(e)}")


def make_chunks(
    sources: List[Union[str, Path]],
    partition_kwargs: dict[str, Any] | None = None,
) -> List[ChunkedDocument]:
    """
    Chunk a list of sources into a list of ChunkedDocuments.
    Args:
        sources: List of file paths or URLs to process
        partition_kwargs: Additional keyword arguments to pass to unstructured's
            partition function (e.g., strategy, languages, pdf_infer_table_structure,
            ocr_languages, headers, ssl_verify, etc.)
    Returns:
        List of ChunkedDocuments
    """

    documents = []
    for source in sources:
        try:
            chunks = parse_source(source, partition_kwargs=partition_kwargs)
            logger.debug(
                f"Source: {source}; No Chunks: {len(chunks)}; Chunks: {chunks};"
            )
            documents.append(ChunkedDocument(chunks=chunks, source=source))
        except Exception as e:
            raise ValueError(f"Failed to parse {source}: {e}")

    # Get statistics about chunks, e.g., important because of the token limits
    # (LLM/embedding).
    all_chunk_lengths = [len(chunk.text) for doc in documents for chunk in doc.chunks]
    if all_chunk_lengths:
        min_chunk = min(all_chunk_lengths)
        max_chunk = max(all_chunk_lengths)
        avg_chunk = sum(all_chunk_lengths) / len(all_chunk_lengths)
        mean_chunk = statistics.mean(all_chunk_lengths)
        logger.info(
            f"Chunk size statistics (chars) - min: {min_chunk}, max: {max_chunk}, avg: {avg_chunk:.2f}, mean: {mean_chunk:.2f}"
        )
    else:
        logger.info("No chunks found, statistics unavailable.")
    return documents


async def from_unstructured(
    sources: List[Union[str, Path]],
    memgraph: Memgraph,
    lightrag_wrapper: MemgraphLightRAGWrapper,
    only_chunks: bool = False,
    link_chunks: bool = False,
    partition_kwargs: dict[str, Any] | None = None,
):
    """
    Process unstructured sources and ingest them into Memgraph using LightRAG.
    Args:
        sources: List of file paths or URLs to process
        memgraph: Memgraph instance for database operations
        lightrag_wrapper: MemgraphLightRAGWrapper instance (requires lightrag-memgraph)
        only_chunks: If True, only create chunk nodes without LightRAG processing
        link_chunks: If True, link chunks in order with NEXT relationship
        partition_kwargs: Additional keyword arguments to pass to unstructured's
            partition function (e.g., strategy, languages, pdf_infer_table_structure,
            ocr_languages, headers, ssl_verify, etc.)
    """

    # TODO(gitbuda): Create all required indexes.
    # TODO(gitbuda): Make the calls idempotent.
    # TODO(gitbuda): Implement batching on the Cypher side as well under memgraph.compute_embeddings
    # NOTE: LightRAG uses { source_id: "chunk-ID..." } to reference its chunks.
    chunked_documents = make_chunks(sources, partition_kwargs=partition_kwargs)
    total_chunks = sum(len(document.chunks) for document in chunked_documents)
    start_time = time.time()
    processed_chunks = 0
    for document in chunked_documents:
        if not document.chunks:
            logger.warning(f"No chunks found in document: {document.source}")
            continue

        logger.info(
            f"Processing {len(document.chunks)} chunks from {document.source}..."
        )
        memgraph_node_props = []
        for chunk in document.chunks:
            logger.debug(f"Chunk: {chunk.hash} - {chunk.text}")
            memgraph_node_props.append({"hash": chunk.hash, "text": chunk.text})
        create_nodes_from_list(memgraph, memgraph_node_props, "Chunk", 100)

        if link_chunks:
            hash_pairs = [
                (document.chunks[i].hash, document.chunks[i + 1].hash)
                for i in range(len(document.chunks) - 1)
            ]
            if hash_pairs:
                relationships = [
                    {"from": from_hash, "to": to_hash}
                    for from_hash, to_hash in hash_pairs
                ]
                link_nodes_in_order(memgraph, "Chunk", "hash", relationships, "NEXT")

        for chunk in document.chunks:
            if not only_chunks:
                await lightrag_wrapper.ainsert(
                    input=chunk.text, file_paths=[chunk.hash]
                )
            if not only_chunks:
                connect_chunks_to_entities(memgraph, "Chunk", "base")

        processed_chunks += len(document.chunks)
        elapsed_time = time.time() - start_time
        estimated_time_remaining = (
            elapsed_time / processed_chunks * (total_chunks - processed_chunks)
        )
        if estimated_time_remaining >= 3600:
            time_str = f"{estimated_time_remaining / 3600:.2f} hours"
        elif estimated_time_remaining >= 60:
            time_str = f"{estimated_time_remaining / 60:.2f} minutes"
        else:
            time_str = f"{estimated_time_remaining:.2f} seconds"
        if total_chunks == processed_chunks:
            logger.info(
                f"All {total_chunks} chunks processed in {elapsed_time:.2f} seconds"
            )
        else:
            logger.info(
                f"Processed {processed_chunks} chunks out of {total_chunks}. Estimated time remaining: {time_str}"
            )
