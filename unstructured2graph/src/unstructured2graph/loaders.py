from pathlib import Path
from typing import List, Union
import logging
from dataclasses import dataclass
import hashlib
import statistics
import os

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from memgraph_toolbox.api.memgraph import Memgraph
from lightrag_memgraph import MemgraphLightRAGWrapper

from .memgraph import create_nodes_from_list, connect_chunks_to_entities

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


def parse_source(source: Union[str, Path]) -> List[str]:
    """
    Parse a source file or URL using the unstructured library. The unstructured
    library supports many types of data sources and various parsing options.
    Args:
        source: Path to file or URL string
    Returns:
        List of text chunks extracted from the source
    """

    source_str = str(source)
    try:
        if source_str.startswith(("http://", "https://")):
            elements = partition(url=source_str)
        else:
            elements = partition(filename=source_str)
        chunks = chunk_by_title(elements)
        text_chunks = [
            Chunk(text=str(chunk), hash=hashlib.sha256(str(chunk).encode()).hexdigest())
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]
        return text_chunks
    except Exception as e:
        raise ValueError(f"Error parsing source {source_str}: {str(e)}")


def make_chunks(sources: List[Union[str, Path]]) -> List[ChunkedDocument]:
    """
    Chunk a list of sources into a list of ChunkedDocuments.
    Args:
        sources: List of file paths or URLs to process
    Returns:
        List of ChunkedDocuments
    """

    documents = []
    for source in sources:
        try:
            chunks = parse_source(source)
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
):
    """
    Process unstructured sources and ingest them into Memgraph using LightRAG.
    Args:
        sources: List of file paths or URLs to process
        memgraph: Memgraph instance for database operations
    """

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key."
        )

    # TODO(gitbuda): Print progress bar and estimete how long it will take to process all chunks.
    # TODO(gitbuda): Add proper error handling.
    # ----> RELEASE READY
    # TODO: set LLM params as defaults under the wrapper
    # TODO(gitbuda): Add option to link chunks coming from the same source.
    # TODO(gitbuda): Make the calls idempotent.
    # TODO(gitbuda): Create all required indexes.
    # NOTE: LightRAG uses { source_id: "chunk-ID..." } to reference its chunks.
    chunked_documents = make_chunks(sources)
    for document in chunked_documents:
        memgraph_node_props = []
        for chunk in document.chunks:
            await lightrag_wrapper.ainsert(input=chunk.text, file_paths=[chunk.hash])
            memgraph_node_props.append({"hash": chunk.hash, "text": chunk.text})
        create_nodes_from_list(memgraph, memgraph_node_props, "Chunk", 100)
    connect_chunks_to_entities(memgraph, "Chunk", "base")
