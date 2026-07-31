import hashlib
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text

from lightrag_memgraph import MemgraphLightRAGWrapper
from memgraph_toolbox.api.memgraph import Memgraph

from .memgraph import (
    connect_chunks_to_entities,
    create_nodes_from_list,
    create_unique_constraint,
    link_nodes_in_order,
    promote_entity_types_to_labels,
)
from .ontology import DEFAULT_ONTOLOGY, load_ontology

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    hash: str


@dataclass
class ChunkedDocument:
    chunks: list[Chunk]
    source: str | Path


def parse_source(
    source: str | Path,
    partition_kwargs: dict[str, Any] | None = None,
) -> list[str]:
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
        raise ValueError(f"Error parsing source {source_str}: {e!s}") from e


def parse_text(
    text: str,
    partition_kwargs: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Parse raw in-memory text (not a file or URL) into chunks.

    Always goes through the same partition_text + chunk_by_title pipeline
    parse_source() uses for files/URLs, regardless of input length, so
    chunking behavior only ever depends on content, never on text size.

    Args:
        text: Raw text to chunk.
        partition_kwargs: Additional keyword arguments to pass to unstructured's
            partition_text function.
    Returns:
        List of text chunks. Empty/whitespace-only input returns an empty list.
    """
    if not text or not text.strip():
        return []

    partition_kwargs = partition_kwargs or {}
    try:
        elements = partition_text(text=text, **partition_kwargs)
        chunks = chunk_by_title(elements)
        return [
            Chunk(text=str(chunk), hash=hashlib.sha256(str(chunk).encode()).hexdigest())
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]
    except Exception as e:
        raise ValueError(f"Error parsing text: {e!s}") from e


def make_chunks(
    sources: list[str | Path],
    partition_kwargs: dict[str, Any] | None = None,
) -> list[ChunkedDocument]:
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
            logger.debug(f"Source: {source}; No Chunks: {len(chunks)}; Chunks: {chunks};")
            documents.append(ChunkedDocument(chunks=chunks, source=source))
        except Exception as e:
            raise ValueError(f"Failed to parse {source}: {e}") from e

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


def _resolve_entity_workspace(
    lightrag_wrapper: MemgraphLightRAGWrapper | None,
    entity_workspace: str | None,
    only_chunks: bool,
) -> str | None:
    if only_chunks or entity_workspace is not None:
        return entity_workspace
    try:
        return lightrag_wrapper.get_lightrag().chunk_entity_relation_graph.workspace
    except Exception as e:
        logger.warning(f"Could not auto-derive LightRAG entity workspace, falling back to 'base': {e}")
        return "base"


async def _ingest_chunks(
    chunks: list[Chunk],
    memgraph: Memgraph,
    lightrag_wrapper: MemgraphLightRAGWrapper | None = None,
    only_chunks: bool = False,
    link_chunks: bool = False,
    entity_workspace: str | None = None,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
) -> list[Chunk]:
    """
    Ingest an already-produced flat list of chunks into Memgraph: upsert Chunk
    nodes, optionally chain them with NEXT, and (unless only_chunks) run
    LightRAG entity extraction, connect the resulting entities back to their
    chunks via MENTIONED_IN, and (if enforce_ontology) promote entity_type to
    a real label for entities that match the ontology.

    Internal helper shared by from_unstructured() and from_texts(). Not
    exported: it relies on its caller having already ensured the Chunk.hash
    uniqueness constraint (see create_unique_constraint) and resolved
    entity_workspace (see _resolve_entity_workspace) once per call rather than
    per chunk batch — an unresolved entity_workspace=None with
    only_chunks=False would silently build a MATCH (n:None) query in
    connect_chunks_to_entities, so this precondition isn't safe to expose on
    a public function.

    Args:
        chunks: Chunks to upsert (e.g. from parse_source/parse_text).
        memgraph: Memgraph instance for database operations.
        lightrag_wrapper: MemgraphLightRAGWrapper instance. Required unless only_chunks=True.
        only_chunks: If True, only create chunk nodes without LightRAG processing.
        link_chunks: If True, link chunks in order with NEXT relationship.
        entity_workspace: Node label LightRAG entities were written under.
        enforce_ontology: If True, promote entity_type to labels per ontology_path (or
            DEFAULT_ONTOLOGY_PATH). If False (default), entities are left exactly as
            LightRAG wrote them -- no label promotion, no ontology_conformant flagging.
        ontology_path: Path to an ontology YAML config file. Only consulted when
            enforce_ontology=True; defaults to DEFAULT_ONTOLOGY_PATH.
    Returns:
        The same chunks that were passed in, for convenience chaining.
    """
    if not chunks:
        logger.warning("No chunks provided to _ingest_chunks")
        return chunks

    if not only_chunks and lightrag_wrapper is None:
        raise ValueError("lightrag_wrapper is required when only_chunks=False")

    if ontology_path and not enforce_ontology:
        logger.warning("ontology_path was provided but enforce_ontology=False; ignoring ontology_path")

    memgraph_node_props = []
    for chunk in chunks:
        logger.debug(f"Chunk: {chunk.hash} - {chunk.text}")
        memgraph_node_props.append({"hash": chunk.hash, "text": chunk.text})
    create_nodes_from_list(memgraph, memgraph_node_props, "Chunk", 100, merge_key="hash")

    if link_chunks:
        hash_pairs = [(chunks[i].hash, chunks[i + 1].hash) for i in range(len(chunks) - 1)]
        if hash_pairs:
            relationships = [{"from": from_hash, "to": to_hash} for from_hash, to_hash in hash_pairs]
            link_nodes_in_order(memgraph, "Chunk", "hash", relationships, "NEXT")

    if not only_chunks:
        for chunk in chunks:
            await lightrag_wrapper.ainsert(input=chunk.text, file_paths=[chunk.hash])
        connect_chunks_to_entities(memgraph, "Chunk", entity_workspace)
        if enforce_ontology:
            ontology = load_ontology(ontology_path) if ontology_path else DEFAULT_ONTOLOGY
            promote_entity_types_to_labels(memgraph, entity_workspace, ontology)

    return chunks


async def from_texts(
    texts: list[str],
    memgraph: Memgraph,
    lightrag_wrapper: MemgraphLightRAGWrapper | None = None,
    only_chunks: bool = False,
    entity_workspace: str | None = None,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
) -> list[list[Chunk]]:
    """
    Ingest raw in-memory strings (not files or URLs) into Memgraph.

    Each text is chunked with parse_text() and the results are fed through the
    same Chunk-node + LightRAG entity-extraction pipeline as from_unstructured().
    Unlike from_unstructured(), texts are treated as independent units rather
    than a single sequential document, so there is no NEXT chunk linking.

    Args:
        texts: Raw strings to ingest. Empty/whitespace-only entries produce no chunks.
        memgraph: Memgraph instance for database operations.
        lightrag_wrapper: MemgraphLightRAGWrapper instance. Required unless only_chunks=True.
        only_chunks: If True, only create chunk nodes without LightRAG processing.
        entity_workspace: Node label LightRAG entities were written under. If None
            (default), auto-derived from lightrag_wrapper's resolved LightRAG
            workspace, falling back to "base" if that fails.
        enforce_ontology: If False (default), entities are left exactly as LightRAG
            wrote them -- no label promotion, no ontology_conformant flagging. If True,
            entity_type gets promoted to a real Memgraph label (e.g. entity_type="person"
            -> :Person) in addition to the entity_workspace label every entity already
            gets, per ontology_path.
        ontology_path: Path to an ontology YAML config file (see load_ontology()). Only
            consulted when enforce_ontology=True; defaults to DEFAULT_ONTOLOGY_PATH,
            which mirrors LightRAG's own built-in type vocabulary. entity_type values
            outside the ontology are never rejected -- the node and its entity_type
            property are kept, stamped ontology_conformant=false instead of getting a
            label. To also steer LightRAG's extraction itself toward the same
            vocabulary, load the same path with load_ontology() and pass its
            addon_params() into MemgraphLightRAGWrapper.initialize() -- using the same
            path at both call sites is what keeps them in sync.
    Returns:
        One list of Chunks per input text, in input order. A text that
        parse_text() splits into several pieces contributes several Chunks in
        its group; empty/whitespace-only input contributes an empty group.
        Grouping (rather than a flat list) is what lets callers trace an
        output Chunk back to the exact source text/record that produced it —
        recomputing a hash from the original text only works while that text
        is short enough for parse_text() to keep it as a single Chunk.
    """
    if not only_chunks and lightrag_wrapper is None:
        raise ValueError("lightrag_wrapper is required when only_chunks=False")

    create_unique_constraint(memgraph, "Chunk", "hash")
    resolved_entity_workspace = _resolve_entity_workspace(lightrag_wrapper, entity_workspace, only_chunks)

    grouped_chunks = [parse_text(text) for text in texts]
    flat_chunks = [chunk for group in grouped_chunks for chunk in group]
    if not flat_chunks:
        logger.warning("No chunks produced from provided texts")
        return grouped_chunks

    await _ingest_chunks(
        flat_chunks,
        memgraph,
        lightrag_wrapper=lightrag_wrapper,
        only_chunks=only_chunks,
        link_chunks=False,
        entity_workspace=resolved_entity_workspace,
        enforce_ontology=enforce_ontology,
        ontology_path=ontology_path,
    )
    return grouped_chunks


async def from_unstructured(
    sources: list[str | Path],
    memgraph: Memgraph,
    lightrag_wrapper: MemgraphLightRAGWrapper | None = None,
    only_chunks: bool = False,
    link_chunks: bool = False,
    entity_workspace: str | None = None,
    partition_kwargs: dict[str, Any] | None = None,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
) -> list[list[Chunk]]:
    """
    Process unstructured sources and ingest them into Memgraph using LightRAG.
    Args:
        sources: List of file paths or URLs to process
        memgraph: Memgraph instance for database operations
        lightrag_wrapper: MemgraphLightRAGWrapper instance (requires lightrag-memgraph).
            Required unless only_chunks=True, since it's only used for entity extraction.
        only_chunks: If True, only create chunk nodes without LightRAG processing
        link_chunks: If True, link chunks in order with NEXT relationship
        entity_workspace: Node label LightRAG entities were written under. If None
            (default), auto-derived from lightrag_wrapper's resolved LightRAG
            workspace, falling back to "base" if that fails.
        partition_kwargs: Additional keyword arguments to pass to unstructured's
            partition function (e.g., strategy, languages, pdf_infer_table_structure,
            ocr_languages, headers, ssl_verify, etc.)
        enforce_ontology: If False (default), no label promotion or ontology_conformant
            flagging happens. If True, entity_type gets promoted to a real Memgraph
            label per ontology_path. See from_texts() for details.
        ontology_path: Path to an ontology YAML config file. Only consulted when
            enforce_ontology=True; defaults to DEFAULT_ONTOLOGY_PATH.
    Returns:
        One list of Chunks per source, in `sources` order — the same
        grouped-return contract as from_texts(). A source that produced no
        chunks contributes an empty group.
    """
    if not only_chunks and lightrag_wrapper is None:
        raise ValueError("lightrag_wrapper is required when only_chunks=False")

    # TODO(gitbuda): Implement batching on the Cypher side as well under memgraph.compute_embeddings
    # NOTE: LightRAG uses { source_id: "chunk-ID..." } to reference its chunks.
    create_unique_constraint(memgraph, "Chunk", "hash")
    resolved_entity_workspace = _resolve_entity_workspace(lightrag_wrapper, entity_workspace, only_chunks)
    chunked_documents = make_chunks(sources, partition_kwargs=partition_kwargs)
    total_chunks = sum(len(document.chunks) for document in chunked_documents)
    start_time = time.time()
    processed_chunks = 0
    grouped_chunks: list[list[Chunk]] = []
    for document in chunked_documents:
        if not document.chunks:
            logger.warning(f"No chunks found in document: {document.source}")
            grouped_chunks.append([])
            continue

        logger.info(f"Processing {len(document.chunks)} chunks from {document.source}...")
        await _ingest_chunks(
            document.chunks,
            memgraph,
            lightrag_wrapper=lightrag_wrapper,
            only_chunks=only_chunks,
            link_chunks=link_chunks,
            entity_workspace=resolved_entity_workspace,
            enforce_ontology=enforce_ontology,
            ontology_path=ontology_path,
        )
        grouped_chunks.append(document.chunks)

        processed_chunks += len(document.chunks)
        elapsed_time = time.time() - start_time
        estimated_time_remaining = elapsed_time / processed_chunks * (total_chunks - processed_chunks)
        if estimated_time_remaining >= 3600:
            time_str = f"{estimated_time_remaining / 3600:.2f} hours"
        elif estimated_time_remaining >= 60:
            time_str = f"{estimated_time_remaining / 60:.2f} minutes"
        else:
            time_str = f"{estimated_time_remaining:.2f} seconds"
        if total_chunks == processed_chunks:
            logger.info(f"All {total_chunks} chunks processed in {elapsed_time:.2f} seconds")
        else:
            logger.info(
                f"Processed {processed_chunks} chunks out of {total_chunks}. Estimated time remaining: {time_str}"
            )

    return grouped_chunks
