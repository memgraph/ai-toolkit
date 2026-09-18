import asyncio
import hashlib
import logging
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text

from lightrag_memgraph import MemgraphLightRAGWrapper
from memgraph_toolbox.api.memgraph import Memgraph

from .extraction_backend import ExtractionBackend
from .memgraph import (
    connect_chunks_to_entities,
    create_entity_type_constraint,
    create_nodes_from_list,
    create_unique_constraint,
    link_nodes_in_order,
    promote_all_entity_types_to_labels,
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
) -> list[Chunk]:
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
    chunk_kwargs: dict[str, Any] | None = None,
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
        chunk_kwargs: Additional keyword arguments to pass to chunk_by_title,
            notably ``max_characters``. Separate from partition_kwargs because
            they reach different functions -- conflating them is why chunk
            sizing was unreachable at all: chunk_by_title was called bare, so
            its ~500-character cap applied to every caller with no way to
            influence it. Feeding it conversation turns, that split each turn
            into roughly 3.6 fragments, each costing two LLM calls downstream
            (#327).

            Left empty by default: documents may legitimately want small
            chunks, so this exposes the choice rather than making it.
    Returns:
        List of text chunks. Empty/whitespace-only input returns an empty list.
    """
    if not text or not text.strip():
        return []

    partition_kwargs = partition_kwargs or {}
    chunk_kwargs = chunk_kwargs or {}
    try:
        elements = partition_text(text=text, **partition_kwargs)
        chunks = chunk_by_title(elements, **chunk_kwargs)
        return [
            Chunk(text=str(chunk), hash=hashlib.sha256(str(chunk).encode()).hexdigest())
            for chunk in chunks
            if chunk.text and chunk.text.strip()
        ]
    except Exception as e:
        raise ValueError(f"Error parsing text: {e!s}") from e


def make_chunks(
    sources: Sequence[str | Path],
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
    extraction_backend: ExtractionBackend | None,
    entity_workspace: str | None,
    only_chunks: bool,
) -> str | None:
    """
    Raises:
        ValueError: if entity_workspace is explicitly given and doesn't match
            extraction_backend.workspace_label. The backend writes entities
            under its own workspace_label regardless of this override, so a
            mismatch would otherwise silently make connect_chunks_to_entities()
            and label promotion scan the wrong label and find nothing --
            failing loudly here is strictly better than that silent no-op.
    """
    if only_chunks:
        return entity_workspace
    if entity_workspace is not None:
        # only_chunks is False here, so callers (from_texts/from_unstructured)
        # have already raised ValueError if extraction_backend were None.
        backend_workspace = cast("ExtractionBackend", extraction_backend).workspace_label
        if entity_workspace != backend_workspace:
            raise ValueError(
                f"entity_workspace={entity_workspace!r} does not match "
                f"extraction_backend.workspace_label={backend_workspace!r}. Pass entity_workspace="
                "None (the default) to auto-derive it from the backend instead of overriding it."
            )
        return entity_workspace
    # only_chunks is False here, so callers (from_texts/from_unstructured)
    # have already raised ValueError if extraction_backend were None.
    return cast("ExtractionBackend", extraction_backend).workspace_label


async def _ingest_chunks(
    chunks: list[Chunk],
    memgraph: Memgraph,
    extraction_backend: ExtractionBackend | None = None,
    only_chunks: bool = False,
    link_chunks: bool = False,
    entity_workspace: str | None = None,
    promote_labels: bool = False,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
) -> list[Chunk]:
    """
    Ingest an already-produced flat list of chunks into Memgraph: upsert Chunk
    nodes, optionally chain them with NEXT, and (unless only_chunks) run the
    extraction backend, connect the resulting entities back to their chunks
    via MENTIONED_IN, and promote entity_type to a real label per
    promote_labels/enforce_ontology.

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
        extraction_backend: An ExtractionBackend (e.g. LightRAGBackend,
            GLiNER2Backend). Required unless only_chunks=True.
        only_chunks: If True, only create chunk nodes without running extraction.
        link_chunks: If True, link chunks in order with NEXT relationship.
        entity_workspace: Node label the extraction backend's entities were written under.
        promote_labels: If True, promote every entity_type to a real Memgraph label,
            with no fixed vocabulary -- no ontology_conformant flagging, since there's
            no ontology to be non-conformant relative to. Ignored if enforce_ontology
            is also True (enforce_ontology is the stricter, ontology-gated mode).
        enforce_ontology: If True, promote entity_type to labels restricted to
            ontology_path's vocabulary (or DEFAULT_ONTOLOGY_PATH), and flag anything
            outside it as ontology_conformant=false. Takes precedence over
            promote_labels. If both are False (default), entities are left exactly as
            the extraction backend wrote them -- no label promotion at all.
        ontology_path: Path to an ontology YAML config file. Only consulted when
            enforce_ontology=True; defaults to DEFAULT_ONTOLOGY_PATH.
    Returns:
        The same chunks that were passed in, for convenience chaining.
    """
    if not chunks:
        logger.warning("No chunks provided to _ingest_chunks")
        return chunks

    if not only_chunks and extraction_backend is None:
        raise ValueError("extraction_backend is required when only_chunks=False")

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
        # Both casts are safe here per this function's documented precondition:
        # callers already raised ValueError for a None extraction_backend, and
        # already resolved entity_workspace (see _resolve_entity_workspace)
        # before calling _ingest_chunks with only_chunks=False.
        backend = cast("ExtractionBackend", extraction_backend)
        resolved_workspace = cast("str", entity_workspace)
        create_entity_type_constraint(memgraph, resolved_workspace)
        for chunk in chunks:
            await backend.aingest_chunk(memgraph, chunk)
        connect_chunks_to_entities(memgraph, "Chunk", resolved_workspace)
        if enforce_ontology:
            ontology = load_ontology(ontology_path) if ontology_path else DEFAULT_ONTOLOGY
            promote_entity_types_to_labels(memgraph, resolved_workspace, ontology)
        elif promote_labels:
            promote_all_entity_types_to_labels(memgraph, resolved_workspace)

    return chunks


async def from_texts(
    texts: list[str],
    memgraph: Memgraph,
    extraction_backend: ExtractionBackend | None = None,
    only_chunks: bool = False,
    entity_workspace: str | None = None,
    promote_labels: bool = False,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
    chunk_kwargs: dict[str, Any] | None = None,
) -> list[list[Chunk]]:
    """
    Ingest raw in-memory strings (not files or URLs) into Memgraph.

    Each text is chunked with parse_text() and the results are fed through the
    same Chunk-node + extraction pipeline as from_unstructured(). Unlike
    from_unstructured(), texts are treated as independent units rather than a
    single sequential document, so there is no NEXT chunk linking.

    Args:
        texts: Raw strings to ingest. Empty/whitespace-only entries produce no chunks.
        memgraph: Memgraph instance for database operations.
        extraction_backend: An ExtractionBackend (e.g. LightRAGBackend,
            GLiNER2Backend). Required unless only_chunks=True.
        only_chunks: If True, only create chunk nodes without running extraction.
        entity_workspace: Node label the extraction backend's entities were written
            under. If None (default), auto-derived from extraction_backend's
            workspace_label.
        promote_labels: Label promotion and ontology enforcement are separate concerns.
            If True, every entity_type gets promoted to a real Memgraph label (e.g.
            entity_type="person" -> :Person), with no fixed vocabulary restricting
            which ones -- and no ontology_conformant flagging, since there's no
            ontology to be non-conformant relative to. Ignored if enforce_ontology is
            also True.
        enforce_ontology: If True, entity_type promotion is restricted to
            ontology_path's vocabulary (or DEFAULT_ONTOLOGY_PATH if omitted), and
            anything outside it is flagged ontology_conformant=false instead of
            getting a label -- this is the stricter, gated mode, and takes precedence
            over promote_labels. If both promote_labels and enforce_ontology are False
            (the default for both), no label promotion happens at all.
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
    if not only_chunks and extraction_backend is None:
        raise ValueError("extraction_backend is required when only_chunks=False")

    create_unique_constraint(memgraph, "Chunk", "hash")
    resolved_entity_workspace = _resolve_entity_workspace(extraction_backend, entity_workspace, only_chunks)

    grouped_chunks = [parse_text(text, chunk_kwargs=chunk_kwargs) for text in texts]
    flat_chunks = [chunk for group in grouped_chunks for chunk in group]
    if not flat_chunks:
        logger.warning("No chunks produced from provided texts")
        return grouped_chunks

    await _ingest_chunks(
        flat_chunks,
        memgraph,
        extraction_backend=extraction_backend,
        only_chunks=only_chunks,
        link_chunks=False,
        entity_workspace=resolved_entity_workspace,
        promote_labels=promote_labels,
        enforce_ontology=enforce_ontology,
        ontology_path=ontology_path,
    )
    return grouped_chunks


async def enqueue_texts(
    texts: list[str],
    memgraph: Memgraph,
    lightrag_wrapper: MemgraphLightRAGWrapper,
    *,
    chunk_kwargs: dict[str, Any] | None = None,
) -> list[list[Chunk]]:
    """Chunk texts and create their Chunk nodes, staging them in LightRAG's
    document queue WITHOUT triggering processing.

    ``from_texts`` calls ``lightrag_wrapper.ainsert()`` per chunk, which
    enqueues *and* immediately processes -- so a caller inserting one
    document at a time (e.g. one session) never gives LightRAG's own
    ``MAX_PARALLEL_INSERT``-sized worker pool more than one document to run
    concurrently over. ``apipeline_process_enqueue_documents`` also holds a
    workspace-level "busy" lock: a second concurrent caller doesn't run its
    own processing pass in parallel, it just sets a pending flag and
    returns having done no work. Real parallelism needs many documents
    enqueued *before* processing starts.

    Call this once per group of texts you want processed together (e.g. many
    sessions in a batch), then :func:`process_enqueued_and_finalize` exactly
    once to trigger LightRAG's processing pass and the post-processing steps
    (``connect_chunks_to_entities``, label promotion) this function skips.

    Args:
        texts: Raw strings to chunk and stage. Empty/whitespace-only entries
            produce no chunks and are not enqueued, matching :func:`from_texts`.
        memgraph: Memgraph instance the Chunk nodes are written to.
        lightrag_wrapper: An initialised ``MemgraphLightRAGWrapper``. Required
            (unlike :func:`from_texts`, this function has no ``only_chunks``
            mode -- staging for LightRAG is the whole point).
        chunk_kwargs: Forwarded to :func:`parse_text` (notably ``max_characters``).

    Returns:
        One list of Chunks per input text, in input order -- same grouping
        contract as :func:`from_texts`, so a caller can trace an output Chunk
        back to the text that produced it. Pass the flattened result to
        :func:`process_enqueued_and_finalize` to verify and finalize it.

    Raises:
        Nothing beyond what ``memgraph`` / ``lightrag_wrapper`` themselves
        raise (e.g. a Memgraph connection error, or LightRAG's own
        enqueue-time validation).
    """
    create_unique_constraint(memgraph, "Chunk", "hash")

    grouped_chunks = [parse_text(text, chunk_kwargs=chunk_kwargs) for text in texts]
    flat_chunks = [chunk for group in grouped_chunks for chunk in group]
    if not flat_chunks:
        logger.warning("No chunks produced from provided texts")
        return grouped_chunks

    memgraph_node_props = [{"hash": chunk.hash, "text": chunk.text} for chunk in flat_chunks]
    create_nodes_from_list(memgraph, memgraph_node_props, "Chunk", 100, merge_key="hash")

    # Deduped by content hash before enqueueing, not left to LightRAG:
    # apipeline_enqueue_documents *rejects* a batch with duplicate ids
    # outright ("IDs must be unique", confirmed against the real API, not
    # assumed) rather than collapsing them. Two chunks with byte-identical
    # text legitimately are the same document -- same reasoning
    # sessions-graph already uses for repeated session ids -- so dedup here,
    # first occurrence wins, same as create_nodes_from_list's merge_key
    # already treats them upstream.
    unique_chunks: dict[str, Chunk] = {}
    for chunk in flat_chunks:
        unique_chunks.setdefault(chunk.hash, chunk)

    # Mirrors LightRAG's own ainsert(): resolve_chunk_options() the same way,
    # so an enqueue_texts + process_enqueued_and_finalize pair behaves
    # identically to N individual ainsert() calls, parallelism aside.
    from lightrag.parser.routing import resolve_chunk_options

    rag = lightrag_wrapper.get_lightrag()
    chunk_opts = resolve_chunk_options(rag.addon_params, split_by_character=None, split_by_character_only=False)
    await rag.apipeline_enqueue_documents(
        input=[chunk.text for chunk in unique_chunks.values()],
        # Explicit, not auto-generated: process_enqueued_and_finalize looks
        # documents back up by exactly this id to verify they actually
        # finished (see its docstring for why the call returning is not
        # proof of that).
        ids=list(unique_chunks.keys()),
        file_paths=list(unique_chunks.keys()),
        chunk_options=chunk_opts,
    )
    return grouped_chunks


async def process_enqueued_and_finalize(
    memgraph: Memgraph,
    lightrag_wrapper: MemgraphLightRAGWrapper,
    chunks: list[Chunk],
    *,
    entity_workspace: str | None = None,
    promote_labels: bool = False,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
    max_attempts: int = 60,
    poll_interval: float = 2.0,
) -> dict[str, dict[str, Any]]:
    """Trigger LightRAG's processing pass over everything staged by prior
    :func:`enqueue_texts` calls, then run the post-processing steps those
    calls deferred.

    ``apipeline_process_enqueue_documents`` returning is not proof the work
    happened: it holds a workspace-level "busy" lock, and a concurrent caller
    that finds it already held just sets a pending flag and returns having
    done no work at all (confirmed by reading its source). Trusting a bare
    return would mark sessions completed while their documents are still
    queued. This instead verifies, by id, that every one of ``chunks``
    reached a *terminal* status (PROCESSED or FAILED) in LightRAG's own
    ``doc_status`` store -- retrying the processing call if not, since the
    other owner finishing will pick up what this call enqueued (LightRAG's
    own "process additional documents due to pending request" handoff) --
    and returns each one's final record so a caller can tell which
    *specific* documents actually succeeded rather than assuming the whole
    batch did because nothing raised.

    ``connect_chunks_to_entities`` and label promotion are workspace-wide
    operations (a MERGE over every matching node under the label), so running
    them once here after the whole batch -- rather than once per chunk, as
    ``_ingest_chunks`` does today -- is strictly more correct as well as
    cheaper: today's per-chunk repetition is redundant work, not a
    correctness requirement. They run regardless of per-document outcome:
    a failed document simply contributed no entities for the MERGE to find.

    Args:
        memgraph: Memgraph instance ``chunks`` were written to (by a prior
            :func:`enqueue_texts` call).
        lightrag_wrapper: The same initialised wrapper ``enqueue_texts`` used.
        chunks: Every chunk staged by the ``enqueue_texts`` call(s) this pass
            should cover -- used only to verify final status by id, never
            re-chunked or re-enqueued. An empty list is a no-op.
        entity_workspace: Node label LightRAG entities were written under. If
            None (default), auto-derived from ``lightrag_wrapper``, falling
            back to ``"base"`` if that fails.
        promote_labels: Passed through to the label-promotion step; see
            :func:`from_texts` for the full semantics.
        enforce_ontology: Passed through to the label-promotion step; takes
            precedence over ``promote_labels``. See :func:`from_texts`.
        ontology_path: Only consulted when ``enforce_ontology=True``.
        max_attempts: How many times to retry ``apipeline_process_enqueue_documents``
            while any chunk is still not in a terminal status. Bounds the
            wait for a concurrent owner to finish rather than looping forever.
        poll_interval: Seconds to wait between retries.

    Returns:
        Each chunk's hash mapped to its final ``doc_status`` record (at least
        a ``"status"`` key; a failed document also carries ``"error_msg"``).
        Callers must inspect this to know which sessions/chunks actually
        succeeded -- a session whose chunk is not ``"processed"`` here did
        not reconcile, whatever this function's own return looked like.

    Raises:
        RuntimeError: if one or more chunks are still not in a terminal
            status after ``max_attempts`` retries -- a real stall (a stuck
            worker, not just another owner mid-pass), surfaced rather than
            reported as silent success.
    """
    if not chunks:
        return {}

    from lightrag.base import DocStatus

    rag = lightrag_wrapper.get_lightrag()
    # Resolved here rather than via _resolve_entity_workspace: that helper
    # is typed against the generic ExtractionBackend Protocol, which has no
    # enqueue/process API -- this function is inherently LightRAG-specific
    # (it drives LightRAG's own document queue directly), so it resolves
    # the workspace straight off the wrapper it already has.
    if entity_workspace is not None:
        resolved_entity_workspace = entity_workspace
    else:
        try:
            resolved_entity_workspace = rag.chunk_entity_relation_graph.workspace
        except Exception as e:
            logger.warning(f"Could not auto-derive LightRAG entity workspace, falling back to 'base': {e}")
            resolved_entity_workspace = "base"
    ids = [chunk.hash for chunk in chunks]
    unique_ids = set(ids)
    terminal = {DocStatus.PROCESSED.value, DocStatus.FAILED.value}

    statuses: dict[str, dict[str, Any]] = {}
    for attempt in range(max_attempts):
        await rag.apipeline_process_enqueue_documents()
        records = await rag.doc_status.get_by_ids(ids)
        statuses = {doc_id: record for doc_id, record in zip(ids, records, strict=True) if record is not None}
        if len(statuses) == len(unique_ids) and all(record.get("status") in terminal for record in statuses.values()):
            break
        if attempt < max_attempts - 1:
            await asyncio.sleep(poll_interval)
    else:
        pending = sorted(doc_id for doc_id in unique_ids if statuses.get(doc_id, {}).get("status") not in terminal)
        raise RuntimeError(
            f"{len(pending)} of {len(unique_ids)} documents never reached a terminal status "
            f"after {max_attempts} attempts (waited {max_attempts * poll_interval:.0f}s total): "
            f"{pending[:5]}{'...' if len(pending) > 5 else ''}. A concurrent owner may be stalled."
        )

    create_entity_type_constraint(memgraph, resolved_entity_workspace)
    connect_chunks_to_entities(memgraph, "Chunk", resolved_entity_workspace)
    if enforce_ontology:
        ontology = load_ontology(ontology_path) if ontology_path else DEFAULT_ONTOLOGY
        promote_entity_types_to_labels(memgraph, resolved_entity_workspace, ontology)
    elif promote_labels:
        promote_all_entity_types_to_labels(memgraph, resolved_entity_workspace)

    return statuses


async def from_unstructured(
    sources: Sequence[str | Path],
    memgraph: Memgraph,
    extraction_backend: ExtractionBackend | None = None,
    only_chunks: bool = False,
    link_chunks: bool = False,
    entity_workspace: str | None = None,
    partition_kwargs: dict[str, Any] | None = None,
    promote_labels: bool = False,
    enforce_ontology: bool = False,
    ontology_path: str | Path | None = None,
) -> list[list[Chunk]]:
    """
    Process unstructured sources and ingest them into Memgraph using an
    extraction backend (e.g. LightRAGBackend, GLiNER2Backend).
    Args:
        sources: List of file paths or URLs to process
        memgraph: Memgraph instance for database operations
        extraction_backend: An ExtractionBackend. Required unless only_chunks=True,
            since it's only used for entity extraction.
        only_chunks: If True, only create chunk nodes without running extraction
        link_chunks: If True, link chunks in order with NEXT relationship
        entity_workspace: Node label the extraction backend's entities were written
            under. If None (default), auto-derived from extraction_backend's
            workspace_label.
        partition_kwargs: Additional keyword arguments to pass to unstructured's
            partition function (e.g., strategy, languages, pdf_infer_table_structure,
            ocr_languages, headers, ssl_verify, etc.)
        promote_labels: If True, promote every entity_type to a real Memgraph label
            with no fixed vocabulary and no ontology_conformant flagging. Ignored if
            enforce_ontology is also True. See from_texts() for details.
        enforce_ontology: If True, restrict entity_type promotion to ontology_path's
            vocabulary and flag anything outside it ontology_conformant=false; takes
            precedence over promote_labels. If both are False (default), no label
            promotion happens at all. See from_texts() for details.
        ontology_path: Path to an ontology YAML config file. Only consulted when
            enforce_ontology=True; defaults to DEFAULT_ONTOLOGY_PATH.
    Returns:
        One list of Chunks per source, in `sources` order — the same
        grouped-return contract as from_texts(). A source that produced no
        chunks contributes an empty group.
    """
    if not only_chunks and extraction_backend is None:
        raise ValueError("extraction_backend is required when only_chunks=False")

    # LightRAG uses `{source_id: "chunk-ID..."}` to reference its chunks.
    create_unique_constraint(memgraph, "Chunk", "hash")
    resolved_entity_workspace = _resolve_entity_workspace(extraction_backend, entity_workspace, only_chunks)
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
            extraction_backend=extraction_backend,
            only_chunks=only_chunks,
            link_chunks=link_chunks,
            entity_workspace=resolved_entity_workspace,
            promote_labels=promote_labels,
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
