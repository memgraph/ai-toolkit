"""Simple test for unstructured2graph loaders."""

import hashlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unstructured2graph import (
    Chunk,
    ChunkedDocument,
    from_texts,
    from_unstructured,
    make_chunks,
    parse_source,
    parse_text,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _fake_document():
    return ChunkedDocument(chunks=[Chunk(text="a", hash="h1")], source="fake.txt")


def _lightrag_wrapper_with_workspace(workspace):
    wrapper = MagicMock()
    wrapper.ainsert = AsyncMock()
    wrapper.get_lightrag.return_value.chunk_entity_relation_graph.workspace = workspace
    return wrapper


def test_parse_source_with_simple_text(tmp_path):
    """Test that parse_source can handle a simple text file."""
    # Create a simple text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a simple test document.\nIt has multiple lines.")

    # Parse the file
    chunks = parse_source(test_file)

    # Assert that we got at least one chunk
    assert len(chunks) > 0
    assert isinstance(chunks, list)

    # Assert that chunks are Chunk objects
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(len(chunk.text.strip()) > 0 for chunk in chunks)
    assert all(isinstance(chunk.hash, str) for chunk in chunks)


def test_parse_source_with_empty_file(tmp_path):
    """Test that parse_source handles empty files gracefully."""
    # Create an empty file
    test_file = tmp_path / "empty.txt"
    test_file.write_text("")

    # Parse the file - should return empty list or handle gracefully
    chunks = parse_source(str(test_file))

    # Should return a list (may be empty)
    assert isinstance(chunks, list)


def test_parse_source_with_invalid_file():
    """Test that parse_source raises an error for non-existent files."""
    with pytest.raises((ValueError, FileNotFoundError)):
        parse_source("/non/existent/file.txt")


def test_make_chunks_with_text_files(tmp_path):
    """Test that make_chunks works with multiple simple text files."""
    # Create multiple text files
    file1 = tmp_path / "doc1.txt"
    file1.write_text("First document content.\nWith multiple sentences.")

    file2 = tmp_path / "doc2.txt"
    file2.write_text("Second document has different content.\nAlso multiple lines.")

    sources = [str(file1), str(file2)]
    chunked_documents = make_chunks(sources)

    assert len(chunked_documents) == 2
    assert all(isinstance(doc, ChunkedDocument) for doc in chunked_documents)
    assert all(isinstance(chunk, Chunk) for doc in chunked_documents for chunk in doc.chunks)


def test_partition_kwargs_passed_through(tmp_path):
    """Test that partition_kwargs are accepted by parse_source."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content for partition kwargs.")

    # Should not raise - just verify kwargs are accepted
    chunks = parse_source(test_file, partition_kwargs={"encoding": "utf-8"})
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_entity_workspace_explicit_override_wins():
    memgraph = MagicMock()
    lightrag_wrapper = _lightrag_wrapper_with_workspace("auto-derived")

    with (
        patch("unstructured2graph.loaders.make_chunks", return_value=[_fake_document()]),
        patch("unstructured2graph.loaders.connect_chunks_to_entities") as mock_connect,
    ):
        await from_unstructured(
            ["fake.txt"], memgraph, lightrag_wrapper, only_chunks=False, entity_workspace="explicit"
        )

    mock_connect.assert_called_once_with(memgraph, "Chunk", "explicit")


@pytest.mark.asyncio
async def test_entity_workspace_auto_derived_from_lightrag_wrapper():
    memgraph = MagicMock()
    lightrag_wrapper = _lightrag_wrapper_with_workspace("tenant-42")

    with (
        patch("unstructured2graph.loaders.make_chunks", return_value=[_fake_document()]),
        patch("unstructured2graph.loaders.connect_chunks_to_entities") as mock_connect,
    ):
        await from_unstructured(["fake.txt"], memgraph, lightrag_wrapper, only_chunks=False)

    mock_connect.assert_called_once_with(memgraph, "Chunk", "tenant-42")


@pytest.mark.asyncio
async def test_entity_workspace_falls_back_to_base_when_auto_derive_fails():
    memgraph = MagicMock()
    lightrag_wrapper = MagicMock()
    lightrag_wrapper.ainsert = AsyncMock()
    lightrag_wrapper.get_lightrag.side_effect = RuntimeError("not initialized")

    with (
        patch("unstructured2graph.loaders.make_chunks", return_value=[_fake_document()]),
        patch("unstructured2graph.loaders.connect_chunks_to_entities") as mock_connect,
    ):
        await from_unstructured(["fake.txt"], memgraph, lightrag_wrapper, only_chunks=False)

    mock_connect.assert_called_once_with(memgraph, "Chunk", "base")


@pytest.mark.asyncio
async def test_connect_chunks_to_entities_called_once_per_document():
    """connect_chunks_to_entities is a full graph scan; it must run once per
    document, not once per chunk."""
    memgraph = MagicMock()
    lightrag_wrapper = _lightrag_wrapper_with_workspace("base")
    fake_document = ChunkedDocument(
        chunks=[Chunk(text="a", hash="h1"), Chunk(text="b", hash="h2"), Chunk(text="c", hash="h3")],
        source="fake.txt",
    )

    with (
        patch("unstructured2graph.loaders.make_chunks", return_value=[fake_document]),
        patch("unstructured2graph.loaders.connect_chunks_to_entities") as mock_connect,
    ):
        await from_unstructured(["fake.txt"], memgraph, lightrag_wrapper, only_chunks=False)

    assert lightrag_wrapper.ainsert.await_count == 3
    mock_connect.assert_called_once_with(memgraph, "Chunk", "base")


@pytest.mark.asyncio
async def test_from_unstructured_requires_lightrag_wrapper_when_not_only_chunks():
    """lightrag_wrapper=None should raise a clear error unless only_chunks=True."""
    memgraph = MagicMock()

    with pytest.raises(ValueError, match="lightrag_wrapper"):
        await from_unstructured(["irrelevant.txt"], memgraph, lightrag_wrapper=None, only_chunks=False)


@pytest.mark.asyncio
async def test_from_unstructured_only_chunks_works_without_lightrag_wrapper(tmp_path):
    """only_chunks=True should not require a lightrag_wrapper at all."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content for chunk-only ingestion.")
    memgraph = MagicMock()

    await from_unstructured([str(test_file)], memgraph, lightrag_wrapper=None, only_chunks=True)

    assert memgraph.query.called


@pytest.mark.asyncio
async def test_from_unstructured_returns_grouped_chunks_per_source():
    """from_unstructured() must mirror from_texts()'s grouped-return contract:
    one list of Chunks per source, in `sources` order."""
    memgraph = MagicMock()
    doc_a = ChunkedDocument(chunks=[Chunk(text="a1", hash="ha1"), Chunk(text="a2", hash="ha2")], source="a.txt")
    doc_b = ChunkedDocument(chunks=[Chunk(text="b1", hash="hb1")], source="b.txt")

    with patch("unstructured2graph.loaders.make_chunks", return_value=[doc_a, doc_b]):
        grouped = await from_unstructured(["a.txt", "b.txt"], memgraph, only_chunks=True)

    assert grouped == [doc_a.chunks, doc_b.chunks]


@pytest.mark.asyncio
async def test_from_unstructured_source_with_no_chunks_contributes_empty_group():
    memgraph = MagicMock()
    empty_doc = ChunkedDocument(chunks=[], source="empty.txt")
    doc = ChunkedDocument(chunks=[Chunk(text="x", hash="hx")], source="x.txt")

    with patch("unstructured2graph.loaders.make_chunks", return_value=[empty_doc, doc]):
        grouped = await from_unstructured(["empty.txt", "x.txt"], memgraph, only_chunks=True)

    assert grouped == [[], doc.chunks]


def test_parse_text_empty_returns_no_chunks():
    assert parse_text("") == []
    assert parse_text("   \n  ") == []


def test_parse_text_short_simple_text_becomes_single_chunk():
    """A short, single-sentence input has nothing for chunk_by_title to split
    on, so it naturally comes back as one chunk -- via the real pipeline, not
    a length-based shortcut."""
    text = "User prefers Python over TypeScript."
    chunks = parse_text(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].hash == hashlib.sha256(text.encode()).hexdigest()


def test_parse_text_long_text_is_chunked_via_unstructured():
    long_text = "Section one.\n\n" + ("Filler sentence. " * 500) + "\n\nSection two.\n\n" + ("More filler. " * 500)

    chunks = parse_text(long_text)

    assert len(chunks) >= 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    # Long text must not collapse into a single chunk identical to the whole input.
    assert not (len(chunks) == 1 and chunks[0].text == long_text)


def test_parse_text_splits_by_content_size_regardless_of_total_length():
    """Regression guard: parse_text() must not fork behavior on an arbitrary
    input-length threshold. This text is short enough that an earlier
    version of parse_text() forced it into a single Chunk unconditionally;
    it must still be split the same way unstructured's chunk_by_title would
    split it on its own (chunk_by_title's default max_characters is ~500,
    well under this text's length)."""
    text = ("This is a filler sentence used to pad content out. " * 20).strip()
    assert len(text) < 2000  # well under the old (now-removed) inline-threshold

    chunks = parse_text(text)

    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_from_texts_only_chunks_creates_chunk_nodes_without_lightrag():
    memgraph = MagicMock()

    grouped = await from_texts(["First memory.", "Second memory."], memgraph, only_chunks=True)

    assert len(grouped) == 2
    assert all(len(group) == 1 for group in grouped)


@pytest.mark.asyncio
async def test_from_texts_requires_lightrag_wrapper_when_not_only_chunks():
    memgraph = MagicMock()

    with pytest.raises(ValueError, match="lightrag_wrapper"):
        await from_texts(["some text"], memgraph, lightrag_wrapper=None, only_chunks=False)


@pytest.mark.asyncio
async def test_from_texts_runs_entity_extraction_and_connects_chunks():
    memgraph = MagicMock()
    lightrag_wrapper = _lightrag_wrapper_with_workspace("base")

    with patch("unstructured2graph.loaders.connect_chunks_to_entities") as mock_connect:
        grouped = await from_texts(["Alice works on the graph engine."], memgraph, lightrag_wrapper)

    assert len(grouped) == 1
    assert len(grouped[0]) == 1
    lightrag_wrapper.ainsert.assert_awaited_once()
    mock_connect.assert_called_once_with(memgraph, "Chunk", "base")


@pytest.mark.asyncio
async def test_from_texts_preserves_grouping_for_empty_texts():
    """Empty inputs must still get an (empty) group, so callers can zip
    grouped results back against their original source list by index."""
    memgraph = MagicMock()

    grouped = await from_texts(["", "   ", "actual content"], memgraph, only_chunks=True)

    assert len(grouped) == 3
    assert grouped[0] == []
    assert grouped[1] == []
    assert len(grouped[2]) == 1
    assert grouped[2][0].text == "actual content"


@pytest.mark.skip(reason="Requires sample-data files and network access - run locally with full deps")
def test_chunking_of_different_sources():
    pypdf_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "pdf", "sample-files")
    docx_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "doc")
    xls_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "xls")
    sources = [
        os.path.join(pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"),
        os.path.join(docx_samples_dir, "sample3.docx"),
        os.path.join(xls_samples_dir, "financial-sample.xlsx"),
        "https://memgraph.com/docs/ai-ecosystem/graph-rag",
    ]

    chunked_documents = make_chunks(sources)
    assert len(chunked_documents) == len(sources)
    assert all(isinstance(document, ChunkedDocument) for document in chunked_documents)
    assert all(len(document.chunks) > 0 for document in chunked_documents)
    assert all(isinstance(chunk, Chunk) for document in chunked_documents for chunk in document.chunks)
    assert all(isinstance(chunk.text, str) for document in chunked_documents for chunk in document.chunks)
    assert all(isinstance(chunk.hash, str) for document in chunked_documents for chunk in document.chunks)
