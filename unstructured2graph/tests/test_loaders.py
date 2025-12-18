"""Simple test for unstructured2graph loaders."""

import os

import pytest
from unstructured2graph import parse_source, make_chunks, Chunk, ChunkedDocument

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


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
    assert all(
        isinstance(chunk, Chunk) for doc in chunked_documents for chunk in doc.chunks
    )


def test_partition_kwargs_passed_through(tmp_path):
    """Test that partition_kwargs are accepted by parse_source."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content for partition kwargs.")

    # Should not raise - just verify kwargs are accepted
    chunks = parse_source(test_file, partition_kwargs={"encoding": "utf-8"})
    assert isinstance(chunks, list)


@pytest.mark.skip(
    reason="Requires sample-data files and network access - run locally with full deps"
)
def test_chunking_of_different_sources():
    pypdf_samples_dir = os.path.join(
        SCRIPT_DIR, "..", "sample-data", "pdf", "sample-files"
    )
    docx_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "doc")
    xls_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "xls")
    sources = [
        os.path.join(
            pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
        ),
        os.path.join(docx_samples_dir, "sample3.docx"),
        os.path.join(xls_samples_dir, "financial-sample.xlsx"),
        "https://memgraph.com/docs/ai-ecosystem/graph-rag",
    ]

    chunked_documents = make_chunks(sources)
    assert len(chunked_documents) == len(sources)
    assert all(isinstance(document, ChunkedDocument) for document in chunked_documents)
    assert all(len(document.chunks) > 0 for document in chunked_documents)
    assert all(
        isinstance(chunk, Chunk)
        for document in chunked_documents
        for chunk in document.chunks
    )
    assert all(
        isinstance(chunk.text, str)
        for document in chunked_documents
        for chunk in document.chunks
    )
    assert all(
        isinstance(chunk.hash, str)
        for document in chunked_documents
        for chunk in document.chunks
    )
