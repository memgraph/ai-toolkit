"""Simple test for unstructured2graph loaders."""

import pytest
from unstructured2graph import parse_source


def test_parse_source_with_simple_text(tmp_path):
    """Test that parse_source can handle a simple text file."""
    # Create a simple text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a simple test document.\nIt has multiple lines.")

    # Parse the file
    chunks = parse_source(str(test_file))

    # Assert that we got at least one chunk
    assert len(chunks) > 0
    assert isinstance(chunks, list)

    # Assert that chunks contain text
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk.strip()) > 0 for chunk in chunks)


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
