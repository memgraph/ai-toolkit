"""Unit tests for LightRAGBackend, the ExtractionBackend adapter over MemgraphLightRAGWrapper."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from unstructured2graph import Chunk, LightRAGBackend


def test_workspace_label_auto_derived_from_lightrag():
    wrapper = MagicMock()
    wrapper.workspace = "tenant-42"

    backend = LightRAGBackend(wrapper)

    assert backend.workspace_label == "tenant-42"


def test_workspace_label_falls_back_to_base_when_lightrag_not_initialized():
    wrapper = MagicMock()
    type(wrapper).workspace = PropertyMock(side_effect=RuntimeError("not initialized"))

    backend = LightRAGBackend(wrapper)

    assert backend.workspace_label == "base"


@pytest.mark.asyncio
async def test_aingest_chunk_delegates_to_wrapper_ainsert():
    wrapper = MagicMock()
    wrapper.ainsert = AsyncMock()
    backend = LightRAGBackend(wrapper)
    chunk = Chunk(text="Alice works on the graph engine.", hash="h1")

    await backend.aingest_chunk(MagicMock(), chunk)

    wrapper.ainsert.assert_awaited_once_with(input=chunk.text, file_paths=[chunk.hash])
