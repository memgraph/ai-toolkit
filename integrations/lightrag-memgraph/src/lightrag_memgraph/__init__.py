"""
LightRAG integration with Memgraph.

This package provides a wrapper around LightRAG that uses Memgraph as the graph storage backend.
"""

from .core import MemgraphLightRAGWrapper

__all__ = ["MemgraphLightRAGWrapper"]
