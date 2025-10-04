"""
LightRAG integration with Memgraph.

This package provides a wrapper around LightRAG that uses Memgraph as the graph storage backend.
"""

from .core import MemgraphLightRAGWrapper

__version__ = "0.1.0"
__all__ = ["MemgraphLightRAGWrapper"]
