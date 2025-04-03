from importlib import metadata

from langchain_memgraph.chains.graph_qa import MemgraphQAChain
from langchain_memgraph.document_loaders import MemgraphLoader
from langchain_memgraph.graphs.memgraph import Memgraph
from langchain_memgraph.retrievers import MemgraphRetriever
from langchain_memgraph.toolkits import MemgraphToolkit
from langchain_memgraph.tools import QueryMemgraphTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "MemgraphLoader",
    "MemgraphQAChain",
    "Memgraph",
    "MemgraphRetriever",
    "MemgraphToolkit",
    "QueryMemgraphTool",
    "__version__",
]
