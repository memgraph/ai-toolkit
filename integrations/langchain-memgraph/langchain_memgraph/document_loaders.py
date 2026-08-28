"""Memgraph document loader."""

from collections.abc import Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class MemgraphLoader(BaseLoader):
    """Load documents from Memgraph.

    The loader is currently a placeholder because the source query and
    document-mapping contract have not been defined yet.
    """

    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError()
