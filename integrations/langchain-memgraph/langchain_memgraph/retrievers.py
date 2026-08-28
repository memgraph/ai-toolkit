"""Memgraph retrievers."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MemgraphRetriever(BaseRetriever):
    """Retrieve documents relevant to a query from Memgraph.

    The current implementation returns placeholder documents. A production
    implementation must define the graph query and result-to-document mapping.
    """

    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        k = kwargs.get("k", self.k)
        return [Document(page_content=f"Result {i} for query: {query}") for i in range(k)]
