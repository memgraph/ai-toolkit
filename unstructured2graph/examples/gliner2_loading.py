import asyncio
import logging
import os

import sources as SOURCES

from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import from_unstructured
from unstructured2graph.gliner2_backend import GLiNER2Backend

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


async def from_unstructured_with_gliner2():
    """Same ingestion as loading.py, but with a local, LLM-free extraction
    backend -- no OPENAI_API_KEY needed. Requires the optional `gliner2`
    dependency: pip install -e ".[gliner2]".
    """
    memgraph = Memgraph(user_agent="unstructured2graph")
    memgraph.query("MATCH (n) DETACH DELETE n;")

    # DEFAULT_ONTOLOGY (used when omitted) is entity-only; pass an ontology
    # with relation_types to also extract typed relationship edges.
    backend = GLiNER2Backend()

    await from_unstructured(
        SOURCES.MEMGRAPH_DOCS_GITHUB_LATEST_RAW,
        memgraph,
        backend,
        only_chunks=False,
        link_chunks=True,
        enforce_ontology=True,  # promote entity_type to real labels (:Person, :Organization, ...)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    asyncio.run(from_unstructured_with_gliner2())
