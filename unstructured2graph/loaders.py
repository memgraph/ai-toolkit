import os
import logging

import asyncio
import shutil
import logging

from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import from_unstructured

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKING_DIR = os.path.join(SCRIPT_DIR, "lightrag_storage.out")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Delete & create LightRAG working directory.
    lightrag_log_file = os.path.join(SCRIPT_DIR, "lightrag.log")
    if os.path.exists(lightrag_log_file):
        os.remove(lightrag_log_file)
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Cleanup Memgraph database.
    memgraph = Memgraph()
    memgraph.query("MATCH (n) DETACH DELETE n;")

    # Define all example sources. NOTE: only paths/urls are needed here.
    pypdf_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "pdf", "sample-files")
    docx_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "doc")
    xls_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "xls")
    sources = [
        os.path.join(
            pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
        ),
        os.path.join(docx_samples_dir, "sample3.docx"),
        # os.path.join(xls_samples_dir, "financial-sample.xlsx"),
        # "https://memgraph.com/docs/ai-ecosystem/graph-rag",
    ]

    # Use the function from the unstructured2graph library
    asyncio.run(from_unstructured(sources, memgraph))
