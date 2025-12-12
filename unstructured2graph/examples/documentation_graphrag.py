import os
import logging
import argparse
import asyncio
import shutil

from dotenv import load_dotenv
from lightrag_memgraph import MemgraphLightRAGWrapper
from memgraph_toolbox.api.memgraph import Memgraph

from unstructured2graph import from_unstructured, create_index

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LIGHTRAG_DIR = os.path.join(SCRIPT_DIR, "..", "lightrag_storage.out")

# Hardcoded list of documentation URLs to process
DOC_URLS = [
    "https://memgraph.com/docs/deployment/workloads",
    "https://memgraph.com/docs/deployment/workloads/memgraph-in-cybersecurity",
    "https://memgraph.com/docs/deployment/workloads/memgraph-in-graphrag",
    "https://memgraph.com/docs/deployment/workloads/memgraph-in-high-throughput-workloads",
    "https://memgraph.com/docs/deployment/workloads/memgraph-in-mission-critical-workloads",
    "https://memgraph.com/docs/deployment/workloads/memgraph-in-fraud-detection",
    "https://memgraph.com/docs/deployment/workloads/memgraph-in-supply-chain",
    "https://memgraph.com/docs/deployment/benchmarking-memgraph",
]

logger = logging.getLogger(__name__)


async def ingest_docs(
    only_chunks: bool = False,
    link_chunks: bool = True,
    cleanup: bool = True
):
    doc_urls = DOC_URLS
    
    if not doc_urls:
        logger.warning("No documentation URLs provided. Exiting.")
        return
    
    logger.info(f"Processing {len(doc_urls)} documentation pages. Starting ingestion...")
    
    lightrag_log_file = os.path.join(LIGHTRAG_DIR, "lightrag.log")
    if cleanup:
        if os.path.exists(lightrag_log_file):
            os.remove(lightrag_log_file)
        if os.path.exists(LIGHTRAG_DIR):
            shutil.rmtree(LIGHTRAG_DIR)
    if not os.path.exists(LIGHTRAG_DIR):
        os.mkdir(LIGHTRAG_DIR)
    
    memgraph = Memgraph()
    
    if cleanup:
        logger.info("Cleaning up existing data in Memgraph...")
        memgraph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        memgraph.query("DROP GRAPH")
        memgraph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

    
    create_index(memgraph, "Chunk", "hash")
    
    lightrag_wrapper = MemgraphLightRAGWrapper(log_level="WARNING", disable_embeddings=True)
    await lightrag_wrapper.initialize(working_dir=LIGHTRAG_DIR)
    
    try:
        await from_unstructured(
            doc_urls,
            memgraph,
            lightrag_wrapper,
            only_chunks=only_chunks,
            link_chunks=link_chunks,
        )
        logger.info("Successfully ingested all documentation pages into Memgraph!")
    finally:
        await lightrag_wrapper.afinalize()


async def main():    
    # Verify OPENAI_API_KEY is set
    # LightRAG Integration currently requires an OpenAI API key.
    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY environment variable is not set. Please set your OpenAI API key.")

    parser = argparse.ArgumentParser(description="Ingest Memgraph documentation URLs into Memgraph")
    parser.add_argument("--only-chunks", action="store_true", help="Only create chunk nodes without LightRAG")
    parser.add_argument("--no-link-chunks", action="store_true", help="Don't link chunks with NEXT relationship")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup existing data")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    await ingest_docs(
        only_chunks=args.only_chunks,
        link_chunks=not args.no_link_chunks,
        cleanup=not args.no_cleanup
    )


if __name__ == "__main__":
    asyncio.run(main())
