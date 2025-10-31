import time
import logging

from tqdm import tqdm
from memgraph_toolbox.api.memgraph import Memgraph

logger = logging.getLogger(__name__)


def create_nodes_from_list(
    memgraph: Memgraph, nodes: list[dict], node_label: str, batch_size: int
) -> None:
    """
    Import data from the given list of dictionaries to Memgraph by batching.
    """
    num_nodes = len(nodes)
    num_batches = (num_nodes + batch_size - 1) // batch_size
    max_retries = 3
    retry_delay = 3
    properties_string = ", ".join([f"{key}: data.{key}" for key in nodes[0].keys()])
    insert_query = f"""
    UNWIND $batch AS data
    CREATE (n:{node_label} {{{properties_string}}})
    """
    with tqdm(total=num_batches, unit="batch") as pbar:
        for offset in range(0, num_nodes, batch_size):
            batch_nodes = nodes[offset : offset + batch_size]
            for attempt in range(max_retries):
                try:
                    memgraph.query(insert_query, params={"batch": batch_nodes})
                    logger.info(f"Created {len(batch_nodes)} nodes")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise e
            pbar.update(1)


def connect_chunks_to_entities(memgraph: Memgraph, chunk_label: str, entity_label: str):
    memgraph.query(
        f"MATCH (n:{entity_label}), (m:{chunk_label}) WHERE n.file_path = m.id CREATE (n)-[:MENTIONED_IN]->(m);"
    )


def create_vector_search_index(memgraph: Memgraph, label: str, property: str):
    # TODO(gitbuda): Add proper error handling.
    try:
        memgraph.query(
            f"CREATE VECTOR INDEX vs_name ON :{label}({property}) WITH CONFIG {{'dimension': 384, 'capacity': 10000}};"
        )
    except Exception as _:
        pass


def compute_embeddings(memgraph: Memgraph, label: str):
    # TODO(gitbuda): Implement batching on the Cypher side as well.
    memgraph.query(
        f"""
            MATCH (n:{label})
            WITH collect(n) AS nodes
            CALL embeddings.node_sentence(nodes) YIELD *;
        """
    )
