import logging
import time

from memgraph_toolbox.api.memgraph import Memgraph

logger = logging.getLogger(__name__)


def create_nodes_from_list(
    memgraph: Memgraph, nodes: list[dict], node_label: str, batch_size: int
) -> None:
    """
    Import data from the given list of dictionaries to Memgraph by batching.
    """
    if not nodes:
        logger.warning(
            f"No nodes provided to create_nodes_from_list for label {node_label}"
        )
        return

    num_nodes = len(nodes)
    max_retries = 3
    retry_delay = 3
    properties_string = ", ".join([f"{key}: data.{key}" for key in nodes[0].keys()])
    insert_query = f"""
    UNWIND $batch AS data
    CREATE (n:{node_label} {{{properties_string}}})
    """
    for offset in range(0, num_nodes, batch_size):
        batch_nodes = nodes[offset : offset + batch_size]
        for attempt in range(max_retries):
            try:
                memgraph.query(insert_query, params={"batch": batch_nodes})
                logger.info(
                    f"Created {len(batch_nodes)} nodes with label :{node_label}"
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e


def connect_chunks_to_entities(memgraph: Memgraph, chunk_label: str, entity_label: str):
    memgraph.query(
        f"""
        MATCH (n:{entity_label}), (m:{chunk_label})
        WHERE n.file_path = m.hash
        MERGE (n)-[:MENTIONED_IN]->(m);
        """
    )


def link_nodes_in_order(
    memgraph: Memgraph,
    find_label: str,
    find_property: str,
    from_to_dicts: list[dict],
    create_edge_type: str,
):
    try:
        memgraph.query(
            f"""
            UNWIND $relationships AS rel
            MATCH (a:{find_label} {{{find_property}: rel.from}}), (b:{find_label} {{{find_property}: rel.to}})
            MERGE (a)-[:{create_edge_type}]->(b)
            """,
            params={"relationships": from_to_dicts},
        )
    except Exception as e:
        logger.error(f"Error creating chunk chain relationships: {e}")


def create_index(memgraph: Memgraph, label: str, property: str):
    try:
        memgraph.query(f"CREATE INDEX ON :{label}({property});")
    except Exception as e:
        logger.warning(f"Error creating index: {e}")


def create_label_index(memgraph: Memgraph, label: str):
    """
    Create a label index for efficient node lookups by label.

    Memgraph does not auto-create label indices, so this should be called
    before performing queries that filter by label (e.g., MATCH (n:Label)).

    Args:
        memgraph: Memgraph instance for database operations
        label: The node label to create an index for
    """
    try:
        memgraph.query(f"CREATE INDEX ON :{label};")
        logger.info(f"Created label index on :{label}")
    except Exception as e:
        # Index may already exist
        logger.warning(f"Could not create label index on :{label}: {e}")


def create_vector_search_index(memgraph: Memgraph, label: str, property: str):
    try:
        memgraph.query(
            f"CREATE VECTOR INDEX vs_name ON :{label}({property}) WITH CONFIG {{'dimension': 384, 'capacity': 10000}};"
        )
    except Exception as e:
        logger.warning(f"Error creating vector search index: {e}")


def compute_embeddings(memgraph: Memgraph, label: str):
    memgraph.query(
        f"""
            MATCH (n:{label})
            WITH collect(n) AS nodes
            CALL embeddings.node_sentence(nodes) YIELD *;
        """
    )
