import logging
import re
import time
from typing import TYPE_CHECKING, Any

from lightrag_memgraph import DEFAULT_EMBEDDING_DIM
from memgraph_toolbox.api.memgraph import Memgraph

if TYPE_CHECKING:
    from .ontology import Ontology

logger = logging.getLogger(__name__)

# A derived label gets f-string-interpolated directly into Cypher
# (SET n:{label}), so it's restricted to safe identifier characters.
_VALID_LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LABEL_WORD_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def _require_valid_identifier(value: str, role: str) -> None:
    """Raise ValueError unless `value` is safe to f-string-interpolate into
    Cypher as a label, relationship type, property key, or variable name --
    Cypher can parameterize values but not these, so any of them built from
    caller-supplied or extracted data (not a compile-time literal) must be
    checked before use. `role` names what was being validated, for a
    diagnosable error message (e.g. "node_label", "relation type")."""
    if not _VALID_LABEL_PATTERN.match(value):
        raise ValueError(
            f"Invalid {role} {value!r}: must be a valid identifier (letters, digits, "
            "underscore, not starting with a digit) to use directly in a Cypher query"
        )


def _entity_type_to_label(entity_type: str) -> str | None:
    """
    Convert a raw entity_type value (e.g. "natural object") into a
    PascalCase Memgraph label ("NaturalObject"). Returns None if nothing
    safe can be derived -- e.g. empty after stripping non-alphanumeric
    characters, or the result would start with a digit -- so callers can
    skip promoting that entity_type rather than risk an invalid label.
    """
    words = [w for w in _LABEL_WORD_SPLIT_PATTERN.split(entity_type.strip()) if w]
    if not words:
        return None
    label = "".join(word[:1].upper() + word[1:].lower() for word in words)
    return label if _VALID_LABEL_PATTERN.match(label) else None


def create_nodes_from_list(
    memgraph: Memgraph,
    nodes: list[dict],
    node_label: str,
    batch_size: int,
    merge_key: str | None = None,
) -> None:
    """
    Import data from the given list of dictionaries to Memgraph by batching.

    Args:
        merge_key: If given, nodes are upserted via MERGE keyed on this
            property (a no-op for nodes that already exist), making re-runs
            over the same data safe. If None (default), nodes are inserted
            via CREATE, so re-running over the same data duplicates them.
    """
    if not nodes:
        logger.warning(f"No nodes provided to create_nodes_from_list for label {node_label}")
        return

    num_nodes = len(nodes)
    max_retries = 3
    retry_delay = 3
    if merge_key:
        set_keys = [key for key in nodes[0] if key != merge_key]
        set_string = ", ".join(f"n.{key} = data.{key}" for key in set_keys)
        on_create_clause = f" ON CREATE SET {set_string}" if set_string else ""
        insert_query = f"""
        UNWIND $batch AS data
        MERGE (n:{node_label} {{{merge_key}: data.{merge_key}}}){on_create_clause}
        """
    else:
        properties_string = ", ".join([f"{key}: data.{key}" for key in nodes[0]])
        insert_query = f"""
        UNWIND $batch AS data
        CREATE (n:{node_label} {{{properties_string}}})
        """
    for offset in range(0, num_nodes, batch_size):
        batch_nodes = nodes[offset : offset + batch_size]
        for attempt in range(max_retries):
            try:
                memgraph.query(insert_query, params={"batch": batch_nodes})
                logger.info(f"Created {len(batch_nodes)} nodes with label :{node_label}")
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


def promote_entity_types_to_labels(memgraph: Memgraph, workspace_label: str, ontology: "Ontology") -> None:
    """
    Additively promote each entity's `entity_type` property to a real
    Memgraph label (e.g. entity_type="person" -> :Person), for entity_type
    values that match the given ontology.

    The workspace label is never touched: LightRAG's own upsert_node()
    re-MERGEs future updates by matching on it, so removing it would break
    LightRAG's ability to recognize this node on subsequent re-ingestion.
    Entities whose entity_type doesn't match any type in the ontology are
    never rejected -- the node and its raw entity_type are always kept, and
    are instead stamped `ontology_conformant: false` so what the ontology
    doesn't recognize stays visible and queryable rather than silently
    indistinguishable from an unprocessed node. Re-running this (e.g. after
    the ontology grows a new type) clears the flag on anything that now
    conforms.
    """
    labels = ontology.allowed_labels()
    for label in labels:
        memgraph.query(
            f"""
            MATCH (n:{workspace_label})
            WHERE toLower(n.entity_type) = toLower($label) AND NOT n:{label}
            SET n:{label}
            """,
            params={"label": label},
        )

    if not labels:
        memgraph.query(f"MATCH (n:{workspace_label}) SET n.ontology_conformant = false")
        return

    conforms_clause = " OR ".join(f"n:{label}" for label in labels)
    memgraph.query(f"MATCH (n:{workspace_label}) WHERE NOT ({conforms_clause}) SET n.ontology_conformant = false")
    memgraph.query(f"MATCH (n:{workspace_label}) WHERE {conforms_clause} REMOVE n.ontology_conformant")


def promote_all_entity_types_to_labels(memgraph: Memgraph, workspace_label: str) -> None:
    """
    Promote every entity's entity_type to a real Memgraph label, with no
    fixed vocabulary to restrict against -- unlike
    promote_entity_types_to_labels(), there's no ontology_conformant
    flagging here, since without an ontology there's nothing to be
    non-conformant relative to; an entity_type is either promoted or
    skipped, never flagged.

    entity_type values that don't sanitize into a safe label (see
    _entity_type_to_label) are skipped and logged; the node, its workspace
    label, and its raw entity_type are always left as-is either way.
    """
    rows = memgraph.query(
        f"MATCH (n:{workspace_label}) WHERE n.entity_type IS NOT NULL RETURN DISTINCT n.entity_type AS entity_type"
    )
    for row in rows:
        entity_type = row["entity_type"]
        label = _entity_type_to_label(entity_type)
        if label is None:
            logger.warning(f"Skipping entity_type {entity_type!r}: could not derive a safe Memgraph label from it")
            continue
        memgraph.query(
            f"""
            MATCH (n:{workspace_label})
            WHERE toLower(n.entity_type) = toLower($entity_type) AND NOT n:{label}
            SET n:{label}
            """,
            params={"entity_type": entity_type},
        )


def upsert_typed_relationships(
    memgraph: Memgraph,
    node_label: str,
    match_key: str,
    relationships_by_type: dict[str, list[dict[str, Any]]],
) -> None:
    """
    Upsert relationships of possibly many distinct Cypher relationship types
    between nodes already present under `node_label`, matched by `match_key`.

    Cypher can parameterize values but not labels, relationship types,
    property keys, or variable names -- so `node_label`, `match_key`, every
    key in `relationships_by_type`, and every extra edge-property key found
    on a relationship dict are all f-string-interpolated into the generated
    query, and are therefore each validated (see `_require_valid_identifier`)
    before use. This matters because none of them are guaranteed to be
    compile-time literals at the call site -- `node_label` in particular is
    commonly an ExtractionBackend's caller-configured `workspace_label`.

    Args:
        node_label: Memgraph label both relationship endpoints are matched under.
        match_key: Node property used to look up each endpoint (e.g. "entity_id").
        relationships_by_type: relation label -> list of
            {"from": <match_key value>, "to": <match_key value>, **extra edge properties}.

    Raises:
        ValueError: if `node_label`, `match_key`, a relation type, or an edge
            property key isn't a valid Cypher identifier.
    """
    _require_valid_identifier(node_label, "node_label")
    _require_valid_identifier(match_key, "match_key")

    for relation_type, relationships in relationships_by_type.items():
        if not relationships:
            continue
        _require_valid_identifier(relation_type, "relation type")
        set_keys = [key for key in relationships[0] if key not in ("from", "to")]
        for key in set_keys:
            _require_valid_identifier(key, "edge property key")
        set_clause = f" SET {', '.join(f'r.{key} = rel.{key}' for key in set_keys)}" if set_keys else ""
        query = f"""
        UNWIND $relationships AS rel
        MATCH (a:{node_label} {{{match_key}: rel.from}}), (b:{node_label} {{{match_key}: rel.to}})
        MERGE (a)-[r:{relation_type}]->(b){set_clause}
        """
        memgraph.query(query, params={"relationships": relationships})


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


def create_property_index(memgraph: Memgraph, label: str, property: str):
    try:
        memgraph.query(f"CREATE INDEX ON :{label}({property});")
    except Exception as e:
        logger.warning(f"Error creating index: {e}")


def create_unique_constraint(memgraph: Memgraph, label: str, property: str):
    """
    Idempotently ensure a uniqueness constraint on :label(property). Unlike
    CREATE INDEX, this actually rejects duplicate values instead of merely
    speeding up lookups, and is safe to call on every run.
    """
    try:
        memgraph.query(f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{property} IS UNIQUE;")
        logger.info(f"Ensured uniqueness constraint on :{label}({property})")
    except Exception as e:
        logger.warning(f"Error creating uniqueness constraint on :{label}({property}): {e}")


def create_entity_type_constraint(memgraph: Memgraph, label: str):
    """
    Idempotently ensure every :label node has an entity_type property typed
    as a string -- the weak Memgraph-level backstop referenced by
    unstructured2graph's own ADR 0002 (enforce-ontology-in-application-code),
    alongside promote_entity_types_to_labels()'s real application-code
    enforcement. Memgraph has no value-membership constraint (no
    `ASSERT n.prop IN [...]`), so this can only catch a missing or
    wrong-typed entity_type, never an out-of-vocabulary one.

    Two constraints, each in its own try/except -- unlike
    create_unique_constraint's uniqueness constraint, a repeated identical
    `IS TYPED STRING` constraint raises ("already exists") rather than
    silently no-op'ing, verified live against a real Memgraph instance. The
    existence constraint (`ASSERT EXISTS`) is idempotent on repeat; kept in
    its own try/except anyway for symmetry and so one failing never blocks
    the other from being attempted.
    """
    try:
        memgraph.query(f"CREATE CONSTRAINT ON (n:{label}) ASSERT EXISTS (n.entity_type);")
        logger.info(f"Ensured entity_type existence constraint on :{label}")
    except Exception as e:
        logger.warning(f"Error creating entity_type existence constraint on :{label}: {e}")

    try:
        memgraph.query(f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.entity_type IS TYPED STRING;")
        logger.info(f"Ensured entity_type typed-string constraint on :{label}")
    except Exception as e:
        logger.warning(f"Error creating entity_type typed-string constraint on :{label}: {e}")


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


def create_vector_search_index(
    memgraph: Memgraph,
    label: str,
    property: str,
    dimension: int = DEFAULT_EMBEDDING_DIM,
    index_name: str = "vs_name",
):
    try:
        memgraph.query(
            f"CREATE VECTOR INDEX {index_name} ON :{label}({property}) "
            f"WITH CONFIG {{'dimension': {dimension}, 'capacity': 10000}};"
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
