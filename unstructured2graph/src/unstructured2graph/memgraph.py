import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hygm import (
    CYPHER_IDENTIFIER_PATTERN,
    ValidationCategory,
    ValidationIssue,
    ValidationSeverity,
    require_valid_identifier,
)
from lightrag_memgraph import DEFAULT_EMBEDDING_DIM
from memgraph_toolbox.api.memgraph import Memgraph

if TYPE_CHECKING:
    from .ontology import Ontology

logger = logging.getLogger(__name__)

_LABEL_WORD_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9]+")

#: Relationship types this package writes for its own structure, never extracted facts.
STRUCTURAL_RELATIONSHIP_TYPES = ("MENTIONED_IN", "NEXT")


def _require_valid_identifier(value: str, role: str) -> None:
    """Raise ValueError unless `value` is safe to f-string-interpolate into
    Cypher as a label, relationship type, property key, or variable name --
    Cypher can parameterize values but not these. See hygm.identifiers."""
    require_valid_identifier(value, role)


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
    return label if CYPHER_IDENTIFIER_PATTERN.match(label) else None


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


def connect_chunks_to_entities(
    memgraph: Memgraph, chunk_label: str, entity_label: str, chunk_hashes: list[str] | None = None
):
    """MERGE (entity)-[:MENTIONED_IN]->(chunk) for every entity whose file_path names a chunk hash.

    With `chunk_hashes`, only entities of those chunks are visited, one
    `file_path` lookup per chunk. Without it, every entity in the workspace
    is, which run once per ingest makes a growing graph's ingestion
    quadratic. Each chunk is looked up by hash, never joined across the two
    labels.
    """
    if chunk_hashes is None:
        memgraph.query(
            f"""
            MATCH (n:{entity_label})
            WHERE n.file_path IS NOT NULL
            MATCH (m:{chunk_label} {{hash: n.file_path}})
            MERGE (n)-[:MENTIONED_IN]->(m);
            """
        )
        return
    memgraph.query(
        f"""
        UNWIND $chunk_hashes AS h
        MATCH (m:{chunk_label} {{hash: h}})
        MATCH (n:{entity_label} {{file_path: h}})
        MERGE (n)-[:MENTIONED_IN]->(m);
        """,
        params={"chunk_hashes": list(chunk_hashes)},
    )


def _entities(workspace_label: str, chunk_hashes: list[str] | None) -> str:
    """A clause binding `n` to workspace entities -- all, or those MENTIONED_IN `chunk_hashes` --
    that a WHERE can follow."""
    if chunk_hashes is None:
        return f"MATCH (n:{workspace_label})"
    return (
        f"MATCH (c:Chunk) WHERE c.hash IN $chunk_hashes "
        f"MATCH (n:{workspace_label})-[:MENTIONED_IN]->(c) WITH DISTINCT n"
    )


def link_mentions(
    memgraph: Memgraph,
    entity_label: str,
    match_key: str,
    chunk_hash: str,
    sources_by_entity: dict[str, list[str]],
):
    """MERGE (entity)-[:MENTIONED_IN]->(:Chunk {hash: chunk_hash}) for each entity id, at ingest.

    Needed once an entity's identity outlives its chunk (#346): a merged node
    keeps only its first chunk's file_path, so connect_chunks_to_entities alone
    would link it to that one chunk and lose every later mention.

    Args:
        sources_by_entity: entity id -> the `Segment.source_id`s of the
            segments mentioning it in this chunk, unioned into
            `MENTIONED_IN.sources` so re-ingesting adds nothing. An empty list
            links the entity without touching `sources`.
    """
    _require_valid_identifier(entity_label, "entity_label")
    _require_valid_identifier(match_key, "match_key")
    if not sources_by_entity:
        return
    rows = [
        {"id": entity_id, "sources": sorted(set(sources))} for entity_id, sources in sorted(sources_by_entity.items())
    ]
    memgraph.query(
        f"""
        MATCH (c:Chunk {{hash: $chunk_hash}})
        UNWIND $rows AS row
        MATCH (n:{entity_label} {{{match_key}: row.id}})
        MERGE (n)-[r:MENTIONED_IN]->(c)
        SET r.sources = CASE
            WHEN size(row.sources) = 0 THEN r.sources
            ELSE [s IN coalesce(r.sources, []) WHERE NOT s IN row.sources] + row.sources
        END
        """,
        params={"chunk_hash": chunk_hash, "rows": rows},
    )


def promote_entity_types_to_labels(
    memgraph: Memgraph, workspace_label: str, ontology: "Ontology", chunk_hashes: list[str] | None = None
) -> None:
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

    With `chunk_hashes`, only entities MENTIONED_IN those chunks are visited,
    so run connect_chunks_to_entities first. Ingestion passes the chunks it
    just wrote; re-projecting a whole workspace after an ontology change
    omits it.
    """
    entities = _entities(workspace_label, chunk_hashes)
    params = {"chunk_hashes": list(chunk_hashes)} if chunk_hashes is not None else {}
    labels = ontology.allowed_labels()
    for label in labels:
        memgraph.query(
            f"""
            {entities}
            WHERE toLower(n.entity_type) = toLower($label) AND NOT n:{label}
            SET n:{label}
            """,
            params={**params, "label": label},
        )

    if not labels:
        memgraph.query(f"{entities} SET n.ontology_conformant = false", params=params)
        return

    conforms_clause = " OR ".join(f"n:{label}" for label in labels)
    memgraph.query(f"{entities} WHERE NOT ({conforms_clause}) SET n.ontology_conformant = false", params=params)
    memgraph.query(f"{entities} WHERE {conforms_clause} REMOVE n.ontology_conformant", params=params)


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


def _any_label(variable: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        _require_valid_identifier(label, "label")
    return " OR ".join(f"{variable}:{label}" for label in labels)


# Deliberately not `(a:ws OR b:ws)`: Memgraph 3.13.1 plans an OR of label
# checks on two different variables as `Filter (a:ws), (b:ws)` -- an AND -- so
# every relationship with one non-workspace endpoint, e.g. onto (:User), is
# silently skipped (verified with EXPLAIN). A label test on one variable is fine.
_TOUCHES_WORKSPACE = "($workspace IN labels(a) OR $workspace IN labels(b))"


def _relationships(workspace_label: str, relation_type: str | None, chunk_hashes: list[str] | None) -> str:
    """A clause binding `r` (and its endpoints `a`, `b`) to extracted relationships touching the
    workspace -- all of them, or those extracted from `chunk_hashes` -- that a WHERE can follow.

    Scoped, it starts from the chunks' own entities (every extracted
    relationship has a workspace endpoint MENTIONED_IN its chunk) rather than
    scanning every relationship of the type, which per ingest would make a
    growing graph's ingestion quadratic. GLiNER2 stamps its edges with
    `chunk`; LightRAG's carry `file_path`, every contributing chunk's hash
    joined by a separator, hence CONTAINS.
    """
    typed = f":{relation_type}" if relation_type else ""
    if chunk_hashes is None:
        return f"MATCH (a)-[r{typed}]->(b) WHERE {_TOUCHES_WORKSPACE} WITH a, r, b"
    return (
        f"MATCH (c:Chunk) WHERE c.hash IN $chunk_hashes "
        f"MATCH (c)<-[:MENTIONED_IN]-(n:{workspace_label})-[r{typed}]-() "
        f"WHERE any(h IN $chunk_hashes WHERE coalesce(r.chunk, r.file_path, '') CONTAINS h) "
        f"WITH DISTINCT r WITH startNode(r) AS a, r, endNode(r) AS b"
    )


def _scope_params(workspace_label: str, chunk_hashes: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {"workspace": workspace_label}
    if chunk_hashes is not None:
        params["chunk_hashes"] = list(chunk_hashes)
    return params


def enforce_relation_domain_range(
    memgraph: Memgraph, workspace_label: str, ontology: "Ontology", chunk_hashes: list[str] | None = None
) -> list[ValidationIssue]:
    """
    The post-hoc half of the typed relation model (#348): check every
    relationship of a declared, constrained relation type against its
    start_labels/end_labels, over already-promoted labels, so run it after
    promote_entity_types_to_labels().

    A mismatch is never removed -- per ADR 0004 it is kept and stamped
    `r.ontology_conformant = false`, and a relationship that now conforms has
    the flag cleared. On a GLiNER2 graph a flag means a bug: the same
    specification constrained its decoding, and domain/range survives the
    window merge (#355). A relation type with neither side constrained is never
    checked.

    Only relationships touching a `workspace_label` node are considered, so
    another package's relationship of the same type is left alone.

    Args:
        chunk_hashes: If given, only relationships extracted from these chunks.

    Returns:
        One WARNING issue per distinct (relation type, start labels, end labels)
        violation, with the count in `details["count"]`.
    """
    _require_valid_identifier(workspace_label, "workspace_label")
    model = ontology.model
    params = _scope_params(workspace_label, chunk_hashes)
    issues: list[ValidationIssue] = []
    for relation in ontology.relation_types:
        if not relation.constrained:
            continue
        _require_valid_identifier(relation.label, "relation type")
        conforms = (
            f"({_any_label('a', model.endpoint_labels(relation, 'start'))}) "
            f"AND ({_any_label('b', model.endpoint_labels(relation, 'end'))})"
        )
        match = _relationships(workspace_label, relation.label, chunk_hashes)
        rows = memgraph.query(
            f"""
            {match}
            WHERE NOT ({conforms})
            SET r.ontology_conformant = false
            RETURN labels(a) AS start_labels, labels(b) AS end_labels, count(r) AS count
            """,
            params=params,
        )
        memgraph.query(f"{match} WHERE {conforms} REMOVE r.ontology_conformant", params=params)
        for row in rows:
            start = tuple(sorted(label for label in row["start_labels"] if label != workspace_label))
            end = tuple(sorted(label for label in row["end_labels"] if label != workspace_label))
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.STRUCTURE,
                    message=(
                        f"{row['count']} :{relation.label} relationship(s) from {start or '(unlabelled)'} to "
                        f"{end or '(unlabelled)'} fall outside its declared domain/range"
                    ),
                    expected=(model.endpoint_labels(relation, "start"), model.endpoint_labels(relation, "end")),
                    actual=(start, end),
                    details={"relation_type": relation.label, "count": row["count"]},
                )
            )
    return issues


@dataclass(frozen=True)
class OntologyReport:
    """What the graph holds relative to an ontology: integrity and coverage counts.

    Reporting only -- nothing reads these to filter (#355). On a GLiNER2 graph
    both non-conformant counts should be zero; anything else is a bug.

    Attributes:
        nonconformant_entities: Workspace entities stamped ontology_conformant=false.
        nonconformant_relations: Relationships stamped ontology_conformant=false.
        relationships: Extracted relationships touching a workspace entity.
        declared_relationships: Those whose type the ontology declares.
        zero_instance_relation_types: Declared types with no instance anywhere
            in the workspace (always workspace-wide, even when scoped).
        issues: The coverage findings, as ValidationIssues (#349).
    """

    nonconformant_entities: int
    nonconformant_relations: int
    relationships: int
    declared_relationships: int
    zero_instance_relation_types: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)


def ontology_report(
    memgraph: Memgraph, workspace_label: str, ontology: "Ontology", chunk_hashes: list[str] | None = None
) -> OntologyReport:
    """
    Read what the graph holds against `ontology`. Run it after label promotion
    and enforce_relation_domain_range(), whose flags it counts.

    Coverage is observed from the graph rather than asked of the backend
    (#349): "relationships, none of a declared type" means the backend emits
    untyped relations (LightRAG's :DIRECTED); "no relationships at all" means
    nothing was extracted. Both are distinct from a declared type that simply
    never materialized.

    Args:
        chunk_hashes: If given, entity and relationship counts cover only these
            chunks (an entity counts if it is MENTIONED_IN one of them).
    """
    _require_valid_identifier(workspace_label, "workspace_label")
    params = _scope_params(workspace_label, chunk_hashes)
    nonconformant_entities = memgraph.query(
        f"{_entities(workspace_label, chunk_hashes)} WHERE n.ontology_conformant = false RETURN count(n) AS count",
        params=params,
    )[0]["count"]
    rows = memgraph.query(
        f"""
        {_relationships(workspace_label, None, chunk_hashes)}
        WHERE NOT type(r) IN $structural
        RETURN type(r) AS type, count(r) AS count, sum(CASE WHEN r.ontology_conformant = false THEN 1 ELSE 0 END) AS flagged
        """,
        params={**params, "structural": list(STRUCTURAL_RELATIONSHIP_TYPES)},
    )
    declared = set(ontology.allowed_relation_labels())
    by_type = {row["type"]: row for row in rows}
    relationships = sum(row["count"] for row in rows)
    declared_relationships = sum(row["count"] for t, row in by_type.items() if t in declared)
    nonconformant_relations = sum(row["flagged"] for t, row in by_type.items() if t in declared)

    # One cheap existence probe per declared type (it stops at the first
    # match), not a scan of every relationship: this runs after every ingest.
    present = {
        label
        for label in declared
        if memgraph.query(
            f"MATCH (a)-[r:{label}]->(b) WHERE {_TOUCHES_WORKSPACE} RETURN 1 AS hit LIMIT 1",
            params={"workspace": workspace_label},
        )
    }
    zero_instance = tuple(label for label in ontology.allowed_relation_labels() if label not in present)

    issues: list[ValidationIssue] = []
    if declared and relationships and not declared_relationships:
        issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                ValidationCategory.COVERAGE,
                f"{relationships} relationship(s), none of a declared relation type: the backend does not "
                "emit typed relations, so domain/range checking is vacuous here",
                details={"types": sorted(by_type)},
            )
        )
    elif declared and not relationships:
        issues.append(
            ValidationIssue(ValidationSeverity.INFO, ValidationCategory.COVERAGE, "No relationships extracted")
        )
    if declared and zero_instance:
        issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                ValidationCategory.COVERAGE,
                f"{len(zero_instance)} declared relation type(s) have no instances: {', '.join(zero_instance)}",
                details={"relation_types": list(zero_instance)},
            )
        )
    return OntologyReport(
        nonconformant_entities=nonconformant_entities,
        nonconformant_relations=nonconformant_relations,
        relationships=relationships,
        declared_relationships=declared_relationships,
        zero_instance_relation_types=zero_instance,
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class Endpoint:
    """How to find one end of an extracted relationship: MATCH (:label {key: value})."""

    label: str
    key: str
    value: str


def upsert_extracted_relationships(memgraph: Memgraph, relationships: list[dict[str, Any]]) -> None:
    """
    MERGE extracted relationships, one per (type, head, tail, source chunk, source_id).

    Keyed on the source chunk as well as the endpoints, so a fact asserted in
    two sessions is two relationships with their own valid_at: superseded
    facts are retained, timestamps only (#347), and a question asking for the
    *initial* value still has it. With a `source_id` the key also takes the
    source turn, so a fact said in two turns of one session is two
    relationships too (#392). Re-ingesting a chunk is idempotent.

    Args:
        relationships: dicts with `type`, `head`/`tail` (Endpoint), `chunk`
            (the source chunk hash), `valid_at` (ISO-8601 string or None, stored
            as a Memgraph datetime so date arithmetic is a subtraction, #364),
            `confidence` (float or None), and optionally `source_id`, `text`
            (the source sentence) and `role` (the speaker), each str or None.

    Raises:
        ValueError: if a relationship type, endpoint label or key isn't a valid Cypher identifier.
    """
    groups: dict[tuple[str, str, str, str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for rel in relationships:
        head, tail = rel["head"], rel["tail"]
        has_source = rel.get("source_id") is not None
        groups[(rel["type"], head.label, head.key, tail.label, tail.key, has_source)].append(
            {
                "from": head.value,
                "to": tail.value,
                "chunk": rel["chunk"],
                "source_id": rel.get("source_id"),
                "valid_at": rel.get("valid_at"),
                "confidence": rel.get("confidence"),
                "text": rel.get("text"),
                "role": rel.get("role"),
            }
        )
    for (relation_type, head_label, head_key, tail_label, tail_key, has_source), rows in groups.items():
        for value, role in (
            (relation_type, "relation type"),
            (head_label, "endpoint label"),
            (head_key, "endpoint key"),
            (tail_label, "endpoint label"),
            (tail_key, "endpoint key"),
        ):
            _require_valid_identifier(value, role)
        # MERGE can't key on a null property, so a relationship with no source
        # turn keeps the chunk-only key.
        key = "{chunk: rel.chunk, source_id: rel.source_id}" if has_source else "{chunk: rel.chunk}"
        memgraph.query(
            f"""
            UNWIND $rows AS rel
            MATCH (a:{head_label} {{{head_key}: rel.from}})
            MATCH (b:{tail_label} {{{tail_key}: rel.to}})
            MERGE (a)-[r:{relation_type} {key}]->(b)
            SET r.valid_at = CASE WHEN rel.valid_at IS NULL THEN null ELSE datetime(rel.valid_at) END,
                r.confidence = rel.confidence,
                r.text = rel.text,
                r.role = rel.role
            """,
            params={"rows": rows},
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


def ensure_lookup(memgraph: Memgraph, label: str, property: str) -> None:
    """Idempotently ensure :label(property) is both unique and indexed.

    Both, because in Memgraph a uniqueness constraint does not create an index
    (verified with EXPLAIN: MATCH (c:Chunk {hash: ...}) plans a label scan plus
    a filter under the constraint alone), so every MERGE on the key would scan
    the whole label; and an index alone does not stop two concurrent MERGEs
    from both creating.
    """
    _require_valid_identifier(label, "label")
    _require_valid_identifier(property, "property")
    create_unique_constraint(memgraph, label, property)
    try:
        memgraph.query(f"CREATE INDEX ON :{label}({property});")
    except Exception as e:
        logger.warning(f"Error creating index on :{label}({property}): {e}")


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
