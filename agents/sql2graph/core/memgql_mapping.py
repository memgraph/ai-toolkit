"""
MemGQL mapping format for ``--mapping`` mode.

The mapping file is a MemGQL graph body: ``vertices`` and ``edges`` backed by
``mappedTableSource`` + ``metaFields``. MemGQL loads it as-is with
``CREATE GRAPH <name> FROM FILE '<path>'`` once a connector to the source
database is registered (``ADD CONNECTOR ...``).

Edge rows come from one table in either of two shapes:

- one-to-many (FK): the table holding the foreign key. ``id`` is that table's
  primary key; the endpoint on the FK table's side is matched by the same key,
  the other endpoint by the FK column.
- many-to-many: the join table. ``from`` / ``to`` are its two FK columns and
  ``id`` is both of them, a composite key.

``mapping_to_graph_model`` also reads the earlier sql2graph format
(``nodes`` / ``id_column`` / ``rel_type``), so mapping files written before
the switch still open in the editor.
"""

import logging
from typing import Any

from core.hygm.models.graph_models import (
    GraphModel,
    GraphNode,
    GraphProperty,
    GraphRelationship,
)
from core.hygm.models.sources import NodeSource, PropertySource, RelationshipSource

logger = logging.getLogger(__name__)


def _split_ref(ref: str) -> tuple[str, str]:
    """Split a ``table.column`` reference; a bare name is a column."""
    if "." in ref:
        table, column = ref.split(".", 1)
        return table, column
    return "", ref


def _attribute(name: str, column: str) -> dict[str, str]:
    """One attribute entry; ``column`` only when it differs from the name."""
    if column and column != name:
        return {"name": name, "column": column}
    return {"name": name}


def _table_source(table: str, meta_fields: dict[str, Any], connector: str | None) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if connector:
        source["connector"] = connector
    source["table"] = table
    source["metaFields"] = meta_fields
    return source


def _edge_meta_fields(
    edge: GraphRelationship,
    vertex_keys: dict[str, tuple[str, str]],
) -> tuple[str, dict[str, Any]] | None:
    """Resolve an edge's backing table and metaFields, or None if unmappable."""
    mapping = edge.source.mapping if edge.source else {}
    start_table, start_col = _split_ref(mapping.get("start_node", ""))
    end_table, end_col = _split_ref(mapping.get("end_node", ""))
    from_label = edge.start_node_labels[0] if edge.start_node_labels else ""
    to_label = edge.end_node_labels[0] if edge.end_node_labels else ""
    from_table, from_id = vertex_keys.get(from_label, ("", ""))
    to_table, to_id = vertex_keys.get(to_label, ("", ""))

    join_table = mapping.get("join_table", "")
    if join_table:
        from_col = mapping.get("join_from_column") or start_col
        to_col = mapping.get("join_to_column") or end_col
        return join_table, {"id": [from_col, to_col], "from": from_col, "to": to_col}

    # FK on the start side: start = <fk table>.<fk column>, end = target key.
    if start_table == from_table and end_col == to_id and start_col != from_id:
        return from_table, {"id": from_id, "from": from_id, "to": start_col}

    # FK on the end side (a reversed relationship): start = source key,
    # end = <fk table>.<fk column>.
    if end_table == to_table and start_col == from_id and end_col != to_id:
        return to_table, {"id": to_id, "from": end_col, "to": to_id}

    return None


def graph_model_to_mapping(graph_model: Any, connector: str | None = None) -> dict[str, Any]:
    """
    Convert an internal GraphModel into a MemGQL graph body.

    ``connector`` stamps every ``mappedTableSource`` with the MemGQL connector
    name; without it the body resolves when exactly one connector is registered.
    """
    vertices: list[dict[str, Any]] = []
    vertex_keys: dict[str, tuple[str, str]] = {}
    for node in graph_model.nodes:
        table = node.source.name if node.source else ""
        id_column = node.source.mapping.get("id_field", "") if node.source else ""
        vertex_keys[node.primary_label] = (table, id_column)

        attributes = []
        for prop in node.properties:
            column = prop.source.field.split(".", 1)[-1] if prop.source and prop.source.field else prop.key
            attributes.append(_attribute(prop.key, column))

        vertex: dict[str, Any] = {
            "label": node.primary_label,
            "mappedTableSource": _table_source(table, {"id": id_column}, connector),
        }
        if attributes:
            vertex["attributes"] = attributes
        vertices.append(vertex)

    edges: list[dict[str, Any]] = []
    for edge in graph_model.edges:
        resolved = _edge_meta_fields(edge, vertex_keys)
        if resolved is None:
            logger.warning(
                "Skipping edge %s: no foreign key or join table backs it in the source database",
                edge.edge_type,
            )
            continue
        table, meta_fields = resolved

        attributes = []
        for prop in edge.properties:
            # Only columns of the edge's own table are addressable.
            field = prop.source.field if prop.source and prop.source.field else ""
            prop_table, column = _split_ref(field)
            if prop_table == table:
                attributes.append(_attribute(prop.key, column))

        entry: dict[str, Any] = {
            "label": edge.edge_type,
            "from": edge.start_node_labels[0] if edge.start_node_labels else "",
            "to": edge.end_node_labels[0] if edge.end_node_labels else "",
            "mappedTableSource": _table_source(table, meta_fields, connector),
        }
        if attributes:
            entry["attributes"] = attributes
        edges.append(entry)

    return {"vertices": vertices, "edges": edges}


def mapping_connector(mapping: dict[str, Any]) -> str | None:
    """The connector a mapping is stamped with, if any."""
    for element in mapping.get("vertices", []) + mapping.get("edges", []):
        connector = element.get("mappedTableSource", {}).get("connector")
        if connector:
            return connector
    return None


def _properties(table: str, attributes: list[dict[str, Any]]) -> list[GraphProperty]:
    return [
        GraphProperty(
            key=a["name"],
            source=PropertySource(field=f"{table}.{a.get('column', a['name'])}"),
        )
        for a in attributes
    ]


def mapping_to_graph_model(mapping: dict[str, Any]) -> GraphModel:
    """Convert a mapping JSON dict (MemGQL graph body or legacy) into a GraphModel."""
    if "nodes" in mapping and "vertices" not in mapping:
        return _legacy_mapping_to_graph_model(mapping)

    nodes = []
    vertex_keys: dict[str, tuple[str, str]] = {}
    for vertex in mapping.get("vertices", []):
        label = vertex.get("label", "")
        source = vertex.get("mappedTableSource", {})
        table = source.get("table", "")
        id_column = source.get("metaFields", {}).get("id", "")
        vertex_keys[label] = (table, id_column)
        nodes.append(
            GraphNode(
                labels=[label],
                properties=_properties(table, vertex.get("attributes", [])),
                source=NodeSource(
                    type="table",
                    name=table,
                    location=f"database.schema.{table}",
                    mapping={"labels": [label], "id_field": id_column},
                ),
            )
        )

    edges = []
    for entry in mapping.get("edges", []):
        source = entry.get("mappedTableSource", {})
        table = source.get("table", "")
        meta = source.get("metaFields", {})
        edge_id, from_col, to_col = meta.get("id"), meta.get("from", ""), meta.get("to", "")
        from_label, to_label = entry.get("from", ""), entry.get("to", "")
        from_table, from_id = vertex_keys.get(from_label, ("", ""))
        to_table, to_id = vertex_keys.get(to_label, ("", ""))

        rel_mapping: dict[str, Any] = {"edge_type": entry.get("label", "")}
        if isinstance(edge_id, list):
            rel_mapping.update(
                start_node=f"{table}.{from_col}",
                end_node=f"{table}.{to_col}",
                join_table=table,
                join_from_column=from_col,
                join_to_column=to_col,
                from_pk=from_id,
            )
        elif from_col == edge_id:
            rel_mapping.update(start_node=f"{table}.{to_col}", end_node=f"{to_table}.{to_id}", from_pk=edge_id)
        else:
            rel_mapping.update(start_node=f"{from_table}.{from_id}", end_node=f"{table}.{from_col}", from_pk=from_id)

        edges.append(
            GraphRelationship(
                edge_type=entry.get("label", ""),
                start_node_labels=[from_label],
                end_node_labels=[to_label],
                properties=_properties(table, entry.get("attributes", [])),
                source=RelationshipSource(
                    type="table",
                    name=table,
                    location=f"database.schema.{table}",
                    mapping=rel_mapping,
                ),
            )
        )

    return GraphModel(nodes=nodes, edges=edges)


def _legacy_mapping_to_graph_model(mapping: dict[str, Any]) -> GraphModel:
    """Read the pre-MemGQL sql2graph mapping (nodes / id_column / rel_type)."""
    nodes = []
    for entry in mapping.get("nodes", []):
        table = entry.get("table", "")
        id_column = entry.get("id_column", "")
        label = entry.get("label", "")
        properties = [
            GraphProperty(key=key, source=PropertySource(field=f"{table}.{column}"))
            for key, column in entry.get("properties", {}).items()
        ]
        nodes.append(
            GraphNode(
                labels=[label],
                properties=properties,
                source=NodeSource(
                    type="table",
                    name=table,
                    location=f"database.schema.{table}",
                    mapping={"labels": [label], "id_field": id_column},
                ),
            )
        )

    node_keys = {n.primary_label: (n.source.name, n.source.mapping["id_field"]) for n in nodes if n.source}
    edges = []
    for entry in mapping.get("edges", []):
        table = entry.get("table", "")
        source_col, target_col = entry.get("source_column", ""), entry.get("target_column", "")
        from_table, from_id = node_keys.get(entry.get("source_label", ""), ("", ""))
        to_table, to_id = node_keys.get(entry.get("target_label", ""), ("", ""))
        rel_mapping: dict[str, Any] = {"edge_type": entry.get("rel_type", ""), "from_pk": from_id}
        if table == from_table and target_col == to_id and source_col != from_id:
            # FK edge: source_column is the FK in the source label's table.
            rel_mapping.update(start_node=f"{table}.{source_col}", end_node=f"{to_table}.{target_col}")
        else:
            rel_mapping.update(
                start_node=f"{table}.{source_col}",
                end_node=f"{table}.{target_col}",
                join_table=table,
            )
        properties = [
            GraphProperty(key=key, source=PropertySource(field=f"{table}.{column}"))
            for key, column in entry.get("properties", {}).items()
        ]
        edges.append(
            GraphRelationship(
                edge_type=entry.get("rel_type", ""),
                start_node_labels=[entry.get("source_label", "")],
                end_node_labels=[entry.get("target_label", "")],
                properties=properties,
                source=RelationshipSource(
                    type="table",
                    name=table,
                    location=f"database.schema.{table}",
                    mapping=rel_mapping,
                ),
            )
        )

    return GraphModel(nodes=nodes, edges=edges)


def print_mapping_summary(mapping: dict[str, Any], max_lines: int = 5) -> None:
    """Print a concise summary of a mapping (first *max_lines* of each section)."""
    vertices = mapping.get("vertices", [])
    edges = mapping.get("edges", [])
    print()
    print(f"  Vertices ({len(vertices)}):")
    for v in vertices[:max_lines]:
        source = v.get("mappedTableSource", {})
        print(f"    :{v['label']}  (table: {source.get('table')}, id: {source.get('metaFields', {}).get('id')})")
        attributes = ", ".join(a["name"] for a in v.get("attributes", []))
        if attributes:
            print(f"      attributes: {attributes}")
    if len(vertices) > max_lines:
        print(f"    ... and {len(vertices) - max_lines} more (use /edit to see all)")

    print(f"  Edges ({len(edges)}):")
    for e in edges[:max_lines]:
        source = e.get("mappedTableSource", {})
        meta = source.get("metaFields", {})
        print(
            f"    (:{e.get('from', '?')})-[:{e['label']}]->(:{e.get('to', '?')})  "
            f"(table: {source.get('table')}, {meta.get('from')} -> {meta.get('to')})"
        )
    if len(edges) > max_lines:
        print(f"    ... and {len(edges) - max_lines} more (use /edit to see all)")
    print()
