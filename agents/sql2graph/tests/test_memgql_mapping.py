"""Unit tests for the MemGQL mapping format written by ``--mapping``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hygm.models.graph_models import (
    GraphModel,
    GraphNode,
    GraphProperty,
    GraphRelationship,
)
from core.hygm.models.sources import (
    NodeSource,
    PropertySource,
    RelationshipSource,
)
from core.memgql_mapping import (
    graph_model_to_mapping,
    mapping_connector,
    mapping_to_graph_model,
)


def _node(label, table, id_field, columns):
    return GraphNode(
        labels=[label],
        properties=[GraphProperty(key=c, source=PropertySource(field=f"{table}.{c}")) for c in columns],
        source=NodeSource(
            type="table",
            name=table,
            location=f"database.schema.{table}",
            mapping={"labels": [label], "id_field": id_field},
        ),
    )


def _edge(edge_type, start, end, source_name, mapping):
    return GraphRelationship(
        edge_type=edge_type,
        start_node_labels=[start],
        end_node_labels=[end],
        source=RelationshipSource(
            type="table",
            name=source_name,
            location=f"database.schema.{source_name}",
            mapping=mapping,
        ),
    )


def _model():
    return GraphModel(
        nodes=[
            _node("Customer", "customers", "id", ["id", "name"]),
            _node("Order", "orders", "id", ["id", "customer_id", "total"]),
            _node("Tag", "tags", "id", ["id", "name"]),
        ],
        edges=[
            # FK on the start side (deterministic strategy shape).
            _edge(
                "PLACED_BY",
                "Order",
                "Customer",
                "orders",
                {"start_node": "orders.customer_id", "end_node": "customers.id", "from_pk": "id"},
            ),
            # FK on the end side (LLM strategy, reversed relationship).
            _edge(
                "PLACED",
                "Customer",
                "Order",
                "orders_customer_fk",
                {"start_node": "customers.id", "end_node": "orders.customer_id", "from_pk": "id"},
            ),
            # Many-to-many through a join table.
            _edge(
                "ORDER_TAGS",
                "Order",
                "Tag",
                "order_tags",
                {
                    "start_node": "order_tags.order_id",
                    "end_node": "order_tags.tag_id",
                    "join_table": "order_tags",
                    "from_pk": "id",
                },
            ),
            # LLM fallback with no FK behind it: not mappable.
            _edge(
                "SIMILAR_TO",
                "Customer",
                "Customer",
                "SIMILAR_TO",
                {"start_node": "customers.id", "end_node": "customers.id", "from_pk": "id"},
            ),
        ],
    )


def test_vertices_are_memgql_table_sources():
    mapping = graph_model_to_mapping(_model())
    order = next(v for v in mapping["vertices"] if v["label"] == "Order")
    assert order["mappedTableSource"] == {"table": "orders", "metaFields": {"id": "id"}}
    assert order["attributes"] == [{"name": "id"}, {"name": "customer_id"}, {"name": "total"}]


def test_edge_shapes():
    edges = {e["label"]: e for e in graph_model_to_mapping(_model())["edges"]}
    assert edges["PLACED_BY"]["mappedTableSource"] == {
        "table": "orders",
        "metaFields": {"id": "id", "from": "id", "to": "customer_id"},
    }
    assert edges["PLACED"]["mappedTableSource"] == {
        "table": "orders",
        "metaFields": {"id": "id", "from": "customer_id", "to": "id"},
    }
    assert edges["ORDER_TAGS"]["mappedTableSource"] == {
        "table": "order_tags",
        "metaFields": {"id": ["order_id", "tag_id"], "from": "order_id", "to": "tag_id"},
    }
    assert "SIMILAR_TO" not in edges


def test_connector_is_stamped():
    mapping = graph_model_to_mapping(_model(), connector="shop")
    assert all(e["mappedTableSource"]["connector"] == "shop" for e in mapping["vertices"] + mapping["edges"])
    assert mapping_connector(mapping) == "shop"
    assert mapping_connector(graph_model_to_mapping(_model())) is None


def test_round_trip():
    mapping = graph_model_to_mapping(_model(), connector="shop")
    assert graph_model_to_mapping(mapping_to_graph_model(mapping), connector="shop") == mapping


def test_legacy_mapping_converts():
    legacy = {
        "nodes": [
            {"label": "Customer", "table": "customers", "id_column": "id", "properties": {"name": "name"}},
            {"label": "Order", "table": "orders", "id_column": "id", "properties": {}},
            {"label": "Tag", "table": "tags", "id_column": "id", "properties": {}},
        ],
        "edges": [
            {
                "rel_type": "PLACED_BY",
                "table": "orders",
                "source_column": "customer_id",
                "target_column": "id",
                "source_label": "Order",
                "target_label": "Customer",
            },
            {
                "rel_type": "ORDER_TAGS",
                "table": "order_tags",
                "source_column": "order_id",
                "target_column": "tag_id",
                "source_label": "Order",
                "target_label": "Tag",
            },
        ],
    }
    mapping = graph_model_to_mapping(mapping_to_graph_model(legacy))
    assert mapping["vertices"][0] == {
        "label": "Customer",
        "mappedTableSource": {"table": "customers", "metaFields": {"id": "id"}},
        "attributes": [{"name": "name"}],
    }
    edges = {e["label"]: e["mappedTableSource"] for e in mapping["edges"]}
    assert edges["PLACED_BY"] == {"table": "orders", "metaFields": {"id": "id", "from": "id", "to": "customer_id"}}
    assert edges["ORDER_TAGS"] == {
        "table": "order_tags",
        "metaFields": {"id": ["order_id", "tag_id"], "from": "order_id", "to": "tag_id"},
    }
