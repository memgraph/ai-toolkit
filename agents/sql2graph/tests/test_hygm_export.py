"""Unit tests for the hygm model and mapping sql2graph writes beside its output."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hygm.models.graph_models import (
    GraphConstraint,
    GraphEnum,
    GraphIndex,
    GraphModel,
    GraphNode,
    GraphProperty,
    GraphRelationship,
)
from core.hygm.models.sources import (
    ConstraintSource,
    EnumSource,
    IndexSource,
    NodeSource,
    PropertySource,
    RelationshipSource,
)
from core.hygm.strategies.deterministic import DeterministicStrategy
from core.hygm_export import Gap, graph_model_to_hygm, write_hygm_files
from database.analyzer import DatabaseAnalyzer
from database.models import ColumnInfo, ForeignKeyInfo

from hygm import (
    ColumnEndpoint,
    Join,
    JoinEndpoint,
    NodeBinding,
    PropertyBinding,
    RelationBinding,
    TableRef,
    check_mapping,
    load_mapping,
    validate_model,
)


def _prop(table, column, *types):
    return GraphProperty(
        key=column,
        types=[{"type": t, "count": 1, "examples": []} for t in types] or [{"type": "String"}],
        source=PropertySource(field=f"{table}.{column}"),
    )


def _node(label, table, id_field, properties):
    return GraphNode(
        labels=[label],
        properties=properties,
        source=NodeSource(
            type="table",
            name=table,
            location=f"database.schema.{table}",
            mapping={"labels": [label], "id_field": id_field},
        ),
    )


def _edge(edge_type, start, end, source_name, mapping, source_type="table", **kwargs):
    return GraphRelationship(
        edge_type=edge_type,
        start_node_labels=[start],
        end_node_labels=[end],
        source=RelationshipSource(
            type=source_type,
            name=source_name,
            location=f"database.schema.{source_name}",
            mapping=mapping,
        ),
        **kwargs,
    )


def _unique(label, prop, table):
    return GraphConstraint(
        type="unique",
        labels=[label],
        properties=[prop],
        source=ConstraintSource(origin="source_database_constraint", migrated_from=f"database.schema.{table}"),
    )


def _index(label, prop):
    return GraphIndex(
        labels=[label],
        properties=[prop],
        source=IndexSource(origin="migration_requirement", reason="performance_optimization", created_by="test"),
    )


def _shop():
    """A small shop, shaped like the deterministic strategy's output: FK columns aren't node properties."""
    return GraphModel(
        nodes=[
            _node(
                "Customers",
                "customers",
                "customer_id",
                [
                    _prop("customers", "customer_id", "Integer"),
                    _prop("customers", "name", "String"),
                    _prop("customers", "phone", "String", "Null"),
                ],
            ),
            _node(
                "Orders",
                "orders",
                "order_id",
                [
                    _prop("orders", "order_id", "Integer"),
                    _prop("orders", "status", "String"),
                    _prop("orders", "placed_at", "LocalDateTime"),
                    _prop("orders", "total", "Double"),
                ],
            ),
            _node(
                "Products",
                "products",
                "product_id",
                [_prop("products", "product_id", "Integer"), _prop("products", "sku", "String")],
            ),
        ],
        edges=[
            _edge(
                "ORDERS_TO_CUSTOMERS",
                "Orders",
                "Customers",
                "orders",
                {
                    "start_node": "orders.customer_id",
                    "end_node": "customers.customer_id",
                    "edge_type": "ORDERS_TO_CUSTOMERS",
                    "from_pk": "order_id",
                },
            ),
            _edge(
                "ORDER_ITEMS",
                "Orders",
                "Products",
                "order_items",
                {
                    "start_node": "order_items.order_id",
                    "end_node": "order_items.product_id",
                    "edge_type": "ORDER_ITEMS",
                    "from_pk": "order_id",
                    "join_table": "order_items",
                },
                properties=[_prop("order_items", "quantity", "Integer")],
            ),
        ],
        node_indexes=[_index("Customers", "customer_id"), _index("Products", "sku")],
        node_constraints=[
            _unique("Customers", "customer_id", "customers"),
            _unique("Orders", "order_id", "orders"),
            _unique("Products", "product_id", "products"),
            _unique("Products", "sku", "products"),
            GraphConstraint(type="existence", labels=["Orders"], properties=["placed_at"]),
            GraphConstraint(type="data_type", labels=["Orders"], properties=["status"], data_type="order_status"),
        ],
        enums=[
            GraphEnum(
                name="order_status",
                values=["pending", "shipped", "delivered"],
                source=EnumSource(origin="source_database_enum", migrated_from="database.schema.orders.status"),
            )
        ],
    )


def _properties(model, label):
    node = model.node_type(label) or next(r for r in model.relation_types if r.label == label)
    return {p.name: p for p in node.properties}


def test_nodes_become_typed_node_types_keyed_by_their_id():
    model, _, _ = graph_model_to_hygm(_shop())

    assert model.node_labels() == ("Customers", "Orders", "Products")
    customers = _properties(model, "Customers")
    assert {name: (p.type, p.required) for name, p in customers.items()} == {
        "customer_id": ("integer", True),
        "name": ("string", False),
        "phone": ("string", False),
    }
    assert model.key_of("Customers") == ("customer_id",)
    orders = _properties(model, "Orders")
    assert orders["placed_at"].type == "datetime"
    assert orders["total"].type == "float"


def test_enum_becomes_values_and_existence_becomes_required():
    model, _, _ = graph_model_to_hygm(_shop())

    orders = _properties(model, "Orders")
    assert orders["status"].values == ("pending", "shipped", "delivered")
    assert orders["placed_at"].required is True


def test_node_sources_become_table_bindings():
    _, mapping, _ = graph_model_to_hygm(_shop(), source="shop")

    assert [(s.name, s.kind) for s in mapping.sources] == [("shop", "table")]
    assert mapping.nodes[0] == NodeBinding(
        type="Customers",
        source="shop",
        table=TableRef("customers"),
        id=("customer_id",),
        properties=(
            PropertyBinding("customer_id", "customer_id"),
            PropertyBinding("name", "name"),
            PropertyBinding("phone", "phone"),
        ),
    )


def test_foreign_key_and_join_table_become_table_relation_bindings():
    model, mapping, _ = graph_model_to_hygm(_shop(), source="shop")

    placed = next(r for r in model.relation_types if r.label == "ORDERS_TO_CUSTOMERS")
    assert (placed.start_labels, placed.end_labels) == (("Orders",), ("Customers",))
    bindings = {b.type: b for b in mapping.relations}
    assert bindings["ORDERS_TO_CUSTOMERS"] == RelationBinding(
        type="ORDERS_TO_CUSTOMERS",
        source="shop",
        table=TableRef("orders"),
        from_=ColumnEndpoint("Orders", ("order_id",)),
        to=ColumnEndpoint("Customers", ("customer_id",)),
    )
    assert bindings["ORDER_ITEMS"] == RelationBinding(
        type="ORDER_ITEMS",
        source="shop",
        table=TableRef("order_items"),
        from_=ColumnEndpoint("Orders", ("order_id",)),
        to=ColumnEndpoint("Products", ("product_id",)),
        properties=(PropertyBinding("quantity", "quantity"),),
    )
    assert _properties(model, "ORDER_ITEMS")["quantity"].type == "integer"


def test_shop_validates_and_reports_only_what_hygm_cannot_carry():
    model, mapping, gaps = graph_model_to_hygm(_shop())

    assert validate_model(model).success
    check_mapping(mapping, model)
    assert {(g.element, g.kind) for g in gaps} == {
        ("constraint unique on Products(sku)", "unique_not_key"),
        ("index on Products(sku)", "index"),
    }


def test_files_are_written_beside_the_output_and_load_back(tmp_path):
    output = tmp_path / "shop.json"
    files = write_hygm_files(_shop(), output)

    assert (files.model.name, files.mapping.name, files.gaps.name) == (
        "shop.model.yaml",
        "shop.mapping.yaml",
        "shop.hygm-gaps.yaml",
    )
    assert yaml.safe_load(files.mapping.read_text())["model"] == "shop.model.yaml"
    model, mapping = load_mapping(files.mapping)
    expected_model, expected_mapping, _ = graph_model_to_hygm(_shop(), model_path="shop.model.yaml")
    assert (model, mapping) == (expected_model, expected_mapping)
    assert len(yaml.safe_load(files.gaps.read_text())["gaps"]) == 2


def _llm_shaped():
    """What the LLM strategy and interactive edits can produce beyond the deterministic shape."""
    customer = _node(
        "Customer",
        "customers",
        "customers.id",
        [
            _prop("customers", "id", "Integer"),
            _prop("customers", "email", "String"),
            GraphProperty(key="full_name", source=PropertySource(field="customers.name", transformation="upper")),
        ],
    )
    customer.labels = ["Customer", "Person"]
    order = _node(
        "Order",
        "orders",
        "id",
        [_prop("orders", "id", "Integer"), _prop("orders", "customer_email", "String"), _prop("orders", "x", "Point")],
    )
    manual = GraphNode(
        labels=["Note"],
        properties=[GraphProperty(key="text")],
        source=NodeSource(type="manual", name="note", location="note", mapping={"labels": ["Note"]}),
    )
    return GraphModel(
        nodes=[customer, order, manual],
        edges=[
            # Reversed: the FK is on the end node's table.
            _edge(
                "PLACED",
                "Customer",
                "Order",
                "orders_customer_fk",
                {"start_node": "customers.id", "end_node": "orders.customer_id"},
                properties=[GraphProperty(key="since", source=PropertySource(field="llm.since"))],
                directionality="undirected",
            ),
            # Equality on bound, non-id properties: a join.
            _edge(
                "ORDERED_BY_EMAIL",
                "Order",
                "Customer",
                "orders",
                {"start_node": "orders.customer_email", "end_node": "customers.email"},
                source_type="derived",
            ),
            # The LLM fallback with no foreign key behind it.
            _edge(
                "SIMILAR_TO",
                "Customer",
                "Customer",
                "SIMILAR_TO",
                {"start_node": "customers.id", "end_node": "customers.id"},
            ),
            _edge("ABOUT", "Note", "Order", "ABOUT", {"start_node": "Note", "end_node": "Order"}, source_type="manual"),
            _edge(
                "SHIPS_TO", "Order", "Address", "orders", {"start_node": "orders.address_id", "end_node": "address.id"}
            ),
        ],
        enums=[GraphEnum(name="tier", values=["gold"])],
    )


def test_what_hygm_cannot_carry_is_listed_and_the_rest_validates():
    model, mapping, gaps = graph_model_to_hygm(_llm_shaped())

    assert validate_model(model).success
    check_mapping(mapping, model)
    assert {(g.element, g.kind) for g in gaps} == {
        ("node Customer", "extra_labels"),
        ("node Customer property full_name", "transformation"),
        ("node Order property x", "property_type"),
        ("node Note", "source_type"),
        ("relationship PLACED", "directionality"),
        ("relationship PLACED property since", "unsourced_property"),
        ("relationship SIMILAR_TO", "no_backing_rows"),
        ("relationship ABOUT", "source_type"),
        ("relationship SHIPS_TO", "undeclared_endpoint"),
        ("enum tier", "unlinked_enum"),
    }
    assert all(isinstance(g, Gap) and g.detail for g in gaps)

    bindings = {b.type: b for b in mapping.relations}
    assert bindings["PLACED"].table == TableRef("orders")
    assert (bindings["PLACED"].from_, bindings["PLACED"].to) == (
        ColumnEndpoint("Customer", ("customer_id",)),
        ColumnEndpoint("Order", ("id",)),
    )
    assert bindings["ORDERED_BY_EMAIL"].join == Join(
        JoinEndpoint("Order", "customer_email"), JoinEndpoint("Customer", "email")
    )
    assert "SIMILAR_TO" in model.relation_labels() and "SIMILAR_TO" not in bindings
    assert "SHIPS_TO" not in model.relation_labels()
    assert _properties(model, "Customer")["full_name"].type == "string"
    assert model.node_type("Note") is not None and all(b.type != "Note" for b in mapping.nodes)


def test_foreign_key_onto_a_non_id_column_becomes_a_join_only_when_both_columns_are_bound():
    graph_model = GraphModel(
        nodes=[
            _node("Customer", "customers", "id", [_prop("customers", "id"), _prop("customers", "email")]),
            _node("Order", "orders", "id", [_prop("orders", "id"), _prop("orders", "customer_email")]),
            _node("Invoice", "invoices", "id", [_prop("invoices", "id")]),
        ],
        edges=[
            _edge(
                "OF",
                "Order",
                "Customer",
                "orders",
                {"start_node": "orders.customer_email", "end_node": "customers.email"},
            ),
            _edge(
                "BILLS",
                "Invoice",
                "Customer",
                "invoices",
                {"start_node": "invoices.email", "end_node": "customers.email"},
            ),
        ],
    )
    model, mapping, gaps = graph_model_to_hygm(graph_model)

    check_mapping(mapping, model)
    assert [b.join for b in mapping.relations] == [
        Join(JoinEndpoint("Order", "customer_email"), JoinEndpoint("Customer", "email"))
    ]
    assert [(g.element, g.kind) for g in gaps] == [("relationship BILLS", "fk_target_not_id")]


def test_one_relationship_type_from_several_foreign_keys_has_a_binding_for_each():
    graph_model = GraphModel(
        nodes=[
            _node("Film", "film", "film_id", [_prop("film", "film_id")]),
            _node("Language", "language", "language_id", [_prop("language", "language_id")]),
        ],
        edges=[
            _edge(
                "FILM_TO_LANGUAGE",
                "Film",
                "Language",
                "film",
                {"start_node": f"film.{fk}", "end_node": "language.language_id"},
            )
            for fk in ("language_id", "original_language_id")
        ],
    )
    model, mapping, gaps = graph_model_to_hygm(graph_model)

    assert model.relation_labels() == ("FILM_TO_LANGUAGE",)
    assert [b.to.columns for b in mapping.relations if b.to] == [("language_id",), ("original_language_id",)]
    assert gaps == ()


def test_one_relationship_type_between_several_pairs_admits_each_and_says_what_it_widens():
    graph_model = GraphModel(
        nodes=[
            _node("Customer", "customers", "id", [_prop("customers", "id")]),
            _node("Order", "orders", "id", [_prop("orders", "id")]),
        ],
        edges=[
            _edge(
                "LINKED",
                "Order",
                "Customer",
                "orders",
                {"start_node": "orders.customer_id", "end_node": "customers.id"},
            ),
            _edge(
                "LINKED",
                "Customer",
                "Order",
                "orders",
                {"start_node": "customers.id", "end_node": "orders.customer_id"},
            ),
        ],
    )
    model, mapping, gaps = graph_model_to_hygm(graph_model)

    linked = model.relation_types[0]
    assert (linked.start_labels, linked.end_labels) == (("Order", "Customer"), ("Customer", "Order"))
    assert [(b.from_.type, b.to.type) for b in mapping.relations if b.from_ and b.to] == [
        ("Order", "Customer"),
        ("Customer", "Order"),
    ]
    assert [(g.element, g.kind) for g in gaps] == [("relationship LINKED", "endpoint_pairs")]


# A sample schema run end to end: sakila, the MySQL sample sql2graph's README
# and defaults point at, through sql2graph's own analyzer and deterministic
# strategy. Tables, columns and foreign keys are sakila's (views left out):
# table -> (columns, primary key columns, foreign keys as (column, table, column)).
_SAKILA: dict[str, tuple[str, tuple[str, ...], tuple[tuple[str, str, str], ...]]] = {
    "actor": ("actor_id first_name last_name last_update", ("actor_id",), ()),
    "address": (
        "address_id address address2 district city_id postal_code phone location last_update",
        ("address_id",),
        (("city_id", "city", "city_id"),),
    ),
    "category": ("category_id name last_update", ("category_id",), ()),
    "city": ("city_id city country_id last_update", ("city_id",), (("country_id", "country", "country_id"),)),
    "country": ("country_id country last_update", ("country_id",), ()),
    "customer": (
        "customer_id store_id first_name last_name email address_id active create_date last_update",
        ("customer_id",),
        (("store_id", "store", "store_id"), ("address_id", "address", "address_id")),
    ),
    "film": (
        "film_id title description release_year language_id original_language_id rental_duration "
        "rental_rate length replacement_cost rating special_features last_update",
        ("film_id",),
        (("language_id", "language", "language_id"), ("original_language_id", "language", "language_id")),
    ),
    "film_actor": (
        "actor_id film_id last_update",
        ("actor_id", "film_id"),
        (("actor_id", "actor", "actor_id"), ("film_id", "film", "film_id")),
    ),
    "film_category": (
        "film_id category_id last_update",
        ("film_id", "category_id"),
        (("film_id", "film", "film_id"), ("category_id", "category", "category_id")),
    ),
    "film_text": ("film_id title description", ("film_id",), ()),
    "inventory": (
        "inventory_id film_id store_id last_update",
        ("inventory_id",),
        (("film_id", "film", "film_id"), ("store_id", "store", "store_id")),
    ),
    "language": ("language_id name last_update", ("language_id",), ()),
    "payment": (
        "payment_id customer_id staff_id rental_id amount payment_date last_update",
        ("payment_id",),
        (
            ("customer_id", "customer", "customer_id"),
            ("staff_id", "staff", "staff_id"),
            ("rental_id", "rental", "rental_id"),
        ),
    ),
    "rental": (
        "rental_id rental_date inventory_id customer_id return_date staff_id last_update",
        ("rental_id",),
        (
            ("inventory_id", "inventory", "inventory_id"),
            ("customer_id", "customer", "customer_id"),
            ("staff_id", "staff", "staff_id"),
        ),
    ),
    "staff": (
        "staff_id first_name last_name address_id picture email store_id active username password last_update",
        ("staff_id",),
        (("address_id", "address", "address_id"), ("store_id", "store", "store_id")),
    ),
    "store": (
        "store_id manager_staff_id address_id last_update",
        ("store_id",),
        (("manager_staff_id", "staff", "staff_id"), ("address_id", "address", "address_id")),
    ),
}


class _SakilaAnalyzer(DatabaseAnalyzer):
    """sql2graph's analyzer over sakila's catalog, held in memory instead of read from MySQL."""

    def _get_database_type(self):
        return "mysql"

    def connect(self):
        return True

    def disconnect(self):
        pass

    def get_tables(self):
        return list(_SAKILA)

    def get_table_schema(self, table_name):
        columns, primary_key, foreign_keys = _SAKILA[table_name]
        fk_columns = {column for column, _, _ in foreign_keys}
        return [
            ColumnInfo(
                name,
                "varchar",
                is_nullable=False,
                is_primary_key=name in primary_key,
                is_foreign_key=name in fk_columns and name not in primary_key,
            )
            for name in columns.split()
        ]

    def get_foreign_keys(self, table_name):
        return [ForeignKeyInfo(column, table, referenced) for column, table, referenced in _SAKILA[table_name][2]]

    def get_table_data(self, table_name, limit=None):
        return []

    def get_table_row_count(self, table_name):
        return 0

    def is_view(self, table_name):
        return False

    def get_indexes(self, table_name):
        return []


def test_sakila_through_the_deterministic_strategy_validates(tmp_path):
    structure = _SakilaAnalyzer({"database": "sakila"}).get_database_structure().to_hygm_format()
    graph_model = DeterministicStrategy().create_model(structure)

    model, mapping, gaps = graph_model_to_hygm(graph_model, source="sakila")

    assert validate_model(model).success
    check_mapping(mapping, model)
    assert set(model.node_labels()) == {t.title() for t in structure["entity_tables"]}
    assert len(mapping.nodes) == len(graph_model.nodes)
    assert all(model.key_of(label) for label in model.node_labels())
    # sql2graph's join-table heuristic makes store and inventory relationships,
    # so the foreign keys onto them end at labels no node has: sql2graph's own
    # model is inconsistent there, and the export says so rather than guessing.
    dropped = {g.element for g in gaps if g.kind == "undeclared_endpoint"}
    assert dropped == {
        "relationship CUSTOMER_TO_STORE",
        "relationship STAFF_TO_STORE",
        "relationship INVENTORY",
        "relationship RENTAL_TO_INVENTORY",
    }
    assert {g.kind for g in gaps} <= {"undeclared_endpoint", "index"}

    files = write_hygm_files(graph_model, tmp_path / "sakila.json", source="sakila")
    assert (
        load_mapping(files.mapping)
        == (graph_model_to_hygm(graph_model, source="sakila", model_path=files.model.name)[:2])
    )
