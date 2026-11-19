"""The Memgraph plan's shape, from a model alone. tests/test_plan_e2e.py applies it to a real Memgraph."""

import pytest

from hygm import (
    HygmModel,
    MemgraphPlan,
    ModelError,
    NodeType,
    Property,
    RelationType,
    Statement,
    enum_for,
    labels_for,
    memgraph_plan,
)


def shop() -> HygmModel:
    return HygmModel(
        node_types=(
            NodeType("Party", properties=(Property("name", required=True),)),
            NodeType(
                "Customer",
                parents=("Party",),
                key=("customer_id",),
                properties=(
                    Property("customer_id", required=True),
                    Property("tier", values=("standard", "gold")),
                    Property("since", type="datetime"),
                ),
            ),
            NodeType("VipCustomer", parents=("Customer",)),
            NodeType(
                "Order",
                key=("shop", "number"),
                properties=(
                    Property("shop", required=True),
                    Property("number", type="integer", required=True),
                    Property("total", type="decimal"),
                    Property("lines", type="json"),
                    Property("placed_on", type="date"),
                    Property("paid", type="boolean"),
                ),
            ),
        ),
        relation_types=(
            RelationType(
                "placed",
                start_labels=("Customer",),
                end_labels=("Order",),
                properties=(Property("at", type="datetime", required=True),),
            ),
        ),
    )


def test_a_shop_model_plans_each_constraint_once_in_declaration_order():
    assert memgraph_plan(shop()).statements == (
        Statement("exists", "Party", ("name",)),
        Statement("typed", "Party", ("name",), data_type="STRING"),
        Statement("unique", "Customer", ("customer_id",)),
        Statement("index", "Customer", ("customer_id",)),
        Statement("exists", "Customer", ("customer_id",)),
        Statement("typed", "Customer", ("customer_id",), data_type="STRING"),
        Statement("enum", "Customer_tier", values=("standard", "gold")),
        Statement("typed", "Customer", ("tier",), data_type="ENUM"),
        Statement("unique", "Order", ("shop", "number")),
        Statement("index", "Order", ("shop", "number")),
        Statement("exists", "Order", ("shop",)),
        Statement("exists", "Order", ("number",)),
        Statement("typed", "Order", ("shop",), data_type="STRING"),
        Statement("typed", "Order", ("number",), data_type="INTEGER"),
        Statement("typed", "Order", ("total",), data_type="FLOAT"),
        Statement("typed", "Order", ("lines",), data_type="MAP"),
        Statement("typed", "Order", ("placed_on",), data_type="DATE"),
        Statement("typed", "Order", ("paid",), data_type="BOOLEAN"),
    )


def test_an_inherited_key_is_constrained_only_on_the_declaring_label():
    plan = memgraph_plan(shop())

    assert [s.target for s in plan.statements if s.kind in {"unique", "index"}] == [
        "Customer",
        "Customer",
        "Order",
        "Order",
    ]
    assert not [s for s in plan.statements if s.target == "VipCustomer"]


def test_a_redeclared_property_is_constrained_on_the_descendant_only_where_it_tightens():
    model = HygmModel(
        node_types=(
            NodeType("Party", properties=(Property("email"),)),
            NodeType("Customer", parents=("Party",), properties=(Property("email", required=True),)),
        )
    )

    assert memgraph_plan(model).statements == (
        Statement("typed", "Party", ("email",), data_type="STRING"),
        Statement("exists", "Customer", ("email",)),
    )


def test_a_datetime_property_gets_no_type_constraint_and_relations_add_nothing():
    statements = memgraph_plan(shop()).statements

    assert not [s for s in statements if "since" in s.properties or "at" in s.properties]


def test_the_statements_render_as_memgraph_ddl_with_quoted_names():
    plan = memgraph_plan(shop())

    assert plan.cypher()[2] == "CREATE CONSTRAINT ON (n:`Customer`) ASSERT n.`customer_id` IS UNIQUE"
    assert plan.cypher()[6] == "CREATE ENUM `Customer_tier` VALUES { `standard`, `gold` }"
    assert plan.cypher()[8] == "CREATE CONSTRAINT ON (n:`Order`) ASSERT n.`shop`, n.`number` IS UNIQUE"
    assert plan.cypher()[9] == "CREATE INDEX ON :`Order`(`shop`, `number`)"


def test_an_enum_value_is_quoted_with_backticks_doubled():
    model = HygmModel(
        node_types=(NodeType("Ticket", properties=(Property("state", values=("in `progress`", "done")),)),)
    )

    assert memgraph_plan(model).cypher()[0] == "CREATE ENUM `Ticket_state` VALUES { `in ``progress```, `done` }"


def test_the_plan_is_deterministic():
    assert memgraph_plan(shop()) == memgraph_plan(shop())
    assert isinstance(memgraph_plan(shop()), MemgraphPlan)


def test_labels_for_are_the_type_and_its_ancestors_nearest_first():
    model = shop()

    assert labels_for(model, "VipCustomer") == ("VipCustomer", "Customer", "Party")
    assert labels_for(model, "Order") == ("Order",)
    with pytest.raises(ValueError, match="not a node type"):
        labels_for(model, "Supplier")


def test_enum_for_names_the_enum_of_the_declaring_type():
    model = shop()

    assert enum_for(model, "VipCustomer", "tier") == "Customer_tier"
    assert enum_for(model, "Customer", "customer_id") is None


def test_an_invalid_model_has_no_plan():
    model = HygmModel(node_types=(NodeType("Customer", key=("customer_id",)),))

    with pytest.raises(ModelError):
        memgraph_plan(model)


def test_enum_names_that_collide_are_refused():
    model = HygmModel(
        node_types=(
            NodeType("A_b", properties=(Property("c", values=("x",)),)),
            NodeType("A", properties=(Property("b_c", values=("y",)),)),
        )
    )

    with pytest.raises(ValueError, match="Enum names"):
        memgraph_plan(model)
