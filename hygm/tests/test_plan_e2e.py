"""The Memgraph plan applied to a real Memgraph: idempotent, and the data it rules out is refused.

Each test's labels carry a fresh suffix, because Memgraph can't drop an enum
yet ("Not yet implemented: Drop enum"): a rerun must not find the enums of
the last one. Nodes, constraints and indexes on those labels are removed afterwards.

Labels and property names below are interpolated into Cypher; they are
identifiers generated here or validated by hygm's types. Values always travel
as parameters.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from hygm import HygmModel, NodeType, PlanConflictError, Property, apply_plan, enum_for, labels_for, memgraph_plan


@dataclass(frozen=True)
class Shop:
    model: HygmModel
    party: str
    customer: str
    vip: str
    order: str


@pytest.fixture
def shop(memgraph_run) -> Iterator[Shop]:
    suffix = uuid.uuid4().hex[:8]
    party, customer, vip, order = (f"{name}{suffix}" for name in ("Party", "Customer", "VipCustomer", "Order"))
    model = HygmModel(
        node_types=(
            NodeType(party, properties=(Property("name", required=True),)),
            NodeType(
                customer,
                parents=(party,),
                key=("customer_id",),
                properties=(Property("customer_id", required=True), Property("tier", values=("standard", "gold"))),
            ),
            NodeType(vip, parents=(customer,)),
            NodeType(
                order,
                key=("shop", "number"),
                properties=(Property("shop", required=True), Property("number", type="integer", required=True)),
            ),
        )
    )
    yield Shop(model, party, customer, vip, order)
    _clean(memgraph_run, set(model.node_labels()))


def _clean(run, labels: set[str]) -> None:
    for label in labels:
        run(f"MATCH (n:`{label}`) DETACH DELETE n")
    for row in run("SHOW CONSTRAINT INFO"):
        if row["label"] not in labels:
            continue
        label, props = row["label"], row["properties"]
        if row["constraint type"] == "unique":
            run(f"DROP CONSTRAINT ON (n:`{label}`) ASSERT {', '.join(f'n.`{p}`' for p in props)} IS UNIQUE")
        elif row["constraint type"] == "exists":
            run(f"DROP CONSTRAINT ON (n:`{label}`) ASSERT EXISTS (n.`{props}`)")
        else:
            data_type = {"BOOL": "BOOLEAN"}.get(row["data_type"], row["data_type"].replace(" ", ""))
            run(f"DROP CONSTRAINT ON (n:`{label}`) ASSERT n.`{props}` IS TYPED {data_type}")
    for row in run("SHOW INDEX INFO"):
        if row["label"] in labels:
            run(f"DROP INDEX ON :`{row['label']}`({', '.join(f'`{p}`' for p in row['property'])})")


def _create(run, shop: Shop, label: str, properties: dict, tier: str | None = None) -> None:
    """Write a `label` node as a writer following the plan does: every ancestor label, enums through ToEnum."""
    labels = ":".join(f"`{name}`" for name in labels_for(shop.model, label))
    set_tier = ", n.tier = ToEnum($enum, $tier)" if tier else ""
    enum = enum_for(shop.model, label, "tier") if tier else None
    run(f"CREATE (n:{labels}) SET n += $properties{set_tier}", properties=properties, enum=enum, tier=tier)


def test_applying_the_plan_twice_runs_nothing_the_second_time(memgraph_run, shop):
    plan = memgraph_plan(shop.model)

    first = apply_plan(plan, memgraph_run)
    second = apply_plan(plan, memgraph_run)

    assert first == plan.statements
    assert second == ()
    assert {"Enum Name": f"{shop.customer}_tier", "Enum Values": ["standard", "gold"]} in memgraph_run("SHOW ENUMS")


def test_a_node_written_to_the_plan_is_accepted(memgraph_run, shop):
    apply_plan(memgraph_plan(shop.model), memgraph_run)

    _create(memgraph_run, shop, shop.vip, {"name": "Ada", "customer_id": "c1"}, tier="gold")

    rows = memgraph_run(f"MATCH (n:`{shop.party}`) RETURN labels(n) AS labels")
    assert sorted(rows[0]["labels"]) == sorted((shop.vip, shop.customer, shop.party))


def test_memgraph_refuses_a_duplicate_key_written_through_a_descendant(memgraph_run, shop):
    apply_plan(memgraph_plan(shop.model), memgraph_run)
    _create(memgraph_run, shop, shop.customer, {"name": "Ada", "customer_id": "c1"})

    with pytest.raises(Exception, match="unique constraint violation"):
        _create(memgraph_run, shop, shop.vip, {"name": "Ada", "customer_id": "c1"})


def test_memgraph_refuses_a_duplicate_composite_key_but_not_a_partial_match(memgraph_run, shop):
    apply_plan(memgraph_plan(shop.model), memgraph_run)
    _create(memgraph_run, shop, shop.order, {"shop": "north", "number": 1})
    _create(memgraph_run, shop, shop.order, {"shop": "south", "number": 1})

    with pytest.raises(Exception, match="unique constraint violation"):
        _create(memgraph_run, shop, shop.order, {"shop": "north", "number": 1})


def test_memgraph_refuses_a_node_missing_an_inherited_required_property(memgraph_run, shop):
    apply_plan(memgraph_plan(shop.model), memgraph_run)

    with pytest.raises(Exception, match="existence constraint violation"):
        _create(memgraph_run, shop, shop.customer, {"customer_id": "c2"})


def test_memgraph_refuses_a_wrongly_typed_value(memgraph_run, shop):
    apply_plan(memgraph_plan(shop.model), memgraph_run)

    with pytest.raises(Exception, match="IS TYPED INTEGER violation"):
        _create(memgraph_run, shop, shop.order, {"shop": "north", "number": "one"})
    with pytest.raises(Exception, match="IS TYPED ENUM violation"):
        _create(memgraph_run, shop, shop.customer, {"name": "Bo", "customer_id": "c4", "tier": "gold"})


def test_a_contradicting_type_constraint_stops_the_plan_before_it_runs_anything(memgraph_run, shop):
    memgraph_run(f"CREATE CONSTRAINT ON (n:`{shop.order}`) ASSERT n.number IS TYPED STRING")

    with pytest.raises(PlanConflictError, match="already typed STRING"):
        apply_plan(memgraph_plan(shop.model), memgraph_run)

    assert not [row for row in memgraph_run("SHOW CONSTRAINT INFO") if row["label"] == shop.party]
