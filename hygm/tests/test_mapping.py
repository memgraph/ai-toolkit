"""Mappings built in code, checked and rendered (docs/contract.md, section 6)."""

from pathlib import Path

import pytest

from hygm import (
    ColumnEndpoint,
    GraphRef,
    HygmModel,
    Join,
    JoinEndpoint,
    ManualStrategy,
    Mapping,
    ModelError,
    NodeBinding,
    PropertyBinding,
    RelationBinding,
    Renderer,
    Source,
    TableRef,
    check_mapping,
    load_mapping,
    mapping_from_mapping,
    mapping_to_mapping,
)

SHOP = Path(__file__).parent.parent / "conformance" / "mapping" / "models" / "shop.model.yaml"


@pytest.fixture(scope="module")
def shop() -> HygmModel:
    return ManualStrategy().create_model(SHOP)


def _customers(*properties: PropertyBinding) -> NodeBinding:
    return NodeBinding(
        type="Customer",
        source="crm",
        table=TableRef("customers"),
        id=("customer_id",),
        properties=properties or (PropertyBinding("customer_id", "customer_id"),),
    )


def test_a_mapping_built_in_code_passes_and_writes_what_a_file_would_hold(shop):
    mapping = Mapping(
        model="shop.model.yaml",
        sources=(Source("crm", "table"),),
        nodes=(_customers(PropertyBinding("customer_id", "id"), PropertyBinding("loyalty_points", "attrs", "$.p")),),
    )

    check_mapping(mapping, shop)
    assert mapping_to_mapping(mapping) == {
        "version": 1,
        "model": "shop.model.yaml",
        "sources": [{"name": "crm", "kind": "table"}],
        "nodes": [
            {
                "type": "Customer",
                "source": "crm",
                "table": {"name": "customers"},
                "id": ["customer_id"],
                "properties": [
                    {"name": "customer_id", "column": "id"},
                    {"name": "loyalty_points", "column": "attrs", "path": "$.p"},
                ],
            }
        ],
    }


def test_a_join_given_table_columns_in_code_is_refused(shop):
    mapping = Mapping(
        model="shop.model.yaml",
        relations=(
            RelationBinding(
                type="placed",
                join=Join(JoinEndpoint("Customer", "customer_id"), JoinEndpoint("Order", "customer_ref")),
                from_=ColumnEndpoint("Customer", ("customer_id",)),
            ),
        ),
    )

    with pytest.raises(ModelError) as raised:
        check_mapping(mapping, shop)

    assert raised.value.codes == ("binding_kind",)


def test_every_problem_is_reported_at_once(shop):
    mapping = Mapping(
        model="shop.model.yaml",
        sources=(Source("crm", "table"),),
        nodes=(
            _customers(PropertyBinding("email", "email")),
            NodeBinding(type="Invoice", source="crm", table=TableRef("invoices"), id=("id",)),
            NodeBinding(type="Order", source="legacy", graph=GraphRef()),
        ),
    )

    with pytest.raises(ModelError) as raised:
        check_mapping(mapping, shop)

    assert set(raised.value.codes) == {"undeclared_property", "unknown_type", "unknown_source"}


def test_a_refusal_names_the_file_and_the_binding(tmp_path):
    (tmp_path / "shop.model.yaml").write_text(SHOP.read_text())
    path = tmp_path / "shop.mapping.yaml"
    path.write_text(
        "version: 1\nmodel: shop.model.yaml\nsources: [{name: crm, kind: table}]\n"
        "nodes: [{type: Customer, source: crm, table: {name: customers}, id: [id], "
        "properties: [{name: email, column: email}]}]\n"
    )

    with pytest.raises(ModelError, match=r"shop\.mapping\.yaml.*nodes\[0\].*'email'"):
        load_mapping(path)


def test_the_model_path_resolves_against_the_mapping_file_not_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "shop.model.yaml").write_text(SHOP.read_text())
    (tmp_path / "maps").mkdir()
    path = tmp_path / "maps" / "empty.yaml"
    path.write_text("version: 1\nmodel: ../models/shop.model.yaml\n")
    monkeypatch.chdir(tmp_path / "models")

    model, mapping = load_mapping(path)

    assert mapping.model == "../models/shop.model.yaml"
    assert "Customer" in model.node_labels()


class _TablesOnly:
    """A renderer whose target knows only table bindings, as the Renderer docs describe."""

    def render(self, model: HygmModel, mapping: Mapping) -> list[str]:
        unrenderable = [b.type for b in mapping.relations if b.kind != "table"]
        if unrenderable:
            raise ValueError(f"can't render non-table relation bindings: {unrenderable}")
        return [f"{b.type}<-{b.table.name}" for b in mapping.nodes if b.table is not None]


def test_a_renderer_refuses_what_its_target_cant_express(shop):
    renderer: Renderer = _TablesOnly()
    join_only = mapping_from_mapping(
        {
            "version": 1,
            "model": "shop.model.yaml",
            "relations": [
                {
                    "type": "placed",
                    "join": {
                        "from": {"type": "Customer", "property": "customer_id"},
                        "to": {"type": "Order", "property": "customer_ref"},
                    },
                }
            ],
        },
        "in a test",
        shop,
    )

    with pytest.raises(ValueError, match="non-table"):
        renderer.render(shop, join_only)
