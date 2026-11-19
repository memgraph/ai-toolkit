import pytest

from hygm import HygmModel, ModelError, NodeType, Property, RelationType


def _shop() -> HygmModel:
    return HygmModel(
        node_types=(
            NodeType("Party", properties=(Property("email", description="any address"),)),
            NodeType(
                "Customer",
                parents=("Party",),
                key=("email",),
                properties=(Property("email", description="billing address", required=True), Property("tier")),
            ),
            NodeType("VipCustomer", parents=("Customer",)),
            NodeType("Supplier", parents=("Party",)),
            NodeType("Order"),
        ),
        relation_types=(
            RelationType("placed", start_labels=("Customer",), end_labels=("Order",)),
            RelationType("contacted", start_labels=("Party",)),
        ),
    )


def test_ancestors_are_nearest_first_and_descendants_in_declaration_order():
    model = _shop()

    assert model.ancestors("VipCustomer") == ("Customer", "Party")
    assert model.descendants("Party") == ("Customer", "VipCustomer", "Supplier")
    assert model.is_a("VipCustomer", "Party")
    assert model.is_a("Order", "Order")
    assert not model.is_a("Party", "Customer")


def test_properties_are_inherited_with_the_nearest_description_and_any_required():
    email, tier = _shop().properties_of("VipCustomer")

    assert (email.name, email.description, email.required) == ("email", "billing address", True)
    assert tier.name == "tier"


def test_the_key_is_inherited():
    model = _shop()

    assert model.key_of("VipCustomer") == ("email",)
    assert model.key_of("Supplier") == ()


def test_an_endpoint_admits_the_named_types_descendants():
    model = _shop()
    placed, contacted = model.relation_types

    assert model.endpoint_labels(placed, "start") == ("Customer", "VipCustomer")
    assert model.endpoint_labels(contacted, "start") == ("Party", "Customer", "VipCustomer", "Supplier")
    assert model.endpoint_labels(contacted, "end") == model.node_labels()


def test_without_parents_endpoints_are_the_raw_field():
    model = HygmModel(
        node_types=(NodeType("A"), NodeType("B")), relation_types=(RelationType("r", start_labels=("B", "A")),)
    )

    assert model.endpoint_labels(model.relation_types[0], "start") == ("B", "A")


def test_a_parent_cycle_does_not_loop_the_queries():
    model = HygmModel(node_types=(NodeType("A", parents=("B",)), NodeType("B", parents=("A",))))

    assert model.ancestors("A") == ("B",)
    assert model.descendants("A") == ("B",)


def test_which_features_a_model_uses():
    assert _shop().uses_hierarchy_or_properties() == ("parents", "properties", "key")
    assert HygmModel(node_types=(NodeType("A"),)).uses_hierarchy_or_properties() == ()


def test_property_types_and_values_are_checked_on_construction():
    with pytest.raises(ModelError) as raised:
        Property("p", type="money")  # ty: ignore[invalid-argument-type]
    assert raised.value.code == "unknown_property_type"
    with pytest.raises(ModelError) as raised:
        Property("p", type="integer", values=("1",))
    assert raised.value.code == "values_not_string"
