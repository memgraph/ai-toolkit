"""Layer 6: operations, apply and diff."""

import json
import random
from dataclasses import replace
from typing import Any

import pytest

from hygm import Change, HygmModel, ModelError, NodeType, Property, RelationType, default_model, validate_model
from hygm.evolution import (
    OPERATIONS,
    AddNodeType,
    AddProperty,
    AddRelationType,
    MergeNodeTypes,
    MergeRelationTypes,
    OrderNodeTypes,
    OrderProperties,
    OrderRelationTypes,
    RemoveNodeType,
    RemoveProperty,
    RemoveRelationType,
    RenameNodeType,
    RenameProperty,
    RenameRelationType,
    SetDescription,
    SetEndpoints,
    SetIri,
    SetKey,
    SetParents,
    SetProfile,
    SetProperty,
    apply,
    diff,
    operation_from_mapping,
    operation_to_mapping,
)

LABELS = ("Person", "Org", "Place", "Tool", "Event", "Doc", "Team", "City")
RELATIONS = ("works_at", "lives_in", "uses", "owns", "attends", "wrote")
PROPERTIES = ("name", "age", "email", "since", "score", "code")
TYPES = ("string", "integer", "float", "boolean", "date", "json")


def _property(rng: random.Random, name: str) -> Property:
    kind = rng.choice(TYPES)
    return Property(
        name,
        type=kind,
        description=rng.choice(["", f"the {name}"]),
        required=rng.random() < 0.4,
        values=("a", "b") if kind == "string" and rng.random() < 0.2 else None,
        iri=rng.choice([None, f"http://example.org/{name}"]),
        profiles=rng.choice([{}, {"x_note": {"seen": rng.randint(0, 3)}}]),
    )


def _node(rng: random.Random, label: str, earlier: list[str]) -> NodeType:
    properties = tuple(_property(rng, name) for name in rng.sample(PROPERTIES, rng.randint(0, 3)))
    required = [p.name for p in properties if p.required]
    profiles: dict = {}
    if rng.random() < 0.3:
        profiles["learning"] = {"protected": rng.random() < 0.5}
    if rng.random() < 0.2:
        profiles["extraction"] = {"value": True}
    return NodeType(
        label,
        description=rng.choice(["", f"a {label.lower()}"]),
        identity=rng.choice(["global", "chunk", "span"]),
        parents=tuple(rng.sample(earlier, rng.randint(0, min(2, len(earlier))))) if rng.random() < 0.5 else (),
        properties=properties,
        key=(rng.choice(required),) if required and rng.random() < 0.5 else (),
        iri=rng.choice([None, f"http://example.org/{label}"]),
        profiles=profiles,
    )


def _node_of(model: HygmModel, label: str) -> NodeType:
    node = model.node_type(label)
    assert node is not None, label
    return node


def _relation(rng: random.Random, label: str, nodes: list[str]) -> RelationType:
    def side() -> tuple[str, ...]:
        return tuple(rng.sample(nodes, rng.randint(0, min(2, len(nodes)))))

    return RelationType(
        label,
        description=rng.choice(["", f"{label} fact"]),
        start_labels=side(),
        end_labels=side(),
        properties=tuple(_property(rng, name) for name in rng.sample(PROPERTIES, rng.randint(0, 2))),
        iri=rng.choice([None, f"http://example.org/{label}"]),
        profiles=rng.choice([{}, {"x_weight": {"w": 1}}]),
    )


def _model(rng: random.Random) -> HygmModel:
    """A random model that passes validate_model()."""
    while True:
        labels = rng.sample(LABELS, rng.randint(0, 6))
        nodes = [_node(rng, label, labels[:index]) for index, label in enumerate(labels)]
        relations = [_relation(rng, label, labels) for label in rng.sample(RELATIONS, rng.randint(0, 4))]
        profiles = {}
        if len(labels) >= 2 and rng.random() < 0.5:
            profiles["conversation"] = {"speaker": labels[0], "other_person": labels[1]}
        model = HygmModel(node_types=tuple(nodes), relation_types=tuple(relations), profiles=profiles)
        if validate_model(model).success:
            return model


def _perturbed(rng: random.Random, model: HygmModel) -> HygmModel:
    """`model` with a few random edits, each kept only if the model stays valid."""
    edits = [
        lambda m: replace(m, node_types=tuple(rng.sample(m.node_types, len(m.node_types)))),
        lambda m: replace(m, relation_types=tuple(rng.sample(m.relation_types, len(m.relation_types)))),
        lambda m: replace(m, node_types=m.node_types[1:]),
        lambda m: replace(m, relation_types=m.relation_types[:-1]),
        lambda m: replace(m, profiles={}),
    ]

    def edit_node(m: HygmModel) -> HygmModel:
        if not m.node_types:
            return m
        index = rng.randrange(len(m.node_types))
        node = m.node_types[index]
        node = rng.choice(
            [
                replace(node, description="changed"),
                replace(node, identity=rng.choice(["global", "chunk", "span"])),
                replace(node, properties=tuple(rng.sample(node.properties, len(node.properties)))),
                replace(node, properties=node.properties[1:], key=()),
                replace(node, properties=(*node.properties, _property(rng, "extra"))),
                replace(node, profiles={"learning": {"catch_all": True}}),
                replace(node, iri=None),
                replace(node, parents=()),
            ]
        )
        return replace(m, node_types=(*m.node_types[:index], node, *m.node_types[index + 1 :]))

    def edit_relation(m: HygmModel) -> HygmModel:
        if not m.relation_types:
            return m
        relation = rng.choice(m.relation_types)
        changed = rng.choice(
            [
                replace(relation, start_labels=(), end_labels=relation.start_labels),
                replace(relation, description="changed", profiles={}),
                replace(relation, properties=tuple(rng.sample(relation.properties, len(relation.properties)))),
            ]
        )
        return replace(m, relation_types=tuple(changed if r is relation else r for r in m.relation_types))

    edits += [edit_node, edit_node, edit_relation]
    for _ in range(rng.randint(1, 5)):
        candidate = rng.choice(edits)(model)
        if validate_model(candidate).success:
            model = candidate
    return model


def _pairs() -> list[tuple[HygmModel, HygmModel]]:
    rng = random.Random(704)
    pairs = [
        (HygmModel(node_types=()), default_model()),
        (default_model(), HygmModel(node_types=())),
        (default_model(), default_model()),
    ]
    for _ in range(150):
        a = _model(rng)
        pairs.append((a, _model(rng)))
        pairs.append((a, _perturbed(rng, a)))
    return pairs


@pytest.mark.parametrize("a, b", _pairs())
def test_applying_the_diff_of_two_models_gives_the_second(a, b):
    operations = diff(a, b)

    assert apply(a, operations) == b
    assert all(operation_from_mapping(json.loads(json.dumps(operation_to_mapping(op)))) == op for op in operations)


def test_a_model_diffed_against_itself_is_no_operations():
    model = _model(random.Random(1))
    assert diff(model, model) == ()


def test_a_renamed_type_diffs_as_a_removal_and_an_addition():
    a = HygmModel(node_types=(NodeType("Org"),))
    b = HygmModel(node_types=(NodeType("Company"),))
    assert diff(a, b) == (RemoveNodeType("Org"), AddNodeType(NodeType("Company")))


NAME = Property("name", required=True)
EVERY_OPERATION = (
    AddNodeType(NodeType("Org", "a company", identity="global", properties=(NAME,), key=("name",))),
    RemoveNodeType("Org"),
    RenameNodeType("Org", "Company"),
    MergeNodeTypes("Org", "Team"),
    AddRelationType(RelationType("works_at", "employment", ("Person",), ("Org",), properties=(Property("since"),))),
    RemoveRelationType("works_at"),
    RenameRelationType("works_at", "employed_by"),
    MergeRelationTypes("employed_by", "works_at"),
    AddProperty("node", "Org", Property("founded", "date", values=None)),
    RemoveProperty("relation", "works_at", "since"),
    RenameProperty("node", "Org", "name", "title"),
    SetProperty("node", "Org", Property("name", "string", "legal name", required=True, values=("x", "y"))),
    SetDescription("relation", "works_at", "employment"),
    SetIri("node", "Org", "http://example.org/Org"),
    SetIri("node", "Org", None),
    SetParents("Org", ("Group",)),
    SetKey("Org", ()),
    SetEndpoints("works_at", ("Person",), ()),
    SetProfile("conversation", {"speaker": "User", "other_person": "Person"}),
    SetProfile("extraction", {"identity": "span", "value": True}, "node", "Duration"),
    SetProfile("learning", None, "node", "Topic"),
    SetProfile("x_private", {"anything": [1, 2]}, "relation", "works_at", "since"),
    OrderNodeTypes(("Person", "Org")),
    OrderRelationTypes(("works_at",)),
    OrderProperties("node", "Org", ("name", "founded")),
)


def test_every_operation_survives_its_json_form():
    assert {type(op) for op in EVERY_OPERATION} == set(OPERATIONS.values())
    for op in EVERY_OPERATION:
        raw = json.loads(json.dumps(operation_to_mapping(op)))
        assert raw["op"] == op.op
        assert operation_from_mapping(raw) == op


def test_the_json_form_is_flat_and_embeds_model_file_entries():
    assert operation_to_mapping(RenameNodeType("Org", "Company")) == {
        "op": "rename_node_type",
        "label": "Org",
        "to": "Company",
    }
    assert operation_to_mapping(AddNodeType(NodeType("Org", "a company", identity="global"))) == {
        "op": "add_node_type",
        "node_type": {"label": "Org", "description": "a company", "profiles": {"extraction": {"identity": "global"}}},
    }
    assert operation_to_mapping(SetProfile("learning", None, "node", "Topic")) == {
        "op": "set_profile",
        "profile": "learning",
        "settings": None,
        "kind": "node",
        "label": "Topic",
    }


@pytest.mark.parametrize(
    "raw, code",
    [
        ("rename_node_type", "invalid_format"),
        ({"label": "Org"}, "invalid_format"),
        ({"op": "split_node_type", "label": "Org"}, "unknown_operation"),
        ({"op": "remove_node_type", "label": "Org", "reason": "x"}, "unknown_key"),
        ({"op": "rename_node_type", "label": "Org"}, "invalid_format"),
        ({"op": "remove_node_type", "label": 3}, "invalid_format"),
        ({"op": "set_parents", "label": "Org", "parents": "Group"}, "invalid_format"),
        ({"op": "remove_node_type", "label": "9lives"}, "invalid_identifier"),
        ({"op": "set_description", "kind": "edge", "label": "Org", "description": ""}, "invalid_format"),
        ({"op": "add_node_type", "node_type": {"label": "Org", "description": "", "colour": "red"}}, "unknown_key"),
        (
            {"op": "add_property", "kind": "node", "label": "Org", "property": {"name": "n", "type": "x"}},
            "unknown_property_type",
        ),
        ({"op": "set_profile", "profile": "extration", "settings": {}}, "unknown_profile"),
        (
            {
                "op": "set_profile",
                "profile": "learning",
                "settings": {"protected": "yes"},
                "kind": "node",
                "label": "Org",
            },
            "invalid_profile_value",
        ),
        (
            {"op": "set_profile", "profile": "learning", "settings": {"pinned": True}, "kind": "node", "label": "Org"},
            "unknown_key",
        ),
        ({"op": "set_profile", "profile": "learning", "settings": {}, "kind": "node"}, "invalid_format"),
        ({"op": "set_profile", "profile": "conversation", "settings": {}, "label": "Org"}, "invalid_format"),
    ],
)
def test_a_malformed_operation_is_refused_with_its_code(raw, code):
    with pytest.raises(ModelError) as refused:
        operation_from_mapping(raw)
    assert refused.value.code == code


BASE = HygmModel(
    node_types=(
        NodeType("User", properties=(NAME,), key=("name",)),
        NodeType("Person"),
        NodeType("Org", properties=(NAME, Property("size", "integer"))),
        NodeType("Team", parents=("Org",)),
    ),
    relation_types=(
        RelationType("works_at", start_labels=("User", "Person"), end_labels=("Org",)),
        RelationType("employed_by", start_labels=("Person",), end_labels=("Team",)),
        RelationType("knows"),
    ),
    profiles={"conversation": {"speaker": "User", "other_person": "Person"}},
)


@pytest.mark.parametrize(
    "op, code",
    [
        (RemoveNodeType("Planet"), "unknown_type"),
        (RenameNodeType("Planet", "World"), "unknown_type"),
        (RenameNodeType("Org", "Team"), "duplicate_label"),
        (RenameNodeType("Org", "Org"), "invalid_operation"),
        (MergeNodeTypes("Org", "Org"), "invalid_operation"),
        (MergeNodeTypes("Org", "Planet"), "unknown_type"),
        (AddNodeType(NodeType("Org")), "duplicate_label"),
        (AddRelationType(RelationType("knows")), "duplicate_label"),
        (RenameRelationType("knows", "works_at"), "duplicate_label"),
        (MergeRelationTypes("likes", "knows"), "unknown_type"),
        (AddProperty("node", "Org", Property("size")), "duplicate_property"),
        (RemoveProperty("node", "Org", "colour"), "undeclared_property"),
        (RenameProperty("node", "Org", "size", "name"), "duplicate_property"),
        (SetProperty("relation", "knows", Property("since")), "undeclared_property"),
        (SetDescription("relation", "likes", ""), "unknown_type"),
        (SetProfile("x_note", {}, "node", "Org", "colour"), "undeclared_property"),
        (OrderNodeTypes(("User", "Person", "Org")), "invalid_operation"),
        (OrderProperties("node", "Org", ("size", "size")), "invalid_operation"),
    ],
)
def test_an_operation_that_does_not_apply_is_refused_with_its_code(op, code):
    with pytest.raises(ModelError) as refused:
        apply(BASE, [op])
    assert refused.value.code == code
    assert f"({op.op})" in str(refused.value)


def test_the_result_is_validated_once_after_every_operation():
    removed = [RemoveNodeType("Person")]
    with pytest.raises(ModelError) as refused:
        apply(BASE, removed)
    assert "undeclared_endpoint" in refused.value.codes

    fixed = [
        *removed,
        SetEndpoints("works_at", ("User",), ("Org",)),
        SetEndpoints("employed_by", (), ("Team",)),
        SetProfile("conversation", None),
    ]
    assert apply(BASE, fixed).node_labels() == ("User", "Org", "Team")
    assert apply(BASE, removed, validate=False).node_labels() == ("User", "Org", "Team")


def test_applying_leaves_the_model_it_was_given_alone():
    before = BASE
    apply(BASE, [RenameNodeType("Org", "Company")])
    assert before == BASE and BASE.node_type("Org")


def test_renaming_a_node_type_carries_through_parents_endpoints_and_profiles():
    renamed = apply(BASE, [RenameNodeType("Org", "Company"), RenameNodeType("User", "Me")])

    assert renamed.node_labels() == ("Me", "Person", "Company", "Team")
    assert _node_of(renamed, "Team").parents == ("Company",)
    assert renamed.relation_types[0].start_labels == ("Me", "Person")
    assert renamed.relation_types[0].end_labels == ("Company",)
    assert renamed.profiles == {"conversation": {"speaker": "Me", "other_person": "Person"}}


def test_renaming_a_property_carries_through_keys_and_descendants():
    model = HygmModel(
        node_types=(
            NodeType("Org", properties=(NAME,)),
            NodeType("Team", parents=("Org",), properties=(Property("name", description="team name"),)),
            NodeType("Squad", parents=("Team",), key=("name",)),
        )
    )

    renamed = apply(model, [RenameProperty("node", "Org", "name", "title")])

    assert _node_of(renamed, "Org").properties == (replace(NAME, name="title"),)
    assert _node_of(renamed, "Team").properties[0].name == "title"
    assert _node_of(renamed, "Squad").key == ("title",)
    assert _node_of(apply(BASE, [RenameProperty("node", "User", "name", "handle")]), "User").key == ("handle",)


def test_merging_node_types_relabels_what_named_the_merged_one():
    merged = apply(BASE, [MergeNodeTypes("Team", "Org"), MergeNodeTypes("Person", "User")])

    assert merged.node_labels() == ("User", "Org")
    assert merged.relation_types[0].start_labels == ("User",)
    assert merged.relation_types[1].end_labels == ("Org",)
    assert merged.profiles == {"conversation": {"speaker": "User", "other_person": "User"}}


def test_merging_a_parent_into_its_child_drops_the_parent_rather_than_a_cycle():
    merged = apply(BASE, [MergeNodeTypes("Org", "Team")])
    assert _node_of(merged, "Team").parents == ()


def test_merging_relation_types_joins_their_endpoints():
    merged = apply(BASE, [MergeRelationTypes("employed_by", "works_at")])
    assert merged.relation_labels() == ("works_at", "knows")
    assert merged.relation_types[0].start_labels == ("User", "Person")
    assert merged.relation_types[0].end_labels == ("Org", "Team")

    into_any = apply(BASE, [MergeRelationTypes("works_at", "knows")])
    assert into_any.relation_types[-1].start_labels == ()


def test_a_node_types_extraction_profile_sets_its_identity():
    model = apply(BASE, [SetProfile("extraction", {"identity": "global", "value": True}, "node", "Org")])
    assert _node_of(model, "Org").identity == "global"
    assert _node_of(model, "Org").profiles == {"extraction": {"value": True}}

    cleared = apply(model, [SetProfile("extraction", None, "node", "Org")])
    assert (_node_of(cleared, "Org").identity, _node_of(cleared, "Org").profiles) == ("chunk", {})


def test_sql2graph_operations_have_a_counterpart():
    """Each of sql2graph's model-level operations, said as hygm operations, on a sql2graph-shaped model."""
    model = HygmModel(
        node_types=(NodeType("Customer", properties=(Property("id", "integer"),)), NodeType("Order")),
        relation_types=(RelationType("PLACED", start_labels=("Customer",), end_labels=("Order",)),),
    )
    steps = [
        RenameNodeType("Customer", "Client"),  # change_node_label
        RenameProperty("node", "Client", "id", "client_id"),  # rename_property
        AddProperty("node", "Order", Property("total")),  # add_property
        RemoveProperty("node", "Order", "total"),  # drop_property
        RenameRelationType("PLACED", "ORDERED"),  # change_relationship_name
        AddNodeType(NodeType("Product", properties=(Property("sku"),))),  # add_node
        AddRelationType(RelationType("CONTAINS", start_labels=("Order",), end_labels=("Product",))),  # add_relationship
        SetProperty("node", "Product", Property("sku", required=True)),  # add_constraint existence
        SetKey("Product", ("sku",)),  # add_constraint unique
        SetProperty("node", "Client", Property("client_id", "string")),  # add_constraint data_type
        SetKey("Product", ()),  # drop_constraint unique
        RemoveRelationType("CONTAINS"),  # drop_relationship; drop_node drops its relationships first
        RemoveNodeType("Product"),  # drop_node
    ]

    result = apply(model, steps)

    assert result.node_labels() == ("Client", "Order")
    assert result.relation_types == (RelationType("ORDERED", start_labels=("Client",), end_labels=("Order",)),)
    assert _node_of(result, "Client").properties == (Property("client_id", "string"),)


def test_a_stored_changelog_entry_reads_as_operations():
    """sessions-graph stores each Change as its fields; that dict reads back and says what it did."""
    model = default_model()
    pool = HygmModel(node_types=(NodeType("Library", "a code library"),))
    stored: list[dict[str, Any]] = [
        {"op": "add", "kind": "node", "label": "Topic", "into": None, "reason": ""},
        {"op": "restore", "kind": "node", "label": "Library", "into": None, "reason": ""},
        {"op": "merge", "kind": "node", "label": "Framework", "into": "Library", "reason": "synonyms"},
        {"op": "merge", "kind": "relation", "label": "built_with", "into": "uses", "reason": ""},
        {"op": "retire", "kind": "relation", "label": "uses", "into": None, "reason": "share 0.0001"},
        {"op": "retire", "kind": "node", "label": "Library", "into": None, "reason": "share 0.0001"},
        {"op": "veto", "kind": "node", "label": "User", "into": "Person", "reason": "User is protected"},
    ]

    operations = [Change(**entry).operations(model, pool) for entry in stored]

    assert operations == [
        (AddNodeType(_node_of(model, "Topic")),),
        (AddNodeType(pool.node_types[0]),),
        (MergeNodeTypes("Framework", "Library"),),
        (MergeRelationTypes("built_with", "uses"),),
        (RemoveRelationType("uses"),),
        (RemoveNodeType("Library"),),
        (),
    ]
    with pytest.raises(ModelError) as missing:
        Change("add", "node", "Planet").operations(model, pool)
    assert missing.value.code == "unknown_type"
