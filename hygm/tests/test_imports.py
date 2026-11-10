"""The ontology block and imports (docs/contract.md, 4.11 and 4.12)."""

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from hygm import (
    AddNodeType,
    Conventions,
    HygmModel,
    ManualStrategy,
    ModelError,
    NodeType,
    Ontology,
    apply,
    diff,
    model_from_mapping,
    model_to_mapping,
    needs_version_2,
)

VALID = Path(__file__).parent.parent / "conformance" / "model" / "valid"


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def test_an_importing_model_uses_imported_types_as_parents_and_endpoints():
    model = ManualStrategy().create_model(VALID / "v2-imports.yaml")

    assert model.node_labels() == ("Food", "Seafood", "Drink", "Wine")
    assert model.ancestors("Wine") == ("Drink",)
    assert [p.name for p in model.properties_of("Wine")] == ["name"]
    pairs_with = next(r for r in model.relation_types if r.label == "pairs_with")
    assert "Wine" in model.endpoint_labels(pairs_with, "start")
    assert model.imported_labels() == (frozenset({"Food", "Seafood", "Drink"}), frozenset({"pairs_with"}))


def test_written_back_an_importing_model_keeps_its_imports_and_inlines_nothing():
    model = ManualStrategy().create_model(VALID / "v2-imports.yaml")

    written = model_to_mapping(model)

    assert written == {
        "version": 2,
        "imports": [{"path": "../imports/drink.model.yaml"}],
        "node_types": [{"label": "Wine", "description": "A drink made from grapes", "parents": ["Drink"]}],
        "relation_types": [
            {
                "label": "served_with",
                "description": "The wine is served with the seafood",
                "start_labels": ["Wine"],
                "end_labels": ["Seafood"],
            }
        ],
    }


def test_imports_resolve_relative_to_the_importing_file(tmp_path):
    _write(tmp_path, "shared/base.yaml", "version: 2\nnode_types: [{label: Base, description: b}]\n")
    _write(tmp_path, "shared/mid.yaml", "version: 2\nimports: [{path: base.yaml}]\nnode_types: []\n")
    top = _write(
        tmp_path,
        "models/top.yaml",
        "version: 2\nimports: [{path: ../shared/mid.yaml}]\nnode_types: [{label: Top, description: t, parents: [Base]}]\n",
    )

    assert ManualStrategy().create_model(top).node_labels() == ("Base", "Top")


def test_a_diamond_of_imports_declares_each_type_once():
    model = ManualStrategy().create_model(VALID / "v2-imports-diamond.yaml")

    assert model.node_labels().count("Food") == 1


def test_given_as_data_a_model_with_imports_is_refused_without_a_resolver():
    raw = {"version": 2, "imports": [{"path": "food.model.yaml"}], "node_types": []}

    with pytest.raises(ModelError) as raised:
        model_from_mapping(raw, "data")

    assert raised.value.codes == ("unknown_model",)


def test_a_caller_supplied_resolver_resolves_imports():
    food = HygmModel(node_types=(NodeType("Food", "f"),))
    raw = {
        "version": 2,
        "imports": [{"path": "food"}],
        "node_types": [{"label": "Fish", "description": "", "parents": ["Food"]}],
    }

    model = model_from_mapping(raw, "data", resolver={"food": food}.__getitem__)

    assert model.node_labels() == ("Food", "Fish")
    assert model.imports[0].model == food


def test_an_invalid_imported_model_is_refused_with_its_own_code(tmp_path):
    _write(tmp_path, "bad.yaml", "version: 2\nnode_types: [{label: A, description: a, parent: [B]}]\n")
    top = _write(tmp_path, "top.yaml", "version: 2\nimports: [{path: bad.yaml}]\nnode_types: []\n")

    with pytest.raises(ModelError) as raised:
        ManualStrategy().create_model(top)

    assert raised.value.codes == ("unknown_key",)


def test_a_cycle_is_named_with_the_files_in_it():
    with pytest.raises(ModelError, match=r"cycle-a\.model\.yaml -> .*cycle-b\.model\.yaml") as raised:
        ManualStrategy().create_model(VALID.parent / "invalid" / "import_cycle" / "two-files.yaml")

    assert raised.value.codes == ("import_cycle",)


def test_the_writer_refuses_a_model_that_changed_an_imported_type():
    model = ManualStrategy().create_model(VALID / "v2-imports.yaml")
    changed = replace(
        model, node_types=tuple(replace(t, description="changed") if t.label == "Food" else t for t in model.node_types)
    )

    with pytest.raises(ModelError) as raised:
        model_to_mapping(changed)

    assert raised.value.codes == ("duplicate_label",)


def test_operations_on_an_importing_model_keep_its_imports():
    model = ManualStrategy().create_model(VALID / "v2-imports.yaml")

    changed = apply(model, [AddNodeType(NodeType("Rose", "a pink wine", parents=("Wine",)))])

    assert changed.imports == model.imports
    assert [t["label"] for t in model_to_mapping(changed)["node_types"]] == ["Wine", "Rose"]
    assert diff(model, changed) == (AddNodeType(NodeType("Rose", "a pink wine", parents=("Wine",))),)


@pytest.mark.parametrize("key", ["ontology", "imports"])
def test_diff_refuses_a_change_no_operation_expresses_yet(key):
    b = ManualStrategy().create_model(VALID / "v2-imports-same-declaration.yaml")
    b = replace(b, ontology=Ontology(name="food"))
    a = replace(b, **{key: None if key == "ontology" else ()})

    with pytest.raises(ModelError, match=key) as raised:
        diff(a, b)

    assert raised.value.codes == ("unknown_operation",)


def test_the_ontology_block_round_trips_and_needs_version_2():
    model = ManualStrategy().create_model(VALID / "v2-ontology.yaml")

    assert model.ontology == Ontology(
        name="wine",
        description="Wines and the foods they pair with, for menu suggestions",
        iri="https://example.org/wine",
        version_info="2026.11",
        includes=("wine characteristics", "food types", "good and bad pairings"),
        excludes=("cellar inventory", "restaurant staff"),
        competency_questions=(
            "Which wine characteristics should I consider when choosing a wine?",
            "Does Cabernet Sauvignon go well with seafood?",
        ),
        maintainers=("the sommelier team",),
        conventions=Conventions("upper_camel", "snake", "snake", "singular"),
    )
    assert needs_version_2(replace(model, node_types=(NodeType("Wine"),)))
    assert model_from_mapping(model_to_mapping(model), "again") == model


def test_an_empty_ontology_block_is_kept():
    model = ManualStrategy().create_model(VALID / "v2-ontology-empty.yaml")

    assert model.ontology == Ontology()
    assert model_to_mapping(model)["ontology"] == {}


@pytest.mark.parametrize("settings", [{"node_case": "kebab"}, {"number": "dual"}])
def test_conventions_outside_the_contract_are_refused(settings):
    with pytest.raises(ModelError) as raised:
        Conventions(**settings)

    assert raised.value.codes == ("invalid_format",)


def test_readers_that_rebuild_types_are_told_about_the_ontology_block_and_imports():
    model = ManualStrategy().create_model(VALID / "v2-imports.yaml")

    assert model.uses_hierarchy_or_properties() == ("parents", "properties", "imports")
    assert replace(model, imports=(), ontology=Ontology()).uses_hierarchy_or_properties() == (
        "parents",
        "properties",
        "ontology",
    )
