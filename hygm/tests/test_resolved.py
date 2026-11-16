"""The resolved form (docs/contract.md, 4.18) and its conformance fixtures (hygm/conformance/README.md)."""

import json
from pathlib import Path

import jsonschema
import pytest

from hygm import (
    HygmModel,
    ManualStrategy,
    ModelError,
    NodeType,
    Property,
    RelationType,
    resolve_model,
    resolved_schema,
    resolved_to_json,
)
from hygm.cli import main

CONFORMANCE = Path(__file__).parent.parent / "conformance"
MODELS = sorted((CONFORMANCE / "model" / "valid").iterdir())
RESOLVED = CONFORMANCE / "resolved"
INVALID = sorted((RESOLVED / "invalid").glob("*/*.json"))


def _expected(model_fixture: Path) -> Path:
    return RESOLVED / "valid" / f"{model_fixture.stem}.json"


def _refusals(raw) -> set[str]:
    """What a reader that doesn't run hygm's validation refuses a resolved file for: the contract's 4.18 checks."""
    if not isinstance(raw, dict) or raw.get("format") != "hygm-resolved":
        return {"invalid_format"}
    if raw.get("version") != 1 or raw.get("model_version") != 2:
        return {"unknown_version"}
    errors = list(jsonschema.Draft202012Validator(resolved_schema()).iter_errors(raw))
    if errors:
        return {"unknown_key" if e.validator == "additionalProperties" else "invalid_format" for e in errors}
    codes = set()
    nodes = [n["label"] for n in raw["node_types"]]
    declared = set(nodes)
    if len(declared) < len(nodes) or len({r["label"] for r in raw["relation_types"]}) < len(raw["relation_types"]):
        codes.add("duplicate_label")
    for node in raw["node_types"]:
        names = [p["name"] for p in node.get("properties", [])]
        if len(set(names)) < len(names):
            codes.add("duplicate_property")
        if not declared.issuperset(node.get("parents", []) + node.get("ancestors", [])):
            codes.add("unknown_parent")
        if not set(names).issuperset(node.get("key", [])):
            codes.add("unknown_key_property")
        for restriction in node.get("relation_restrictions", []):
            if not declared.issuperset(restriction.get("end_types", [])):
                codes.add("undeclared_endpoint")
    for relation in raw["relation_types"]:
        names = [p["name"] for p in relation.get("properties", [])]
        if len(set(names)) < len(names):
            codes.add("duplicate_property")
        if not declared.issuperset(relation["start_types"] + relation["end_types"]):
            codes.add("undeclared_endpoint")
    return codes


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.name)
def test_every_valid_model_fixture_resolves_to_its_expected_output_byte_for_byte(path):
    assert resolved_to_json(ManualStrategy().create_model(path)) == _expected(path).read_text(encoding="utf-8")


def test_every_expected_output_has_a_model_fixture():
    assert {p.name for p in (RESOLVED / "valid").iterdir()} == {_expected(p).name for p in MODELS}


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.name)
def test_every_expected_output_passes_the_schema_and_a_readers_checks(path):
    raw = json.loads(_expected(path).read_text(encoding="utf-8"))

    jsonschema.validate(raw, resolved_schema())
    assert _refusals(raw) == set()


@pytest.mark.parametrize("path", INVALID, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_an_invalid_resolved_fixture_is_refused_with_exactly_its_code(path):
    assert _refusals(json.loads(path.read_text(encoding="utf-8"))) == {path.parent.name}


def test_the_schema_refuses_unknown_keys_at_every_level():
    for path in (RESOLVED / "invalid" / "unknown_key").iterdir():
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), resolved_schema())


def test_resolving_an_invalid_model_raises_its_codes():
    model = HygmModel(node_types=(NodeType("Wine", parents=("Drink",)),))

    with pytest.raises(ModelError) as raised:
        resolve_model(model)

    assert raised.value.codes == ("unknown_parent",)


def test_an_unknown_resolved_version_has_no_schema():
    with pytest.raises(ModelError) as raised:
        resolved_schema(2)

    assert raised.value.codes == ("unknown_version",)


def test_a_property_inherited_and_narrowed_names_the_type_that_introduced_it():
    model = HygmModel(
        node_types=(
            NodeType("Wine", properties=(Property("region"), Property("grapes", max_count=None))),
            NodeType("Blend", parents=("Wine",), properties=(Property("grapes", min_count=2, max_count=None),)),
        )
    )

    blend = resolve_model(model)["node_types"][1]

    assert blend["ancestors"] == ["Wine"]
    assert blend["properties"] == [
        {"name": "region", "type": "string", "declared_by": "Wine"},
        {
            "name": "grapes",
            "type": "string",
            "required": True,
            "min_count": 2,
            "max_count": None,
            "declared_by": "Wine",
        },
    ]


def test_an_unconstrained_endpoint_resolves_to_every_declared_type():
    model = HygmModel(
        node_types=(NodeType("Wine"), NodeType("Grape")),
        relation_types=(RelationType("mentions", end_labels=("Grape",)),),
    )

    relation = resolve_model(model)["relation_types"][0]

    assert "start_labels" not in relation
    assert relation["start_types"] == ["Wine", "Grape"]
    assert relation["end_types"] == ["Grape"]


def test_resolve_prints_to_standard_output_by_default(capsys):
    shop = CONFORMANCE / "model" / "valid" / "v2-shop.yaml"

    assert main(["resolve", str(shop)]) == 0

    assert capsys.readouterr().out == _expected(shop).read_text(encoding="utf-8")


def test_resolve_writes_the_output_file(tmp_path):
    shop = CONFORMANCE / "model" / "valid" / "v2-shop.yaml"
    out = tmp_path / "shop.resolved.json"

    assert main(["resolve", str(shop), "-o", str(out)]) == 0

    assert out.read_text(encoding="utf-8") == _expected(shop).read_text(encoding="utf-8")


def test_resolve_writes_nothing_and_exits_1_for_an_invalid_model(tmp_path, capsys):
    invalid = CONFORMANCE / "model" / "invalid" / "parent_cycle" / "two-types.yaml"
    out = tmp_path / "out.json"

    assert main(["resolve", str(invalid), "-o", str(out)]) == 1

    assert not out.exists()
    assert "parent_cycle" in capsys.readouterr().err
