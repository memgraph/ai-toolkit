"""The model conformance fixtures every reader runs (hygm/conformance/README.md)."""

from pathlib import Path

import jsonschema
import pytest
import yaml

from hygm import ManualStrategy, ModelError, model_from_mapping, model_schema, model_to_mapping

FIXTURES = Path(__file__).parent.parent / "conformance" / "model"
VALID = sorted((FIXTURES / "valid").iterdir())
INVALID = sorted((FIXTURES / "invalid").glob("*/*"))

#: Codes for problems in a file's structure alone, which its version's JSON Schema rejects too.
STRUCTURAL = {
    "invalid_format",
    "unknown_key",
    "invalid_identifier",
    "unknown_property_type",
    "values_not_string",
    "unknown_profile",
    "invalid_profile_value",
}


def _raw(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _version(raw) -> int:
    return raw.get("version", 1) if isinstance(raw, dict) else 1


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_a_valid_fixture_loads_and_round_trips(path):
    model = ManualStrategy().create_model(path)

    written = model_to_mapping(model)

    assert model_from_mapping(written, "round trip", resolver=ManualStrategy().resolver(path)) == model


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_a_valid_fixture_passes_its_versions_schema(path):
    raw = _raw(path)

    jsonschema.validate(raw, model_schema(_version(raw)))


@pytest.mark.parametrize("path", INVALID, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_an_invalid_fixture_fails_with_its_code(path):
    with pytest.raises(ModelError) as raised:
        ManualStrategy().create_model(path)

    assert path.parent.name in raised.value.codes


@pytest.mark.parametrize(
    "path",
    [p for p in INVALID if p.parent.name in STRUCTURAL and isinstance(_raw(p), list | dict)],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_a_structurally_invalid_fixture_fails_its_versions_schema_too(path):
    raw = _raw(path)
    version = _version(raw)
    if version not in (1, 2):
        pytest.skip("no schema for a malformed version")

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, model_schema(version))


def test_every_code_in_the_contract_has_a_fixture():
    model_codes = {
        "invalid_format",
        "unknown_version",
        "unknown_key",
        "invalid_identifier",
        "duplicate_label",
        "duplicate_property",
        "unknown_parent",
        "parent_cycle",
        "undeclared_endpoint",
        "unknown_property_type",
        "values_not_string",
        "property_conflict",
        "unknown_key_property",
        "key_not_required",
        "two_keys",
        "unknown_profile",
        "invalid_profile_value",
        "import_cycle",
        "unknown_model",
        "unknown_type",
        "disjoint_conflict",
        "duplicate_name",
    }

    assert {p.parent.name for p in INVALID} == model_codes
