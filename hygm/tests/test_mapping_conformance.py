"""The mapping conformance fixtures every reader runs (hygm/conformance/README.md)."""

from pathlib import Path

import jsonschema
import pytest
import yaml

from hygm import ModelError, load_mapping, mapping_from_mapping, mapping_schema, mapping_to_mapping

FIXTURES = Path(__file__).parent.parent / "conformance" / "mapping"
VALID = sorted((FIXTURES / "valid").iterdir())
INVALID = sorted((FIXTURES / "invalid").glob("*/*"))

#: Codes for problems in a file's structure alone, which the JSON Schema rejects too.
STRUCTURAL = {"invalid_format", "unknown_key", "unknown_version"}

LAYER_3_CODES = {
    "invalid_format",
    "unknown_version",
    "unknown_key",
    "unknown_model",
    "unknown_type",
    "unknown_source",
    "binding_kind",
    "undeclared_property",
    "duplicate_property",
    "duplicate_source",
    "unbound_key",
    "endpoint_not_admitted",
    "endpoint_columns",
}


def _raw(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_a_valid_fixture_loads_and_round_trips(path):
    model, mapping = load_mapping(path)

    assert mapping_from_mapping(mapping_to_mapping(mapping), "round trip", model) == mapping


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_a_valid_fixture_passes_the_schema(path):
    jsonschema.validate(_raw(path), mapping_schema(1))


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.name)
def test_writing_a_valid_fixture_keeps_what_it_says(path):
    _, mapping = load_mapping(path)
    raw = _raw(path)

    assert mapping_to_mapping(mapping) == {key: value for key, value in raw.items() if value != []}


@pytest.mark.parametrize("path", INVALID, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_an_invalid_fixture_fails_with_its_code(path):
    with pytest.raises(ModelError) as raised:
        load_mapping(path)

    assert path.parent.name in raised.value.codes


@pytest.mark.parametrize("path", INVALID, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_an_invalid_fixture_holds_one_problem(path):
    with pytest.raises(ModelError) as raised:
        load_mapping(path)

    assert set(raised.value.codes) == {path.parent.name}


@pytest.mark.parametrize(
    "path",
    [p for p in INVALID if p.parent.name in STRUCTURAL],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_a_structurally_invalid_fixture_fails_the_schema_too(path):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_raw(path), mapping_schema(1))


def test_every_mapping_code_has_a_fixture():
    assert {p.parent.name for p in INVALID} == LAYER_3_CODES


def test_an_unknown_schema_version_is_refused():
    with pytest.raises(ModelError) as raised:
        mapping_schema(2)

    assert raised.value.codes == ("unknown_version",)
