"""The `hygm` command line: exit 1 only for a model that fails validation."""

import json
from pathlib import Path

import pytest

from hygm import ManualStrategy, OwlImportStrategy
from hygm.cli import main

CONFORMANCE = Path(__file__).parent.parent / "conformance" / "model"
SHOP = CONFORMANCE / "valid" / "v2-shop.yaml"
WINE = Path(__file__).parent / "fixtures" / "wine.ttl"


def test_validate_reports_a_valid_model_and_exits_0(capsys):
    assert main(["validate", str(SHOP)]) == 0
    assert capsys.readouterr().out.startswith("valid:")


@pytest.mark.parametrize("path", sorted((CONFORMANCE / "invalid" / "parent_cycle").iterdir()), ids=lambda p: p.name)
def test_validate_exits_1_with_the_code_for_an_invalid_model(path, capsys):
    assert main(["validate", str(path)]) == 1
    assert "parent_cycle" in capsys.readouterr().err


def test_validate_reads_owl_by_its_suffix(capsys):
    assert main(["validate", str(WINE)]) == 0
    assert "9 node types, 5 relation types" in capsys.readouterr().out


def test_lint_reports_findings_and_still_exits_0(capsys):
    assert main(["lint", str(WINE)]) == 0
    out = capsys.readouterr().out
    assert "single_subtype" in out and out.rstrip().endswith("5 finding(s)")


def test_lint_prints_json_on_request(capsys):
    assert main(["lint", "--json", str(WINE)]) == 0
    findings = json.loads(capsys.readouterr().out)
    assert {"code", "severity", "element", "message", "suggestion"} == set(findings[0])


@pytest.mark.parametrize("suffix", [".ttl", ".owl", ".jsonld", ".json", ".yaml"])
def test_convert_writes_the_format_its_suffix_names(tmp_path, suffix):
    out = tmp_path / f"shop{suffix}"

    assert main(["convert", str(SHOP), str(out)]) == 0

    read = ManualStrategy() if suffix in (".json", ".yaml") else OwlImportStrategy()
    back = read.create_model(out)
    assert back.node_labels() and sorted(back.node_labels()) == sorted(
        ManualStrategy().create_model(SHOP).node_labels()
    )


def test_convert_from_owl_to_a_model_file(tmp_path):
    out = tmp_path / "wine.yaml"

    assert main(["convert", str(WINE), str(out)]) == 0
    assert ManualStrategy().create_model(out) == OwlImportStrategy().create_model(WINE)


def test_diff_prints_the_operations_as_json(tmp_path, capsys):
    smaller = tmp_path / "smaller.yaml"
    smaller.write_text("version: 2\nnode_types:\n  - {label: Wine, description: a wine}\n")
    bigger = tmp_path / "bigger.yaml"
    bigger.write_text(
        "version: 2\nnode_types:\n  - {label: Wine, description: a wine}\n  - {label: Grape, description: a grape}\n"
    )

    assert main(["diff", str(smaller), str(bigger)]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"op": "add_node_type", "node_type": {"label": "Grape", "description": "a grape"}}
    ]


@pytest.mark.parametrize(
    "argv",
    [["validate", "missing.yaml"], ["convert", str(SHOP), "shop.csv"]],
    ids=["missing-file", "unknown-suffix"],
)
def test_a_usage_problem_exits_2_not_1(argv, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("hygm:")
