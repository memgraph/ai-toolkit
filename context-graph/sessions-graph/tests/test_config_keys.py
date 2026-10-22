"""The hub's config table lists sessions-graph's ``[recall]`` and ``[extraction]`` keys; it must match them."""

import dataclasses

import pytest
from sessions_graph.extraction import ExtractionConfig
from sessions_graph.recall import RecallConfig

config_schema = pytest.importorskip("memgraph_context_graph.config_schema")


def _table(section: str) -> dict[str, str]:
    return {key.key: key.example for key in config_schema.CONFIG_KEYS if key.section == section and key.field is None}


def test_recall_keys_match_recall_config_and_its_defaults():
    defaults = {
        field.name: ", ".join(field.default) if isinstance(field.default, tuple) else str(field.default or "")
        for field in dataclasses.fields(RecallConfig)
    }
    assert _table("recall") == defaults


def test_extraction_keys_match_extraction_config():
    assert set(_table("extraction")) == {field.name for field in dataclasses.fields(ExtractionConfig)}


@pytest.fixture()
def config_file(monkeypatch, tmp_path):
    from memgraph_context_graph.adapters import _identity

    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity._reset_cache()
    yield tmp_path / "config.toml"
    _identity._reset_cache()


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("extraction.entity_threshold", "1.5", "between 0 and 1"),
        ("extraction.chunk_size", "0", "> 0"),
        ("recall.lanes", "turns, gossip", "unknown recall lanes"),
    ],
)
def test_config_set_refuses_what_sessions_graph_would_reject(config_file, capsys, key, value, reason):
    """Accepted here, the value would fail every embed, reconcile or recall until someone edits the file."""
    from memgraph_context_graph.cli import main

    assert main(["config", "set", key, value]) == 2
    assert reason in capsys.readouterr().err
    assert not config_file.exists()
