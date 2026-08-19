"""Tests for the `agent-context-graph config set/get/show` CLI subcommand."""

import pytest

from agent_context_graph.adapters import _identity
from agent_context_graph.cli import main as top_level_main


@pytest.fixture(autouse=True)
def _reset_cache():
    _identity._reset_cache()
    yield
    _identity._reset_cache()


@pytest.fixture()
def config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "context-graph"
    config_file = config_dir / "config.toml"
    monkeypatch.setattr(_identity, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(_identity, "_CONFIG_FILE", config_file)
    return config_dir


def test_config_set_reconcile_auto_reconcile_true(config_dir, capsys):
    assert top_level_main(["config", "set", "reconcile.auto_reconcile", "true"]) == 0
    assert "Wrote reconcile.auto_reconcile = True" in capsys.readouterr().out
    assert _identity.load_config().auto_reconcile is True


def test_config_set_reconcile_auto_reconcile_false(config_dir, capsys):
    _identity.write_full_config(auto_reconcile=True)
    _identity._reset_cache()

    assert top_level_main(["config", "set", "reconcile.auto_reconcile", "false"]) == 0
    assert "Wrote reconcile.auto_reconcile = False" in capsys.readouterr().out
    assert _identity.load_config().auto_reconcile is False


def test_config_set_reconcile_auto_reconcile_rejects_invalid_value(config_dir, capsys):
    assert top_level_main(["config", "set", "reconcile.auto_reconcile", "banana"]) == 2
    assert "Invalid value for reconcile.auto_reconcile" in capsys.readouterr().err


def test_config_get_reconcile_auto_reconcile(config_dir, capsys):
    _identity.write_full_config(auto_reconcile=True)
    _identity._reset_cache()

    assert top_level_main(["config", "get", "reconcile.auto_reconcile"]) == 0
    assert capsys.readouterr().out.strip() == "True"


def test_config_show_includes_reconcile_auto_reconcile(config_dir, capsys):
    assert top_level_main(["config", "show"]) == 0
    assert "reconcile.auto_reconcile = False" in capsys.readouterr().out
