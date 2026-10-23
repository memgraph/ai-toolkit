"""Tests for the `memgraph-context-graph config set/get/show` CLI subcommand."""

import re

import pytest

from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.cli import main as top_level_main


@pytest.fixture(autouse=True)
def _reset_cache():
    _identity._reset_cache()
    yield
    _identity._reset_cache()


@pytest.fixture()
def config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "context-graph"
    # Uses the supported override (ADR 0003) rather than monkeypatching module
    # privates, so these tests exercise the same path a real isolated session
    # takes instead of a shape only tests can produce.
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config_dir / "config.toml"))
    return config_dir


def test_config_set_reconcile_auto_reconcile_true(config_dir, capsys):
    assert top_level_main(["config", "set", "reconcile.auto_reconcile", "true"]) == 0
    assert "Wrote reconcile.auto_reconcile = true" in capsys.readouterr().out
    assert _identity.load_config().auto_reconcile is True


def test_doctor_passes_auto_reconcile_without_an_llm_key(config_dir):
    """Reconciliation extracts locally; following the docs' `config set reconcile.auto_reconcile true` must not fail."""
    from memgraph_context_graph.cli import _check_config

    _identity.write_config(user_id="alice", auto_reconcile=True)
    _identity._reset_cache()

    check = _check_config()

    assert check["ok"] is True
    assert "auto_reconcile=ON" in check["detail"]
    assert "needed only to learn the schema" in check["detail"]


def test_config_set_reconcile_auto_reconcile_false(config_dir, capsys):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()

    assert top_level_main(["config", "set", "reconcile.auto_reconcile", "false"]) == 0
    assert "Wrote reconcile.auto_reconcile = false" in capsys.readouterr().out
    assert _identity.load_config().auto_reconcile is False


def test_config_set_reconcile_auto_reconcile_rejects_invalid_value(config_dir, capsys):
    assert top_level_main(["config", "set", "reconcile.auto_reconcile", "banana"]) == 2
    assert "Invalid value for reconcile.auto_reconcile" in capsys.readouterr().err


def test_config_get_reconcile_auto_reconcile(config_dir, capsys):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()

    assert top_level_main(["config", "get", "reconcile.auto_reconcile"]) == 0
    assert capsys.readouterr().out.strip() == "true"


def test_config_get_reconcile_auto_reconcile_reports_unset(config_dir, capsys):
    assert top_level_main(["config", "get", "reconcile.auto_reconcile"]) == 1
    assert "not set" in capsys.readouterr().err


def test_config_show_reports_unset_reconcile_auto_reconcile(config_dir, capsys):
    assert top_level_main(["config", "show"]) == 0
    assert "reconcile.auto_reconcile = unset (defaults to false)" in capsys.readouterr().out


def test_config_show_includes_explicit_reconcile_auto_reconcile(config_dir, capsys):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()

    assert top_level_main(["config", "show"]) == 0
    assert "reconcile.auto_reconcile = true" in capsys.readouterr().out


def test_config_set_recall_embedding_model_round_trips(config_dir, capsys):
    assert top_level_main(["config", "set", "recall.embedding_model", "BAAI/bge-m3"]) == 0
    assert "Wrote recall.embedding_model = 'BAAI/bge-m3'" in capsys.readouterr().out
    assert _identity.resolve_embedding_model() == "BAAI/bge-m3"


def test_recall_embedding_model_is_unset_by_default(config_dir, capsys):
    assert _identity.resolve_embedding_model() is None
    assert top_level_main(["config", "show"]) == 0
    assert "recall.embedding_model = unset" in capsys.readouterr().out


def test_bootstrap_rewrite_keeps_recall_embedding_model(config_dir):
    """Like auto_reconcile, the model is only ever set by hand; re-running bootstrap must not drop it."""
    _identity.write_config(embedding_model="BAAI/bge-m3")
    _identity._reset_cache()

    _identity.write_config(user_id="alice")
    _identity._reset_cache()

    assert _identity.resolve_embedding_model() == "BAAI/bge-m3"


def test_config_set_ontology_round_trips_and_survives_bootstrap(config_dir, capsys):
    assert top_level_main(["config", "set", "ontology.path", "/schemas/coding.yaml"]) == 0
    assert top_level_main(["config", "set", "ontology.derive", "off"]) == 0
    _identity.write_config(user_id="alice")
    _identity._reset_cache()

    assert _identity.resolve_ontology() == ("/schemas/coding.yaml", "off")


def test_config_set_ontology_derive_rejects_other_values(config_dir, capsys):
    assert top_level_main(["config", "set", "ontology.derive", "sometimes"]) == 2
    assert "expected extend/off" in capsys.readouterr().err
    assert _identity.resolve_ontology() == (None, None)


def test_ontology_is_unset_by_default(config_dir, capsys):
    assert top_level_main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "ontology.path = unset" in out
    assert "ontology.derive = unset (defaults to extend)" in out


def test_config_set_github_token_is_secret_and_survives_bootstrap(config_dir, capsys):
    assert top_level_main(["config", "set", "github.token", "ghp_example"]) == 0
    assert "Wrote github.token = ***" in capsys.readouterr().out
    _identity.write_config(user_id="alice")
    _identity._reset_cache()

    assert _identity.load_config().github_token == "ghp_example"
    assert top_level_main(["config", "show"]) == 0
    shown = capsys.readouterr().out
    assert "github.token = ***" in shown and "ghp_example" not in shown


def test_github_token_is_unset_by_default(config_dir, capsys):
    assert _identity.load_config().github_token is None
    assert top_level_main(["config", "show"]) == 0
    assert "github.token = unset" in capsys.readouterr().out


def test_config_set_memory_backend_round_trips_and_survives_bootstrap(config_dir, capsys):
    assert top_level_main(["config", "set", "memory.backend", "context-graph"]) == 0
    _identity.write_config(user_id="ante")
    _identity._reset_cache()

    assert _identity.load_config().graph_memory
    assert top_level_main(["config", "show"]) == 0
    assert "memory.backend = 'context-graph'" in capsys.readouterr().out


def test_config_set_memory_backend_rejects_other_values(config_dir, capsys):
    assert top_level_main(["config", "set", "memory.backend", "files"]) == 2
    assert "expected context-graph/native" in capsys.readouterr().err


def test_memory_backend_is_native_by_default(config_dir, capsys):
    assert not _identity.load_config().graph_memory
    assert top_level_main(["config", "show"]) == 0
    assert "memory.backend = unset (native)" in capsys.readouterr().out


def test_config_set_capture_connectors_normalizes_and_validates(config_dir, capsys):
    assert top_level_main(["config", "set", "capture.connectors", "actions_graph, resources-graph"]) == 0
    assert _identity.load_config().connectors == ("actions-graph", "resources-graph")

    assert top_level_main(["config", "set", "capture.connectors", "actions-graph,nope-graph"]) == 2
    assert "Unsupported connector: nope-graph" in capsys.readouterr().err
    assert _identity.load_config().connectors == ("actions-graph", "resources-graph")


def test_config_show_names_the_default_connectors_while_unset(config_dir, capsys):
    assert top_level_main(["config", "show"]) == 0
    assert (
        "capture.connectors = unset (defaults to skills-graph, actions-graph, sessions-graph)"
        in capsys.readouterr().out
    )


def test_config_set_llm_derive_model(config_dir):
    assert top_level_main(["config", "set", "llm.derive_model", "gpt-5-mini"]) == 0
    assert _identity.load_config().derive_model == "gpt-5-mini"


def test_config_set_capture_policy_keys_validate_round_trip_and_keep_connectors(config_dir, capsys):
    assert top_level_main(["config", "set", "capture.connectors", "actions-graph"]) == 0
    assert top_level_main(["config", "set", "capture.redact", "OFF"]) == 0
    assert top_level_main(["config", "set", "capture.max_chars", "5000"]) == 0
    assert top_level_main(["config", "set", "capture.exclude_tools", "TodoWrite,  mcp__x__*"]) == 0
    assert top_level_main(["config", "set", "capture.max_chars", "lots"]) == 2
    assert top_level_main(["config", "set", "capture.redact", "maybe"]) == 2
    capsys.readouterr()

    assert top_level_main(["config", "get", "capture.exclude_tools"]) == 0
    assert capsys.readouterr().out.strip() == "TodoWrite, mcp__x__*"
    config = _identity.load_config()
    assert config.capture_settings == {"redact": "false", "max_chars": "5000", "exclude_tools": "TodoWrite, mcp__x__*"}
    assert config.connectors == ("actions-graph",)

    assert top_level_main(["config", "set", "capture.exclude_tools", ""]) == 0
    assert "exclude_tools" not in _identity.load_config().capture_settings


def test_config_show_describes_unset_capture_policy_keys(config_dir, capsys):
    assert top_level_main(["config", "show"]) == 0
    out = capsys.readouterr().out

    assert "capture.redact = unset (defaults to true)" in out
    assert "capture.exclude_projects = unset (none excluded)" in out


def test_the_written_file_lists_every_key_and_a_reread_keeps_them_unset(config_dir):
    from memgraph_context_graph.config_schema import CONFIG_KEYS

    _identity.write_config(user_id="ante")
    text = (config_dir / "config.toml").read_text()

    written, section = set(), ""
    for line in text.splitlines():
        if line.startswith("["):
            section = line.strip("[]")
        elif match := re.match(r"#? ?(\w+) = ", line):
            written.add(f"{section}.{match[1]}" if section else match[1])
    assert {key.name for key in CONFIG_KEYS} <= written
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.auto_reconcile is None and config.connectors is None and config.session_resume is None
    assert config.recall_settings == {} and config.extraction_settings == {} and config.capture_settings == {}
    assert config.present == frozenset(
        {"identity.user_id", "memgraph.url", "memgraph.database"}
    )  # empty strings aren't "set"


def test_config_set_recall_and_extraction_keys_validate_and_round_trip(config_dir, capsys):
    assert top_level_main(["config", "set", "recall.turns_k", "12"]) == 0
    assert top_level_main(["config", "set", "extraction.entity_threshold", "0.4"]) == 0
    assert top_level_main(["config", "set", "recall.turns_k", "many"]) == 2
    assert top_level_main(["config", "set", "extraction.entity_threshold", "high"]) == 2
    capsys.readouterr()

    assert _identity.load_config().recall_settings == {"turns_k": "12"}
    assert _identity.resolve_extraction() == {"entity_threshold": "0.4"}
    assert top_level_main(["config", "get", "recall.turns_k"]) == 0
    assert capsys.readouterr().out.strip() == "12"


def test_keys_the_table_does_not_know_survive_a_rewrite(config_dir):
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("[recall]\nfuture_width = \"3\"\n\n[redact]\nticket = 'ACME-[0-9]+'\n")

    _identity.write_config(user_id="u")
    _identity._reset_cache()

    config = _identity.load_config()
    assert config.recall_settings == {"future_width": "3"}
    assert config.redact_patterns == {"ticket": "ACME-[0-9]+"}


def test_config_show_and_get_mask_every_secret(config_dir, capsys):
    from memgraph_context_graph.config_schema import CONFIG_KEYS

    for key in CONFIG_KEYS:
        if key.secret:
            assert top_level_main(["config", "set", key.name, "s3cret-value"]) == 0
            assert top_level_main(["config", "get", key.name]) == 0
    assert top_level_main(["config", "show"]) == 0
    assert "s3cret-value" not in capsys.readouterr().out
