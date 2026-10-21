"""Tests for the generic command hook CLI."""

import io
import json

from memgraph_context_graph.cli import main as top_level_main
from memgraph_context_graph.hooks.cli import main


def test_generic_cli_dispatches_codex_hook(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert main(["run", "codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_top_level_cli_dispatches_hook_run(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert top_level_main(["hook", "run", "codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_top_level_cli_prints_version(monkeypatch, capsys):
    monkeypatch.setattr("memgraph_context_graph.cli._package_version", lambda package_name: "1.2.3")

    assert top_level_main(["--version"]) == 0

    assert capsys.readouterr().out.strip() == "1.2.3"


def test_top_level_cli_doctor_reports_checks(monkeypatch, capsys):
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_cli",
        lambda: {"name": "memgraph-context-graph executable", "ok": True, "detail": "/bin/memgraph-context-graph"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_package",
        lambda package_name: {"name": package_name, "ok": True, "detail": "1.2.3"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_config",
        lambda: {"name": "config", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_memgraph",
        lambda: {"name": "memgraph", "ok": True, "detail": "bolt://localhost:7687 reachable"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_connector",
        lambda connector: {"name": f"connector:{connector}", "ok": True, "detail": "installed; memgraph=reachable"},
    )
    monkeypatch.setattr("memgraph_context_graph.cli._check_plugin", lambda runtime: None)
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_runtime",
        lambda runtime, connectors: {"name": f"runtime:{runtime}", "ok": True, "detail": "strict hook smoke passed"},
    )

    assert top_level_main(["doctor", "--runtime", "claude-code", "--connector", "skills-graph"]) == 0

    output = capsys.readouterr().out
    assert "OK memgraph-context-graph executable" in output
    assert "OK connector:skills-graph" in output
    assert "OK runtime:claude-code" in output


def test_top_level_cli_doctor_fails_when_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_cli",
        lambda: {"name": "memgraph-context-graph executable", "ok": True, "detail": "/bin/memgraph-context-graph"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_package",
        lambda package_name: {"name": package_name, "ok": True, "detail": "1.2.3"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_config",
        lambda: {"name": "config", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_connector",
        lambda connector: {"name": f"connector:{connector}", "ok": False, "detail": "missing"},
    )
    monkeypatch.setattr("memgraph_context_graph.cli._check_plugin", lambda runtime: None)
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_runtime",
        lambda runtime, connectors: {"name": f"runtime:{runtime}", "ok": True, "detail": "strict hook smoke passed"},
    )

    assert top_level_main(["doctor", "--runtime", "codex", "--connector", "skills-graph"]) == 1

    assert "FAIL connector:skills-graph" in capsys.readouterr().out


def test_top_level_cli_doctor_checks_embeddings_only_with_sessions_graph(monkeypatch, capsys):
    for name, result in {
        "_check_cli": {
            "name": "memgraph-context-graph executable",
            "ok": True,
            "detail": "/bin/memgraph-context-graph",
        },
        "_check_config": {"name": "config", "ok": True, "detail": "ok"},
        "_check_memgraph": {"name": "memgraph", "ok": True, "detail": "reachable"},
        "_check_embeddings": {"name": "embeddings", "ok": False, "detail": "no MAGE"},
        "_check_mcp": {"name": "mcp", "ok": True, "detail": "serves recall"},
    }.items():
        monkeypatch.setattr(f"memgraph_context_graph.cli.{name}", lambda result=result: result)
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_package",
        lambda package_name: {"name": package_name, "ok": True, "detail": "1.2.3"},
    )
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_connector",
        lambda connector: {"name": f"connector:{connector}", "ok": True, "detail": "installed"},
    )
    monkeypatch.setattr("memgraph_context_graph.cli._check_plugin", lambda runtime: None)
    monkeypatch.setattr(
        "memgraph_context_graph.cli._check_runtime",
        lambda runtime, connectors: {"name": f"runtime:{runtime}", "ok": True, "detail": "ok"},
    )

    assert top_level_main(["doctor", "--connector", "skills-graph"]) == 0
    assert "embeddings" not in capsys.readouterr().out

    assert top_level_main(["doctor", "--connector", "sessions-graph"]) == 1
    assert "FAIL embeddings: no MAGE" in capsys.readouterr().out


def test_generic_cli_requires_runtime(capsys):
    assert main([]) == 2

    assert "command" in capsys.readouterr().out


def test_init_codex_writes_private_config(tmp_path, capsys):
    assert (
        main(
            [
                "init",
                "codex",
                "--project-dir",
                str(tmp_path),
                "--connector",
                "skills-graph",
                "--hook-command",
                "/venv/bin/memgraph-context-graph hook run codex --connector skills-graph",
            ]
        )
        == 0
    )

    assert (tmp_path / ".codex" / "config.toml").read_text() == "[features]\nhooks = true\n"
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == "/venv/bin/memgraph-context-graph hook run codex --connector skills-graph"
    assert hooks["hooks"]["SessionStart"][0]["matcher"] == "startup|resume|clear|compact"
    assert "Wrote" in capsys.readouterr().out


def test_init_codex_does_not_bake_memgraph_connection_into_hook_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("memgraph_context_graph.adapters._spec.shutil.which", lambda _: "/bin/memgraph-context-graph")

    assert (
        main(
            [
                "init",
                "codex",
                "--project-dir",
                str(tmp_path),
                "--memgraph-url",
                "bolt://memgraph.example:7687",
                "--memgraph-user",
                "neo",
                "--memgraph-password",
                "secret",
                "--memgraph-database",
                "skills",
            ]
        )
        == 0
    )

    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == "/bin/memgraph-context-graph hook run codex"
    assert "MEMGRAPH_URL" not in command
    assert "MEMGRAPH_USER" not in command
    assert "MEMGRAPH_PASSWORD" not in command
    assert "MEMGRAPH_DATABASE" not in command
    assert "secret" not in capsys.readouterr().out


def _register(monkeypatch, name, graph_class):
    """Stand *graph_class* in for the installed component that registers connector *name*."""
    from memgraph_context_graph import connectors

    def never_built(graph, config):
        raise AssertionError("init only sets up the schema")

    spec = connectors.ConnectorSpec(name=name, graph=graph_class, build=never_built)
    monkeypatch.setattr(connectors, "load_connectors", lambda: {name: spec})


def test_init_codex_uses_memgraph_flags_only_for_setup_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("memgraph_context_graph.adapters._spec.shutil.which", lambda _: "/bin/memgraph-context-graph")
    captured = {}

    class _SkillGraph:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def setup(self):
            captured["setup"] = True

    _register(monkeypatch, "skills-graph", _SkillGraph)

    assert (
        main(
            [
                "init",
                "codex",
                "--project-dir",
                str(tmp_path),
                "--connector",
                "skills-graph",
                "--memgraph-url",
                "bolt://memgraph.example:7687",
                "--memgraph-user",
                "neo",
                "--memgraph-password",
                "secret",
                "--memgraph-database",
                "skills",
                "--setup-schema",
            ]
        )
        == 0
    )

    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "MEMGRAPH" not in command
    assert captured == {
        "url": "bolt://memgraph.example:7687",
        "username": "neo",
        "password": "secret",
        "database": "skills",
        "setup": True,
    }


def test_init_codex_sets_up_actions_graph_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("memgraph_context_graph.adapters._spec.shutil.which", lambda _: "/bin/memgraph-context-graph")
    captured = {}

    class _ActionsGraph:
        def __init__(self, **kwargs):
            captured.update(url=kwargs["url"], database=kwargs["database"])

        def setup(self):
            captured["setup"] = True

    _register(monkeypatch, "actions-graph", _ActionsGraph)

    assert (
        main(
            [
                "init",
                "codex",
                "--project-dir",
                str(tmp_path),
                "--connector",
                "actions-graph",
                "--memgraph-url",
                "bolt://memgraph.example:7687",
                "--memgraph-database",
                "actions",
                "--setup-schema",
            ]
        )
        == 0
    )

    assert captured == {"url": "bolt://memgraph.example:7687", "database": "actions", "setup": True}


def test_init_codex_refuses_to_overwrite_without_force(tmp_path, capsys):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text("existing", encoding="utf-8")

    assert main(["init", "codex", "--project-dir", str(tmp_path)]) == 1

    assert (codex_dir / "config.toml").read_text() == "existing"
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_init_codex_force_overwrites(tmp_path):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text("existing", encoding="utf-8")

    assert main(["init", "codex", "--project-dir", str(tmp_path), "--force"]) == 0

    assert (codex_dir / "config.toml").read_text() == "[features]\nhooks = true\n"


def _enabled_connectors(monkeypatch, argv):
    """The connector names `hook run` hands create_link, for a turn-end payload."""
    from memgraph_context_graph.link import AgentLink

    enabled = []

    def recording(names, **kwargs):
        enabled.append(list(names))
        return AgentLink()

    monkeypatch.setattr("memgraph_context_graph.hooks.runner.create_link", recording)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))
    assert main(["run", "codex", *argv]) == 0
    (names,) = enabled
    return names


def test_hook_run_enables_the_configured_connectors(monkeypatch):
    from memgraph_context_graph.adapters import _identity

    _identity.write_config(connectors=("actions-graph", "resources-graph"))

    assert _enabled_connectors(monkeypatch, []) == ["actions-graph", "resources-graph"]
    assert _enabled_connectors(monkeypatch, ["--connector", "skills-graph"]) == ["skills-graph"]


def test_hook_run_enables_the_defaults_while_capture_connectors_is_unset(monkeypatch, tmp_path):
    from memgraph_context_graph.adapters import _identity
    from memgraph_context_graph.connectors import DEFAULT_CONNECTORS

    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "unset.toml"))
    _identity._reset_cache()

    assert _enabled_connectors(monkeypatch, []) == list(DEFAULT_CONNECTORS)
