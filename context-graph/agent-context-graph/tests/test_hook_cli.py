"""Tests for the generic command hook CLI."""

import io
import json
import os
import subprocess
import sys
from types import ModuleType

from agent_context_graph.cli import main as top_level_main
from agent_context_graph.hooks.cli import main


def test_generic_cli_dispatches_codex_hook(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert main(["run", "codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_top_level_cli_dispatches_hook_run(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    assert top_level_main(["hook", "run", "codex"]) == 0

    assert capsys.readouterr().out.strip() == '{"continue": true}'


def test_top_level_cli_prints_version(monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.cli._package_version", lambda package_name: "1.2.3")

    assert top_level_main(["--version"]) == 0

    assert capsys.readouterr().out.strip() == "1.2.3"


def test_top_level_cli_doctor_reports_checks(monkeypatch, capsys):
    monkeypatch.setattr(
        "agent_context_graph.cli._check_cli",
        lambda: {"name": "agent-context-graph executable", "ok": True, "detail": "/bin/agent-context-graph"},
    )
    monkeypatch.setattr(
        "agent_context_graph.cli._check_package",
        lambda package_name: {"name": package_name, "ok": True, "detail": "1.2.3"},
    )
    monkeypatch.setattr(
        "agent_context_graph.cli._check_connector",
        lambda connector: {"name": f"connector:{connector}", "ok": True, "detail": "installed; memgraph=reachable"},
    )
    monkeypatch.setattr(
        "agent_context_graph.cli._check_runtime",
        lambda runtime, connectors: {"name": f"runtime:{runtime}", "ok": True, "detail": "strict hook smoke passed"},
    )

    assert top_level_main(["doctor", "--runtime", "claude-code", "--connector", "skills-graph"]) == 0

    output = capsys.readouterr().out
    assert "OK agent-context-graph executable" in output
    assert "OK connector:skills-graph" in output
    assert "OK runtime:claude-code" in output


def test_top_level_cli_doctor_fails_when_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        "agent_context_graph.cli._check_cli",
        lambda: {"name": "agent-context-graph executable", "ok": True, "detail": "/bin/agent-context-graph"},
    )
    monkeypatch.setattr(
        "agent_context_graph.cli._check_package",
        lambda package_name: {"name": package_name, "ok": True, "detail": "1.2.3"},
    )
    monkeypatch.setattr(
        "agent_context_graph.cli._check_connector",
        lambda connector: {"name": f"connector:{connector}", "ok": False, "detail": "missing"},
    )
    monkeypatch.setattr(
        "agent_context_graph.cli._check_runtime",
        lambda runtime, connectors: {"name": f"runtime:{runtime}", "ok": True, "detail": "strict hook smoke passed"},
    )

    assert top_level_main(["doctor", "--runtime", "codex", "--connector", "skills-graph"]) == 1

    assert "FAIL connector:skills-graph" in capsys.readouterr().out


def test_top_level_cli_bootstrap_installs_and_runs_doctor(monkeypatch, capsys):
    commands = []
    doctor_args = []

    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", lambda host, port: True)

    def fake_run(command, *, check):
        commands.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    def fake_doctor(args):
        doctor_args.extend(args)
        return 0

    monkeypatch.setattr("agent_context_graph.cli.subprocess.run", fake_run)
    monkeypatch.setattr("agent_context_graph.cli._doctor", fake_doctor)

    assert top_level_main(["bootstrap", "--runtime", "codex", "--connector", "skills-graph"]) == 0

    assert commands == [
        [
            "/bin/uv",
            "tool",
            "install",
            "agent-context-graph",
            "--with",
            "skills-graph[agent-context-graph]",
            "--reinstall",
        ]
    ]
    assert doctor_args == ["--runtime", "codex", "--connector", "skills-graph"]
    output = capsys.readouterr().out
    assert "OK uv: /bin/uv" in output
    assert "OK memgraph: bolt://localhost:7687 reachable" in output


def test_top_level_cli_bootstrap_no_reinstall_omits_reinstall(monkeypatch):
    commands = []

    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", lambda host, port: True)
    monkeypatch.setattr("agent_context_graph.cli._doctor", lambda args: 0)

    def fake_run(command, *, check):
        commands.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agent_context_graph.cli.subprocess.run", fake_run)

    assert top_level_main(["bootstrap", "--runtime", "codex", "--connector", "skills-graph", "--no-reinstall"]) == 0

    assert "--reinstall" not in commands[0]


def test_top_level_cli_bootstrap_reports_memgraph_start_command(monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: f"/bin/{name}" if name == "uv" else None)
    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", lambda host, port: False)

    assert top_level_main(["bootstrap", "--runtime", "claude-code", "--connector", "skills-graph"]) == 1

    error = capsys.readouterr().err
    assert "FAIL memgraph: bolt://localhost:7687 is not reachable" in error
    assert "docker run --rm -p 7687:7687 memgraph/memgraph" in error


def test_top_level_cli_bootstrap_reports_missing_uv(monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: None)

    assert top_level_main(["bootstrap", "--runtime", "codex", "--connector", "skills-graph"]) == 1

    error = capsys.readouterr().err
    assert "FAIL uv: not found on PATH" in error
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in error
    assert "install Python 3.10+" in error


def test_top_level_cli_bootstrap_reports_unknown_connector(monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", lambda host, port: True)

    assert top_level_main(["bootstrap", "--runtime", "codex", "--connector", "typo-graph"]) == 1

    assert "FAIL connector:typo-graph: unsupported connector" in capsys.readouterr().err


def test_top_level_cli_bootstrap_reports_install_failure(monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", lambda host, port: True)

    def fail_run(command, *, check):
        raise subprocess.CalledProcessError(42, command)

    monkeypatch.setattr("agent_context_graph.cli.subprocess.run", fail_run)

    assert top_level_main(["bootstrap", "--runtime", "codex", "--connector", "skills-graph"]) == 42

    assert "FAIL install: uv tool install exited with status 42" in capsys.readouterr().err


def test_top_level_cli_bootstrap_reports_executable_missing_after_install(monkeypatch, capsys):
    def fake_which(name):
        if name == "uv":
            return "/bin/uv"
        return None

    monkeypatch.setattr("agent_context_graph.cli.shutil.which", fake_which)
    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", lambda host, port: True)
    monkeypatch.setattr(
        "agent_context_graph.cli.subprocess.run", lambda command, *, check: subprocess.CompletedProcess(command, 0)
    )

    assert top_level_main(["bootstrap", "--runtime", "codex", "--connector", "skills-graph"]) == 1

    assert "FAIL agent-context-graph: installed but not on PATH" in capsys.readouterr().err


def test_top_level_cli_bootstrap_accepts_zero_port(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr("agent_context_graph.cli.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        "agent_context_graph.cli.subprocess.run", lambda command, *, check: subprocess.CompletedProcess(command, 0)
    )
    monkeypatch.setattr("agent_context_graph.cli._doctor", lambda args: 0)

    def fake_reachable(host, port):
        captured["host"] = host
        captured["port"] = port
        return True

    monkeypatch.setattr("agent_context_graph.cli._memgraph_reachable", fake_reachable)

    assert (
        top_level_main(["bootstrap", "--runtime", "codex", "--connector", "skills-graph", "--memgraph-port", "0"]) == 0
    )

    assert captured == {"host": "localhost", "port": 0}
    assert "bolt://localhost:0 reachable" in capsys.readouterr().out


def test_top_level_cli_setup_aliases_codex_init(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_context_graph.hooks.cli.shutil.which", lambda _: "/bin/agent-context-graph")

    assert top_level_main(["setup", "codex", "--project-dir", str(tmp_path)]) == 0

    assert (tmp_path / ".codex" / "config.toml").read_text() == "[features]\ncodex_hooks = true\n"


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
                "/venv/bin/agent-context-graph hook run codex --connector skills-graph",
            ]
        )
        == 0
    )

    assert (tmp_path / ".codex" / "config.toml").read_text() == "[features]\ncodex_hooks = true\n"
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == "/venv/bin/agent-context-graph hook run codex --connector skills-graph"
    assert hooks["hooks"]["SessionStart"][0]["matcher"] == "startup|resume|clear"
    assert "Wrote" in capsys.readouterr().out


def test_init_codex_does_not_bake_memgraph_connection_into_hook_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("agent_context_graph.hooks.cli.shutil.which", lambda _: "/bin/agent-context-graph")

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
    assert command == "/bin/agent-context-graph hook run codex --connector skills-graph"
    assert "MEMGRAPH_URL" not in command
    assert "MEMGRAPH_USER" not in command
    assert "MEMGRAPH_PASSWORD" not in command
    assert "MEMGRAPH_DATABASE" not in command
    assert "secret" not in capsys.readouterr().out


def test_init_codex_uses_memgraph_env_only_for_setup_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_context_graph.hooks.cli.shutil.which", lambda _: "/bin/agent-context-graph")
    captured = {}

    class _SkillGraph:
        def setup(self):
            captured["url"] = os.environ.get("MEMGRAPH_URL")
            captured["user"] = os.environ.get("MEMGRAPH_USER")
            captured["password"] = os.environ.get("MEMGRAPH_PASSWORD")
            captured["database"] = os.environ.get("MEMGRAPH_DATABASE")

    fake_skills_graph = ModuleType("skills_graph")
    fake_skills_graph.SkillGraph = _SkillGraph
    monkeypatch.setitem(sys.modules, "skills_graph", fake_skills_graph)

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
        "user": "neo",
        "password": "secret",
        "database": "skills",
    }


def test_init_codex_uses_memgraph_toolbox_env_helper(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_context_graph.hooks.cli.shutil.which", lambda _: "/bin/agent-context-graph")
    monkeypatch.setenv("MEMGRAPH_URL", "bolt://env-memgraph:7687")
    monkeypatch.setenv("MEMGRAPH_DATABASE", "env-skills")

    assert main(["init", "codex", "--project-dir", str(tmp_path)]) == 0

    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    command = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "MEMGRAPH_URL" not in command
    assert "MEMGRAPH_DATABASE" not in command


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

    assert (codex_dir / "config.toml").read_text() == "[features]\ncodex_hooks = true\n"
