"""setup and uninstall without Docker: agents' wiring, the plan, the command line, and the TUI.

The plugin agents run against the fake `claude`/`codex` from ``fake_agent_cli.py``; project wiring writes
real files into a temp project. The Memgraph step's real-Docker cases are in ``test_setup_docker.py``.
"""

from __future__ import annotations

import json

import pytest

from memgraph_context_graph.cli import main
from memgraph_context_graph.setup import agents, engine, memgraph
from memgraph_context_graph.setup.agents import BY_RUNTIME, WiringError


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "billing"
    path.mkdir()
    return path


def _logged():
    lines: list[str] = []
    return lines, lines.append


# --- project agents -----------------------------------------------------------


def test_cursor_wiring_keeps_the_users_own_hooks_and_writes_once(project, setup_home):
    hooks = project / ".cursor" / "hooks.json"
    hooks.parent.mkdir()
    users = {"command": "./lint.sh"}
    hooks.write_text(json.dumps({"version": 1, "custom": True, "hooks": {"preToolUse": [users]}}))
    cursor = BY_RUNTIME["cursor"]

    _, log = _logged()
    assert agents.wire_project(cursor, project, log) == f"wired {project}"
    assert agents.wire_project(cursor, project, log) == f"rewired {project}"

    document = json.loads(hooks.read_text())
    assert document["custom"] is True
    pre = document["hooks"]["preToolUse"]
    assert pre[0] == users
    assert len(pre) == 2 and "hook run cursor" in pre[1]["command"]
    assert agents.project_wired(cursor, project)

    assert agents.unwire_project(cursor, project)
    assert json.loads(hooks.read_text()) == {"version": 1, "custom": True, "hooks": {"preToolUse": [users]}}
    assert not agents.unwire_project(cursor, project)


def test_unwiring_deletes_a_hook_file_that_held_only_our_hooks(project, setup_home):
    for runtime, relative in (
        ("antigravity-cli", ".agents/hooks.json"),
        ("copilot-cli", ".github/hooks/memgraph-context-graph.json"),
        ("grok", ".grok/hooks/memgraph-context-graph.json"),
    ):
        agent = BY_RUNTIME[runtime]
        agents.wire_project(agent, project, lambda _: None)
        assert (project / relative).is_file(), runtime
        assert agents.unwire_project(agent, project)
        assert not (project / relative).exists(), runtime


def test_opencode_wiring_is_its_own_plugin_file(project, setup_home):
    opencode = BY_RUNTIME["opencode"]
    agents.wire_project(opencode, project, lambda _: None)
    agents.wire_project(opencode, project, lambda _: None)
    assert agents.project_wired(opencode, project)
    assert agents.unwire_project(opencode, project)
    assert not agents.project_wired(opencode, project)


def test_wiring_refuses_the_home_directory(setup_home):
    with pytest.raises(WiringError, match="home directory"):
        agents.wire_project(BY_RUNTIME["cursor"], setup_home, lambda _: None)
    assert not (setup_home / ".cursor").exists()


# --- plugin agents ------------------------------------------------------------


@pytest.mark.parametrize("runtime, binary", [("claude-code", "claude"), ("codex", "codex")])
def test_plugin_wiring_registers_installs_and_comes_off_again(fake_agents, runtime, binary):
    agent = BY_RUNTIME[runtime]
    lines, log = _logged()

    assert agents.wire_plugin(agent, log) == "plugin installed"
    assert agents.plugin_wired(agent)
    assert agents.wire_plugin(agent, log) == "plugin already installed"
    state = fake_agents.of(binary)
    assert state["plugins"] == [agents.PLUGIN_ID]
    assert ["plugin", "marketplace", "add", "memgraph/ai-toolkit", "--sparse", agent.sparse] in state["calls"]
    assert f"Installed {agents.PLUGIN_ID}" in lines

    assert agents.unwire_plugin(agent, log) == "removed the plugin and marketplace"
    assert fake_agents.of(binary)["plugins"] == []
    assert fake_agents.of(binary)["marketplaces"] == []
    assert agents.unwire_plugin(agent, log) == "wasn't installed"


def test_a_failing_plugin_install_says_what_the_agent_said(fake_agents, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_FAIL", "install")
    with pytest.raises(WiringError, match="network unreachable"):
        agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)


def test_only_agents_on_path_are_detected(fake_agents):
    assert [agent.runtime for agent in agents.detected()] == ["claude-code", "codex"]


# --- choices, plan and command line -------------------------------------------


def test_defaults_come_from_path_the_project_and_git(fake_agents, setup_home, project):
    env = engine.look_around()
    choices = engine.default_choices(env, cwd=project, user_id=None)
    assert choices.runtimes == ["claude-code", "codex"]
    assert choices.project_dirs == [project.resolve()]
    assert choices.user_id
    assert engine.default_choices(env, cwd=setup_home).project_dirs == []


def test_the_configured_user_wins_over_git(setup_home):
    from memgraph_context_graph.adapters._identity import write_config

    write_config(user_id="alice", memgraph_url="bolt://localhost:7700")
    env = engine.look_around()
    assert (env.configured_user, env.configured_url) == ("alice", "bolt://localhost:7700")
    assert engine.default_choices(env).user_id == "alice"


def test_plan_changes_nothing_and_names_every_step(fake_agents, setup_home, project, capsys):
    assert main(["setup", "--plan", "cursor", "claude-code", "--project-dir", str(project), "--user-id", "bob"]) == 0
    out = capsys.readouterr().out
    assert "Memgraph: use one on localhost:7687, else start the context-graph-memgraph container" in out
    assert "Identity: identity.user_id = bob" in out
    assert f"Cursor: wire {project.resolve()}" in out
    assert "Claude Code: install the Context Graph plugin (user-wide)" in out
    assert fake_agents.of("claude")["calls"] == []
    assert not (project / ".cursor").exists()
    assert not (setup_home.parent / "config.toml").exists()


def test_a_per_project_agent_without_a_project_is_skipped_not_failed(monkeypatch, setup_home, capsys):
    monkeypatch.chdir(setup_home)
    assert main(["setup", "--plan", "cursor"]) == 0
    assert "Cursor: skipped, no project directory (pass --project-dir)" in capsys.readouterr().out


def test_unknown_runtime_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["setup", "--plan", "notepad"])
    assert exit_info.value.code == 2
    assert "unknown runtime notepad" in capsys.readouterr().err


def test_bootstrap_is_gone(capsys):
    assert main(["bootstrap"]) == 2
    assert "Unknown command: bootstrap" in capsys.readouterr().err


def test_no_docker_and_nothing_running_says_how_to_go_on(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(memgraph.ProvisionError, match="Docker is needed") as error:
        memgraph.find_or_start(configured_url=None, default_port=1)
    assert any("--memgraph-url" in hint for hint in error.value.hints)


def test_an_unreachable_remote_memgraph_is_an_error_not_a_container(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # no docker: proves nothing local is even tried
    with pytest.raises(memgraph.ProvisionError, match="doesn't answer"):
        memgraph.find_or_start(configured_url="bolt://setup-test.invalid:7687")


# --- uninstall ----------------------------------------------------------------


def test_uninstall_finds_what_is_wired_and_keeps_data(fake_agents, setup_home, project, capsys):
    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    agents.wire_project(BY_RUNTIME["cursor"], project, lambda _: None)

    assert main(["uninstall", "--project-dir", str(project)]) == 0

    out = capsys.readouterr().out
    assert "OK    Claude Code: removed the plugin and marketplace" in out
    assert f"OK    Cursor: unwired {project.resolve()}" in out
    assert "docker rm -f context-graph-memgraph && docker volume rm context-graph-memgraph-data" in out
    assert fake_agents.of("claude")["plugins"] == []
    assert not (project / ".cursor" / "hooks.json").exists()
    assert fake_agents.of("codex")["calls"] == [["plugin", "list", "--json"]]  # looked, found nothing, left it


# --- the TUI ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_plan_screen_preselects_found_agents_and_follows_edits(fake_agents, setup_home, project):
    from textual.widgets import Checkbox, Input, SelectionList, Static

    from memgraph_context_graph.setup.tui import PlanScreen, SetupApp

    env = engine.look_around()
    app = SetupApp(engine.default_choices(env, cwd=project, user_id="carol"), env)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, PlanScreen)
        selection = screen.query_one("#agents", SelectionList)
        assert sorted(selection.selected) == ["claude-code", "codex"]
        plan = str(screen.query_one("#plan", Static).render())
        assert "Identity: identity.user_id = carol" in plan

        selection.deselect("codex")
        selection.select("cursor")
        screen.query_one("#user-id", Input).value = "dave"
        screen.query_one("#memory", Checkbox).value = True
        await pilot.pause()
        plan = str(screen.query_one("#plan", Static).render())
        assert "Codex" not in plan
        assert f"Cursor: wire {project.resolve()}" in plan
        assert "Identity: identity.user_id = dave" in plan
        assert "Memory: memory.backend = context-graph for Claude Code" in plan
        assert screen.read_choices().runtimes == ["claude-code", "cursor"]


@pytest.mark.asyncio
async def test_the_run_screen_shows_each_outcome_and_the_fixes(fake_agents, setup_home, project, monkeypatch):
    """The engine is stubbed here only: what it does for real is covered above and in the Docker tests."""
    from textual.widgets import DataTable, Static

    from memgraph_context_graph.setup.tui import RunScreen, SetupApp

    def fake_run(choices, env, log, report):
        outcomes = [
            engine.Outcome("Memgraph", "ok", "started Memgraph 3.13.1 at bolt://localhost:7687"),
            engine.Outcome(
                "Claude Code", "fail", "network unreachable", fix="memgraph-context-graph setup claude-code"
            ),
        ]
        log("docker run ...")
        for outcome in outcomes:
            report(outcome)
        return outcomes

    monkeypatch.setattr(engine, "run", fake_run)
    env = engine.look_around()
    app = SetupApp(engine.default_choices(env, cwd=project, user_id="erin"), env)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        steps = screen.query_one("#steps", DataTable)
        assert [str(steps.get_row_at(i)[1]) for i in range(steps.row_count)] == ["Memgraph", "Claude Code"]
        result = str(screen.query_one("#result", Static).render())
        assert "1 step(s) failed" in result
        assert "Claude Code: memgraph-context-graph setup claude-code" in result
        await pilot.press("q")
    assert app.return_value == 1
