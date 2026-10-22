"""setup, update and uninstall without Docker: agents' wiring, the plan, the command line, and the questions.

The plugin agents run against the fake `claude`/`codex` from ``fake_agent_cli.py``; project wiring writes
real files into a temp project. The Memgraph step's real-Docker cases are in ``test_setup_docker.py``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from memgraph_context_graph.cli import main
from memgraph_context_graph.connectors import DEFAULT_CONNECTORS, INSTALLABLE
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


def test_agent_clis_never_read_setups_stdin(fake_agents, monkeypatch):
    """Claude Code on macOS crashes on the cbreak-mode terminal setup's questions leave, so a wired plugin read as
    not installed; every agent command gets a null stdin instead of inheriting setup's."""
    monkeypatch.setenv("FAKE_AGENT_NEEDS_NULL_STDIN", "1")
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        assert agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None) == "plugin installed"
        assert agents.plugin_wired(BY_RUNTIME["claude-code"])
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)


def test_doctor_fails_until_the_plugin_is_installed_and_enabled(fake_agents, monkeypatch):
    """doctor's runtime probe runs the connectors in its own process, so only this says the agent calls the hooks."""
    from memgraph_context_graph.cli import _check_plugin

    check = _check_plugin("claude-code")
    assert check is not None
    assert (check["ok"], check["detail"]) == (
        False,
        "not installed, so Claude Code records nothing — run: memgraph-context-graph setup claude-code",
    )

    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    assert _check_plugin("claude-code") == {
        "name": "plugin:claude-code",
        "ok": True,
        "detail": f"{agents.PLUGIN_ID} installed and enabled",
    }

    monkeypatch.setenv("FAKE_AGENT_DISABLED", "1")
    check = _check_plugin("claude-code")
    assert check is not None
    assert not check["ok"]
    assert f"claude plugin enable {agents.PLUGIN_ID}" in check["detail"]

    assert _check_plugin("cursor") is None


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


def test_no_agent_on_path_is_named_in_the_plan_and_fails_the_run(monkeypatch, setup_home, tmp_path, capsys):
    """Setup's checks follow the agents it wires: with none, every check passed and nothing was ever recorded."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    env = engine.look_around(cwd=setup_home)
    choices = engine.default_choices(env, cwd=setup_home, user_id="carol")
    assert choices.runtimes == []
    assert "Agents: no coding agent found on PATH, so nothing is recorded; install one, or name it" in engine.plan(
        choices, env
    )

    monkeypatch.setattr(engine, "_memgraph", lambda *args: (engine.Outcome("Memgraph", "ok", "found"), "bolt://x:1"))
    monkeypatch.setattr(engine, "_config", lambda *args: engine.Outcome("Config", "ok", "written"))
    monkeypatch.setattr(engine, "_packages", lambda *args: engine.Outcome("Packages", "ok", "all installed"))
    monkeypatch.setattr(engine, "verify", lambda runtimes: [])
    outcomes = engine.run(choices, env, log=lambda _: None, report=lambda _: None)
    assert engine.failed(outcomes)
    (agents_outcome,) = [outcome for outcome in outcomes if outcome.step == "Agents"]
    assert agents_outcome.fix == "memgraph-context-graph setup claude-code"


def test_the_configured_user_wins_over_git(setup_home):
    from memgraph_context_graph.adapters._identity import write_config

    write_config(user_id="alice", memgraph_url="bolt://localhost:7700")
    env = engine.look_around()
    assert (env.configured_user, env.configured_url) == ("alice", "bolt://localhost:7700")
    assert engine.default_choices(env).user_id == "alice"


def test_a_configured_default_url_is_not_a_choice_but_another_port_is(setup_home, capsys):
    from memgraph_context_graph.adapters._identity import write_config

    write_config(memgraph_url="bolt://localhost:7687")
    assert main(["setup", "--plan"]) == 0
    assert "Memgraph: use the context-graph-memgraph container, started if it isn't running" in capsys.readouterr().out

    write_config(memgraph_url="bolt://localhost:7700")
    assert main(["setup", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "Memgraph: use bolt://localhost:7700 (configured), else start the context-graph-memgraph container" in out


def test_plan_changes_nothing_and_names_every_step(fake_agents, setup_home, project, capsys):
    assert main(["setup", "--plan", "cursor", "claude-code", "--project-dir", str(project), "--user-id", "bob"]) == 0
    out = capsys.readouterr().out
    assert "Memgraph: use the context-graph-memgraph container, started if it isn't running" in out
    assert "Identity: identity.user_id = bob" in out
    assert f"Cursor: register the memory tools (user-wide) and wire {project.resolve()}" in out
    assert "Claude Code: install the Context Graph plugin (user-wide)" in out
    assert fake_agents.of("claude")["calls"] == [["plugin", "list", "--json"]]  # looked, changed nothing
    assert not (project / ".cursor").exists()
    assert not (setup_home.parent / "config.toml").exists()


def test_a_per_project_agent_without_a_project_is_skipped_not_failed(monkeypatch, setup_home, capsys):
    monkeypatch.chdir(setup_home)
    assert main(["setup", "--plan", "cursor"]) == 0
    assert "Cursor: register the memory tools (user-wide); no project to record (pass --project-dir)" in (
        capsys.readouterr().out
    )


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


def test_uninstall_deletes_the_config_and_log_only_when_asked(fake_agents, setup_home, project, capsys):
    from memgraph_context_graph import capture_log
    from memgraph_context_graph.adapters._identity import config_file, write_config

    write_config(user_id="alice", memgraph_url="bolt://localhost:7700")
    capture_log.record_failure("connect", error_type="ValueError", error="unreachable", connector=None)
    log = capture_log.path()
    assert config_file().exists() and log.exists()

    assert main(["uninstall", "--config", "--tool", "--project-dir", str(project)]) == 0

    out = capsys.readouterr().out
    assert f"OK    Config: deleted {config_file()}, {log}" in out
    assert "SKIP  Tool: not running as the uv tool" in out
    assert not config_file().exists() and not log.exists()
    assert "the graph:   docker rm -f context-graph-memgraph" in out
    assert "the config:" not in out and "the tool:" not in out


def test_without_docker_there_is_no_graph_of_ours_to_delete(fake_agents, setup_home, monkeypatch, capsys):
    import shutil

    which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: None if name == "docker" else which(name))
    assert main(["uninstall", "--graph"]) == 0
    assert "SKIP  Graph: Docker isn't installed, so setup never started a Memgraph here" in capsys.readouterr().out


def test_uninstall_asks_about_each_removal_and_keeps_by_default(fake_agents, setup_home):
    from memgraph_context_graph.setup.prompts import ENTER, Prompts, ask_removals
    from memgraph_context_graph.setup.uninstall import Removals

    everything = Removals(graph=True, config=True, tool=True)
    # Graph: Enter keeps it. Config: yes. Tool: Enter keeps it. Remove now: Enter.
    term = ScriptedTerminal(keys=[ENTER, "y", ENTER, ENTER])
    assert ask_removals(Prompts(term), ["claude-code"], everything, "cg") == Removals(config=True)
    assert "Delete the graph too?  y/N" in term.text
    assert "The cg container and its data: every session and memory in it, for good." in term.text
    assert "◇  Agents           unwire Claude Code" in term.text
    assert "Delete: config" in term.text


def test_esc_takes_back_a_yes_and_no_at_the_end_changes_nothing(fake_agents, setup_home):
    from memgraph_context_graph.setup.prompts import ENTER, ESC, Prompts, ask_removals
    from memgraph_context_graph.setup.uninstall import Removals

    everything = Removals(graph=True, config=True, tool=True)
    # Graph: yes, then Esc on the config question to take it back. Then Enter through, and no at the end.
    term = ScriptedTerminal(keys=["y", ESC, "n", ENTER, ENTER, ENTER])
    assert ask_removals(Prompts(term), ["claude-code"], everything, "cg") == Removals()
    term = ScriptedTerminal(keys=[ENTER, ENTER, ENTER, "n"])
    assert ask_removals(Prompts(term), ["claude-code"], everything, "cg") is None
    assert "└  Nothing changed." in term.text


def test_only_what_is_there_is_asked_about(fake_agents, setup_home):
    from memgraph_context_graph.setup.prompts import ENTER, Prompts, ask_removals
    from memgraph_context_graph.setup.uninstall import Removals

    term = ScriptedTerminal(keys=[ENTER, ENTER])
    assert ask_removals(Prompts(term), [], Removals(config=True), "cg") == Removals()
    assert "Delete the graph" not in term.text and "Uninstall the tool" not in term.text
    assert "◇  Agents           nothing is wired" in term.text

    term = ScriptedTerminal(keys=[])
    assert ask_removals(Prompts(term), [], Removals(), "cg") is None
    assert "└  Nothing to remove." in term.text


def test_the_interactive_uninstall_unwires_removes_and_lists_what_it_kept(fake_agents, setup_home, capsys):
    from memgraph_context_graph.adapters._identity import config_file, write_config
    from memgraph_context_graph.setup.prompts import ENTER, run_uninstall_prompts
    from memgraph_context_graph.setup.uninstall import Removals

    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    write_config(user_id="alice", memgraph_url="bolt://localhost:7700")
    # Config: yes. Remove now: Enter.
    term = ScriptedTerminal(keys=["y", ENTER])
    assert run_uninstall_prompts(["claude-code"], [], Removals(config=True), "cg", term) == 0

    assert fake_agents.of("claude")["plugins"] == []
    assert not config_file().exists()
    assert "✓  Claude Code      removed the plugin and marketplace" in term.text
    assert "the graph:   docker rm -f cg && docker volume rm cg-data" in term.text
    assert "└  Done." in term.text


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


# --- update -------------------------------------------------------------------


@pytest.fixture
def no_schema(monkeypatch):
    """Update's schema step and doctor checks need a Memgraph and the connectors; ``test_setup_docker.py`` has them."""
    monkeypatch.setattr("memgraph_context_graph.hooks.runner.setup_connector_schemas", lambda *a, **k: None)
    monkeypatch.setattr(engine, "verify", lambda runtimes: [])


@pytest.mark.parametrize("runtime, binary, verbs", [
    ("claude-code", "claude", [["plugin", "marketplace", "update", agents.MARKETPLACE], ["plugin", "update", agents.PLUGIN_ID]]),
    ("codex", "codex", [["plugin", "marketplace", "upgrade", agents.MARKETPLACE], ["plugin", "add", agents.PLUGIN_ID]]),
])  # fmt: skip
def test_update_refreshes_the_marketplace_and_the_installed_plugin(fake_agents, runtime, binary, verbs):
    agent = BY_RUNTIME[runtime]
    agents.wire_plugin(agent, lambda _: None)
    before = len(fake_agents.of(binary)["calls"])

    assert agents.update_plugin(agent, lambda _: None) == "plugin updated"

    calls = fake_agents.of(binary)["calls"][before:]
    assert [
        call for call in calls if call[:2] != ["plugin", "list"] and call[:3] != ["plugin", "marketplace", "list"]
    ] == verbs
    assert fake_agents.of(binary)["plugins"] == [agents.PLUGIN_ID]


def test_update_leaves_a_plugin_that_is_not_installed_alone(fake_agents):
    assert agents.update_plugin(BY_RUNTIME["claude-code"], lambda _: None) == "plugin not installed; nothing to update"
    assert fake_agents.of("claude")["calls"] == [["plugin", "list", "--json"]]


def test_update_rewires_projects_and_keeps_the_users_own_hooks(fake_agents, setup_home, project, no_schema, capsys):
    hooks = project / ".cursor" / "hooks.json"
    hooks.parent.mkdir()
    users = {"command": "./lint.sh"}
    hooks.write_text(json.dumps({"version": 1, "hooks": {"preToolUse": [users]}}))
    agents.wire_project(BY_RUNTIME["cursor"], project, lambda _: None)
    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    stale = json.loads(hooks.read_text())
    stale["hooks"]["preToolUse"][1]["command"] = "/old/venv/bin/memgraph-context-graph hook run cursor"
    hooks.write_text(json.dumps(stale))

    assert main(["update", "--no-upgrade", "--project-dir", str(project)]) == 0

    out = capsys.readouterr().out
    assert "OK    Claude Code: plugin updated" in out
    assert f"OK    Cursor: rewired {project.resolve()}" in out
    assert "OK    Schema: up to date on" in out
    assert "Restart running agent sessions" in out
    pre = json.loads(hooks.read_text())["hooks"]["preToolUse"]
    assert pre[0] == users and len(pre) == 2
    assert "/old/venv" not in pre[1]["command"]


def test_update_finds_nothing_wired_and_changes_nothing(fake_agents, setup_home, project, no_schema, capsys):
    assert main(["update", "--no-upgrade", "--project-dir", str(project)]) == 0
    assert "Nothing is wired" in capsys.readouterr().out
    assert fake_agents.of("claude")["plugins"] == []
    assert not (project / ".cursor").exists()


def test_update_reports_a_failed_plugin_and_still_refreshes_the_rest(
    fake_agents, setup_home, project, no_schema, monkeypatch, capsys
):
    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    agents.wire_plugin(BY_RUNTIME["codex"], lambda _: None)
    monkeypatch.setenv("FAKE_AGENT_FAIL", "install")  # codex re-adds the plugin, which now fails

    assert main(["update", "--no-upgrade", "--project-dir", str(project)]) == 1

    out = capsys.readouterr().out
    assert "OK    Claude Code: plugin updated" in out
    assert "FAIL  Codex:" in out and "network unreachable" in out
    assert "To fix Codex: memgraph-context-graph update codex" in out
    assert "Restart running agent sessions" not in out


def test_update_says_to_delete_duplicates_when_a_constraint_cannot_be_created(
    fake_agents, setup_home, project, monkeypatch, capsys
):
    def refuse(*args, **kwargs):
        raise RuntimeError(
            "Unable to create unique constraint :Action(action_id), because an existing node violates it."
        )

    monkeypatch.setattr("memgraph_context_graph.hooks.runner.setup_connector_schemas", refuse)
    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)

    assert main(["update", "--no-upgrade", "--project-dir", str(project)]) == 1

    out = capsys.readouterr().out
    assert "To fix Schema: nodes already in the graph share the value" in out
    assert "--no-upgrade" not in out


def test_update_says_when_the_package_is_not_the_uv_tool(fake_agents, setup_home, project, no_schema, capsys):
    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    assert main(["update", "--project-dir", str(project)]) == 0  # a test venv is not the uv tool
    assert "WARN  Packages: memgraph-context-graph" in capsys.readouterr().out


def test_upgrading_asks_uv_to_refresh_every_package(monkeypatch, default_connectors):
    from memgraph_context_graph.setup import tool

    seen = {}

    class Fake:
        stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr(tool.shutil, "which", lambda _: "/bin/uv")
    monkeypatch.setattr(tool.subprocess, "Popen", lambda argv, **_: seen.setdefault("argv", argv) and Fake())
    monkeypatch.setattr(tool, "installed", lambda package: False)

    tool.install(viewer=False, log=lambda _: None, upgrade=True)

    argv = seen["argv"]
    assert "--upgrade" in argv
    refreshed = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--refresh-package"]
    assert refreshed == ["memgraph-context-graph", "skills-graph", "actions-graph", "sessions-graph"]


# --- the questions --------------------------------------------------------------


class ScriptedTerminal:
    """Answers the prompts from a script of keypresses; "ctrl-c" interrupts as the real terminal would."""

    color = ""

    def __init__(self, keys=(), columns=100):
        self.columns = columns
        self.keys = list(keys)
        self.out: list[str] = []

    def write(self, text):
        self.out.append(text)

    def read_key(self):
        key = self.keys.pop(0)
        if key == "ctrl-c":
            raise KeyboardInterrupt
        return key

    def keypresses(self):
        return contextlib.nullcontext()

    @property
    def text(self):
        return "".join(self.out)


def typed(text):
    return list(text)


def test_a_first_run_ticks_the_plugin_agents_found_and_typing_replaces_the_default_name(
    fake_agents, setup_home, project
):
    from memgraph_context_graph.setup.prompts import DOWN, ENTER, SPACE, Prompts, ask

    env = engine.look_around()
    # Name: type "dave". Agents: Claude Code and Codex start ticked; untick Codex. Viewer: no.
    # Memory switch: default (no). Install: yes.
    term = ScriptedTerminal(keys=[*typed("dave"), ENTER, DOWN, SPACE, ENTER, "n", ENTER, ENTER])
    choices = ask(Prompts(term), engine.default_choices(env, cwd=project, user_id="carol"), env, preselect=False)

    assert choices is not None
    assert (choices.runtimes, choices.user_id, choices.viewer, choices.memory_backend) == (
        ["claude-code"],
        "dave",
        False,
        None,
    )
    assert "enter keeps carol · type to replace · esc back" in term.text
    assert "◇  Your name        dave" in term.text
    assert "◇  Agents           Claude Code\n" in term.text
    assert re.search(r"◻ Cursor +per project · not found", term.text)
    assert "Identity: identity.user_id = dave" in term.text
    assert fake_agents.of("claude")["calls"] == [["plugin", "list", "--json"]]  # looked, changed nothing


def test_enter_on_an_empty_field_keeps_the_default(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import BACKSPACE, ENTER, Prompts, ask

    env = engine.look_around()
    # Name: type "x", erase it, Enter. Agents, viewer, memory switch, install: Enter.
    term = ScriptedTerminal(keys=["x", BACKSPACE, ENTER, ENTER, ENTER, ENTER, ENTER])
    choices = ask(Prompts(term), engine.default_choices(env, cwd=project, user_id="carol"), env, preselect=False)

    assert choices is not None
    assert (choices.user_id, choices.runtimes) == ("carol", ["claude-code", "codex"])


def test_esc_steps_back_and_keeps_the_answers(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import DOWN, ENTER, ESC, SPACE, Prompts, ask

    env = engine.look_around()
    keys = [
        ESC,  # on the first question: stays there
        ENTER,  # name: carol
        DOWN,  # agents: both start ticked; untick Codex ...
        SPACE,
        ESC,  # ... back to the name before confirming: the ticks stay
        ENTER,
        ENTER,  # agents: Codex still unticked
        ESC,  # viewer: back to the agents, Claude Code still ticked
        ENTER,
        "n",  # viewer: no
        ENTER,  # memory switch: default (no)
        ESC,  # install now: back to the memory switch
        "y",
        ENTER,  # install now: yes
    ]
    term = ScriptedTerminal(keys=keys)
    choices = ask(Prompts(term), engine.default_choices(env, cwd=project, user_id="carol"), env, preselect=False)

    assert choices is not None
    assert (choices.user_id, choices.runtimes, choices.viewer, choices.memory_backend) == (
        "carol",
        ["claude-code"],
        False,
        "context-graph",
    )
    assert term.keys == []


def test_a_per_project_agent_asks_which_projects(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import ENTER, Prompts, ask

    env = engine.look_around()
    defaults = engine.default_choices(env, runtimes=["claude-code", "cursor"], cwd=project, user_id="carol")
    # Agents named on the command line start ticked. Everything else keeps its default but the memory switch.
    # Wide enough that the plan's line for the temp project's long path isn't cut short.
    term = ScriptedTerminal(keys=[ENTER, ENTER, ENTER, ENTER, "y", ENTER], columns=300)
    choices = ask(Prompts(term), defaults, env, preselect=True)

    assert choices is not None
    assert choices.runtimes == ["claude-code", "cursor"]
    assert choices.project_dirs == [project.resolve()]
    assert choices.memory_backend == "context-graph"
    assert f"Cursor: register the memory tools (user-wide) and wire {project.resolve()}" in term.text


def test_saying_no_or_ctrl_c_changes_nothing(fake_agents, setup_home, project, monkeypatch):
    from memgraph_context_graph.setup.prompts import ENTER, run_setup_prompts

    monkeypatch.setattr(engine, "run", lambda *args, **kwargs: pytest.fail("nothing should run"))
    env = engine.look_around()
    defaults = engine.default_choices(env, cwd=project, user_id="carol")

    declined = ScriptedTerminal(keys=[ENTER, ENTER, ENTER, ENTER, "n"])
    assert run_setup_prompts(defaults, env, declined) == 0
    assert "└  Nothing changed." in declined.text

    interrupted = ScriptedTerminal(keys=["ctrl-c"])
    assert run_setup_prompts(defaults, env, interrupted) == 0
    assert "└  Stopped. Nothing changed." in interrupted.text


def test_the_run_prints_each_outcome_and_the_fixes(fake_agents, setup_home, project, monkeypatch):
    """The engine is stubbed here only: what it does for real is covered above and in the Docker tests."""
    from memgraph_context_graph.setup.prompts import ENTER, run_setup_prompts

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
    term = ScriptedTerminal(keys=[ENTER] * 5)
    assert run_setup_prompts(engine.default_choices(env, cwd=project, user_id="erin"), env, term) == 1
    assert "│    docker run ..." in term.text
    assert "✓  Memgraph         started Memgraph 3.13.1" in term.text
    assert "✗  Claude Code      network unreachable" in term.text
    assert "Claude Code: memgraph-context-graph setup claude-code" in term.text


def test_the_wordmark_is_drawn_large_only_when_it_fits(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import WORDMARK, Prompts

    wide = ScriptedTerminal(columns=100)
    Prompts(wide).banner()
    assert all(row in wide.text for row in WORDMARK)

    narrow = ScriptedTerminal(columns=60)
    Prompts(narrow).banner()
    assert WORDMARK[0] not in narrow.text
    assert "memgraph" in narrow.text


def test_every_edge_of_the_banner_graph_joins_two_nodes_so_the_signals_can_walk_all_of_it():
    from memgraph_context_graph.setup.prompts import GRAPH, graph_edges

    paths = graph_edges(GRAPH)
    walked = [cell for node_paths in paths.values() for path in node_paths for cell in path[:-1]]
    glyphs = [(row, column) for row, line in enumerate(GRAPH) for column, char in enumerate(line) if char not in " ●"]
    # Each edge is found once from either end.
    assert sorted(walked) == sorted(glyphs * 2)

    reached, frontier = set(), [next(iter(paths))]
    while frontier:
        node = frontier.pop()
        if node not in reached:
            reached.add(node)
            frontier.extend(path[-1] for path in paths[node])
    assert reached == set(paths)


def test_the_graph_hangs_from_the_wordmark_and_every_signal_flows_down_a_stem_first():
    import random

    from memgraph_context_graph.setup.prompts import (
        _STEM_COLUMNS,
        BANNER_GRAPH,
        WORDMARK,
        _signal,
        graph_edges,
        hang,
        stems,
    )

    starts = stems(BANNER_GRAPH, _STEM_COLUMNS)
    for column, stem in zip(_STEM_COLUMNS, starts, strict=True):
        assert WORDMARK[-1][column] != " ", "a stem hangs from a letter"
        assert BANNER_GRAPH[stem[-1][0]][column] == "●"

    with pytest.raises(ValueError, match="column 1"):
        hang(["   ", "●──●"], [1], gap=1)

    paths = graph_edges(BANNER_GRAPH)
    for seed in range(20):
        cells = [cell for cell in itertools.islice(_signal(paths, starts, random.Random(seed)), 200) if cell]
        assert any(cells[: len(stem)] == stem for stem in starts)


def test_the_animated_banner_comes_to_rest_on_the_static_graph():
    from memgraph_context_graph.setup.prompts import Prompts

    class ColorTerminal(ScriptedTerminal):
        color = "truecolor"

    still = ColorTerminal()
    Prompts(still).banner()

    animated = ColorTerminal()
    with Prompts(animated).animated_banner():
        deadline = time.monotonic() + 5
        while len(animated.out) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert len(animated.out) >= 5, "no frames were drawn"
    assert animated.out[0:2] == still.out
    assert animated.out[-1].endswith(still.out[-1] + "\033[?25h")

    plain = ScriptedTerminal()
    with Prompts(plain).animated_banner():
        pass
    assert "\033" not in plain.text


def test_the_questions_start_apart_from_what_came_before(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import RULE, run_setup_prompts

    env = engine.look_around()
    term = ScriptedTerminal(keys=["ctrl-c"])
    run_setup_prompts(engine.default_choices(env, cwd=project, user_id="carol"), env, term)
    before_title = term.text.split("◆  Context Graph setup")[0]
    assert before_title.endswith(RULE + "\n\n")
    assert before_title.startswith("\n")


def test_the_divider_always_ends_on_a_node(fake_agents, setup_home):
    from memgraph_context_graph.setup.prompts import Prompts

    for columns in range(10, 100):
        term = ScriptedTerminal(columns=columns)
        Prompts(term).intro("Title")
        divider = term.text.strip().splitlines()[0]
        assert divider.endswith("●"), (columns, divider)


def test_a_second_run_starts_from_what_is_set_up(fake_agents, setup_home, project):
    from memgraph_context_graph.adapters._identity import write_config
    from memgraph_context_graph.setup.prompts import ENTER, Prompts, ask

    agents.wire_plugin(BY_RUNTIME["claude-code"], lambda _: None)
    write_config(user_id="carol", memgraph_url="bolt://localhost:7700", memory_backend="context-graph")
    env = engine.look_around(cwd=project)
    assert env.wired_agents == ("claude-code",)

    # Enter all the way: nothing the user didn't change is changed.
    term = ScriptedTerminal(keys=[ENTER] * 5, columns=200)
    choices = ask(Prompts(term), engine.default_choices(env, cwd=project), env, preselect=False)
    assert choices is not None
    assert (choices.runtimes, choices.memory_backend) == (["claude-code"], "context-graph")
    assert re.search(r"◻ Codex +user-wide plugin\n", term.text)
    assert re.search(r"◼ Claude Code +user-wide plugin · set up", term.text)
    assert "Claude Code: plugin already installed; checked again" in term.text
    assert "for Claude Code; already set, checked again" in term.text

    # Saying no to the memory question hands memory back rather than leaving it on.
    term = ScriptedTerminal(keys=[ENTER, ENTER, ENTER, "n", ENTER])
    choices = ask(Prompts(term), engine.default_choices(env, cwd=project), env, preselect=False)
    assert choices is not None
    assert choices.memory_backend == "native"


def test_the_uv_tool_is_recognised_even_when_the_shell_forces_colour(monkeypatch):
    """FORCE_COLOR makes uv colour what it prints into a pipe, which broke comparing its tool directory."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from memgraph_context_graph.setup import tool

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv")
    tool_dir = subprocess.run([uv, "--color", "never", "tool", "dir"], capture_output=True, text=True, check=True)
    monkeypatch.setenv("FORCE_COLOR", "3")
    monkeypatch.setattr(sys, "prefix", str(Path(tool_dir.stdout.strip()) / "memgraph-context-graph"))

    assert tool.uv_tool_environment()


def test_grok_is_reported_with_the_trust_it_still_needs(fake_agents, setup_home, project):
    env = engine.look_around(cwd=project)
    choices = engine.default_choices(env, cwd=project)

    grok = engine._agent(BY_RUNTIME["grok"], choices, lambda _: None)

    assert grok.status == "ok"
    assert "/hooks-trust" in grok.summary


def test_codex_is_reported_with_the_trust_it_still_needs_and_claude_code_is_not(fake_agents, setup_home, project):
    env = engine.look_around(cwd=project)
    choices = engine.default_choices(env, cwd=project)

    codex = engine._agent(BY_RUNTIME["codex"], choices, lambda _: None)
    claude = engine._agent(BY_RUNTIME["claude-code"], choices, lambda _: None)

    assert codex.status == "ok"
    assert "records nothing until you trust its hooks" in codex.summary
    assert "trust" not in claude.summary


def test_reinstalling_the_tool_keeps_the_spaces_package_once_it_is_there(monkeypatch, default_connectors):
    from memgraph_context_graph.setup import tool

    monkeypatch.setattr(tool, "installed", lambda package: package == "context-graph-spaces")

    argv = tool.requirements()

    assert tool.BASE in argv
    assert all(INSTALLABLE[name] in argv for name in DEFAULT_CONNECTORS)
    assert INSTALLABLE["resources-graph"] not in argv
    assert tool.SPACES[1] in argv
    assert tool.VIEWER[1] not in argv

    monkeypatch.setattr(tool, "installed", lambda package: False)
    assert tool.SPACES[1] not in tool.requirements()
    assert tool.SPACES[1] in tool.requirements(spaces=True)


def test_the_enterprise_hint_names_the_whole_install_not_just_the_spaces_package(monkeypatch, default_connectors):
    import pytest

    from memgraph_context_graph import spaces
    from memgraph_context_graph.adapters._identity import HookConfig

    monkeypatch.setattr(spaces, "_edition_factory", lambda edition: None)
    spaces._reset_cache()

    with pytest.raises(spaces.SpaceError) as excinfo:
        spaces.spaces_for(HookConfig(edition="enterprise"))

    hint = str(excinfo.value)
    assert "context-graph-spaces" in hint
    assert "skills-graph" in hint and "sessions-graph" in hint
    spaces._reset_cache()


def test_join_installs_the_spaces_package_inside_the_uv_tool_and_not_elsewhere(monkeypatch, setup_home, capsys):
    from memgraph_context_graph import spaces
    from memgraph_context_graph.setup import tool

    installs = []
    monkeypatch.setattr(spaces, "_edition_factory", lambda edition: None)
    monkeypatch.setattr(tool, "install", lambda **kwargs: installs.append(kwargs))

    monkeypatch.setattr(tool, "uv_tool_environment", lambda: False)
    spaces._reset_cache()
    assert main(["join", "bolt://localhost:1", "ante", "--password=x"]) == 1
    assert installs == []
    assert "uv tool install" in capsys.readouterr().err

    monkeypatch.setattr(tool, "uv_tool_environment", lambda: True)
    spaces._reset_cache()
    main(["join", "bolt://localhost:1", "ante", "--password=x"])
    assert [kwargs["spaces"] for kwargs in installs] == [True]
    spaces._reset_cache()


def test_the_welcome_says_what_the_user_is_getting_before_any_question(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import ENTER, Prompts, ask

    env = engine.look_around(cwd=project)
    term = ScriptedTerminal(keys=[ENTER] * 5, columns=100)

    ask(Prompts(term), engine.default_choices(env, cwd=project), env, preselect=False)

    first_question = term.text.index("Your name")
    for line in ("about to get a memory", "No account or cloud needed", "Nothing changes until you confirm"):
        assert term.text.index(line) < first_question


def _last_screen(choices, env=None):
    env = env or engine.Environment(edition="community", configured_url=None, configured_user=None, found_agents=())
    return engine.next_steps(choices, env, "bolt://localhost:7687", Path.home() / ".config" / "cg" / "config.toml")


def test_the_last_screen_says_what_is_set_up_what_to_expect_and_only_what_applies(tmp_path):
    choices = engine.Choices(
        runtimes=["claude-code", "codex", "grok"],
        project_dirs=[tmp_path / "billing"],
        user_id="carol",
        memory_backend="context-graph",
    )

    steps = _last_screen(choices)

    assert dict(steps.setup) == {
        "Memgraph": "bolt://localhost:7687",
        "You": "carol",
        "Agents": "Claude Code, Codex, Grok Build in billing",
        "Memory": "the graph, shared by every agent",
        "Config": str(Path("~") / ".config" / "cg" / "config.toml"),
    }
    tips = "\n".join(steps.tips)
    assert "what did we do last time?" in tips
    assert 'Say "remember that ..."' in tips
    assert "in Codex, open it and accept its hooks" in tips
    assert "in Grok, run /hooks-trust" in tips
    assert [name for name, _ in steps.commands] == ["ui", "recall <question>", "doctor", "setup", "uninstall"]

    quiet = _last_screen(engine.Choices(runtimes=["opencode"], project_dirs=[], user_id="carol", viewer=False))

    assert dict(quiet.setup)["Memory"] == "each agent's own"
    assert dict(quiet.setup)["Agents"] == "OpenCode"
    assert [tip.splitlines()[0] for tip in quiet.tips] == ["Finish a session, start another, and ask"]
    assert "ui" not in dict(quiet.commands)


def test_every_line_of_the_welcome_and_the_plain_last_screen_fits_a_narrow_terminal():
    choices = engine.Choices(
        runtimes=["claude-code", "codex", "grok"], project_dirs=[], user_id="carol", memory_backend="context-graph"
    )

    for line in (*engine.WELCOME, *_last_screen(choices).lines()):
        assert len(line) <= 78, line


@pytest.mark.parametrize("columns", [53, 60, 80, 200])
def test_the_last_screen_is_a_graph_shaped_box_that_fits_the_terminal(columns):
    from memgraph_context_graph.setup.prompts import Prompts, _card_sections

    class ColorTerminal(ScriptedTerminal):
        color = "truecolor"

    steps = _last_screen(
        engine.Choices(
            runtimes=["claude-code", "codex"], project_dirs=[], user_id="carol", memory_backend="context-graph"
        )
    )
    term = ColorTerminal(columns=columns)

    Prompts(term).card(steps.heading, _card_sections(steps))

    rows = re.sub(r"\033\[[0-9;]*m", "", term.text).splitlines()
    assert {len(row) for row in rows} == {min(columns - 1, 76)}
    edges = [row for row in rows if row.startswith("●")]
    assert [rows[0], rows[-1]] == [edges[0], edges[-1]] and len(edges) == 4
    assert all(row.endswith("●") and "●●" not in row and "●─●─●" not in row for row in edges)
    assert all(row[:2] in {"│ ", "├─"} and row.endswith(" │") for row in rows if row not in edges)
    assert "●─● Your agents now remember ●" in rows[0]
    assert any(row.startswith("├─● Memgraph  bolt://localhost:7687") for row in rows)
    commands = rows.index(next(row for row in rows if row.startswith("├─● memgraph-context-graph ")))
    assert "Use and configure it" in rows[commands - 2]
    subcommands = [row for row in rows[commands + 1 : -1] if "─● " in row]
    assert all(row.startswith("│   ") for row in subcommands)
    assert [re.sub(r"  .*", "", row[4:]) for row in subcommands] == [
        "├─● ui",
        "├─● recall <question>",
        "├─● doctor",
        "├─● setup",
        "└─● uninstall",
    ]
    assert any(row.startswith('│   "what did we do last time?"') for row in rows)


def test_a_last_screen_too_narrow_for_its_frame_keeps_every_line_of_it():
    from memgraph_context_graph.setup.prompts import Prompts, _card_sections

    steps = _last_screen(engine.Choices(runtimes=["claude-code"], project_dirs=[], user_id="carol"))
    term = ScriptedTerminal(columns=40)

    Prompts(term).card(steps.heading, _card_sections(steps))

    assert "●" not in term.text and "│" not in term.text
    assert all(len(row) <= 39 for row in term.text.splitlines())
    for text in ("Your agents now remember", "Memgraph: bolt://localhost:7687", "uninstall: undo the wiring"):
        assert text in term.text
    assert "config.toml" in term.text


def test_a_listing_that_fails_once_does_not_make_a_wired_agent_look_unwired(monkeypatch):
    listing = json.dumps([{"id": agents.PLUGIN_ID}])
    replies = iter(
        [
            subprocess.CompletedProcess([], 1, "", "busy"),
            subprocess.CompletedProcess([], 0, f"Checking for updates...\n{listing}", ""),
        ]
    )
    monkeypatch.setattr(agents.subprocess, "run", lambda *a, **k: next(replies))
    monkeypatch.setattr(agents.time, "sleep", lambda _: None)
    monkeypatch.setattr(agents.shutil, "which", lambda name: f"/bin/{name}")

    assert agents.plugin_wired(BY_RUNTIME["claude-code"])


def test_codex_wiring_lets_recall_through_a_never_policy_and_unwiring_takes_it_back(fake_agents, setup_home):
    import tomlkit

    config = setup_home / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('approval_policy = "never"\n\n[features]\nhooks = true\n')
    codex = BY_RUNTIME["codex"]

    agents.wire_plugin(codex, lambda _: None)
    agents.wire_plugin(codex, lambda _: None)

    parsed = tomlkit.parse(config.read_text()).unwrap()
    assert parsed["plugins"][agents.PLUGIN_ID]["mcp_servers"]["context-graph"] == {
        "default_tools_approval_mode": "approve"
    }
    assert parsed["approval_policy"] == "never" and parsed["features"] == {"hooks": True}

    agents.unwire_plugin(codex, lambda _: None)

    assert tomlkit.parse(config.read_text()).unwrap() == {"approval_policy": "never", "features": {"hooks": True}}


def test_codex_wiring_keeps_an_approval_mode_the_user_chose(fake_agents, setup_home):
    import tomlkit

    config = setup_home / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        f'[plugins."{agents.PLUGIN_ID}".mcp_servers.context-graph]\ndefault_tools_approval_mode = "prompt"\n'
    )

    agents.wire_plugin(BY_RUNTIME["codex"], lambda _: None)
    agents.unwire_plugin(BY_RUNTIME["codex"], lambda _: None)

    mode = tomlkit.parse(config.read_text()).unwrap()["plugins"][agents.PLUGIN_ID]["mcp_servers"]["context-graph"]
    assert mode == {"default_tools_approval_mode": "prompt"}
