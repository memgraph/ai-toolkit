"""setup and uninstall without Docker: agents' wiring, the plan, the command line, and the questions.

The plugin agents run against the fake `claude`/`codex` from ``fake_agent_cli.py``; project wiring writes
real files into a temp project. The Memgraph step's real-Docker cases are in ``test_setup_docker.py``.
"""

from __future__ import annotations

import contextlib
import json
import re

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
    assert fake_agents.of("claude")["calls"] == [["plugin", "list", "--json"]]  # looked, changed nothing
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


def test_nothing_is_ticked_and_typing_replaces_the_default_name(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import ENTER, SPACE, Prompts, ask

    env = engine.look_around()
    # Name: type "dave". Agents: tick Claude Code. Viewer: no. Memory switch: default (no). Install: yes.
    term = ScriptedTerminal(keys=[*typed("dave"), ENTER, SPACE, ENTER, "n", ENTER, ENTER])
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
    # Name: type "x", erase it, Enter. Agents: none. Viewer, install: Enter.
    term = ScriptedTerminal(keys=["x", BACKSPACE, ENTER, ENTER, ENTER, ENTER])
    choices = ask(Prompts(term), engine.default_choices(env, cwd=project, user_id="carol"), env, preselect=False)

    assert choices is not None
    assert (choices.user_id, choices.runtimes) == ("carol", [])


def test_esc_steps_back_and_keeps_the_answers(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import DOWN, ENTER, ESC, SPACE, Prompts, ask

    env = engine.look_around()
    keys = [
        ESC,  # on the first question: stays there
        ENTER,  # name: carol
        SPACE,  # agents: tick Claude Code and Codex ...
        DOWN,
        SPACE,
        ESC,  # ... back to the name before confirming: the ticks stay
        ENTER,
        DOWN,  # agents: both still ticked, so untick Codex
        SPACE,
        ENTER,
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
    assert f"Cursor: wire {project.resolve()}" in term.text


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


def test_no_banner_leaves_the_wordmark_to_whoever_drew_it(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import WORDMARK, run_setup_prompts

    env = engine.look_around()
    term = ScriptedTerminal(keys=["ctrl-c"])
    run_setup_prompts(engine.default_choices(env, cwd=project, user_id="carol"), env, term, banner=False)
    assert WORDMARK[0] not in term.text
    assert "◆  Context Graph setup" in term.text


def test_the_questions_start_apart_from_what_came_before(fake_agents, setup_home, project):
    from memgraph_context_graph.setup.prompts import RULE, run_setup_prompts

    env = engine.look_around()
    term = ScriptedTerminal(keys=["ctrl-c"])
    run_setup_prompts(engine.default_choices(env, cwd=project, user_id="carol"), env, term, banner=False)
    before_title = term.text.split("◆  Context Graph setup")[0]
    assert before_title.endswith(RULE[:71] + "\n\n")
    assert before_title.startswith("\n")


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
