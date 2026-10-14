"""What setup will do and does: the choices, the steps they turn into, and the outcome of each.

Both front ends (plain, and the questions) build :class:`Choices`, show :func:`plan` and call :func:`run`; neither
does any work itself. Every step checks before it changes anything, so running setup again reports
"already" instead of redoing work (#533).
"""

from __future__ import annotations

import getpass
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from memgraph_context_graph.setup import agents as agents_mod
from memgraph_context_graph.setup import memgraph as memgraph_mod
from memgraph_context_graph.setup import tool as tool_mod

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Status = Literal["ok", "skip", "warn", "fail"]


@dataclass
class Choices:
    """Everything setup lets the user decide; each field is also a flag."""

    runtimes: list[str]
    project_dirs: list[Path]
    user_id: str
    memgraph_url: str | None = None
    memory_backend: str | None = None
    viewer: bool = True
    container: str = memgraph_mod.CONTAINER


@dataclass(frozen=True)
class Outcome:
    """What one step did."""

    step: str
    status: Status
    summary: str
    #: The command that retries or fixes this step, for failures.
    fix: str = ""
    lines: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Environment:
    """What setup found before asking anything."""

    edition: str
    configured_url: str | None
    configured_user: str | None
    found_agents: tuple[str, ...]
    #: Found agents that already capture: their plugin is installed, or the current project is wired.
    wired_agents: tuple[str, ...] = ()
    configured_memory_backend: str | None = None


def look_around(cwd: Path | None = None) -> Environment:
    """Read the config file, PATH and what is already wired (plugins, and *cwd*'s hook files); changes nothing."""
    from memgraph_context_graph.adapters._identity import load_config

    config = load_config()
    cwd = (cwd or Path.cwd()).resolve()
    found = agents_mod.detected()
    return Environment(
        edition=config.edition or "community",
        configured_url=config.memgraph_url if "memgraph.url" in config.present else None,
        configured_user=config.user_id if "identity.user_id" in config.present and config.user_id else None,
        found_agents=tuple(agent.runtime for agent in found),
        wired_agents=tuple(agent.runtime for agent in found if _wired(agent, cwd)),
        configured_memory_backend=config.memory_backend,
    )


def _wired(agent: agents_mod.Agent, cwd: Path) -> bool:
    if agent.wiring == "plugin":
        return agents_mod.plugin_wired(agent)
    return cwd != Path.home().resolve() and agents_mod.project_wired(agent, cwd)


def default_choices(
    env: Environment,
    *,
    runtimes: Sequence[str] = (),
    project_dirs: Sequence[Path] = (),
    user_id: str | None = None,
    memgraph_url: str | None = None,
    memory_backend: str | None = None,
    viewer: bool = True,
    container: str = memgraph_mod.CONTAINER,
    cwd: Path | None = None,
) -> Choices:
    """Fill in what wasn't asked for: found agents, the current project, the configured or git identity."""
    cwd = (cwd or Path.cwd()).resolve()
    dirs = [path.expanduser().resolve() for path in project_dirs]
    if not dirs and cwd != Path.home().resolve():
        dirs = [cwd]
    return Choices(
        runtimes=list(runtimes) or list(env.found_agents),
        project_dirs=dirs,
        user_id=user_id or env.configured_user or default_user_id(),
        memgraph_url=memgraph_url,
        memory_backend=memory_backend,
        viewer=viewer,
        container=container,
    )


def default_user_id() -> str:
    """Git's ``user.name``, else the login name."""
    try:
        name = subprocess.run(
            ["git", "config", "--get", "user.name"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        name = ""
    return name or os.environ.get("USER") or getpass.getuser()


def plan(choices: Choices, env: Environment) -> list[str]:
    """What :func:`run` will do with *choices*, one line per step, without doing any of it."""
    lines = []
    if env.edition == "enterprise":
        lines.append("Memgraph and identity: from your Enterprise login (edition = enterprise); unchanged")
    else:
        if choices.memgraph_url:
            lines.append(f"Memgraph: use {choices.memgraph_url}")
        elif env.configured_url:
            lines.append(
                f"Memgraph: use {env.configured_url} (configured), else start the {choices.container} container"
            )
        else:
            lines.append(f"Memgraph: use one on localhost:7687, else start the {choices.container} container")
        lines.append(f"Identity: identity.user_id = {choices.user_id}")
    packages = tool_mod.missing(viewer=choices.viewer)
    lines.append(f"Packages: install {', '.join(packages)}" if packages else "Packages: all installed")
    for runtime in choices.runtimes:
        agent = agents_mod.BY_RUNTIME[runtime]
        if agent.wiring == "plugin" and runtime in env.wired_agents:
            lines.append(f"{agent.label}: plugin already installed; checked again")
        elif agent.wiring == "plugin":
            lines.append(f"{agent.label}: install the Context Graph plugin (user-wide)")
        elif choices.project_dirs:
            lines.append(f"{agent.label}: wire {', '.join(str(path) for path in choices.project_dirs)}")
        else:
            lines.append(f"{agent.label}: skipped, no project directory (pass --project-dir)")
    if choices.memory_backend:
        switchable = [
            agents_mod.BY_RUNTIME[runtime].label for runtime in choices.runtimes if has_native_memory(runtime)
        ]
        already = "; already set, checked again" if choices.memory_backend == env.configured_memory_backend else ""
        if switchable:
            lines.append(f"Memory: memory.backend = {choices.memory_backend} for {' and '.join(switchable)}{already}")
        else:
            lines.append("Memory: no selected agent has a built-in memory to switch")
    lines.append("Verify: doctor's checks, and each agent's hook probe")
    return lines


def run(
    choices: Choices, env: Environment, log: Callable[[str], None], report: Callable[[Outcome], None]
) -> list[Outcome]:
    """Carry out *choices*, reporting each step as it finishes; returns every outcome.

    A Memgraph that can't be found or started stops the run, since nothing after it could be verified;
    every agent step is independent of the others.
    """
    outcomes: list[Outcome] = []

    def done(outcome: Outcome) -> Outcome:
        outcomes.append(outcome)
        report(outcome)
        return outcome

    if env.edition == "enterprise":
        done(Outcome("Memgraph", "skip", "Enterprise: your login decides the Memgraph and the user"))
    else:
        outcome, url = _memgraph(choices, env, log)
        done(outcome)
        if url is None:
            return outcomes
        done(_config(choices, url))
    done(_packages(choices, log))
    for runtime in choices.runtimes:
        done(_agent(agents_mod.BY_RUNTIME[runtime], choices, log))
    if choices.memory_backend:
        for runtime in choices.runtimes:
            if (outcome := _memory(runtime, choices.memory_backend)) is not None:
                done(outcome)
    for outcome in _verify(choices):
        done(outcome)
    return outcomes


def has_native_memory(runtime: str) -> bool:
    """Whether *runtime* has a built-in memory that ``--memory-backend`` can switch."""
    from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin
    from memgraph_context_graph.memory_backend import native_memory_of

    return native_memory_of(get_runtime_plugin(runtime)) is not None


def failed(outcomes: Sequence[Outcome]) -> bool:
    """Whether any step failed."""
    return any(outcome.status == "fail" for outcome in outcomes)


# --- steps --------------------------------------------------------------------


def _memgraph(choices: Choices, env: Environment, log: Callable[[str], None]) -> tuple[Outcome, str | None]:
    """The Memgraph step's outcome, and the URL hooks should use (None when there is none)."""
    from memgraph_context_graph.adapters._identity import load_config

    config = load_config()
    try:
        found = memgraph_mod.find_or_start(
            configured_url=env.configured_url,
            explicit_url=choices.memgraph_url,
            user=config.memgraph_user,
            password=config.memgraph_password,
            container=choices.container,
            log=log,
        )
    except memgraph_mod.ProvisionError as exc:
        return Outcome("Memgraph", "fail", str(exc), fix="\n".join(exc.hints)), None
    summary = found.describe() + "".join(f"; {note}" for note in found.notes)
    return Outcome("Memgraph", "ok", summary), found.url


def _config(choices: Choices, url: str) -> Outcome:
    from memgraph_context_graph import spaces
    from memgraph_context_graph.adapters._identity import config_file, write_config

    write_config(memgraph_url=url, user_id=choices.user_id)
    spaces._reset_cache()
    return Outcome("Config", "ok", f"{config_file()}: memgraph.url = {url}, identity.user_id = {choices.user_id}")


def _packages(choices: Choices, log: Callable[[str], None]) -> Outcome:
    missing = tool_mod.missing(viewer=choices.viewer)
    if not missing:
        return Outcome("Packages", "ok", "already installed")
    if not tool_mod.uv_tool_environment():
        return Outcome(
            "Packages",
            "warn",
            f"missing {', '.join(missing)}, and this isn't the memgraph-context-graph uv tool, so install them yourself",
        )
    try:
        tool_mod.install(viewer=choices.viewer, log=log)
    except tool_mod.ToolError as exc:
        connectors_missing = [package for package in missing if package in tool_mod.CONNECTORS]
        if connectors_missing:
            return Outcome("Packages", "fail", str(exc), fix="memgraph-context-graph setup --no-tui")
        # Only the viewer was missing: capture works without it.
        return Outcome("Packages", "warn", f"the viewer didn't install ({exc}); capture works without it")
    return Outcome("Packages", "ok", f"installed {', '.join(missing)}")


def _agent(agent: agents_mod.Agent, choices: Choices, log: Callable[[str], None]) -> Outcome:
    fix = f"memgraph-context-graph setup {agent.runtime}"
    if agent.wiring == "plugin":
        try:
            return Outcome(agent.label, "ok", _with_trust_note(agent, agents_mod.wire_plugin(agent, log)))
        except agents_mod.WiringError as exc:
            return Outcome(agent.label, "fail", str(exc), fix=fix)
    if not choices.project_dirs:
        return Outcome(agent.label, "skip", "no project directory; run setup inside a project or pass --project-dir")
    summaries, errors = [], []
    for project_dir in choices.project_dirs:
        try:
            summaries.append(agents_mod.wire_project(agent, project_dir, log))
        except agents_mod.WiringError as exc:
            errors.append(str(exc))
    if errors:
        dirs = " ".join(f"--project-dir {path}" for path in choices.project_dirs)
        return Outcome(agent.label, "fail", "; ".join(errors + summaries), fix=f"{fix} {dirs}")
    return Outcome(agent.label, "ok", _with_trust_note(agent, "; ".join(summaries)))


#: Agents that skip hooks they haven't been told to trust, without a word: nothing is recorded until then.
_TRUST_NOTES = {
    "codex": "Codex records nothing until you trust its hooks: open codex and accept them (/hooks)",
    "grok": "Grok records nothing until you trust this project's hooks: run /hooks-trust in grok",
}


def _with_trust_note(agent: agents_mod.Agent, summary: str) -> str:
    note = _TRUST_NOTES.get(agent.runtime)
    return f"{summary}; {note}" if note else summary


def _memory(runtime: str, backend: str) -> Outcome | None:
    from memgraph_context_graph.adapters._identity import MEMORY_BACKEND_GRAPH
    from memgraph_context_graph.hooks.runtime_plugin import get_runtime_plugin
    from memgraph_context_graph.memory_backend import (
        MemoryBackendError,
        native_memory_of,
        use_graph_memory,
        use_native_memory,
    )

    plugin = get_runtime_plugin(runtime)
    if native_memory_of(plugin) is None:
        return None
    label = f"{agents_mod.BY_RUNTIME[runtime].label} memory"
    switch = use_graph_memory if backend == MEMORY_BACKEND_GRAPH else use_native_memory
    try:
        lines = switch(plugin)
    except MemoryBackendError as exc:
        return Outcome(
            label, "fail", str(exc), fix=f"memgraph-context-graph setup {runtime} --memory-backend {backend}"
        )
    return Outcome(label, "ok", f"memory.backend = {backend}", lines=tuple(lines))


def _verify(choices: Choices) -> list[Outcome]:
    from memgraph_context_graph.cli import doctor_checks

    checks = doctor_checks(choices.runtimes, list(agents_mod.CONNECTORS))
    outcomes = []
    for check in checks:
        if check.get("info"):
            continue
        status: Status = "ok" if check["ok"] else "fail"
        outcomes.append(Outcome(f"Check {check['name']}", status, check["detail"], fix="memgraph-context-graph doctor"))
    return outcomes
