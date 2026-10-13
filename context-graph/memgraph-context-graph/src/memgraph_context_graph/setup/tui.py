"""The interactive front end: one screen to choose, one to watch it run (#533).

It only fills in :class:`~memgraph_context_graph.setup.engine.Choices` and calls the same
:func:`~memgraph_context_graph.setup.engine.run` the plain front end does, from a worker thread so the
screen keeps drawing while Docker and uv work.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, DataTable, Footer, Header, Input, Label, RichLog, SelectionList, Static
from textual.widgets.selection_list import Selection

from memgraph_context_graph.setup import engine
from memgraph_context_graph.setup.agents import AGENTS

if TYPE_CHECKING:
    from memgraph_context_graph.setup.engine import Choices, Environment, Outcome

_MARKS = {"ok": ("✓", "green"), "skip": ("–", "dim"), "warn": ("!", "yellow"), "fail": ("✗", "red")}
_MEMORY_GRAPH = "context-graph"

CSS = """
Screen { padding: 0 1; }
.field { margin-top: 1; }
.hint { color: $text-muted; }
#agents { height: auto; max-height: 9; }
#plan { margin-top: 1; padding: 0 1; border: round $primary; height: auto; }
#buttons { height: auto; margin-top: 1; }
#buttons Button { margin-right: 2; }
#steps { height: auto; max-height: 14; }
#log { height: 1fr; min-height: 6; border: round $panel; }
#result { margin-top: 1; height: auto; }
"""


class PlanScreen(Screen[None]):
    """Choose what to install; the plan below updates as you go."""

    BINDINGS: ClassVar = [
        Binding("ctrl+r", "install", "Install", priority=True),
        Binding("escape", "app.quit", "Quit"),
    ]

    def __init__(self, choices: Choices, env: Environment) -> None:
        super().__init__()
        self.choices = choices
        self.env = env

    def compose(self) -> ComposeResult:
        yield Header()
        enterprise = self.env.edition == "enterprise"
        with VerticalScroll():
            yield Label("Memgraph", classes="field")
            yield Input(
                value=self.choices.memgraph_url or "",
                placeholder=self.env.configured_url or "find one on localhost:7687, or start one in Docker",
                id="memgraph-url",
                disabled=enterprise,
            )
            yield Label("Your name in the graph", classes="field")
            yield Input(value=self.choices.user_id, id="user-id", disabled=enterprise)
            yield Label("Agents (found on PATH are ticked; space to toggle)", classes="field")
            yield SelectionList[str](
                *(
                    Selection(
                        f"{agent.label}  ({'user-wide plugin' if agent.wiring == 'plugin' else 'per project'})"
                        + ("" if agent.binary() else "  — not found"),
                        agent.runtime,
                        agent.runtime in self.choices.runtimes,
                        disabled=agent.binary() is None and agent.runtime not in self.choices.runtimes,
                    )
                    for agent in AGENTS
                ),
                id="agents",
            )
            yield Label("Projects to wire for per-project agents (comma-separated)", classes="field")
            yield Input(value=", ".join(str(path) for path in self.choices.project_dirs), id="project-dirs")
            yield Checkbox("Install the terminal viewer (memgraph-context-graph ui)", self.choices.viewer, id="viewer")
            yield Checkbox(
                "Make Context Graph the memory of Claude Code and Codex (turns their own memory off)",
                self.choices.memory_backend == _MEMORY_GRAPH,
                id="memory",
            )
            yield Static(id="plan")
            with Horizontal(id="buttons"):
                yield Button("Install", variant="primary", id="install")
                yield Button("Quit", id="quit")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Context Graph setup"
        self.sub_title = "nothing changes until you press Install"
        self._show_plan()

    def on_input_changed(self) -> None:
        self._show_plan()

    def on_selection_list_selected_changed(self) -> None:
        self._show_plan()

    def on_checkbox_changed(self) -> None:
        self._show_plan()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "install":
            self.action_install()
        else:
            self.app.exit(0)

    def action_install(self) -> None:
        self.app.push_screen(RunScreen(self.read_choices(), self.env))

    def read_choices(self) -> Choices:
        """The choices as the widgets hold them now."""
        dirs = [part.strip() for part in self.query_one("#project-dirs", Input).value.split(",") if part.strip()]
        if self.query_one("#memory", Checkbox).value:
            memory_backend = _MEMORY_GRAPH
        else:  # unticked: no switch, unless the flags asked to switch back to native
            memory_backend = None if self.choices.memory_backend == _MEMORY_GRAPH else self.choices.memory_backend
        return engine.Choices(
            runtimes=list(self.query_one("#agents", SelectionList).selected),
            project_dirs=[Path(part).expanduser().resolve() for part in dirs],
            user_id=self.query_one("#user-id", Input).value.strip() or self.choices.user_id,
            memgraph_url=self.query_one("#memgraph-url", Input).value.strip() or None,
            memory_backend=memory_backend,
            viewer=self.query_one("#viewer", Checkbox).value,
            container=self.choices.container,
        )

    def _show_plan(self) -> None:
        lines = engine.plan(self.read_choices(), self.env)
        self.query_one("#plan", Static).update(Text("\n".join(["Plan", *(f"  {line}" for line in lines)])))


class RunScreen(Screen[None]):
    """Each step as it finishes, the commands' output, then the result."""

    BINDINGS: ClassVar = [Binding("q", "finish", "Close"), Binding("escape", "finish", "Close", show=False)]

    def __init__(self, choices: Choices, env: Environment) -> None:
        super().__init__()
        self.choices = choices
        self.env = env
        self.outcomes: list[Outcome] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="steps", cursor_type="none", show_header=False)
        yield RichLog(id="log", markup=False, wrap=True)
        yield Static(id="result")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Context Graph setup"
        self.sub_title = "installing"
        self.query_one("#steps", DataTable).add_columns("", "step", "result")
        self.query_one("#result", Static).update(Text("Working… (the first Memgraph start downloads ~3 GB)"))
        self.install()

    @work(thread=True, exclusive=True)
    def install(self) -> None:
        """Run the engine off the UI thread; every line and outcome is handed back to it."""
        outcomes = engine.run(
            self.choices,
            self.env,
            log=lambda line: self.app.call_from_thread(self._log, line),
            report=lambda outcome: self.app.call_from_thread(self._add, outcome),
        )
        self.app.call_from_thread(self._done, outcomes)

    def action_finish(self) -> None:
        if self.outcomes is not None:
            self.app.exit(1 if engine.failed(self.outcomes) else 0)

    def _log(self, line: str) -> None:
        self.query_one("#log", RichLog).write(Text(line))

    def _add(self, outcome: Outcome) -> None:
        mark, style = _MARKS[outcome.status]
        self.query_one("#steps", DataTable).add_row(Text(mark, style=style), outcome.step, Text(outcome.summary))
        for line in outcome.lines:
            self._log(line)

    def _done(self, outcomes: list[Outcome]) -> None:
        self.outcomes = outcomes
        failures = [outcome for outcome in outcomes if outcome.status == "fail"]
        self.sub_title = "done" if not failures else f"{len(failures)} failed"
        if not failures:
            text = Text(
                "Context Graph is set up. Use your agents as usual; `memgraph-context-graph ui` shows what they capture.",
                style="green",
            )
        else:
            text = Text(f"{len(failures)} step(s) failed. To fix:\n", style="red")
            for outcome in failures:
                for line in (outcome.fix or "memgraph-context-graph setup").splitlines():
                    text.append(f"  {outcome.step}: {line}\n", style="default")
        text.append("\nPress q to close.", style="dim")
        self.query_one("#result", Static).update(text)


class SetupApp(App[int]):
    """Plan, then run."""

    CSS = CSS

    def __init__(self, choices: Choices, env: Environment) -> None:
        super().__init__()
        self.choices = choices
        self.env = env

    def on_mount(self) -> None:
        self.push_screen(PlanScreen(self.choices, self.env))


def run_setup_app(choices: Choices, env: Environment) -> int:
    """Run the interactive setup; returns its exit code (0 when nothing failed, or when the user quit)."""
    return SetupApp(choices, env).run() or 0
