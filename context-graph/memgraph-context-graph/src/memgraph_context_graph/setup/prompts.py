"""The interactive front end: a few questions in the terminal's own flow, then the run, line by line.

Like the plain front end it only fills in :class:`~memgraph_context_graph.setup.engine.Choices` and calls
:func:`~memgraph_context_graph.setup.engine.run`. Nothing takes over the screen: each answered question
collapses to one line and stays in the scrollback, so what setup asked and did is still there afterwards.
Esc steps back to the previous question, with its answer kept. Only the name, the agents, the viewer and
(where they apply) the projects and the memory switch are asked; the Memgraph URL and the rest stay flags.
"""

from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from memgraph_context_graph.adapters._identity import MEMORY_BACKEND_GRAPH, MEMORY_BACKEND_NATIVE
from memgraph_context_graph.setup import engine
from memgraph_context_graph.setup.agents import AGENTS, BY_RUNTIME

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from contextlib import AbstractContextManager

    from memgraph_context_graph.setup.engine import Choices, Environment, Outcome

_MEMORY_GRAPH = MEMORY_BACKEND_GRAPH
_LABEL_WIDTH = 16
# "◆  " plus the padded label and a space: where an answer starts on its line.
_ANSWER_COLUMN = 3 + _LABEL_WIDTH + 1

# Key names a Terminal returns from read_key(); anything else is the character typed.
UP, DOWN, SPACE, ENTER, ESC, BACKSPACE, CLEAR = "up", "down", "space", "enter", "esc", "backspace", "ctrl-u"


class _Back:
    """What a question returns when the user pressed Esc."""


BACK = _Back()


class Terminal(Protocol):
    """What the prompts need from a terminal; tests script one."""

    #: "truecolor", "256" or "" (no escapes at all: not a terminal, or NO_COLOR).
    color: str
    columns: int

    def write(self, text: str) -> None:
        """Write *text* as is, flushing it."""

    def read_key(self) -> str:
        """One keypress: one of the key names above, or the character typed. Ctrl+C raises KeyboardInterrupt."""

    def keypresses(self) -> AbstractContextManager[object]:
        """Hold the terminal in key-by-key mode for a whole session of questions."""


# Memgraph's brand tokens, with the nearest xterm-256 colour for terminals without truecolor. Orange is the
# accent only (the wordmark, the active question, the cursor); grays carry everything else, and green, red
# and yellow appear only as a step's status.
_PALETTE = {
    "orange": ((0xFB, 0x6E, 0x00), 208),
    "muted": ((0x85, 0x7F, 0x87), 245),  # gray-2
    "faint": ((0xBA, 0xB8, 0xBB), 250),  # gray-3: placeholders, disabled rows
    "success": ((0x19, 0xAF, 0x66), 35),
    "error": ((0xDC, 0x22, 0x23), 160),
    "warning": ((0xFF, 0xC5, 0x00), 220),
}
# The brand gradient (yellow, orange, red, purple), used once: on the nodes of the graph under the wordmark.
_GRADIENT = ((0xFF, 0xC5, 0x00), (0xFB, 0x6E, 0x00), (0xDC, 0x22, 0x23), (0x72, 0x00, 0x96))
# The nearest xterm-256 colour of each stop, for terminals without truecolor.
_GRADIENT_XTERM = (220, 208, 160, 90)

WORDMARK = (
    "███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗",
    "████╗ ████║██╔════╝████╗ ████║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║",
    "██╔████╔██║█████╗  ██╔████╔██║██║  ███╗██████╔╝███████║██████╔╝███████║",
    "██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║",
    "██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║",
    "╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝",
)
# Under the wordmark, as wide as it: a graph of nodes joined by edges.
GRAPH = (
    "●───────────────●                 ●─────────────────●─────────────────●",
    " ╲             ╱ ╲               ╱ ╲               ╱ ╲               ╱",
    "  ●           ●───●             ●───●             ●───●             ●",
    "   ╲         ╱     ╲           ╱     ╲           ╱     ╲           ╱",
    "    ●───────●       ●─────────●       ●─────────●       ●─────────●",
    "     ╲     ╱       ╱         ╱         ╲         ╲     ╱         ╱",
    "      ●───●           ●─────●           ●─────●           ●─────●",
)
# The divider between the banner and the first question: a short stretch of graph, in orange.
RULE = "●──────●─────────●────●───────●──────────●─────●────────●───────●─────●"


class _Style:
    def __init__(self, color: str) -> None:
        self.color = color

    def _fg(self, rgb: tuple[int, int, int], xterm: int, text: str) -> str:
        if self.color == "truecolor":
            return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"
        if self.color:
            return f"\033[38;5;{xterm}m{text}\033[0m"
        return text

    def _token(self, name: str, text: str) -> str:
        rgb, xterm = _PALETTE[name]
        return self._fg(rgb, xterm, text)

    def bold(self, text: str) -> str:
        return f"\033[1m{text}\033[0m" if self.color else text

    def accent(self, text: str) -> str:
        return self._token("orange", text)

    def muted(self, text: str) -> str:
        return self._token("muted", text)

    def faint(self, text: str) -> str:
        return self._token("faint", text)

    def success(self, text: str) -> str:
        return self._token("success", text)

    def error(self, text: str) -> str:
        return self._token("error", text)

    def warning(self, text: str) -> str:
        return self._token("warning", text)

    def wordmark(self, line: str) -> str:
        """A wordmark line: the letters' faces in orange, their drop shadow in gray."""
        if not self.color:
            return line
        return "".join(self.accent(char) if char == "█" else self.muted(char) if char != " " else char for char in line)

    def graph(self, line: str, width: int) -> str:
        """A line of :data:`GRAPH`: nodes in the brand gradient, edges in gray."""
        if not self.color:
            return line
        cells = []
        for column, char in enumerate(line):
            if char == " ":
                cells.append(char)
            elif char != "●":
                cells.append(self.muted(char))
            else:
                position = column / max(width - 1, 1)
                rgb = _gradient_at(position)
                xterm = _GRADIENT_XTERM[round(position * (len(_GRADIENT) - 1))]
                cells.append(self._fg(rgb, xterm, char))
        return "".join(cells)


def _gradient_at(position: float) -> tuple[int, int, int]:
    """The brand gradient's colour at *position*, from 0 (yellow) through red to 1 (purple)."""
    scaled = position * (len(_GRADIENT) - 1)
    start = min(int(scaled), len(_GRADIENT) - 2)
    fraction = scaled - start
    a, b = _GRADIENT[start], _GRADIENT[start + 1]
    return (
        round(a[0] + (b[0] - a[0]) * fraction),
        round(a[1] + (b[1] - a[1]) * fraction),
        round(a[2] + (b[2] - a[2]) * fraction),
    )


@dataclass
class _Option:
    value: str
    label: str
    note: str
    selected: bool
    enabled: bool


class Prompts:
    """Draws the questions on *terminal* and reads the answers.

    It counts the lines it has written, so that stepping back can erase exactly what was drawn since a
    question was asked. Lines are kept shorter than the terminal for the same reason: a wrapped line would
    throw the count off.
    """

    def __init__(self, terminal: Terminal) -> None:
        self.term = terminal
        self.style = _Style(terminal.color)
        self.written = 0

    # --- drawing ----------------------------------------------------------------

    def write(self, text: str) -> None:
        self.term.write(text)
        self.written += text.count("\n")

    def erase_since(self, mark: int) -> None:
        """Clear every line written since :attr:`written` was *mark*, leaving the cursor where it was then."""
        lines = self.written - mark
        if self.term.color and lines > 0:
            self.term.write(f"\033[{lines}F\033[J")
        self.written = mark

    def fit(self, text: str, indent: int = 3) -> str:
        """*text* cut to what fits on one line after *indent* columns."""
        room = max(self.term.columns - indent - 1, 10)
        return text if len(text) <= room else text[: room - 1] + "…"

    def line(self, text: str = "") -> None:
        self.write(f"{self.style.muted('│')}  {text}".rstrip() + "\n")

    def banner(self) -> None:
        """The Memgraph wordmark, drawn large when the terminal is wide enough, else in one word."""
        width = len(WORDMARK[0])
        if self.term.columns < width + 2:
            self.write(f"\n{self.style.bold(self.style.accent('memgraph'))}\n\n")
            return
        self.write("\n" + "".join(f"{self.style.wordmark(row)}\n" for row in WORDMARK))
        self.write("".join(f"{self.style.graph(row, width)}\n" for row in GRAPH) + "\n")

    def intro(self, title: str, subtitle: str | Sequence[str] = "") -> None:
        """The orange graph divider, some space, then the title and its lines: the questions start apart from what came before."""
        rule = RULE[: min(len(WORDMARK[0]), self.term.columns - 1)]
        rule = rule[: rule.rfind("●") + 1]  # cut short, it still ends on a node
        self.write(f"\n{self.style.accent(rule)}\n\n")
        self.write(f"{self.style.accent('◆')}  {self.style.bold(title)}\n")
        for text in [subtitle] if isinstance(subtitle, str) else subtitle:
            self.line(self.style.muted(self.fit(text)))

    def answered(self, label: str, value: str) -> None:
        value = self.fit(value, _ANSWER_COLUMN)
        self.write(f"{self.style.muted('◇')}  {label.ljust(_LABEL_WIDTH)} {self.style.muted(value)}\n")

    def outro(self, text: str) -> None:
        self.line()
        self.write(f"{self.style.muted('└')}  {text}\n")

    def _hint(self, parts: Sequence[str], value: str) -> str:
        """The key hint after *value* on a question's line, short enough not to wrap.

        The first part is a description and goes first when space runs out; what the keys do stays.
        """
        room = self.term.columns - 1 - _ANSWER_COLUMN - len(value) - 3
        parts = [part for part in parts if part]
        text = " · ".join(parts)
        if len(text) > room and len(parts) > 1:
            text = " · ".join(parts[1:])
        return text if len(text) <= room else text[: max(room - 1, 0)] + "…" if room > 1 else ""

    def _redraw_line(self, text: str, cursor_column: int | None = None) -> None:
        """Replace the line the cursor is on with *text*; with colour off, only the first draw is written."""
        if self.term.color:
            move = f"\033[{cursor_column}G" if cursor_column else ""
            self.term.write(f"\r\033[2K{text}{move}")

    # --- questions --------------------------------------------------------------

    def text(self, label: str, default: str, hint: str = "") -> str | _Back:
        """A line of text. *default* shows as a placeholder: typing replaces it, Enter on nothing keeps it."""
        keeps = f"enter keeps {default}" if default else "enter leaves it empty"
        prefix = f"{self.style.accent('◆')}  {label.ljust(_LABEL_WIDTH)} "
        typed = ""
        if not self.term.color:
            self.term.write(f"{prefix}{default}   {self._hint((hint, keeps, 'type to replace', 'esc back'), default)}")
        while True:
            shown = typed or self.style.faint(default)
            help_text = self._hint((hint, keeps, "type to replace", "esc back"), typed or default)
            self._redraw_line(f"{prefix}{shown}   {self.style.muted(help_text)}", _ANSWER_COLUMN + 1 + len(typed))
            key = self.term.read_key()
            if key == ENTER:
                break
            if key == ESC:
                self._redraw_line("")
                if not self.term.color:
                    self.term.write("\n")
                return BACK
            if key == BACKSPACE:
                typed = typed[:-1]
            elif key == CLEAR:
                typed = ""
            elif len(key) == 1 and key.isprintable():
                typed += key
        self._redraw_line("")
        if not self.term.color:
            self.term.write("\n")
        answer = typed.strip() or default
        self.answered(label, answer or "none")
        return answer

    def confirm(
        self, label: str, default: bool, back_hint: str = "esc back", *, note: str = "", short: str = ""
    ) -> bool | _Back:
        """Yes or no; Enter keeps *default*.

        *note* is one line of context shown above the question while it is open; the answer collapses to
        *short* (else *label*) and the note goes with the question.
        """
        mark = self.written
        if note:
            self.line(self.style.muted(self.fit(note)))
        hint = f"{'Y/n' if default else 'y/N'} · {back_hint}"
        if 3 + len(label) + 2 + len(hint) >= self.term.columns:
            hint = "Y/n" if default else "y/N"
        label = self.fit(label, 3 + 2 + len(hint))
        self._redraw_line(f"{self.style.accent('◆')}  {label}  {self.style.muted(hint)}")
        if not self.term.color:
            self.term.write(f"◆  {label}  {hint}")
        while True:
            key = self.term.read_key()
            if key == ESC:
                answer: bool | _Back = BACK
            elif key == ENTER:
                answer = default
            elif key.lower() in ("y", "n"):
                answer = key.lower() == "y"
            else:
                continue
            break
        self._redraw_line("")
        if not self.term.color:
            self.term.write("\n")
        if not isinstance(answer, _Back):
            self.erase_since(mark)
            self.answered(short or label, "yes" if answer else "no")
        return answer

    def multiselect(self, label: str, options: Sequence[_Option]) -> list[str] | _Back:
        """Pick any of *options* with the arrows, space and Enter; returns the picked values in order."""
        cursor = next((i for i, option in enumerate(options) if option.enabled), 0)
        width = max(len(option.label) for option in options) + 2
        help_text = self.fit("↑/↓ move · space select · enter confirm · esc back", _ANSWER_COLUMN)
        mark = self.written
        self.term.write("\033[?25l" if self.term.color else "")
        try:
            while True:
                self.erase_since(mark)
                self.write(f"{self.style.accent('◆')}  {label.ljust(_LABEL_WIDTH)} {self.style.muted(help_text)}\n")
                for i, option in enumerate(options):
                    pointer = self.style.accent("›") if i == cursor else " "
                    if option.enabled:
                        box = self.style.accent("◼") if option.selected else "◻"
                        text = f"{option.label.ljust(width)}{self.style.muted(option.note)}"
                    else:
                        box, text = self.style.faint("◻"), self.style.faint(f"{option.label.ljust(width)}{option.note}")
                    self.write(f"{self.style.muted('│')} {pointer} {box} {text}\n")
                key = self.term.read_key()
                if key in (ENTER, ESC):
                    break
                if key in (UP, "k"):
                    cursor = self._step(options, cursor, -1)
                elif key in (DOWN, "j"):
                    cursor = self._step(options, cursor, 1)
                elif key == SPACE and options[cursor].enabled:
                    options[cursor].selected = not options[cursor].selected
        finally:
            self.term.write("\033[?25h" if self.term.color else "")
        self.erase_since(mark)
        if key == ESC:
            return BACK
        picked = [option for option in options if option.selected]
        self.answered(label, ", ".join(option.label for option in picked) or "none")
        return [option.value for option in picked]

    @staticmethod
    def _step(options: Sequence[_Option], cursor: int, direction: int) -> int:
        for offset in range(1, len(options) + 1):
            candidate = (cursor + direction * offset) % len(options)
            if options[candidate].enabled:
                return candidate
        return cursor


# --- the flow -------------------------------------------------------------------


@dataclass
class _Answers:
    """The answers so far; each question starts from what is here, so stepping back keeps them."""

    user_id: str
    runtimes: list[str]
    project_dirs: list[Path]
    viewer: bool
    use_graph_memory: bool
    install: bool = False
    #: The questions asked, by index, and how many lines had been written when each was asked.
    asked_at: dict[int, int] = field(default_factory=dict)


# A question returns DONE, SKIPPED (it doesn't apply, and drew nothing), BACK or STOP.
DONE, SKIPPED, STOP = "done", "skipped", "stop"


def ask(prompts: Prompts, choices: Choices, env: Environment, *, preselect: bool = True) -> Choices | None:
    """Ask for what the flags left open, show the plan and confirm; None when the user says no.

    *preselect* starts the agent list with *choices*' agents ticked; off, the user ticks every one.
    """
    prompts.intro("Context Graph setup", engine.WELCOME)
    enterprise = env.edition == "enterprise"
    prompts.answered("Memgraph", "from your Enterprise login" if enterprise else _memgraph_summary(choices, env))

    # Every question starts from how things are now, so running setup again shows what is set up and
    # changes only what the user changes.
    configured_graph = choices.memory_backend is None and env.configured_memory_backend == _MEMORY_GRAPH
    answers = _Answers(
        user_id=choices.user_id,
        runtimes=list(choices.runtimes) if preselect else [r for r in env.found_agents if r in env.wired_agents],
        project_dirs=list(choices.project_dirs),
        viewer=choices.viewer,
        use_graph_memory=choices.memory_backend == _MEMORY_GRAPH or configured_graph,
    )

    def name(answers: _Answers) -> str | _Back:
        if enterprise:
            return SKIPPED
        answer = prompts.text("Your name", answers.user_id, hint="your memories are filed under it")
        if isinstance(answer, str):
            answers.user_id = answer
        return answer if isinstance(answer, _Back) else DONE

    def agents(answers: _Answers) -> str | _Back:
        options = _agent_options(answers.runtimes, env.wired_agents)
        answer = prompts.multiselect("Agents", options)
        # Ticks survive Esc too, so coming back to the list finds them where they were left.
        answers.runtimes = [option.value for option in options if option.selected]
        return answer if isinstance(answer, _Back) else DONE

    def projects(answers: _Answers) -> str | _Back:
        if not any(BY_RUNTIME[runtime].wiring == "project" for runtime in answers.runtimes):
            return SKIPPED
        default = ", ".join(str(path) for path in answers.project_dirs)
        answer = prompts.text("Projects", default, hint="comma-separated")
        if isinstance(answer, _Back):
            return answer
        answers.project_dirs = [Path(part.strip()).expanduser().resolve() for part in answer.split(",") if part.strip()]
        return DONE

    def viewer(answers: _Answers) -> str | _Back:
        answer = prompts.confirm(
            "Install the terminal viewer?",
            answers.viewer,
            note="Watch your agents' memory grow, right in this terminal: memgraph-context-graph ui.",
            short="Viewer",
        )
        if isinstance(answer, bool):
            answers.viewer = answer
        return answer if isinstance(answer, _Back) else DONE

    def memory(answers: _Answers) -> str | _Back:
        # Every agent setup knows can switch; one with no memory of its own just gains the memory tool.
        switchable = [BY_RUNTIME[runtime].label for runtime in answers.runtimes if engine.has_native_memory(runtime)]
        if not switchable:
            return SKIPPED
        answer = prompts.confirm(
            f"Use Context Graph as the memory of {engine.listed(switchable)}?",
            answers.use_graph_memory,
            note="Their memories move into the graph, shared across agents. Switch back any time.",
            short="Agent memory",
        )
        if isinstance(answer, bool):
            answers.use_graph_memory = answer
        return answer if isinstance(answer, _Back) else DONE

    def confirm(answers: _Answers) -> str | _Back:
        prompts.line()
        prompts.answered("Plan", "")
        for step in engine.plan(_choices(answers, choices, env), env):
            prompts.line(prompts.style.muted(prompts.fit(step)))
        prompts.line()
        answer = prompts.confirm("Install now?", True, back_hint="esc to change an answer")
        if isinstance(answer, _Back):
            return answer
        return DONE if answer else STOP

    questions: list[Callable[[_Answers], str | _Back]] = [name, agents, projects, viewer, memory, confirm]
    skipped: set[int] = set()
    index = 0
    while index < len(questions):
        answers.asked_at[index] = prompts.written
        result = questions[index](answers)
        if isinstance(result, _Back):
            previous = next((i for i in range(index - 1, -1, -1) if i not in skipped), None)
            if previous is not None:
                prompts.erase_since(answers.asked_at[previous])
                index = previous
            else:
                prompts.erase_since(answers.asked_at[index])
            continue
        if result == STOP:
            prompts.outro("Nothing changed.")
            return None
        skipped.discard(index)
        if result == SKIPPED:
            skipped.add(index)
        index += 1
    return _choices(answers, choices, env)


def _choices(answers: _Answers, flags: Choices, env: Environment) -> Choices:
    if answers.use_graph_memory:
        memory_backend = _MEMORY_GRAPH
    elif env.configured_memory_backend == _MEMORY_GRAPH:  # it is on, and the user said no: hand memory back
        memory_backend = MEMORY_BACKEND_NATIVE
    else:  # no switch, unless the flags asked for one
        memory_backend = None if flags.memory_backend == _MEMORY_GRAPH else flags.memory_backend
    return engine.Choices(
        runtimes=answers.runtimes,
        project_dirs=answers.project_dirs,
        user_id=answers.user_id or flags.user_id,
        memgraph_url=flags.memgraph_url,
        memory_backend=memory_backend,
        viewer=answers.viewer,
        container=flags.container,
    )


def run(prompts: Prompts, choices: Choices, env: Environment) -> int:
    """Carry out *choices*, one line per step as it finishes; 0 when nothing failed, 1 otherwise."""
    style = prompts.style
    marks = {"ok": style.success("✓"), "skip": style.muted("–"), "warn": style.warning("!"), "fail": style.error("✗")}
    prompts.line()
    prompts.line(style.muted("Setting up… (a first Memgraph start downloads about 1 GB)"))

    def log(text: str) -> None:
        prompts.line(style.muted(f"  {text}"))

    def report(outcome: Outcome) -> None:
        prompts.write(f"{marks[outcome.status]}  {outcome.step.ljust(_LABEL_WIDTH)} {outcome.summary}\n")
        for extra in outcome.lines:
            log(extra)

    outcomes = engine.run(choices, env, log=log, report=report)
    failures = [outcome for outcome in outcomes if outcome.status == "fail"]
    if not failures:
        steps = engine.next_steps(choices, env)
        prompts.line()
        prompts.line(style.bold(steps[0]))
        for text in steps[1:]:
            prompts.line(prompts.fit(text))
        prompts.outro(style.success("Context Graph is set up."))
        return 0
    prompts.line()
    prompts.line(style.error(f"{len(failures)} step(s) failed. To fix:"))
    for outcome in failures:
        for fix in (outcome.fix or "memgraph-context-graph setup").splitlines():
            prompts.line(f"  {outcome.step}: {fix}")
    prompts.outro("Run setup again after fixing; it skips what is already done.")
    return 1


def run_setup_prompts(
    choices: Choices,
    env: Environment,
    terminal: Terminal | None = None,
    *,
    preselect: bool = True,
) -> int:
    """Ask, then run; 0 when nothing failed or the user stopped, 1 when a step failed."""
    prompts = Prompts(terminal or StdTerminal())
    try:
        with prompts.term.keypresses():
            answered = ask(prompts, choices, env, preselect=preselect)
    except KeyboardInterrupt:
        prompts.write("\n")
        prompts.outro("Stopped. Nothing changed.")
        return 0
    if answered is None:
        return 0
    return run(prompts, answered, env)


def _memgraph_summary(choices: Choices, env: Environment) -> str:
    if choices.memgraph_url:
        return choices.memgraph_url
    if env.configured_url:
        return f"{env.configured_url} (configured)"
    return "uses Memgraph on localhost:7687, or starts one in Docker for you"


def _agent_options(selected: Sequence[str], wired: Sequence[str]) -> list[_Option]:
    options = []
    for agent in AGENTS:
        found = agent.binary() is not None
        wiring = "user-wide plugin" if agent.wiring == "plugin" else "per project"
        state = " · set up" if agent.runtime in wired else "" if found else " · not found"
        options.append(
            _Option(
                value=agent.runtime,
                label=agent.label,
                note=f"{wiring}{state}",
                selected=agent.runtime in selected,
                enabled=found or agent.runtime in selected,
            )
        )
    return options


# --- the real terminal ----------------------------------------------------------


class StdTerminal:
    """stdin/stdout, reading single keys without echo (termios on POSIX, msvcrt on Windows)."""

    def __init__(self) -> None:
        if os.name == "nt":
            # An empty os.system call is the usual way to switch on ANSI escapes in a Windows console.
            os.system("")
        if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
            self.color = ""
        elif os.environ.get("COLORTERM") in ("truecolor", "24bit") or os.environ.get("WT_SESSION"):
            self.color = "truecolor"
        else:  # macOS Terminal.app, for one, has 256 colours but no truecolor
            self.color = "256"
        self.columns = shutil.get_terminal_size().columns

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def read_key(self) -> str:
        if sys.platform == "win32":
            import msvcrt

            char = msvcrt.getwch()
            if char == "\x03":
                raise KeyboardInterrupt
            if char in ("\x00", "\xe0"):
                return {"H": UP, "P": DOWN}.get(msvcrt.getwch(), "")
            return {" ": SPACE, "\r": ENTER, "\x1b": ESC, "\x08": BACKSPACE, "\x15": CLEAR}.get(char, char)
        return self._read_key_posix()

    @contextmanager
    def keypresses(self) -> Iterator[object]:
        """Keys arrive one by one and unechoed until the block ends; Ctrl+C still interrupts.

        Held for the whole session rather than per key: switching modes between keys loses what was typed
        or pasted faster than one key per read. The saved mode comes back however the block ends.
        """
        if sys.platform == "win32" or not sys.stdin.isatty():
            yield None
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            yield None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    @staticmethod
    def _read_key_posix() -> str:
        import select

        fd = sys.stdin.fileno()
        char = os.read(fd, 1)
        if char == b"\x1b":
            # An arrow key is ESC [ A; a lone ESC is the key itself.
            if not select.select([fd], [], [], 0.05)[0]:
                return ESC
            char += os.read(fd, 2)
        elif char[0] >= 0xC0:  # the first byte of a multi-byte UTF-8 character
            char += os.read(fd, 1 if char[0] < 0xE0 else 2 if char[0] < 0xF0 else 3)
        keys = {
            b"\x1b[A": UP,
            b"\x1b[B": DOWN,
            b" ": SPACE,
            b"\n": ENTER,
            b"\r": ENTER,
            b"\x7f": BACKSPACE,
            b"\x08": BACKSPACE,
            b"\x15": CLEAR,
        }
        if char in keys:
            return keys[char]
        return "" if char.startswith(b"\x1b") else char.decode(errors="replace")
