"""The viewer app: one reader per space, a scope shared by every view, and a polling tick."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar

from textual.app import App
from textual.binding import Binding

from . import fmt, queries
from .base import PolledScreen
from .dashboard import DashboardScreen
from .source import Identity, Reader, describe

if TYPE_CHECKING:
    from collections.abc import Callable

POLL_SECONDS = 3.0

CSS = """
#left { width: 2fr; }
#sessions, #entities, #nb { height: 1fr; }
#side { width: 1fr; }
#banner { padding: 0 1; border: round $warning; height: auto; }
#filter { height: 3; }
.pane { border: round $primary-darken-2; padding: 0 1; height: 1fr; }
.pane:focus-within, .pane:focus { border: round $accent; }
#growth-pane { height: auto; }
#growth { height: 3; }
#timeline { height: 3fr; border: round $primary-darken-2; }
#timeline:focus { border: round $accent; }
#produced-pane { height: 1fr; max-height: 12; }
#main { width: 3fr; }
#details-pane { width: 2fr; }
#entities { width: 3fr; border: round $primary-darken-2; }
#entities:focus, #sessions:focus { border: round $accent; }
#sessions { border: round $primary-darken-2; }
#prov-pane { width: 2fr; }
#nbmain { width: 3fr; }
#nb { border: round $primary-darken-2; }
#nb:focus { border: round $accent; }
#evidence-pane { width: 2fr; }
#view-status { height: 1; padding: 0 1; background: $panel; }
ProjectPicker { align: center middle; }
#picker { width: 80%; max-width: 100; height: 70%; border: round $accent; background: $surface; padding: 0 1; }
#picker .title { text-style: bold; padding: 0 0 1 0; }
"""


class ViewerApp(App[None]):
    """Read-only Context Graph viewer. Opens on the dashboard and polls the active view."""

    TITLE = "Context Graph"
    CSS = CSS
    BINDINGS: ClassVar = [
        Binding("a", "widen_scope", "scope"),
        Binding("w", "next_space", "space", show=False),
        Binding("question_mark", "help", "keys"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(
        self,
        identity: Identity,
        *,
        poll_seconds: float = POLL_SECONDS,
        clock: Callable[[], datetime] | None = None,
        reader_factory: Callable[..., Reader] = Reader,
    ) -> None:
        """Args:
        identity: Who is looking, and which spaces they may read (personal first).
        poll_seconds: How often the active view reloads; 0 turns polling off (tests drive it by hand).
        clock: The current time, for "3m ago" text; tests pin it.
        reader_factory: Builds the reader for a space.
        """
        super().__init__()
        self.identity = identity
        self.scope = queries.Scope(user_id=identity.user_id)
        self.poll_seconds = poll_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._reader_factory = reader_factory
        self.space_index = 0
        self.reader = reader_factory(identity.spaces[0])

    def now(self) -> datetime:
        """The current time, from the injected clock."""
        return self._clock()

    def on_mount(self) -> None:
        """Open the dashboard and start polling."""
        self.push_screen(DashboardScreen())
        if self.poll_seconds > 0:
            self.set_interval(self.poll_seconds, self.poll_active)

    def on_unmount(self) -> None:
        """Close the connection."""
        self.reader.close()

    def poll_active(self) -> None:
        """Reload the view on top; views underneath catch up when they're shown again."""
        if isinstance(self.screen, PolledScreen):
            self.screen.action_poll()

    def context_line(self) -> str:
        """Where, as whom, and what scope: shown under every view's title."""
        space = self.identity.spaces[self.space_index]
        user = self.identity.user_id or "no user configured"
        parts = [f"{describe(space)}", f"{space.name} space", f"user {user}", f"scope {self.scope.mode}"]
        if self.scope.project:
            parts.append(f"project {fmt.project_name(self.scope.project)}")
        return " · ".join(parts)

    def _rescoped(self) -> None:
        for screen in self.screen_stack:
            if isinstance(screen, PolledScreen):
                screen.refresh_titles()
        self.poll_active()

    def action_widen_scope(self) -> None:
        """Cycle the scope: your sessions, plus unattributed ones, everyone's."""
        self.scope = self.scope.widened()
        self._rescoped()

    def set_project(self, project: str | None) -> None:
        """Narrow every view to one working directory, or None for all."""
        self.scope = replace(self.scope, project=project)
        self._rescoped()

    def action_next_space(self) -> None:
        """Enterprise: switch to the next space this login may read; Community has only one."""
        if len(self.identity.spaces) < 2:
            self.notify("Only one space: shared spaces need the Enterprise edition.", timeout=3)
            return
        self.reader.close()
        self.space_index = (self.space_index + 1) % len(self.identity.spaces)
        self.reader = self._reader_factory(self.identity.spaces[self.space_index])
        self._rescoped()

    def action_help(self) -> None:
        """Toggle the key help panel."""
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()
