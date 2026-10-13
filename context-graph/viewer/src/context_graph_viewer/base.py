"""What every view shares: polling with its own error state, pane navigation, and in-place table updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from rich.text import Text
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import get_current_worker

from . import fmt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

    from textual.app import ComposeResult
    from textual.widget import Widget

    from .app import ViewerApp
    from .source import Reader

T = TypeVar("T")


class Pane(VerticalScroll):
    """A titled, focusable, scrollable pane that ``f`` can fill the screen with."""

    ALLOW_MAXIMIZE = True


class ViewStatus(Static):
    """One line under a view: live, or the error from this view's last poll and how old its data is."""

    def show(self, *, error: str | None, last_ok: datetime | None, now: datetime, note: str = "") -> None:
        """Redraw for the view's latest poll outcome."""
        line = Text()
        if error:
            line.append("⚠ ", style="bold red").append(fmt.clip(error, 160), style="red")
            line.append(f"  · data from {fmt.ago(last_ok, now)}" if last_ok else "  · no data yet", style="dim")
        elif last_ok is None:
            line.append("… loading", style="dim")
        else:
            line.append("● ", style="green").append(f"live · updated {fmt.ago(last_ok, now)}", style="dim")
        if note:
            line.append(f"  · {note}", style="dim")
        self.update(line)


class PolledScreen(Screen[None], Generic[T]):
    """A view that loads its data off the UI thread on every poll and keeps its own error state.

    Subclasses implement :meth:`fetch` (runs in a worker thread, may raise) and :meth:`show` (runs on the UI
    thread with the fetched data). A failed poll leaves the last good data on screen and says so on this view's
    status line; another view's successful poll can't hide it.
    """

    #: Selectors of the panes number keys jump between, in order.
    PANES: ClassVar[Sequence[str]] = ()
    BINDINGS: ClassVar = [
        *(Binding(str(n), f"pane({n})", f"pane {n}", show=False) for n in range(1, 10)),
        Binding("f", "fullscreen", "fullscreen"),
        Binding("r", "poll", "refresh", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.data: T | None = None
        self.error: str | None = None
        self.last_ok: datetime | None = None

    @property
    def viewer(self) -> ViewerApp:
        """The app, typed."""
        return cast("ViewerApp", self.app)

    def compose(self) -> ComposeResult:
        """Header, the view's own body, its status line, and the key footer."""
        yield Header()
        yield from self.compose_body()
        yield ViewStatus(id="view-status")
        yield Footer()

    def compose_body(self) -> Iterable[Widget]:
        """The view's widgets between the header and the status line."""
        return ()

    def fetch(self, reader: Reader) -> T:
        """Load this view's data. Runs in a worker thread."""
        raise NotImplementedError

    def show(self, data: T) -> None:
        """Draw *data*. Runs on the UI thread."""
        raise NotImplementedError

    def status_note(self) -> str:
        """Extra context for the status line (e.g. counts)."""
        return ""

    def on_mount(self) -> None:
        """Load immediately rather than waiting for the first poll tick."""
        self.refresh_titles()
        self.draw_status()
        self.action_poll()

    def on_screen_resume(self) -> None:
        """Coming back along the breadcrumb: catch up at once."""
        self.refresh_titles()
        if self.is_mounted:
            self.action_poll()

    def refresh_titles(self) -> None:
        """Set the header from :meth:`title_text` and the viewer's context line."""
        self.title = self.title_text()
        self.sub_title = self.viewer.context_line()

    def title_text(self) -> str:
        """This view's breadcrumb title."""
        return "Context Graph"

    def action_poll(self) -> None:
        """Start a background load; a newer poll replaces one still running."""
        reader = self.viewer.reader
        self.run_worker(lambda: self._load(reader), thread=True, exclusive=True, group="poll")

    def _load(self, reader: Reader) -> None:
        worker = get_current_worker()
        try:
            data = self.fetch(reader)
        except Exception as exc:  # A poll must never reach Textual's handler, which would end the app (#382).
            if not worker.is_cancelled:
                self.app.call_from_thread(self._failed, f"{type(exc).__name__}: {exc}")
            return
        if not worker.is_cancelled:
            self.app.call_from_thread(self._loaded, data)

    def _loaded(self, data: T) -> None:
        self.data, self.error, self.last_ok = data, None, self.viewer.now()
        self.show(data)
        self.refresh_titles()
        self.draw_status()

    def _failed(self, error: str) -> None:
        self.error = error
        self.draw_status()

    def draw_status(self) -> None:
        """Redraw this view's status line."""
        for status in self.query(ViewStatus):
            status.show(error=self.error, last_ok=self.last_ok, now=self.viewer.now(), note=self.status_note())

    def action_pane(self, number: int) -> None:
        """Focus the *number*-th pane."""
        if 0 < number <= len(self.PANES):
            for widget in self.query(self.PANES[number - 1]):
                widget.focus()
                return

    def action_fullscreen(self) -> None:
        """Toggle the focused pane filling the screen."""
        if self.maximized is not None:
            self.minimize()
        elif self.focused is not None:
            self.maximize(self.focused)


def sync_table(table: DataTable, columns: Sequence[tuple[str, str]], rows: Sequence[tuple[str, list[Any]]]) -> None:
    """Make *table* show *rows* without losing the reader's place.

    Cells change in place when the same rows arrive in the same order, so the cursor and scroll stay put. When
    rows are added, removed or reordered the table is rebuilt, and the cursor follows the row it was on.

    Args:
        columns: ``(key, label)`` per column.
        rows: ``(row_key, cells)`` in display order; row keys must be unique.
    """
    column_keys = [key for key, _ in columns]
    if [column.key.value for column in table.columns.values()] != column_keys:
        table.clear(columns=True)
        for key, label in columns:
            table.add_column(label, key=key)
    current = [row.value for row in table.rows]
    wanted = [key for key, _ in rows]
    if current == wanted:
        for key, cells in rows:
            for column, cell in zip(column_keys, cells, strict=True):
                if table.get_cell(key, column) != cell:
                    table.update_cell(key, column, cell)
        return
    selected = cursor_key(table)
    scroll_y = table.scroll_y
    table.clear()
    for key, cells in rows:
        table.add_row(*cells, key=key)
    if selected in wanted:
        table.move_cursor(row=wanted.index(selected), animate=False, scroll=False)
    table.scroll_to(y=scroll_y, animate=False)


def cursor_key(table: DataTable) -> str | None:
    """The key of the row under *table*'s cursor, or None."""
    if not table.row_count:
        return None
    try:
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
    except Exception:  # The cursor sits outside the rows while the table is empty or shrinking.
        return None
