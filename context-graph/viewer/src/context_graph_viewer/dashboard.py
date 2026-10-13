"""The landing screen: capture health and pipeline state over the sessions in scope (#381)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, OptionList, Sparkline, Static
from textual.widgets.option_list import Option

from . import fmt, queries
from .base import Pane, PolledScreen, cursor_key, sync_table
from .failures import Failures, recent_failures

if TYPE_CHECKING:
    from collections.abc import Iterable

    from textual.widget import Widget

    from .source import Reader

#: An in-progress session with no event for this long is reported as never ended.
STALE_AFTER = timedelta(minutes=30)
GROWTH_DAYS = 14

_RECONCILE_STYLE = {"completed": ("done", "green"), "failed": ("failed", "red"), "pending": ("pending", "yellow")}


@dataclass
class DashboardData:
    """One poll of the landing screen: the graph's side and the local failure log's side."""

    graph: queries.Dashboard
    failures: Failures


def reconcile_cell(status: str | None) -> Text:
    """A session's reconciliation status, coloured."""
    label, style = _RECONCILE_STYLE.get(status or "", ("—", "dim"))
    return Text(label, style=style)


def session_columns(show_owner: bool) -> list[tuple[str, str]]:
    """The sessions table's columns; the owner column only once the scope reaches past one user."""
    columns = [("id", "session"), ("project", "project"), ("last", "last event"), ("actions", "actions")]
    columns += [("errors", "tool errs"), ("reconcile", "reconcile"), ("chunks", "chunks"), ("touches", "touches")]
    return [*columns, ("owner", "owner")] if show_owner else columns


def session_cells(session: dict[str, Any], now: datetime, show_owner: bool) -> list[Any]:
    """One sessions-table row."""
    touches = session["touches"] or 0
    open_touches = session["touches_open"] or 0
    cells: list[Any] = [
        session["id"][:8],
        fmt.project_name(session["wd"]),
        fmt.ago(fmt.to_datetime(session["last_event"] or session["started"]), now),
        session["actions"],
        Text(str(session["errors"]), style="red") if session["errors"] else "",
        reconcile_cell(session["rstatus"]),
        session["chunks"] or "",
        Text(f"{touches} ({open_touches} open)", style="magenta") if open_touches else (touches or ""),
    ]
    if show_owner:
        cells.append(Text("unattributed", style="dim") if session["unattributed"] else str(session["owner"]))
    return cells


def matches(session: dict[str, Any], needle: str) -> bool:
    """Whether the ``/`` filter text appears in the session's id, project, owner or status."""
    if not needle:
        return True
    haystack = " ".join(
        str(value or "") for value in (session["id"], session["wd"], session["owner"], session["rstatus"])
    )
    return needle.casefold() in haystack.casefold()


def health_lines(data: DashboardData, now: datetime) -> list[Text]:
    """The capture-health hints: the local failure log first, then what the graph shows came in."""
    lines = [*data.failures.describe(now)]
    sessions = data.graph.sessions
    last = max((t for s in sessions if (t := fmt.to_datetime(s["last_event"]))), default=None)
    lines.append(Text("last event in scope: ").append(fmt.ago(last, now), style="bold"))
    empty = [s for s in sessions if s["actions"] == 0]
    if empty:
        ids = ", ".join(s["id"][:8] for s in empty[:3])
        lines.append(Text(f"{len(empty)} session(s) with no actions", style="yellow").append(f" ({ids})"))
    stale = [
        s
        for s in sessions
        if s["status"] == "in_progress"
        and (seen := fmt.to_datetime(s["last_event"] or s["started"]))
        and now - seen > STALE_AFTER
    ]
    if stale:
        lines.append(Text(f"{len(stale)} session(s) never ended", style="yellow").append(" (no event for 30m+)"))
    errors = sum(s["errors"] for s in sessions)
    if errors:
        lines.append(Text(f"{errors} failed tool result(s)", style="red").append(" (tool errors, not capture errors)"))
    open_touches = data.graph.touches.get("pending", 0) + data.graph.touches.get("unresolved", 0)
    if open_touches:
        lines.append(Text(f"{open_touches} resource touch(es) pending or unresolved", style="yellow"))
    if data.graph.unattributed:
        lines.append(Text(f"{data.graph.unattributed} unattributed session(s) in the graph · a widens the scope"))
    return lines


def _counts(label: str, values: Iterable[tuple[str | None, int]]) -> Text:
    counts: Counter[str] = Counter()
    for value, n in values:
        counts[value or "not marked"] += n
    line = Text(f"{label}: ")
    if not counts:
        return line.append("none", style="dim")
    for i, (value, n) in enumerate(sorted(counts.items())):
        line.append("  " if i else "").append(f"{value} ").append(str(n), style="bold")
    return line


def pipeline_lines(data: queries.Dashboard) -> list[Text]:
    """The pipeline-state panel: reconcile and embedding queues, extraction, ontology, tokens and failures."""
    rows = data.pipeline
    lines = [
        _counts("sessions", ((r["rstatus"], r["n"]) for r in rows)).append(
            "  (pending returns after every turn)", style="dim"
        ),
        _counts("embeddings", ((r["estatus"], r["n"]) for r in rows)),
        _counts("memory files", ((m["status"], 1) for m in data.memories)),
    ]
    backends = Counter[str]()
    for r in rows:
        if r["backend"]:
            backends[r["backend"]] += r["n"]
    if backends:
        lines.append(Text("extracted by: " + ", ".join(f"{b} {n}" for b, n in sorted(backends.items()))))
    for o in data.ontology:
        version = "default (v0)" if o["version"] is None else f"v{o['version']}"
        line = Text(f"ontology for {o['user']}: ").append(version, style="bold")
        lines.append(line.append(f"  derive milestone {o['milestone'] if o['milestone'] is not None else '—'}"))
    tokens = Text("capture tokens: ")
    lines.append(tokens.append(f"{data.tokens:,}", style="bold") if data.tokens else tokens.append("none reported"))
    for f in data.failures[:5]:
        for kind, error, failed in (
            ("reconcile", f["rerror"], f["rstatus"] == "failed"),
            ("re-extract", f["reerror"], f["reerror"] is not None),
            ("embed", f["eerror"], f["estatus"] == "failed"),
        ):
            if failed:
                lines.append(Text(f"✕ {f['id'][:8]} {kind} ", style="red").append(fmt.clip(error or "", 70)))
    for m in data.memories:
        if m["status"] == "failed":
            lines.append(Text(f"✕ memory {m['path']} ", style="red").append(fmt.clip(m["error"] or "", 50)))
    return lines


def growth_series(growth: dict[str, int], now: datetime, days: int = GROWTH_DAYS) -> list[float]:
    """New entities per day for the last *days* days, oldest first."""
    today = now.date()
    return [float(growth.get((today - timedelta(days=i)).isoformat(), 0)) for i in range(days - 1, -1, -1)]


def empty_scope_banner(data: queries.Dashboard, scope: queries.Scope) -> Text | None:
    """A call-out when the default scope shows nothing but the graph does hold sessions (#383)."""
    if scope.mode != queries.USER or data.sessions:
        return None
    if scope.user_id is None:
        text = Text("No user is configured (identity.user_id), so no session is yours. ", style="bold yellow")
    else:
        text = Text(f"No sessions belong to {scope.user_id}. ", style="bold yellow")
    if data.unattributed:
        text.append(f"{data.unattributed} session(s) in the graph have no user: press a to include them.")
    else:
        text.append("Nothing captured yet: run `memgraph-context-graph doctor` to check the hooks.")
    return text


class ProjectPicker(ModalScreen[str | None]):
    """Choose a working directory to narrow every view to, or all projects."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss_picker", "cancel")]
    ALL = "\0all"

    def __init__(self, projects: list[str], current: str | None) -> None:
        super().__init__()
        self.projects = projects
        self.current = current

    def compose(self) -> Iterable[Widget]:
        """A list of projects with "all projects" first."""
        options = [Option("all projects", id=self.ALL)]
        options += [Option(Text(f"{fmt.project_name(p)}  ").append(p, style="dim"), id=p) for p in self.projects]
        with Vertical(id="picker"):
            yield Static("Narrow to a project", classes="title")
            yield OptionList(*options, id="projects")

    def on_mount(self) -> None:
        """Start on the current choice."""
        option_list = self.query_one(OptionList)
        if self.current in self.projects:
            option_list.highlighted = self.projects.index(self.current) + 1
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Return the chosen working directory, or None for all projects."""
        chosen = event.option.id
        self.dismiss(None if chosen == self.ALL else chosen)

    def action_dismiss_picker(self) -> None:
        """Keep the current choice."""
        self.dismiss(self.current)


class DashboardScreen(PolledScreen[DashboardData]):
    """Sessions in scope beside the pipeline, health and growth panes; the viewer opens here."""

    PANES: ClassVar = ("#sessions", "#pipeline-pane", "#health-pane", "#growth-pane")
    BINDINGS: ClassVar = [
        Binding("enter", "open", "open session", show=False),
        Binding("slash", "filter", "filter"),
        Binding("p", "project", "project"),
        Binding("k", "knowledge", "knowledge graph"),
    ]

    def compose_body(self) -> Iterable[Widget]:
        """The sessions table on the left, the three panels on the right."""
        with Horizontal():
            with Vertical(id="left"):
                yield Static(id="banner")
                yield Input(placeholder="filter sessions (esc clears)", id="filter")
                sessions: DataTable = DataTable(id="sessions", cursor_type="row")
                sessions.border_title = "1 Sessions"
                yield sessions
            with Vertical(id="side"):
                with Pane(id="pipeline-pane", classes="pane") as pane:
                    pane.border_title = "2 Pipeline"
                    yield Static(id="pipeline")
                with Pane(id="health-pane", classes="pane") as pane:
                    pane.border_title = "3 Capture health"
                    yield Static(id="health")
                with Pane(id="growth-pane", classes="pane") as pane:
                    pane.border_title = "4 Growth"
                    yield Static(id="growth-label")
                    yield Sparkline([0.0], id="growth")

    def on_mount(self) -> None:
        """Hide the filter and banner until needed, focus the table."""
        self.query_one("#filter").display = False
        self.query_one("#banner").display = False
        self.query_one("#sessions", DataTable).focus()
        super().on_mount()

    def title_text(self) -> str:
        """The dashboard is the root of the breadcrumb."""
        return "Context Graph"

    def fetch(self, reader: Reader) -> DashboardData:
        """Query the graph and read the local failure log."""
        scope = self.viewer.scope
        return DashboardData(
            graph=queries.dashboard(reader, scope),
            failures=recent_failures(self.viewer.now(), project=scope.project),
        )

    def show(self, data: DashboardData) -> None:
        """Draw every pane from one poll."""
        now = self.viewer.now()
        self._fill_sessions(data.graph, now)
        banner = empty_scope_banner(data.graph, self.viewer.scope)
        banner_widget = self.query_one("#banner", Static)
        banner_widget.display = banner is not None
        if banner is not None:
            banner_widget.update(banner)
        self.query_one("#pipeline", Static).update(Text("\n").join(pipeline_lines(data.graph)))
        self.query_one("#health", Static).update(Text("\n").join(health_lines(data, now)))
        series = growth_series(data.graph.growth, now)
        self.query_one("#growth", Sparkline).data = series
        self.query_one("#growth-label", Static).update(
            Text(f"new entities per day, last {GROWTH_DAYS} days: ").append(f"{int(sum(series))}", style="bold")
        )

    def _fill_sessions(self, data: queries.Dashboard, now: datetime) -> None:
        show_owner = self.viewer.scope.mode != queries.USER
        needle = self.query_one("#filter", Input).value
        rows = [(s["id"], session_cells(s, now, show_owner)) for s in data.sessions if matches(s, needle)]
        sync_table(self.query_one("#sessions", DataTable), session_columns(show_owner), rows)

    def status_note(self) -> str:
        """How many sessions the table shows out of how many are in scope."""
        if self.data is None:
            return ""
        shown = self.query_one("#sessions", DataTable).row_count
        return f"{shown} of {self.data.graph.total_sessions} sessions in scope"

    def action_open(self) -> None:
        """Open the session under the cursor."""
        if key := cursor_key(self.query_one("#sessions", DataTable)):
            self._open(key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a row opens that session."""
        if event.row_key.value:
            self._open(event.row_key.value)

    def _open(self, session_id: str) -> None:
        from .session import SessionScreen

        self.app.push_screen(SessionScreen(session_id))

    def action_filter(self) -> None:
        """Show and focus the ``/`` filter."""
        box = self.query_one("#filter", Input)
        box.display = True
        box.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Re-filter the table as the text changes."""
        if self.data is not None:
            self._fill_sessions(self.data.graph, self.viewer.now())
            self.draw_status()

    def on_input_submitted(self, _: Input.Submitted) -> None:
        """Back to the table, keeping the filter."""
        self.query_one("#sessions", DataTable).focus()

    def on_key(self, event: Any) -> None:
        """Escape in the filter clears and hides it."""
        box = self.query_one("#filter", Input)
        if event.key == "escape" and box.has_focus:
            box.value = ""
            box.display = False
            self.query_one("#sessions", DataTable).focus()
            event.stop()

    def action_project(self) -> None:
        """Pick a project to narrow every view to."""
        projects = self.data.graph.projects if self.data else []
        self.app.push_screen(ProjectPicker(projects, self.viewer.scope.project), self._project_chosen)

    def _project_chosen(self, project: str | None) -> None:
        self.viewer.set_project(project)

    def action_knowledge(self) -> None:
        """Open the in-scope entity list."""
        from .knowledge import EntityListScreen

        self.app.push_screen(EntityListScreen())
