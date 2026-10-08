"""PROTOTYPE — throwaway. Answers #383: do the #380/#381/#382 view designs read well on a real graph?

Two structurally different variants per screen, switched with `v`:
  Dashboard  A = lazygit-style panes (sessions | pipeline / health / growth)
             B = k9s-style single wide table under a one-line meter strip
  Session    A = Tree with time bars (folding subagents)        (#382 option a)
             B = DataTable with indent + bar column             (#382 option b)

Run:  uv run --no-project --python 3.12 --with textual --with neo4j context-graph/viewer-prototype/prototype_viewer.py
Keys: v variant · a scope (user → user+unattributed → all) · enter drill in · esc back · r refresh · q quit

Reads the same config file as the hooks (CONTEXT_GRAPH_CONFIG or ~/.config/context-graph/config.toml).
Read-only. Polls every 3 s. The local capture-failure log (#378) does not exist yet and is stubbed.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import tomllib
from neo4j import GraphDatabase
from prototype_knowledge import EntityList
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Sparkline, Static, Tree

POLL_SECONDS = 3
SCOPES = ["user", "user+unattributed", "all"]
BAR_WIDTH = 36


def load_config() -> dict:
    path = Path(os.environ.get("CONTEXT_GRAPH_CONFIG", Path.home() / ".config/context-graph/config.toml"))
    with path.open("rb") as f:
        return tomllib.load(f)


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def is_true(value) -> bool:
    return value is True or str(value).lower() == "true"


def ago(ts) -> str:
    if ts is None:
        return "—"
    secs = (datetime.now(timezone.utc) - ts).total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{int(secs // size)}{unit} ago"
    return f"{int(secs)}s ago"


def bar(start, end, t0, t1, width=BAR_WIDTH) -> str:
    if not (start and t0 and t1) or t1 <= t0:
        return ""
    span = (t1 - t0).total_seconds()
    a = int((start - t0).total_seconds() / span * width)
    b = int(((end or start) - t0).total_seconds() / span * width)
    return " " * a + ("█" * max(b - a, 0) or "▏")


class Graph:
    """All Cypher lives here (query/renderer split required by #381)."""

    def __init__(self, cfg: dict):
        mg = cfg.get("memgraph", {})
        self.url = mg.get("url", "bolt://localhost:7687")
        self.user_id = cfg.get("identity", {}).get("user_id")
        self.driver = GraphDatabase.driver(self.url, auth=(mg.get("user", ""), mg.get("password", "")))

    def rows(self, query: str, **params) -> list[dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(query, **params)]

    def dashboard(self, scope: str) -> dict:
        sessions = self.rows(
            """
            MATCH (s:Session)
            OPTIONAL MATCH (u:User)-[:HAD_SESSION]->(s)
            WITH s, u.user_id AS owner
            WHERE $scope = 'all' OR owner = $uid OR ($scope = 'user+unattributed' AND owner IS NULL)
            OPTIONAL MATCH (s)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action)
            WITH s, owner, count(CASE WHEN a.action_type <> 'tool_result' THEN 1 END) AS actions,
                 max(a.timestamp) AS last_event,
                 sum(CASE WHEN toString(a.is_error) = 'True' OR a.is_error = true THEN 1 ELSE 0 END) AS errors
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
            WITH s, owner, actions, last_event, errors, count(DISTINCT c) AS chunks
            RETURN s.session_id AS id, owner, s.working_directory AS wd, s.status AS status,
                   s.started_at AS started, s.ended_at AS ended, s.reconciliation_status AS rstatus,
                   s.reconciliation_error AS rerror, s.reconciled_at AS reconciled_at,
                   s.reconciliation_input_tokens AS in_tok, s.reconciliation_output_tokens AS out_tok,
                   actions, last_event, errors, chunks
            ORDER BY coalesce(last_event, s.started_at) DESC
            """,
            scope=scope,
            uid=self.user_id,
        )
        hidden = self.rows(
            "MATCH (s:Session) WHERE NOT EXISTS { MATCH (:User)-[:HAD_SESSION]->(s) } RETURN count(s) AS n"
        )[0]["n"]
        ids = [s["id"] for s in sessions]
        new_entities = self.rows(
            """
            MATCH (s:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
            WHERE s.session_id IN $ids AND s.reconciled_at IS NOT NULL
            WITH e, min(s.reconciled_at) AS first
            RETURN substring(first, 0, 10) AS day, count(e) AS n
            """,
            ids=ids,
        )
        return {"sessions": sessions, "unattributed": hidden, "new_entities": new_entities}

    def session(self, session_id: str) -> dict:
        actions = self.rows(
            """
            MATCH (s:Session {session_id: $id})
            CALL {
              WITH s MATCH (s)-[:HAS_ACTION]->(a:Action) RETURN a, null AS agent
              UNION
              WITH s MATCH (s)-[:HAS_AGENT]->(g:Agent)-[:HAS_ACTION]->(a:Action)
              RETURN a, coalesce(g.agent_type, g.agent_id) AS agent
            }
            OPTIONAL MATCH (a)-[:PARENT_OF]->(r:Action)
            RETURN a.action_id AS id, a.action_type AS type, a.tool_name AS tool, a.timestamp AS at,
                   a.status AS status, a.properties AS props, agent,
                   r.timestamp AS result_at, r.is_error AS result_error
            ORDER BY at
            """,
            id=session_id,
        )
        derived = self.rows(
            """
            MATCH (s:Session {session_id: $id})
            OPTIONAL MATCH (s)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(e)
            RETURN s.started_at AS started, s.ended_at AS ended, s.reconciliation_status AS rstatus,
                   ep.summary AS summary, count(DISTINCT c) AS chunks,
                   collect(DISTINCT coalesce(e.name, e.entity_id, e.id))[..12] AS entities,
                   count(DISTINCT e) AS entity_count
            """,
            id=session_id,
        )
        return {"actions": [a for a in actions if a["type"] != "tool_result"], "derived": derived[0] if derived else {}}


# ---------------------------------------------------------------- dashboard


def health_hints(data: dict) -> list[str]:
    now = datetime.now(timezone.utc)
    hints = ["[dim]local failure log: not built yet (#378) — stubbed[/]"]
    sessions = data["sessions"]
    last = max((parse_ts(s["last_event"]) for s in sessions if s["last_event"]), default=None)
    hints.append(f"last event in scope: [b]{ago(last)}[/]")
    no_actions = [s for s in sessions if s["actions"] == 0]
    if no_actions:
        hints.append(
            f"[yellow]{len(no_actions)} session(s) with no actions[/] ({', '.join(s['id'][:8] for s in no_actions[:3])})"
        )
    stale = [
        s
        for s in sessions
        if s["status"] == "in_progress" and (lp := parse_ts(s["last_event"])) and now - lp > timedelta(minutes=30)
    ]
    if stale:
        hints.append(f"[yellow]{len(stale)} session(s) never ended[/] (no event for 30m+)")
    errs = sum(s["errors"] for s in sessions)
    if errs:
        hints.append(f"[red]{errs} failed tool result(s)[/] (tool errors, not capture errors)")
    if data["unattributed"]:
        hints.append(f"{data['unattributed']} unattributed session(s) in graph — press [b]a[/] to widen scope")
    return hints


def pipeline_lines(data: dict) -> list[str]:
    counts = Counter(s["rstatus"] or "not marked" for s in data["sessions"])
    lines = ["  ".join(f"{k}: [b]{v}[/]" for k, v in sorted(counts.items()))]
    tok = [(s["in_tok"] or 0) + (s["out_tok"] or 0) for s in data["sessions"]]
    lines.append(
        f"memory cost: [b]{sum(tok)}[/] tokens"
        if any(tok)
        else "memory cost: [dim]not recorded yet (#381 write-side task)[/]"
    )
    for s in data["sessions"]:
        if s["rstatus"] == "failed":
            lines.append(f"[red]✕ {s['id'][:8]}[/] {str(s['rerror'])[:70]}")
    return lines


def growth_series(data: dict, days: int = 14) -> tuple[list[float], list[float]]:
    today = datetime.now(timezone.utc).date()
    keys = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    rec = Counter(str(s["reconciled_at"])[:10] for s in data["sessions"] if s["reconciled_at"])
    ent = {r["day"]: r["n"] for r in data["new_entities"]}
    return [float(rec.get(k, 0)) for k in keys], [float(ent.get(k, 0)) for k in keys]


SESSION_COLUMNS = {
    "A": ["session", "last event", "actions", "reconcile"],
    "B": [
        "session",
        "owner",
        "project",
        "status",
        "actions",
        "tool errs",
        "last event",
        "reconcile",
        "chunks",
        "tokens",
    ],
}


def session_row(s: dict, variant: str) -> list:
    rstatus = {"completed": "[green]done[/]", "failed": "[red]failed[/]", "pending": "[yellow]pending[/]"}.get(
        s["rstatus"], "[dim]—[/]"
    )
    last = ago(parse_ts(s["last_event"]))
    if variant == "A":
        return [s["id"][:8], last, s["actions"], Text.from_markup(rstatus)]
    tokens = (s["in_tok"] or 0) + (s["out_tok"] or 0)
    return [
        s["id"][:8],
        s["owner"] or Text("unattributed", style="dim"),
        Path(s["wd"]).name if s["wd"] else Text("—", style="dim"),
        s["status"] or "—",
        s["actions"],
        s["errors"] or "",
        last,
        Text.from_markup(rstatus),
        s["chunks"] or "",
        tokens or "",
    ]


class Dashboard(Screen):
    BINDINGS: ClassVar = [
        Binding("enter", "open", "open session", show=True),
        Binding("k", "knowledge", "knowledge graph"),
    ]

    def action_knowledge(self) -> None:
        self.app.push_screen(EntityList())

    def compose(self) -> ComposeResult:
        app: Viewer = self.app  # type: ignore[assignment]  # prototype
        yield Header()
        if app.variant["dash"] == "A":
            with Horizontal():
                yield DataTable(id="sessions", cursor_type="row")
                with Vertical(id="side"):
                    yield Static(id="pipeline", classes="box")
                    yield Static(id="health", classes="box")
                    yield Static("growth · sessions reconciled / new entities per day (14d)", classes="label")
                    yield Sparkline([0], id="spark_rec")
                    yield Sparkline([0], id="spark_ent")
        else:
            yield Static(id="meters", classes="box")
            with Horizontal(id="sparkrow"):
                yield Sparkline([0], id="spark_rec")
                yield Sparkline([0], id="spark_ent")
            yield DataTable(id="sessions", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sessions", DataTable).focus()
        self.render_data()

    def render_data(self) -> None:
        app: Viewer = self.app  # type: ignore[assignment]  # prototype
        data, variant = app.dash_data, app.variant["dash"]
        self.title = f"Context Graph · dashboard {variant} · scope: {app.scope}"
        self.sub_title = app.status_line()
        if data is None:
            return
        table = self.query_one("#sessions", DataTable)
        cursor = table.cursor_row
        table.clear(columns=True)
        table.add_columns(*SESSION_COLUMNS[variant])
        for s in data["sessions"]:
            table.add_row(*session_row(s, variant), key=s["id"])
        if data["sessions"]:
            table.move_cursor(row=min(cursor, len(data["sessions"]) - 1))
        rec, ent = growth_series(data)
        self.query_one("#spark_rec", Sparkline).data = rec
        self.query_one("#spark_ent", Sparkline).data = ent
        if variant == "A":
            self.query_one("#pipeline", Static).update("[b]Pipeline[/]\n" + "\n".join(pipeline_lines(data)))
            self.query_one("#health", Static).update("[b]Capture health[/]\n" + "\n".join(health_hints(data)))
        else:
            self.query_one("#meters", Static).update(
                " │ ".join(health_hints(data)[1:3] + pipeline_lines(data)[:2])
                + "\n[dim]growth ↓ reconciled · new entities[/]"
            )

    def action_open(self) -> None:
        table = self.query_one("#sessions", DataTable)
        if table.row_count:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
            self.app.push_screen(SessionScreen(key))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(SessionScreen(event.row_key.value))


# ---------------------------------------------------------------- session anatomy


def action_label(a: dict) -> str:
    if a["type"] == "tool_call":
        detail = ""
        try:
            inp = json.loads(a["props"] or "{}").get("tool_input", {})
            detail = (
                inp.get("description")
                or inp.get("command")
                or inp.get("file_path")
                or inp.get("pattern")
                or inp.get("url")
                or ""
            )
        except (ValueError, AttributeError):
            pass
        return f"{a['tool']} {str(detail)[:40]}".strip()
    return a["type"] or "?"


def duration(a: dict) -> str:
    s, e = parse_ts(a["at"]), parse_ts(a["result_at"])
    if not (s and e):
        return "…" if a["type"] == "tool_call" else ""
    d = (e - s).total_seconds()
    return f"{d:.1f}s" if d < 60 else f"{int(d // 60)}m{int(d % 60):02d}s"


class SessionScreen(Screen):
    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("g", "entities", "entities of session"),
    ]

    def action_entities(self) -> None:
        self.app.push_screen(EntityList(self.session_id))

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id

    def compose(self) -> ComposeResult:
        app: Viewer = self.app  # type: ignore[assignment]  # prototype
        yield Header()
        with Horizontal():
            with Vertical(id="main"):
                if app.variant["session"] == "A":
                    yield Tree("session", id="timeline")
                else:
                    yield DataTable(id="timeline", cursor_type="row")
                yield Static(id="derived", classes="box")
            yield Static(id="details", classes="box")
        yield Footer()

    def on_mount(self) -> None:
        self.app.load_session(self.session_id)  # type: ignore[attr-defined]  # prototype

    def render_data(self) -> None:
        app: Viewer = self.app  # type: ignore[assignment]  # prototype
        variant = app.variant["session"]
        self.title = f"Context Graph · session {self.session_id[:8]} · variant {variant}"
        self.sub_title = app.status_line()
        data = app.session_data.get(self.session_id)
        if data is None:
            return
        actions = data["actions"]
        times = [parse_ts(a["at"]) for a in actions] + [parse_ts(a["result_at"]) for a in actions]
        times = [t for t in times if t]
        t0, t1 = (min(times), max(times)) if times else (None, None)
        groups = defaultdict(list)
        for a in actions:
            groups[a["agent"]].append(a)
        self._actions = {a["id"]: a for a in actions}

        if variant == "A":
            tree = self.query_one("#timeline", Tree)
            tree.clear()
            tree.root.set_label(f"session {self.session_id[:8]} · {len(actions)} actions · {ago(t1)}")
            tree.root.expand()
            for agent, items in sorted(groups.items(), key=lambda kv: kv[0] is not None):
                parent = (
                    tree.root if agent is None else tree.root.add(f"▾ subagent {agent} ({len(items)})", expand=True)
                )
                for a in items:
                    err = " [red]✕[/]" if is_true(a["result_error"]) else ""
                    b = bar(parse_ts(a["at"]), parse_ts(a["result_at"]), t0, t1)
                    parent.add_leaf(
                        Text.from_markup(f"{action_label(a)[:44]:<44} {duration(a):>7}{err} [cyan]{b}[/]"), data=a["id"]
                    )
        else:
            table = self.query_one("#timeline", DataTable)
            table.clear(columns=True)
            table.add_columns("", "action", "dur", "err", "timeline")
            for agent, items in sorted(groups.items(), key=lambda kv: kv[0] is not None):
                if agent is not None:
                    table.add_row("▾", Text(f"subagent {agent}", style="bold"), "", "", "")
                for a in items:
                    indent = "  " if agent is None else "    "
                    table.add_row(
                        indent,
                        action_label(a)[:50],
                        duration(a),
                        Text("✕", style="red") if is_true(a["result_error"]) else "",
                        Text(bar(parse_ts(a["at"]), parse_ts(a["result_at"]), t0, t1), style="cyan"),
                        key=a["id"],
                    )

        d = data["derived"]
        ents = ", ".join(str(e) for e in d.get("entities", []) if e)
        self.query_one("#derived", Static).update(
            f"[b]Produced[/] (linked at session level, #381)  reconcile: {d.get('rstatus') or '—'}"
            f"  ·  Chunks {d.get('chunks', 0)} · Entities {d.get('entity_count', 0)}\n"
            f"[dim]{ents[:220]}[/]\n"
            f"Episode: {d.get('summary') or '[dim]none yet[/]'}"
        )

    def _show_details(self, action_id) -> None:
        a = self._actions.get(action_id) if action_id else None
        if not a:
            return
        try:
            inp = json.dumps(json.loads(a["props"] or "{}").get("tool_input", {}), indent=1)[:900]
        except ValueError:
            inp = str(a["props"])[:900]
        self.query_one("#details", Static).update(
            f"[b]{a['type']}[/] {a['tool'] or ''}\nat  {a['at']}\ndur {duration(a)}\nagent {a['agent'] or 'main'}\n\n{inp}"
        )

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self._show_details(event.node.data)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_details(event.row_key.value if event.row_key else None)


# ---------------------------------------------------------------- app


class Viewer(App):
    CSS = """
    #sessions { width: 2fr; }
    #side { width: 1fr; }
    .box { border: round $primary; padding: 0 1; height: auto; }
    .label { color: $text-muted; padding: 0 1; }
    Sparkline { height: 3; margin: 0 1; }
    #sparkrow { height: 3; }
    #main { width: 3fr; }
    #details { width: 1fr; height: 1fr; }
    #derived { height: auto; max-height: 12; }
    #entities { width: 3fr; }
    #prov { width: 2fr; height: 1fr; }
    #nbmain { width: 3fr; }
    #evidence { width: 2fr; height: 1fr; }
    #filter { height: 3; }
    """
    BINDINGS: ClassVar = [
        Binding("v", "variant", "variant"),
        Binding("a", "scope", "scope"),
        Binding("r", "poll", "refresh"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self):
        super().__init__()
        self.graph = Graph(load_config())
        self.scope = "user"
        self.variant = {"dash": "A", "session": "A"}
        self.dash_data = None
        self.session_data: dict = {}
        self.last_ok = None
        self.error = None

    def status_line(self) -> str:
        state = f"⚠ stale: {self.error[:60]}" if self.error else "● live"
        return (
            f"{self.graph.url} · user {self.graph.user_id} · {state} · ⟳ {POLL_SECONDS}s · updated {ago(self.last_ok)}"
        )

    def on_mount(self) -> None:
        self.push_screen(Dashboard())
        self.set_interval(POLL_SECONDS, self.action_poll)
        self.action_poll()

    def action_poll(self) -> None:
        self.load_dashboard()
        if isinstance(self.screen, SessionScreen):
            self.load_session(self.screen.session_id)

    # Poll errors must never reach Textual's default handler, which ends the app (#382).
    @work(thread=True, exclusive=True, group="dash")
    def load_dashboard(self) -> None:
        try:
            data = self.graph.dashboard(self.scope)
            self.call_from_thread(self._apply, "dash", data)
        except Exception as exc:
            self.call_from_thread(self._fail, exc)

    @work(thread=True, exclusive=True, group="session")
    def load_session(self, session_id: str) -> None:
        try:
            data = self.graph.session(session_id)
            self.call_from_thread(self._apply, session_id, data)
        except Exception as exc:
            self.call_from_thread(self._fail, exc)

    def _apply(self, key: str, data: dict) -> None:
        if key == "dash":
            self.dash_data = data
        else:
            self.session_data[key] = data
        self.error, self.last_ok = None, datetime.now(timezone.utc)
        if hasattr(self.screen, "render_data"):
            self.screen.render_data()

    def _fail(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        if hasattr(self.screen, "render_data"):
            self.screen.render_data()

    def action_variant(self) -> None:
        if not hasattr(self.screen, "render_data"):
            return
        key = "session" if isinstance(self.screen, SessionScreen) else "dash"
        self.variant[key] = "B" if self.variant[key] == "A" else "A"
        self.screen.refresh(recompose=True)
        self.call_after_refresh(self.screen.render_data)

    def action_scope(self) -> None:
        self.scope = SCOPES[(SCOPES.index(self.scope) + 1) % len(SCOPES)]
        self.load_dashboard()


if __name__ == "__main__":
    Viewer().run()
