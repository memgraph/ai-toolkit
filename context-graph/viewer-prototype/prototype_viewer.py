"""PROTOTYPE — throwaway. Answers #383: do the #380/#381/#382 view designs read well on a real graph?

Two structurally different variants per screen, switched with `v`:
  Dashboard  A = lazygit-style panes (sessions | pipeline / health / growth)
             B = k9s-style single wide table under a one-line meter strip
  Session    A = Tree with time bars (folding subagents)        (#382 option a)
             B = DataTable with indent + bar column             (#382 option b)

Run:  uv run --no-project --python 3.12 --with textual --with neo4j context-graph/viewer-prototype/prototype_viewer.py
Keys: v variant · a scope (user → user+unattributed → all) · enter drill in · esc back · r refresh · q quit

Reads the same config file as memgraph-context-graph's hooks (CONTEXT_GRAPH_CONFIG or
~/.config/context-graph/config.toml), including `edition`: Community scopes to [identity] user_id, Enterprise to the
Memgraph login. A real viewer would ship as `memgraph-context-graph ui`.
Read-only. Polls every 3 s. The local capture-failure log (#378) does not exist yet and is stubbed.

Tracks main@1b046f6: no Episodes (#520), no reconciliation LLM call (so no memory-cost tokens), GLiNER2 entities keyed
by hash with `text` as the label, memory files reconciled on their own, resources-graph Touches, ontology versions.
Not covered: Enterprise shared spaces (SpaceMembership databases), which need a licensed Memgraph to try.
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
from rich.markup import escape as esc
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
    """All Cypher lives here (query/renderer split required by #381).

    Never RETURN whole Action/Memory/entity nodes or properties(): since main@1b046f6 they carry embedding vectors.
    """

    def __init__(self, cfg: dict):
        mg = cfg.get("memgraph", {})
        self.url = mg.get("url", "bolt://localhost:7687")
        self.edition = str(cfg.get("edition", "community")).lower()
        # Identity follows memgraph-context-graph's SpaceResolver: Community reads [identity] user_id; Enterprise
        # makes the Memgraph login the user and its personal space the login's main database (database=None).
        if self.edition == "enterprise":
            self.user_id = mg.get("user") or None
            self.database = None
        else:
            self.user_id = cfg.get("identity", {}).get("user_id")
            database = mg.get("database") or "memgraph"
            self.database = None if database == "memgraph" else database
        self.driver = GraphDatabase.driver(self.url, auth=(mg.get("user", ""), mg.get("password", "")))

    def rows(self, query: str, **params) -> list[dict]:
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(query, **params)]

    def dashboard(self, scope: str) -> dict:
        # Reconcile synthesizes `anon-<session_id>` users for sessions captured without one, so those count as
        # unattributed alongside sessions with no owner at all.
        sessions = self.rows(
            """
            MATCH (s:Session)
            OPTIONAL MATCH (u:User)-[:HAD_SESSION]->(s)
            // A session can end up with two owners: the anon-<id> user reconcile synthesizes, and the real user
            // linked by a later SessionStart. Prefer the real one.
            WITH s, collect(u.user_id) AS owners
            WITH s, head([o IN owners WHERE NOT o STARTS WITH 'anon-'] + owners) AS owner
            WITH s, owner, owner IS NULL OR owner STARTS WITH 'anon-' AS unattributed
            WHERE $scope = 'all' OR owner = $uid OR ($scope = 'user+unattributed' AND unattributed)
            OPTIONAL MATCH (s)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action)
            WITH s, owner, unattributed, count(CASE WHEN a.action_type <> 'tool_result' THEN 1 END) AS actions,
                 max(a.timestamp) AS last_event,
                 sum(CASE WHEN toString(a.is_error) = 'True' OR a.is_error = true THEN 1 ELSE 0 END) AS errors
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
            WITH s, owner, unattributed, actions, last_event, errors, count(DISTINCT c) AS chunks
            RETURN s.session_id AS id, owner, unattributed, s.working_directory AS wd, s.status AS status,
                   s.started_at AS started, s.ended_at AS ended, s.reconciliation_status AS rstatus,
                   s.reconciliation_error AS rerror, s.reconciled_at AS reconciled_at,
                   s.reextraction_error AS reerror, s.extraction_backend AS backend,
                   s.ontology_version AS ontology_version, s.embedding_status AS estatus,
                   s.embedding_error AS eerror, s.total_input_tokens AS in_tok, s.total_output_tokens AS out_tok,
                   actions, last_event, errors, chunks
            ORDER BY coalesce(last_event, s.started_at) DESC
            """,
            scope=scope,
            uid=self.user_id,
        )
        hidden = self.rows(
            """
            MATCH (s:Session) OPTIONAL MATCH (u:User)-[:HAD_SESSION]->(s)
            WITH s, collect(u.user_id) AS owners
            WHERE all(o IN owners WHERE o STARTS WITH 'anon-') RETURN count(s) AS n
            """
        )[0]["n"]
        ids = [s["id"] for s in sessions]
        # Memory files are reconciled on their own (#498) and belong to the user, not to a session.
        memories = self.rows(
            """
            MATCH (u:User)-[:HAS_MEMORY]->(m:Memory)
            WHERE $scope = 'all' OR u.user_id = $uid
            RETURN m.path AS path, m.extraction_status AS status, m.extraction_error AS error,
                   m.updated_at AS updated_at
            """,
            scope=scope,
            uid=self.user_id,
        )
        touches = self.rows(
            """
            MATCH (t:Touch)-[:IN_SESSION]->(s:Session) WHERE s.session_id IN $ids
            RETURN coalesce(t.status, 'unknown') AS status, count(t) AS n
            """,
            ids=ids,
        )
        ontology = self.rows(
            """
            MATCH (u:User) WHERE $scope = 'all' OR u.user_id = $uid
            OPTIONAL MATCH (u)-[:ADOPTED]->(v:OntologyVersion)
            RETURN u.user_id AS user, max(v.version) AS version, u.derive_milestone AS milestone
            """,
            scope=scope,
            uid=self.user_id,
        )
        new_entities = self.rows(
            """
            MATCH (s:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
            WHERE s.session_id IN $ids AND s.reconciled_at IS NOT NULL
            WITH e, min(s.reconciled_at) AS first
            RETURN substring(toString(first), 0, 10) AS day, count(e) AS n
            """,
            ids=ids,
        )
        return {
            "sessions": sessions,
            "unattributed": hidden,
            "memories": memories,
            "touches": {r["status"]: r["n"] for r in touches},
            "ontology": [o for o in ontology if not str(o["user"]).startswith("anon-")],
            "new_entities": new_entities,
        }

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
        # No Episode since #520: what a session produced is its chunks, the entities they mention, and memory files.
        derived = self.rows(
            """
            MATCH (s:Session {session_id: $id})
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(e)
            WITH s, count(DISTINCT c) AS chunks, count(DISTINCT e) AS entity_count,
                 collect(DISTINCT coalesce(e.text, e.name, e.entity_id))[..12] AS entities
            OPTIONAL MATCH (s)-[:PRODUCED_MEMORY]->(m:Memory)
            RETURN s.reconciliation_status AS rstatus, s.reconciliation_error AS rerror,
                   s.extraction_backend AS backend, s.ontology_version AS ontology_version,
                   s.embedding_status AS estatus, chunks, entity_count, entities,
                   collect(DISTINCT m.path) AS memories
            """,
            id=session_id,
        )
        touches = self.rows(
            """
            MATCH (t:Touch)-[:IN_SESSION]->(:Session {session_id: $id})
            OPTIONAL MATCH (t)-[:CAUSED_BY]->(c:Action)
            OPTIONAL MATCH (t)-[:TOUCHED]->(r)
            RETURN c.action_id AS action_id, t.status AS status, t.served_from_memory AS from_memory,
                   [l IN labels(r) WHERE l <> 'Resource'][0] AS kind, coalesce(r.address, r.node_id) AS address
            """,
            id=session_id,
        )
        by_action = defaultdict(list)
        for t in touches:
            by_action[t["action_id"]].append(t)
        return {
            "actions": [a for a in actions if a["type"] != "tool_result"],
            "derived": derived[0] if derived else {},
            "touches": by_action,
        }


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
    unresolved = data["touches"].get("pending", 0) + data["touches"].get("unresolved", 0)
    if unresolved:
        hints.append(f"[yellow]{unresolved} resource touch(es) pending/unresolved[/] (resources-graph sweep)")
    if data["unattributed"]:
        hints.append(f"{data['unattributed']} unattributed session(s) in graph — press [b]a[/] to widen scope")
    return hints


def count_line(label: str, values) -> str:
    counts = Counter(v or "not marked" for v in values)
    return f"{label}: " + ("  ".join(f"{k} [b]{v}[/]" for k, v in sorted(counts.items())) or "[dim]none[/]")


def pipeline_lines(data: dict) -> list[str]:
    sessions = data["sessions"]
    # Every TURN_END flips a session back to pending, so an active session reads `pending` even after a reconcile.
    lines = [
        count_line("sessions", (s["rstatus"] for s in sessions)) + "  [dim](pending returns after each turn)[/]",
        count_line("memory files", (m["status"] for m in data["memories"])),
        count_line("embeddings", (s["estatus"] for s in sessions)),
    ]
    for o in data["ontology"]:
        version = "default (v0)" if o["version"] is None else f"v{o['version']}"
        lines.append(f"ontology for {o['user']}: [b]{version}[/]  derive milestone {o['milestone'] or '—'}")
    backends = Counter(s["backend"] for s in sessions if s["backend"])
    if backends:
        lines.append("extracted by: " + ", ".join(f"{b} {n}" for b, n in backends.items()))
    tokens = sum((s["in_tok"] or 0) + (s["out_tok"] or 0) for s in sessions)
    # Reconcile makes no LLM call since #520, so the only token spend to show is the harness's own (capture side).
    lines.append(f"capture tokens: [b]{tokens}[/]" if tokens else "capture tokens: [dim]not reported by the runtime[/]")
    for s in sessions:
        for kind, err in (("reconcile", s["rerror"]), ("re-extract", s["reerror"]), ("embed", s["eerror"])):
            if err and (kind != "reconcile" or s["rstatus"] == "failed"):
                lines.append(f"[red]✕ {s['id'][:8]} {kind}[/] {esc(str(err)[:60])}")
    for m in data["memories"]:
        if m["status"] == "failed":
            lines.append(f"[red]✕ memory {esc(str(m['path']))}[/] {esc(str(m['error'])[:50])}")
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
        Text("unattributed", style="dim") if s["unattributed"] else s["owner"],
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


def touch_marker(touches: list[dict] | None) -> str:
    """A resources-graph Touch caused by this tool call: ◆ resolved, ◇ still pending/unresolved."""
    if not touches:
        return ""
    return "".join("◆" if t["status"] == "resolved" else "◇" for t in touches) + " "


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
                    label = Text(f"{action_label(a)[:44]:<44} {duration(a):>7}")
                    if is_true(a["result_error"]):
                        label.append(" ✕", style="red")
                    label.append(f" {touch_marker(data['touches'].get(a['id']))}", style="magenta")
                    label.append(bar(parse_ts(a["at"]), parse_ts(a["result_at"]), t0, t1), style="cyan")
                    parent.add_leaf(label, data=a["id"])
        else:
            table = self.query_one("#timeline", DataTable)
            table.clear(columns=True)
            table.add_columns("", "action", "dur", "err", "res", "timeline")
            for agent, items in sorted(groups.items(), key=lambda kv: kv[0] is not None):
                if agent is not None:
                    table.add_row("▾", Text(f"subagent {agent}", style="bold"), "", "", "", "")
                for a in items:
                    indent = "  " if agent is None else "    "
                    table.add_row(
                        indent,
                        action_label(a)[:50],
                        duration(a),
                        Text("✕", style="red") if is_true(a["result_error"]) else "",
                        Text(touch_marker(data["touches"].get(a["id"])), style="magenta"),
                        Text(bar(parse_ts(a["at"]), parse_ts(a["result_at"]), t0, t1), style="cyan"),
                        key=a["id"],
                    )

        self._touches = data["touches"]
        d = data["derived"]
        version = d.get("ontology_version")
        out = Text()
        out.append("Produced", style="bold").append(" (linked at session level, #381)  ")
        out.append(f"reconcile {d.get('rstatus') or '—'}  ·  backend {d.get('backend') or '—'}  ·  ")
        out.append(f"ontology {'—' if version is None else f'v{version}'}  ·  embeddings {d.get('estatus') or '—'}\n")
        out.append(f"Chunks {d.get('chunks', 0)} · Entities {d.get('entity_count', 0)}  ")
        out.append(", ".join(str(e) for e in d.get("entities", []) if e)[:200], style="dim")
        if d.get("memories"):
            out.append("\nMemory files written: " + ", ".join(str(p) for p in d["memories"])[:200])
        if d.get("rstatus") == "failed" and d.get("rerror"):
            out.append(f"\n✕ {str(d['rerror'])[:200]}", style="red")
        self.query_one("#derived", Static).update(out)

    def _show_details(self, action_id) -> None:
        a = self._actions.get(action_id) if action_id else None
        if not a:
            return
        try:
            inp = json.dumps(json.loads(a["props"] or "{}").get("tool_input", {}), indent=1)[:900]
        except ValueError:
            inp = str(a["props"])[:900]
        out = Text()
        out.append(f"{a['type']}", style="bold").append(f" {a['tool'] or ''}\nat  {a['at']}\ndur {duration(a)}\n")
        out.append(f"agent {a['agent'] or 'main'}\n")
        for t in getattr(self, "_touches", {}).get(a["id"], []):
            source = "memory" if is_true(t["from_memory"]) else "live"
            out.append(f"touched {t['kind']} {t['address']} · {t['status']} · {source}\n", style="magenta")
        out.append(f"\n{inp}")
        self.query_one("#details", Static).update(out)

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

    def scope_user(self):
        """The user whose memory files count, or None for scope `all`."""
        return None if self.scope == "all" else self.graph.user_id

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
