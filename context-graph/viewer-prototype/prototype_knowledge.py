"""PROTOTYPE — throwaway. Knowledge-graph view for the #383 viewer prototype (map #374).

Two screens, reached from prototype_viewer.py:
  EntityList     `k` on the dashboard (global, in-scope) or `g` on a session (that session's entities).
                 Sorted by support = distinct in-scope sessions grounding the entity (#381). `/` filters.
  Neighbourhood  `enter` on an entity. Focused-entity ego view (#382) with a provenance pane;
                 `enter` on a neighbour refocuses, the screen stack is the breadcrumb, `esc` pops it.
                 `v` cycles three groupings of the same neighbourhood:
                   A  by relation (type, or LightRAG's first keyword since its edges are all :DIRECTED)
                   B  by the neighbour's entity type
                   C  flat table: direction, relation, neighbour, type, weight, description
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import ClassVar

from rich.markup import escape as esc
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree

PER_GROUP = 12
VARIANTS = ["A", "B", "C"]

# Entities are whatever MENTIONED_IN a Chunk (unstructured2graph owns that contract), so the queries
# don't depend on LightRAG's `base` label. Name falls back across backends' id properties.
NAME = "coalesce(e.entity_id, e.name, e.id)"
LABELS: dict[str, str] = {}  # entity key -> display label, filled from every query result


def name(key) -> str:
    return LABELS.get(key) or str(key)


def entity_list(graph, session_ids: list[str], only_session: str | None) -> list[dict]:
    return graph.rows(
        f"""
        MATCH (s:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
        WHERE s.session_id IN $ids AND ($only IS NULL OR s.session_id = $only)
        WITH e, count(DISTINCT s) AS support
        OPTIONAL MATCH (e)-[r]-(f)
        WHERE type(r) <> 'MENTIONED_IN'
        WITH e, support, count(DISTINCT r) AS degree
        RETURN {NAME} AS id, coalesce(e.text, {NAME}) AS label, e.entity_type AS type, support, degree, e.created_at AS created
        ORDER BY support DESC, degree DESC, id
        LIMIT 1000
        """,
        ids=session_ids,
        only=only_session,
    )


def entity_provenance(graph, entity_id: str, session_ids: list[str]) -> dict:
    head = graph.rows(
        f"MATCH (e) WHERE {NAME} = $id AND exists((e)-[:MENTIONED_IN]->()) "
        "RETURN coalesce(e.description, e.text) AS description, e.entity_type AS type, labels(e) AS labels LIMIT 1",
        id=entity_id,
    )
    sessions = graph.rows(
        f"""
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)<-[:HAS_CHUNK]-(:Action)<-[:HAS_ACTION]-(s:Session)
        WHERE {NAME} = $id AND s.session_id IN $ids
        OPTIONAL MATCH (s)-[:HAS_EPISODE]->(ep:Episode)
        RETURN s.session_id AS session, s.reconciled_at AS reconciled_at, ep.summary AS summary,
               count(DISTINCT c) AS chunks
        ORDER BY reconciled_at DESC
        """,
        id=entity_id,
        ids=session_ids,
    )
    return {**(head[0] if head else {}), "sessions": sessions}


def neighbours(graph, entity_id: str) -> list[dict]:
    return graph.rows(
        f"""
        MATCH (e)-[r]-(f)
        WHERE {NAME} = $id AND type(r) <> 'MENTIONED_IN' AND exists((f)-[:MENTIONED_IN]->())
        RETURN startNode(r) = e AS outgoing, type(r) AS rel_type, r.keywords AS keywords,
               coalesce(r.weight, r.confidence) AS weight, coalesce(r.description, r.text) AS description,
               coalesce(r.created_at, r.valid_at) AS created, r.ontology_conformant AS conformant,
               coalesce(f.entity_id, f.name, f.id) AS id, coalesce(f.text, f.entity_id, f.name, f.id) AS label,
               f.entity_type AS type
        """,
        id=entity_id,
    )


def relation_name(n: dict) -> str:
    if n["rel_type"] and n["rel_type"] != "DIRECTED":
        return n["rel_type"]
    return (n["keywords"] or "related").split(",")[0].strip() or "related"


def arrow(n: dict) -> str:
    # LightRAG's :DIRECTED is explicitly undirected (#349), so its direction is shown but not trusted.
    if n["rel_type"] == "DIRECTED":
        return "–"
    return "→" if n["outgoing"] else "←"


def off_ontology(n: dict) -> bool:
    return str(n.get("conformant")).lower() == "false"


def fmt_epoch(value) -> str:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value or "—")[:16]


def scoped_session_ids(app) -> list[str]:
    return [s["id"] for s in (app.dash_data or {}).get("sessions", [])]


def provenance_markup(entity_id: str, prov: dict) -> Text:
    # Graph text never goes through markup: source spans contain `[...]` and `\` that the parser would act on.
    out = Text()
    out.append(name(entity_id), style="bold").append(f"  {prov.get('type') or ''}\n", style="dim")
    if prov.get("description"):
        out.append(str(prov["description"])[:400] + "\n")
    sessions = prov.get("sessions", [])
    out.append(f"\nSessions in scope ({len(sessions)})", style="bold")
    out.append("  provenance is session-level (#381)\n", style="dim")
    for s in sessions[:6]:
        out.append(f"▸ {s['session'][:8]}  {str(s['reconciled_at'] or '')[:10]}  chunks {s['chunks']}\n")
        if s["summary"]:
            out.append(f"  {str(s['summary'])[:160]}\n", style="dim")
    if not sessions:
        out.append("none in the current scope — press a on the dashboard to widen\n", style="dim")
    return out


class EntityList(Screen):
    AUTO_FOCUS = "#entities"
    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("slash", "filter", "filter"),
        Binding("r", "reload", "reload"),
    ]

    def __init__(self, session_id: str | None = None):
        super().__init__()
        self.session_id = session_id
        self.rows: list[dict] = []
        self._prov_target = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="/ filter entities", id="filter")
        with Horizontal():
            yield DataTable(id="entities", cursor_type="row")
            yield Static(id="prov", classes="box")
        yield Footer()

    def on_mount(self) -> None:
        where = f"session {self.session_id[:8]}" if self.session_id else f"scope: {self.app.scope}"
        self.title = f"Context Graph ▸ Knowledge ▸ entities ({where})"
        self.sub_title = self.app.status_line()
        self.query_one("#entities", DataTable).focus()
        self.action_reload()

    def action_reload(self) -> None:
        self.load()

    @work(thread=True, exclusive=True, group="entities")
    def load(self) -> None:
        try:
            rows = entity_list(self.app.graph, scoped_session_ids(self.app), self.session_id)
            self.app.call_from_thread(self._apply, rows)
        except Exception as exc:
            self.app.call_from_thread(self.app._fail, exc)

    def _apply(self, rows: list[dict]) -> None:
        LABELS.update({r["id"]: r["label"] for r in rows})
        self.rows = rows
        self._fill(self.query_one("#filter", Input).value)

    def _fill(self, needle: str) -> None:
        table = self.query_one("#entities", DataTable)
        table.clear(columns=True)
        table.add_columns("entity", "type", "support", "degree", "first seen")
        shown = [r for r in self.rows if needle.lower() in str(r["label"]).lower()]
        for r in shown:
            table.add_row(
                str(r["label"])[:48], r["type"] or "—", r["support"], r["degree"], fmt_epoch(r["created"]), key=r["id"]
            )
        self.sub_title = f"{len(shown)} of {len(self.rows)} entities · {self.app.status_line()}"

    def action_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._fill(event.value)

    def on_input_submitted(self, _: Input.Submitted) -> None:
        self.query_one("#entities", DataTable).focus()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value:
            self.load_prov(event.row_key.value)

    @work(thread=True, exclusive=True, group="prov")
    def load_prov(self, entity_id: str) -> None:
        try:
            prov = entity_provenance(self.app.graph, entity_id, scoped_session_ids(self.app))
            self.app.call_from_thread(self.query_one("#prov", Static).update, provenance_markup(entity_id, prov))
        except Exception as exc:
            self.app.call_from_thread(self.app._fail, exc)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.app.push_screen(Neighbourhood(event.row_key.value, [event.row_key.value]))


class Neighbourhood(Screen):
    AUTO_FOCUS = "#nb"
    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("v", "cycle_variant", "grouping"),
    ]
    variant = "A"  # shared across the stack so the grouping survives refocusing

    def __init__(self, entity_id: str, path: list[str]):
        super().__init__()
        self.entity_id = entity_id
        self.path = path
        self.items: list[dict] = []
        self.prov: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="nbmain"):
                if Neighbourhood.variant == "C":
                    yield DataTable(id="nb", cursor_type="row")
                else:
                    yield Tree(Text(name(self.entity_id)), id="nb")
            yield Static(id="evidence", classes="box")
        yield Footer()

    def on_mount(self) -> None:
        crumbs = " ▸ ".join(name(p)[:24] for p in self.path[-5:])
        self.title = f"Knowledge ▸ {crumbs}   [grouping {Neighbourhood.variant}]"
        self.sub_title = self.app.status_line()
        self.query_one("#nb").focus()
        self.load()

    @work(thread=True, exclusive=True, group="nb")
    def load(self) -> None:
        try:
            items = neighbours(self.app.graph, self.entity_id)
            prov = entity_provenance(self.app.graph, self.entity_id, scoped_session_ids(self.app))
            self.app.call_from_thread(self._apply, items, prov)
        except Exception as exc:
            self.app.call_from_thread(self.app._fail, exc)

    def _apply(self, items: list[dict], prov: dict) -> None:
        LABELS.update({n["id"]: n["label"] for n in items})
        self.items, self.prov = items, prov
        self._render_all()

    def _render_all(self) -> None:
        self.query_one("#evidence", Static).update(provenance_markup(self.entity_id, self.prov))
        if Neighbourhood.variant == "C":
            self._render_table()
        else:
            self._render_tree()

    def _render_tree(self) -> None:
        tree = self.query_one("#nb", Tree)
        tree.clear()
        tree.root.set_label(Text(f"{name(self.entity_id)}  ·  {len(self.items)} neighbours"))
        tree.root.expand()
        groups: dict[str, list[dict]] = defaultdict(list)
        for n in self.items:
            key = f"{arrow(n)} {relation_name(n)}" if Neighbourhood.variant == "A" else (n["type"] or "untyped")
            groups[key].append(n)
        for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            node = tree.root.add(Text(f"{key} ({len(members)})"), expand=len(groups) <= 6)
            members.sort(key=lambda n: -float(n["weight"] or 0))
            for n in members[:PER_GROUP]:
                on_path = "  [yellow]↺ on path[/]" if n["id"] in self.path else ""
                if off_ontology(n):
                    on_path += "  [red]✕ off-ontology[/]"
                detail = n["type"] if Neighbourhood.variant == "A" else f"{arrow(n)} {relation_name(n)}"
                label = Text(str(n["label"])[:40]) + Text.from_markup(f"  [dim]{esc(str(detail))}[/]{on_path}")
                node.add_leaf(label, data=n)
            if len(members) > PER_GROUP:
                node.add_leaf(Text(f"… + {len(members) - PER_GROUP} more", style="dim"))

    def _render_table(self) -> None:
        table = self.query_one("#nb", DataTable)
        table.clear(columns=True)
        table.add_columns("", "relation", "neighbour", "type", "w", "description")
        for i, n in enumerate(sorted(self.items, key=lambda n: (relation_name(n), -float(n["weight"] or 0)))):
            table.add_row(
                arrow(n),
                relation_name(n),
                str(n["label"])[:32] + (" ↺" if n["id"] in self.path else ""),
                n["type"] or "—",
                f"{float(n['weight']):.2f}" if n["weight"] else "",
                Text(str(n["description"] or "")[:60], style="red" if off_ontology(n) else ""),
                key=str(i),
            )

    def _evidence(self, n: dict | None) -> None:
        if not n:
            return
        out = Text()
        out.append(name(self.entity_id), style="bold").append(f" {arrow(n)} ")
        out.append(relation_name(n), style="bold").append(f" {arrow(n)} ").append(str(n["label"]), style="bold")
        out.append(
            f"\nrel type {n['rel_type']} · keywords {n['keywords'] or '—'} · weight {n['weight'] or '—'}"
            f" · {fmt_epoch(n['created'])}\n",
            style="dim",
        )
        if off_ontology(n):
            out.append("outside the ontology's domain/range\n", style="red")
        out.append("\nEvidence", style="bold").append(" (relation description, or GLiNER2's source span)\n")
        out.append(
            "\n".join(f"• {part.strip()}" for part in str(n["description"] or "—").replace("\\n", "\n").split("<SEP>"))[
                :600
            ]
        )
        out.append("\n\nenter: refocus on this neighbour · esc: back along the breadcrumb", style="dim")
        self.query_one("#evidence", Static).update(out)

    def _selected(self) -> dict | None:
        widget = self.query_one("#nb")
        if isinstance(widget, Tree):
            return widget.cursor_node.data if widget.cursor_node else None
        if widget.row_count:
            key = widget.coordinate_to_cell_key(widget.cursor_coordinate).row_key.value
            return sorted(self.items, key=lambda n: (relation_name(n), -float(n["weight"] or 0)))[int(key)]
        return None

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if event.node.data:
            self._evidence(event.node.data)
        elif event.node.is_root:
            self.query_one("#evidence", Static).update(provenance_markup(self.entity_id, self.prov))

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data:
            self._refocus(event.node.data)

    def on_data_table_row_highlighted(self, _: DataTable.RowHighlighted) -> None:
        self._evidence(self._selected())

    def on_data_table_row_selected(self, _: DataTable.RowSelected) -> None:
        if n := self._selected():
            self._refocus(n)

    def _refocus(self, n: dict) -> None:
        self.app.push_screen(Neighbourhood(n["id"], [*self.path, n["id"]]))

    def action_cycle_variant(self) -> None:
        Neighbourhood.variant = VARIANTS[(VARIANTS.index(Neighbourhood.variant) + 1) % len(VARIANTS)]
        self.refresh(recompose=True)
        self.call_after_refresh(self._after_recompose)

    def _after_recompose(self) -> None:
        self.title = self.title.rsplit("[grouping", 1)[0] + f"[grouping {Neighbourhood.variant}]"
        self.query_one("#nb").focus()
        self._render_all()
