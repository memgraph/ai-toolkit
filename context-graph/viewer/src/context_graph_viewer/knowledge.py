"""The knowledge graph: an entity list leading to one entity's neighbourhood, one hop at a time (#381, #382, #383)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, Static, Tree

from . import fmt, queries
from .base import Pane, PolledScreen, cursor_key, sync_table

if TYPE_CHECKING:
    from collections.abc import Iterable

    from textual.widget import Widget

    from .source import Reader

PER_GROUP = 12
DEFAULT_FLOOR = 0.5
FLOOR_STEP = 0.1

TREE = "tree"
TABLE = "table"


def relation_name(row: dict[str, Any]) -> str:
    """The relation's type, or for LightRAG's untyped ``:DIRECTED`` edges its first keyword."""
    if row["rel_type"] and row["rel_type"] != "DIRECTED":
        return row["rel_type"]
    return (str(row["keywords"] or "related").split(",")[0].strip()) or "related"


def arrow(row: dict[str, Any]) -> str:
    """An arrow from the focused entity, or a dash for LightRAG's ``:DIRECTED``, which means undirected."""
    if row["rel_type"] == "DIRECTED":
        return "–"
    return "→" if row["outgoing"] else "←"


def strength(row: dict[str, Any]) -> float:
    """What a neighbour row sorts by: GLiNER2's confidence, else LightRAG's weight."""
    value = row["confidence"] if row["confidence"] is not None else row["weight"]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def above_floor(row: dict[str, Any], floor: float) -> bool:
    """Whether a row clears the confidence floor; rows without a confidence (LightRAG) always do."""
    return row["confidence"] is None or strength(row) >= floor


def is_typed(rows: list[dict[str, Any]]) -> bool:
    """Whether relations carry real types; LightRAG's are all ``:DIRECTED``, which can't be grouped (#383)."""
    return any(row["rel_type"] != "DIRECTED" for row in rows)


def provenance_text(focus: queries.Focus, prov: dict[str, Any]) -> Text:
    """The provenance pane: description, in-scope sessions and memory files grounding the entity.

    Graph text is appended as plain text, never parsed as markup: source spans hold ``[``, ``]`` and ``\\``.
    """
    out = Text()
    out.append(focus.label, style="bold").append(f"  {prov.get('type') or ''}", style="dim")
    if len(focus.keys) > 1:
        out.append(f"  ({len(focus.keys)} nodes with this name)", style="dim")
    out.append("\n")
    if prov.get("description"):
        out.append(fmt.clip(prov["description"], 400) + "\n")
    sessions = prov.get("sessions", [])
    out.append(f"\nsessions in scope ({len(sessions)})", style="bold")
    out.append("  provenance is per session", style="dim")
    for s in sessions[:8]:
        out.append(f"\n▸ {s['session'][:8]}  {str(s['reconciled_at'] or '')[:10]}  {fmt.project_name(s['wd'])}")
        out.append(f"  chunks {s['chunks']}", style="dim")
    if not sessions:
        out.append("\nnone in the current scope · a on the dashboard widens it", style="dim")
    if prov.get("memories"):
        out.append(f"\n\nmemory files ({len(prov['memories'])})", style="bold")
        for path in prov["memories"][:8]:
            out.append(f"\n▸ {path}")
    return out


def evidence_text(focus: queries.Focus, row: dict[str, Any]) -> Text:
    """The evidence pane for one relation: its strongest edge's text and where it sits in the ontology."""
    out = Text()
    out.append(focus.label, style="bold").append(f" {arrow(row)} ").append(relation_name(row), style="bold")
    out.append(f" {arrow(row)} ").append(str(row["label"]), style="bold")
    facts = [f"{row['edges']} edge(s)", f"role {row['role'] or '—'}"]
    if row["confidence"] is not None:
        facts.append(f"confidence {strength(row):.2f}")
    if row["weight"] is not None:
        facts.append(f"weight {row['weight']}")
    if row["keywords"]:
        facts.append(f"keywords {row['keywords']}")
    if row["created"]:
        facts.append(str(row["created"])[:16])
    out.append("\n" + " · ".join(facts) + "\n", style="dim")
    if row["nonconformant"]:
        out.append("outside the ontology's domain or range\n", style="red")
    out.append("\nevidence\n", style="bold")
    parts = str(row["evidence"] or "—").replace("\\n", "\n").split("<SEP>")
    out.append(fmt.clip("\n".join(f"• {part.strip()}" for part in parts), 800) if len(parts) > 1 else parts[0][:800])
    out.append("\n\nenter refocuses here · esc goes back", style="dim")
    return out


class EntityListScreen(PolledScreen[list[dict[str, Any]]]):
    """In-scope entities merged by name: globally from the dashboard, or one session's from its anatomy."""

    PANES: ClassVar = ("#entities", "#prov-pane", "#filter")
    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("slash", "filter", "filter"),
        Binding("h", "toggle_values", "value types"),
        Binding("s", "toggle_sort", "sort"),
    ]

    def __init__(self, session_id: str | None = None) -> None:
        super().__init__()
        self.session_id = session_id
        self.hide_values = True
        self.by_recency = False
        self._rows: dict[str, dict[str, Any]] = {}

    def compose_body(self) -> Iterable[Widget]:
        """Filter box over the entity table, provenance on the right."""
        yield Input(placeholder="/ filter entities", id="filter")
        with Horizontal():
            table: DataTable = DataTable(id="entities", cursor_type="row")
            table.border_title = "1 Entities"
            yield table
            with Pane(id="prov-pane", classes="pane") as pane:
                pane.border_title = "2 Provenance"
                yield Static(id="prov")

    def on_mount(self) -> None:
        """Focus the table."""
        self.query_one("#entities", DataTable).focus()
        super().on_mount()

    def title_text(self) -> str:
        """Where this list came from."""
        where = f"session {self.session_id[:8]}" if self.session_id else "all in scope"
        return f"Context Graph ▸ knowledge ▸ entities ({where})"

    def fetch(self, reader: Reader) -> list[dict[str, Any]]:
        """Load the merged entity list."""
        return queries.entities(reader, self.viewer.scope, session_id=self.session_id, hide_values=self.hide_values)

    def show(self, data: list[dict[str, Any]]) -> None:
        """Fill the table from the latest list and the current filter."""
        self._fill()

    def _fill(self) -> None:
        rows = list(self.data or [])
        if self.by_recency:
            rows.sort(key=lambda e: e["last_seen"] or "", reverse=True)
        needle = self.query_one("#filter", Input).value.casefold()
        shown = [e for e in rows if needle in e["label"].casefold()]
        self._rows = {e["label"].casefold(): e for e in shown}
        columns = [("entity", "entity"), ("type", "type"), ("support", "support"), ("degree", "degree")]
        columns += [("mentions", "mentions"), ("seen", "last seen")]
        cells = [
            (
                e["label"].casefold(),
                [
                    fmt.clip(e["label"], 48) + (f" ×{len(e['keys'])}" if len(e["keys"]) > 1 else ""),
                    ", ".join(e["types"]) or "—",
                    e["support"],
                    e["degree"],
                    e["mentions"],
                    str(e["last_seen"] or "—")[:10],
                ],
            )
            for e in shown
        ]
        sync_table(self.query_one("#entities", DataTable), columns, cells)
        self.draw_status()

    def status_note(self) -> str:
        """Counts, sort and value-type visibility."""
        total = len(self.data or [])
        order = "most recent" if self.by_recency else "support"
        values = "shown" if not self.hide_values else "hidden (h)"
        return f"{len(self._rows)} of {total} entities · sorted by {order} (s) · value types {values}"

    def action_filter(self) -> None:
        """Focus the filter."""
        self.query_one("#filter", Input).focus()

    def on_input_changed(self, _: Input.Changed) -> None:
        """Re-filter as the text changes."""
        self._fill()

    def on_input_submitted(self, _: Input.Submitted) -> None:
        """Back to the table."""
        self.query_one("#entities", DataTable).focus()

    def action_toggle_values(self) -> None:
        """Show or hide value types (dates, durations, amounts)."""
        self.hide_values = not self.hide_values
        self.action_poll()

    def action_toggle_sort(self) -> None:
        """Sort by support or by most recently seen."""
        self.by_recency = not self.by_recency
        self._fill()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Load the highlighted entity's provenance."""
        entity = self._rows.get(event.row_key.value or "") if event.row_key else None
        if entity:
            focus = queries.Focus(tuple(entity["keys"]), entity["label"])
            reader, scope = self.viewer.reader, self.viewer.scope
            self.run_worker(
                lambda: self._load_provenance(reader, scope, focus), thread=True, exclusive=True, group="prov"
            )

    def _load_provenance(self, reader: Reader, scope: queries.Scope, focus: queries.Focus) -> None:
        try:
            prov = queries.provenance(reader, focus, scope)
        except Exception as exc:  # Same rule as a poll: report on this view, never end the app.
            self.app.call_from_thread(self._failed, f"{type(exc).__name__}: {exc}")
            return
        self.app.call_from_thread(self.query_one("#prov", Static).update, provenance_text(focus, prov))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter opens the entity's neighbourhood."""
        entity = self._rows.get(event.row_key.value or "")
        if entity:
            focus = queries.Focus(tuple(entity["keys"]), entity["label"])
            self.app.push_screen(NeighbourhoodScreen(focus, [focus]))


@dataclass
class Neighbourhood:
    """One poll of a neighbourhood screen."""

    rows: list[dict[str, Any]]
    provenance: dict[str, Any]


class NeighbourhoodScreen(PolledScreen[Neighbourhood]):
    """One entity and its neighbours, grouped by relation type; ``enter`` refocuses, the screen stack is the path."""

    PANES: ClassVar = ("#nb", "#evidence-pane")
    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("v", "toggle_layout", "tree/table"),
        Binding("plus,equals_sign", "floor(1)", "raise floor"),
        Binding("minus", "floor(-1)", "lower floor"),
    ]
    #: Shared across the stack so the layout and floor survive refocusing.
    view_mode: ClassVar[str] = TREE
    floor: ClassVar[float] = DEFAULT_FLOOR

    def __init__(self, focus: queries.Focus, path: list[queries.Focus]) -> None:
        super().__init__()
        self.focus_entity = focus
        self.path = path
        self._shown: list[dict[str, Any]] = []
        self._signature: tuple[Any, ...] | None = None
        self._mounted_layout = TREE

    def _wanted_layout(self) -> str:
        """The table when asked for, or when relations are untyped (LightRAG) and can't be grouped (#383)."""
        if NeighbourhoodScreen.view_mode == TABLE:
            return TABLE
        if self.data is not None and not is_typed(self.data.rows):
            return TABLE
        return TREE

    def compose_body(self) -> Iterable[Widget]:
        """The neighbours (tree or table) and the evidence pane."""
        self._mounted_layout = self._wanted_layout()
        with Horizontal():
            with Vertical(id="nbmain"):
                neighbours: Widget = (
                    DataTable(id="nb", cursor_type="row")
                    if self._mounted_layout == TABLE
                    else Tree(Text(self.focus_entity.label), id="nb")
                )
                neighbours.border_title = "1 Neighbours"
                yield neighbours
            with Pane(id="evidence-pane", classes="pane") as pane:
                pane.border_title = "2 Evidence"
                yield Static(id="evidence")

    def on_mount(self) -> None:
        """Focus the neighbours."""
        self.query_one("#nb").focus()
        super().on_mount()

    def title_text(self) -> str:
        """The breadcrumb of entities visited, most recent last."""
        crumbs = " ▸ ".join(fmt.clip(f.label, 24) for f in self.path[-5:])
        return f"Context Graph ▸ knowledge ▸ {crumbs}"

    def fetch(self, reader: Reader) -> Neighbourhood:
        """Load the neighbours within scope and the focused entity's provenance."""
        scope = self.viewer.scope
        allowed = queries.entity_keys(reader, scope)
        return Neighbourhood(
            rows=queries.neighbours(reader, self.focus_entity, scope, allowed),
            provenance=queries.provenance(reader, self.focus_entity, scope),
        )

    def show(self, data: Neighbourhood) -> None:
        """Redraw only when the neighbours changed, so the cursor stays where the reader left it."""
        if self._wanted_layout() != self._mounted_layout:
            self._recompose()
            return
        signature = (self._mounted_layout, NeighbourhoodScreen.floor, repr(data.rows))
        if signature == self._signature:
            return
        self._signature = signature
        self._shown = sorted(
            (row for row in data.rows if above_floor(row, NeighbourhoodScreen.floor)),
            key=lambda row: (relation_name(row), -strength(row)),
        )
        self.query_one("#evidence", Static).update(provenance_text(self.focus_entity, data.provenance))
        if self._mounted_layout == TABLE:
            self._fill_table()
        else:
            self._fill_tree()

    def status_note(self) -> str:
        """How many neighbours the floor hides."""
        if self.data is None:
            return ""
        hidden = len(self.data.rows) - len(self._shown)
        note = f"{len(self._shown)} neighbours · confidence floor {NeighbourhoodScreen.floor:.1f} (+/-)"
        if hidden:
            note += f" · {hidden} below it hidden"
        if self.data.rows and not is_typed(self.data.rows) and NeighbourhoodScreen.view_mode == TREE:
            note += " · untyped relations (LightRAG), shown as a table"
        return note

    def _fill_tree(self) -> None:
        tree = self.query_one("#nb")
        if not isinstance(tree, Tree):
            return
        tree.clear()
        tree.root.set_label(Text(f"{self.focus_entity.label}  ·  {len(self._shown)} neighbours", style="bold"))
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._shown:
            groups[f"{arrow(row)} {relation_name(row)}"].append(row)
        on_path = {key for f in self.path for key in f.keys}
        for name, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            node = tree.root.add(Text(f"{name} ({len(members)})"), expand=len(groups) <= 6)
            members.sort(key=lambda row: -strength(row))
            for row in members[:PER_GROUP]:
                label = Text(fmt.clip(row["label"], 40)).append(f"  {row['type'] or ''}", style="dim")
                if row["confidence"] is not None:
                    label.append(f"  {strength(row):.2f}", style="dim")
                if on_path & set(row["keys"]):
                    label.append("  ↺ on path", style="yellow")
                if row["nonconformant"]:
                    label.append("  ✕ off-ontology", style="red")
                node.add_leaf(label, data=row)
            if len(members) > PER_GROUP:
                node.add_leaf(Text(f"… {len(members) - PER_GROUP} more (v shows them all as a table)", style="dim"))
        tree.root.expand()

    def _fill_table(self) -> None:
        table = self.query_one("#nb")
        if not isinstance(table, DataTable):
            return
        on_path = {key for f in self.path for key in f.keys}
        columns = [("dir", ""), ("relation", "relation"), ("neighbour", "neighbour"), ("type", "type")]
        columns += [("strength", "conf/w"), ("evidence", "evidence")]
        rows = [
            (
                str(i),
                [
                    arrow(row),
                    relation_name(row),
                    fmt.clip(row["label"], 32) + (" ↺" if on_path & set(row["keys"]) else ""),
                    row["type"] or "—",
                    f"{strength(row):.2f}" if (row["confidence"] is not None or row["weight"] is not None) else "",
                    Text(fmt.clip(row["evidence"] or "", 60), style="red" if row["nonconformant"] else ""),
                ],
            )
            for i, row in enumerate(self._shown)
        ]
        sync_table(table, columns, rows)

    def _selected(self) -> dict[str, Any] | None:
        widget = self.query_one("#nb")
        if isinstance(widget, Tree):
            node = widget.cursor_node
            return node.data if node is not None and isinstance(node.data, dict) else None
        if isinstance(widget, DataTable) and (key := cursor_key(widget)) is not None:
            return self._shown[int(key)] if int(key) < len(self._shown) else None
        return None

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        """Show the relation's evidence, or the entity's provenance on the root."""
        if isinstance(event.node.data, dict):
            self.query_one("#evidence", Static).update(evidence_text(self.focus_entity, event.node.data))
        elif event.node.is_root and self.data is not None:
            self.query_one("#evidence", Static).update(provenance_text(self.focus_entity, self.data.provenance))

    def on_tree_node_selected(self, event: Tree.NodeSelected[Any]) -> None:
        """Enter on a neighbour refocuses on it."""
        if isinstance(event.node.data, dict):
            self._refocus(event.node.data)

    def on_data_table_row_highlighted(self, _: DataTable.RowHighlighted) -> None:
        """Show the highlighted relation's evidence."""
        if row := self._selected():
            self.query_one("#evidence", Static).update(evidence_text(self.focus_entity, row))

    def on_data_table_row_selected(self, _: DataTable.RowSelected) -> None:
        """Enter on a neighbour refocuses on it."""
        if row := self._selected():
            self._refocus(row)

    def _refocus(self, row: dict[str, Any]) -> None:
        focus = queries.Focus(tuple(row["keys"]), str(row["label"]))
        self.app.push_screen(NeighbourhoodScreen(focus, [*self.path, focus]))

    def action_toggle_layout(self) -> None:
        """Switch between the relation tree and the flat table."""
        NeighbourhoodScreen.view_mode = TABLE if NeighbourhoodScreen.view_mode == TREE else TREE
        self._recompose()

    def _recompose(self) -> None:
        self._signature = None
        self.refresh(recompose=True)
        self.call_after_refresh(self._after_recompose)

    def action_floor(self, step: int) -> None:
        """Raise or lower the confidence floor."""
        NeighbourhoodScreen.floor = round(min(max(NeighbourhoodScreen.floor + step * FLOOR_STEP, 0.0), 0.9), 1)
        self._signature = None
        if self.data is not None:
            self.show(self.data)
        self.draw_status()

    def _after_recompose(self) -> None:
        self.query_one("#nb").focus()
        if self.data is not None:
            self.show(self.data)
        self.draw_status()
