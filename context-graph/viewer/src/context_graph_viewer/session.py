"""Session anatomy: one session's timeline, what each action touched, and what the session produced (#381)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Tree

from . import fmt, queries
from .base import Pane, PolledScreen
from .failures import session_failures

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from textual.widget import Widget
    from textual.widgets.tree import TreeNode

    from memgraph_context_graph.capture_log import Failure

    from .source import Reader

LABEL_WIDTH = 44
BAR_WIDTH = 30


def touch_marker(touches: list[dict[str, Any]] | None) -> str:
    """One mark per resources-graph Touch the action caused: ◆ resolved, ◇ not yet."""
    return "".join("◆" if t["status"] == "resolved" else "◇" for t in touches or [])


def action_line(
    action: dict[str, Any], touches: list[dict[str, Any]] | None, window: tuple[datetime | None, datetime | None]
) -> Text:
    """A timeline row: label, duration, error mark, Touch marks, and a time bar."""
    start, end = fmt.to_datetime(action["at"]), fmt.to_datetime(action["result_at"])
    took = fmt.duration(start, end) or ("…" if action["type"] == "tool_call" else "")
    line = Text(f"{fmt.action_label(action, LABEL_WIDTH):<{LABEL_WIDTH}} {took:>7}")
    if action["result_error"] is True or action["is_error"] is True:
        line.append(" ✕", style="red")
    if marks := touch_marker(touches):
        line.append(f" {marks}", style="magenta")
    return line.append(" ").append(fmt.bar(start, end, *window, width=BAR_WIDTH), style="cyan")


def time_window(anatomy: queries.SessionAnatomy) -> tuple[datetime | None, datetime | None]:
    """The first and last moment any action started or finished."""
    moments = [
        moment
        for action in anatomy.actions
        for moment in (fmt.to_datetime(action["at"]), fmt.to_datetime(action["result_at"]))
        if moment is not None
    ]
    return (min(moments), max(moments)) if moments else (None, None)


def agent_line(agent: dict[str, Any], count: int, spawned: bool) -> Text:
    """A subagent's folding row."""
    line = Text("▾ subagent ", style="bold").append(str(agent["type"] or agent["id"]), style="bold")
    if agent["description"]:
        line.append(f"  {fmt.clip(agent['description'], 40)}", style="dim")
    line.append(f"  ({count} actions)")
    if not spawned:
        line.append("  spawning call unknown", style="dim")
    return line


def build_timeline(tree: Tree[Any], anatomy: queries.SessionAnatomy) -> None:
    """Fill *tree*: top-level actions in order, each subagent folded under the call that spawned it.

    Node data is ``("action", id)`` or ``("agent", id)``.
    """
    window = time_window(anatomy)
    by_agent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for action in anatomy.actions:
        by_agent[action["agent_id"]].append(action)
    spawned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in anatomy.agents:
        if agent["spawned_by"]:
            spawned[agent["spawned_by"]].append(agent)
    placed: set[str] = set()

    def add_actions(parent: TreeNode[Any], actions: list[dict[str, Any]]) -> None:
        for action in actions:
            line = action_line(action, anatomy.touches.get(action["id"]), window)
            children = spawned.get(action["id"], [])
            if not children:
                parent.add_leaf(line, data=("action", action["id"]))
                continue
            node = parent.add(line, data=("action", action["id"]), expand=True)
            for agent in children:
                add_agent(node, agent, spawned_by_known=True)

    def add_agent(parent: TreeNode[Any], agent: dict[str, Any], *, spawned_by_known: bool) -> None:
        if agent["id"] in placed:
            return
        placed.add(agent["id"])
        own = by_agent.get(agent["id"], [])
        node = parent.add(agent_line(agent, len(own), spawned_by_known), data=("agent", agent["id"]), expand=True)
        add_actions(node, own)

    session = anatomy.session
    root_label = Text(f"session {session['id'][:8]}", style="bold")
    root_label.append(f"  ·  {len(anatomy.actions)} actions  ·  {fmt.project_name(session['wd'])}")
    root_label.append(f"  ·  {session['status'] or '—'}", style="dim")
    tree.root.set_label(root_label)
    tree.root.data = None
    add_actions(tree.root, by_agent.get(None, []))
    for agent in anatomy.agents:
        add_agent(tree.root, agent, spawned_by_known=False)
    tree.root.expand()


def timeline_signature(anatomy: queries.SessionAnatomy) -> tuple[Any, ...]:
    """What the timeline tree depends on; the tree is rebuilt only when this changes."""
    return (
        tuple((a["id"], a["result_at"], a["result_error"]) for a in anatomy.actions),
        tuple((g["id"], g["spawned_by"]) for g in anatomy.agents),
        tuple((key, touch_marker(ts)) for key, ts in sorted(anatomy.touches.items())),
    )


def action_details(action: dict[str, Any], touches: list[dict[str, Any]] | None) -> Text:
    """The details pane for one action: when, how long, what it touched, and its input or text."""
    out = Text()
    out.append(str(action["type"]), style="bold").append(f" {action['tool'] or ''}\n")
    out.append(f"at   {action['at']}\n")
    took = fmt.duration(fmt.to_datetime(action["at"]), fmt.to_datetime(action["result_at"]))
    out.append(f"took {took or '—'}\n")
    out.append(f"in   {action['agent_id'] or 'main agent'}\n")
    if action["result_error"] is True:
        out.append("result was an error\n", style="red")
    for touch in touches or []:
        source = "served from memory" if touch["from_memory"] is True else "live"
        out.append(f"touched {touch['kind'] or 'resource'} {touch['address']}", style="magenta")
        out.append(f" · {touch['status']} · {source}\n", style="dim")
    if action["type"] == "tool_call":
        out.append("\ninput\n", style="bold")
        out.append(json.dumps(fmt.tool_input(action["props"]), indent=1, ensure_ascii=False)[:1500])
    elif action["text"]:
        out.append("\n").append(str(action["text"]))
    return out


def agent_details(agent: dict[str, Any]) -> Text:
    """The details pane for one subagent."""
    out = Text("subagent ", style="bold").append(str(agent["type"] or agent["id"]), style="bold")
    out.append(f"\nid      {agent['id']}\nstatus  {agent['status'] or '—'}\nstarted {agent['started'] or '—'}")
    out.append(f"\nended   {agent['ended'] or '—'}\n")
    if agent["description"]:
        out.append(f"\n{agent['description']}")
    return out


def produced_text(anatomy: queries.SessionAnatomy, failures: list[Failure]) -> Text:
    """What the session produced, linked at session level only (#381), and its capture failures."""
    p = anatomy.produced
    version = p.get("ontology_version")
    out = Text()
    out.append("reconcile ").append(str(p.get("rstatus") or "—"), style="bold")
    out.append(f"  ·  backend {p.get('backend') or '—'}  ·  ontology {'—' if version is None else f'v{version}'}")
    out.append(f"  ·  embeddings {p.get('estatus') or '—'}\n")
    out.append(f"chunks {p.get('chunks', 0)}  ·  entities {p.get('entity_count', 0)}  ")
    out.append(fmt.clip(", ".join(str(e) for e in p.get("entities") or [] if e), 160), style="dim")
    if p.get("memories"):
        out.append("\nmemory files written: " + ", ".join(str(m) for m in p["memories"]))
    for kind, error in (("reconcile", p.get("rerror")), ("re-extract", p.get("reerror")), ("embed", p.get("eerror"))):
        if error and (kind != "reconcile" or p.get("rstatus") == "failed"):
            out.append(f"\n✕ {kind}: ", style="red").append(fmt.clip(error, 200))
    if anatomy.loose_touches:
        out.append(f"\n{len(anatomy.loose_touches)} touch(es) not matched to an action", style="dim")
    if failures:
        latest = failures[0]
        out.append(f"\n✕ {len(failures)} capture failure(s) in the last 24h", style="bold red")
        out.append(f", latest {latest.stage}/{latest.connector or '—'} {latest.error_type}: ", style="red")
        out.append(fmt.clip(latest.error, 120))
    return out


@dataclass
class SessionView:
    """One poll of a session screen: the graph's anatomy and the local log's failures for it."""

    anatomy: queries.SessionAnatomy | None
    failures: list[Failure]


class SessionScreen(PolledScreen[SessionView]):
    """One session end to end; ``g`` opens the entities it produced."""

    PANES: ClassVar = ("#timeline", "#details-pane", "#produced-pane")
    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("g", "entities", "entities of session"),
    ]

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self._signature: tuple[Any, ...] | None = None

    def compose_body(self) -> Iterable[Widget]:
        """The timeline over the produced panel, the details pane on the right."""
        with Horizontal():
            with Vertical(id="main"):
                tree: Tree[Any] = Tree("session", id="timeline")
                tree.border_title = "1 Timeline"
                yield tree
                with Pane(id="produced-pane", classes="pane") as pane:
                    pane.border_title = "3 Produced"
                    yield Static(id="produced")
            with Pane(id="details-pane", classes="pane") as pane:
                pane.border_title = "2 Details"
                yield Static(id="details")

    def on_mount(self) -> None:
        """Focus the timeline."""
        self.query_one("#timeline", Tree).focus()
        super().on_mount()

    def title_text(self) -> str:
        """Dashboard ▸ this session."""
        return f"Context Graph ▸ session {self.session_id[:8]}"

    def fetch(self, reader: Reader) -> SessionView:
        """Load the session and its capture failures."""
        return SessionView(
            anatomy=queries.session_anatomy(reader, self.session_id),
            failures=session_failures(self.session_id, self.viewer.now()),
        )

    def show(self, data: SessionView) -> None:
        """Rebuild the timeline only when it changed, keeping the cursor on the same node."""
        anatomy = data.anatomy
        if anatomy is None:
            self.query_one("#produced", Static).update(Text("This session is not in the graph.", style="yellow"))
            return
        signature = timeline_signature(anatomy)
        if signature != self._signature:
            tree = self.query_one("#timeline", Tree)
            selected = tree.cursor_node.data if tree.cursor_node else None
            tree.clear()
            build_timeline(tree, anatomy)
            self._signature = signature
            if selected is not None:
                self._select(tree, selected)
        self.query_one("#produced", Static).update(produced_text(anatomy, data.failures))

    @staticmethod
    def _select(tree: Tree[Any], data: Any) -> None:
        stack = [tree.root]
        while stack:
            node = stack.pop()
            if node.data == data:
                tree.move_cursor(node)
                return
            stack.extend(node.children)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        """Show the highlighted action or subagent in the details pane."""
        anatomy = self.data.anatomy if self.data else None
        if anatomy is None or not event.node.data:
            return
        kind, key = event.node.data
        if kind == "action":
            action = next((a for a in anatomy.actions if a["id"] == key), None)
            if action:
                self.query_one("#details", Static).update(action_details(action, anatomy.touches.get(key)))
        else:
            agent = next((g for g in anatomy.agents if g["id"] == key), None)
            if agent:
                self.query_one("#details", Static).update(agent_details(agent))

    def action_entities(self) -> None:
        """Open the entities this session produced."""
        from .knowledge import EntityListScreen

        self.app.push_screen(EntityListScreen(session_id=self.session_id))
