"""The ``resource`` tool: the harness's model reads a GitHub resource from memory.

Registered under agent-context-graph's ``agent_context_graph.tools`` entry
point, so ``agent-context-graph mcp`` serves it. It never calls GitHub and
writes nothing: it returns what is stored with ``fetched_at`` and the source's
``updated_at`` — facts, so the model judges staleness for its own task. The
Cache Read is recorded by the connector from the first line of the answer
(:data:`resources_graph.connector.OUTCOME_LINE`), where the session is known.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar

from agent_context_graph.tools import ToolError, ToolResult

from .address import parse_address
from .core import ResourcesGraph
from .models import MISS, SUBSUMED, Served

if TYPE_CHECKING:
    from agent_context_graph.adapters._identity import HookConfig

_DESCRIPTION = """\
Read a public GitHub issue, pull request, repository or issue/PR list from memory instead of \
fetching it. Pass what you would fetch: a URL, owner/repo#123, owner/repo, or the gh command \
(gh issue list -R owner/repo --label bug ...). An issue or PR comes with all its comments; a \
list comes as an index of its members, page by page — read the ones you need one at a time with \
owner/repo#number. Every answer says when it was fetched (fetched_at) and, for an item, when \
GitHub last changed it (updated_at); decide yourself whether that is fresh enough. On a miss, \
fetch it as usual."""
#: Index rows per page of a Listing answer.
PAGE_SIZE = 100


class ResourceTool:
    """``resource(address)`` over the shared Resources; keeps one connection per config."""

    name = "resource"
    connector = "resources-graph"
    description = _DESCRIPTION
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "address": {
                "type": "string",
                "description": "A GitHub URL, owner/repo#123, owner/repo, or a gh issue/pr/repo view or list command.",
            },
            "page": {"type": "integer", "minimum": 1, "description": "For a list: which page of its index (1 first)."},
        },
        "required": ["address"],
    }
    session_hint = (
        "Public GitHub issues, pull requests, repositories and issue/PR lists you have read before are in "
        "Context Graph memory. Before fetching one, call the `resource` tool with its URL, owner/repo#number "
        "or gh command."
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._graph: ResourcesGraph | None = None
        self._graph_key: tuple[str, ...] | None = None

    def call(self, arguments: dict[str, Any], config: HookConfig) -> ToolResult:
        """Serve ``arguments["address"]`` from memory.

        Raises:
            ToolError: no address, one that names no single resource, or no ``identity.user_id`` configured.
        """
        text = str(arguments.get("address") or "").strip()
        if not text:
            raise ToolError("resource needs an address: a GitHub URL, owner/repo#123 or owner/repo.")
        if not config.user_id:
            raise ToolError(
                "No user is configured, so memory can't be read. "
                "Set one with: agent-context-graph config set identity.user_id <name>"
            )
        address = parse_address(text)
        if address is None:
            raise ToolError(
                f"{text!r} doesn't name a GitHub issue, pull request, repository or issue/PR list I can read. "
                "Lists need the repository (gh issue list -R owner/repo ...)."
            )
        try:
            page = max(1, int(arguments.get("page") or 1))
        except (TypeError, ValueError) as exc:
            raise ToolError("page must be a whole number, 1 or more.") from exc
        served = self._graph_for(config).read(address)
        return ToolResult(text=render(served, page=page), structured=_structured(served, page=page))

    def _graph_for(self, config: HookConfig) -> ResourcesGraph:
        key = (config.memgraph_url, config.memgraph_user, config.memgraph_password, config.memgraph_database)
        with self._lock:
            if self._graph is None or self._graph_key != key:
                self._graph = ResourcesGraph(
                    url=config.memgraph_url,
                    username=config.memgraph_user,
                    password=config.memgraph_password,
                    database=config.memgraph_database,
                )
                self._graph_key = key
            return self._graph


def render(served: Served, *, page: int = 1) -> str:
    """The model-facing answer. Its first line is the outcome the connector records."""
    head = f"[resource {served.outcome}] {served.address}"
    if served.outcome == SUBSUMED:
        head += f" from {served.served_from}"
    if served.outcome == MISS:
        return f"{head}\nNot in memory. Fetch it as usual; it will be remembered for next time."
    r = served.resource
    lines = [head]
    if served.kind == "Listing":
        return "\n".join(lines + _listing_lines(served, page))
    if served.kind == "Repository":
        lines += [
            f"Repository {r['name_with_owner']}: {r.get('description') or ''}".rstrip(),
            f"url: {r['url']}",
            f"stars: {r.get('stars')} · pushed_at: {r.get('pushed_at')}",
            _freshness(r),
            "",
            r.get("readme") or "(no README.md)",
        ]
        return "\n".join(lines)
    lines += [
        f"{served.kind} #{r['number']}: {r['title']}",
        f"url: {r['url']}",
        " · ".join(
            part
            for part in (
                f"state: {r['state']}",
                f"author: {r.get('author')}",
                f"labels: {', '.join(r.get('labels') or []) or '-'}",
                f"assignees: {', '.join(r.get('assignees') or []) or '-'}",
                f"milestone: {r.get('milestone') or '-'}",
            )
        ),
        f"created_at: {r['created_at']} · closed_at: {r.get('closed_at') or '-'}",
        _freshness(r),
    ]
    if served.kind == "PullRequest":
        lines.append(f"merged: {r.get('merged')} · base: {r.get('base_ref')}")
        lines.append(f"changed files ({len(r.get('changed_files') or [])}): {', '.join(r.get('changed_files') or [])}")
    lines += ["", r.get("body") or "(no description)", "", f"{len(served.comments)} comments:"]
    for comment in served.comments:
        where = f" on {comment['path']}" if comment.get("path") else ""
        lines += ["", f"--- {comment.get('author')} at {comment['created_at']}{where} ---", comment.get("body") or ""]
    return "\n".join(lines)


def _listing_lines(served: Served, page: int) -> list[str]:
    listing, rows = served.resource, _page(served.index, page)
    pages = max(1, -(-len(served.index) // PAGE_SIZE))
    lines = []
    if served.outcome == SUBSUMED:
        lines.append(f"Derived from {served.served_from} by filtering its stored members locally.")
    lines += [
        f"fetched_at: {listing.get('fetched_at')} · {len(served.index)} members"
        + (
            ""
            if listing.get("fully_expanded")
            else f" (the first {listing.get('member_count')} of {listing.get('total_count')})"
        ),
        f"page {min(page, pages)} of {pages}; newest created first. Read one with owner/repo#number.",
        "",
    ]
    lines += [
        f"#{row['number']} [{row['state']}] {row['title']} · labels: {', '.join(row.get('labels') or []) or '-'}"
        f" · updated_at: {row.get('updated_at')} · comments: {row.get('comment_count')}"
        for row in rows
    ]
    return lines


def _page(rows: list[dict[str, Any]], page: int) -> list[dict[str, Any]]:
    return rows[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]


def _freshness(resource: dict[str, Any]) -> str:
    return f"fetched_at: {resource.get('fetched_at')} · updated_at (GitHub): {resource.get('updated_at')}"


def _structured(served: Served, *, page: int) -> dict[str, Any]:
    return {
        "address": served.address,
        "outcome": served.outcome,
        "kind": served.kind,
        "served_from": served.served_from,
        "resource": served.resource,
        "comments": served.comments,
        "index": _page(served.index, page),
        "index_size": len(served.index),
        "page": page,
    }


RESOURCE = ResourceTool()
