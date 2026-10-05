"""Print the graph shape each live runtime session produced on the e2e Memgraph.

Run through ``live-hooks-e2e.sh verify``, which points ``MEMGRAPH_URL`` at the
disposable instance. Provider identity lives only in each node's metadata JSON
(issue #385), hence the ``CONTAINS`` match.
"""

import json
import sys

from actions_graph import ActionsGraph, ToolResult

RUNTIMES = ["claude-code", "codex", "copilot-cli", "opencode", "antigravity-cli", "grok"]


def _session_ids(db, runtime: str) -> list[str]:
    tag = f'"source_sdk": "{runtime}"'
    rows = db.query(
        "MATCH (s:Session) OPTIONAL MATCH (s)-[:HAS_ACTION]->(a:Action) "
        "WITH s, collect(a.metadata) AS action_metadata "
        "WHERE s.metadata CONTAINS $tag OR any(m IN action_metadata WHERE m CONTAINS $tag) "
        "RETURN s.session_id AS id",
        params={"tag": tag},
    )
    return [row["id"] for row in rows]


def _report(graph: ActionsGraph, runtime: str, session_id: str) -> None:
    db = graph._db
    session = graph.get_session(session_id)
    status = db.query(
        "MATCH (s:Session {session_id: $sid}) RETURN s.reconciliation_status AS rs, s.embedding_status AS es",
        params={"sid": session_id},
    )[0]
    actions = graph.get_session_actions(session_id)
    linked = db.query(
        "MATCH (s:Session {session_id: $sid})-[:HAS_ACTION]->(c:ToolCall) "
        "OPTIONAL MATCH (c)-[:PARENT_OF]->(r:ToolResult) RETURN c.tool_name AS tool, r IS NOT NULL AS linked",
        params={"sid": session_id},
    )
    embedded = db.query(
        "MATCH (s:Session {session_id: $sid})-[:HAS_ACTION]->(m:Message) "
        "RETURN m.action_type AS kind, m.embedding IS NOT NULL AS embedded",
        params={"sid": session_id},
    )
    print(f"\n== {runtime} session {session_id}")
    print(
        f"   status={session.status.value if session else None} ended={bool(session and session.ended_at)} "
        f"cwd={'set' if session and session.working_directory else 'MISSING'} "
        f"model={session.model if session else None}"
    )
    print(f"   reconciliation_status={status['rs']} embedding_status={status['es']}")
    print(f"   actions: {[action.action_type.value for action in actions]}")
    print(f"   tool calls linked to results: {[(row['tool'], row['linked']) for row in linked]}")
    print(f"   messages embedded: {[(row['kind'], row['embedded']) for row in embedded]}")
    for result in (action for action in actions if isinstance(action, ToolResult)):
        content = result.content if isinstance(result.content, str) else json.dumps(result.content)
        print(f"   result {result.tool_name}: {str(content)[:60]!r} error={result.is_error}")


def main(runtimes: list[str]) -> int:
    """Print every listed runtime's sessions; return non-zero if one has none."""
    graph = ActionsGraph()
    missing = []
    for runtime in runtimes:
        session_ids = _session_ids(graph._db, runtime)
        if not session_ids:
            print(f"\n== {runtime}: NO SESSION")
            missing.append(runtime)
        for session_id in session_ids:
            _report(graph, runtime, session_id)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or RUNTIMES))
