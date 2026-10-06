"""Print, and with ``--check`` assert, the graph each live runtime session produced.

Run through ``live-hooks-e2e.sh verify``, which points ``MEMGRAPH_URL`` at the
disposable instance. Provider identity lives only in each node's metadata JSON
(issue #385), hence the ``CONTAINS`` match.
"""

import argparse
import json
import time
from dataclasses import dataclass

from actions_graph import ActionsGraph, ToolResult

RUNTIMES = ["claude-code", "codex", "copilot-cli", "opencode", "antigravity-cli", "grok"]


@dataclass(frozen=True)
class Expected:
    """What a runtime's hooks reliably capture, as observed in live sessions."""

    prompt: bool = True
    reply: bool = True
    result_text: bool = True
    ends_session: bool = False


# Per runtime capture limits: Copilot's hooks carry no reply text, Antigravity's
# no prompt, reply, or tool output; Codex, OpenCode `run`, and Antigravity have
# no session-end hook, so their sessions stay open.
EXPECTED = {
    "claude-code": Expected(ends_session=True),
    "codex": Expected(),
    "copilot-cli": Expected(reply=False, ends_session=True),
    "opencode": Expected(),
    "antigravity-cli": Expected(prompt=False, reply=False, result_text=False),
    "grok": Expected(ends_session=True),
}


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


def _embedding_status(db, session_id: str, timeout: float) -> str | None:
    # Embedding runs in a detached process after the turn or session ends, and
    # MAGE downloads its model on first use, so it can lag the agent run.
    deadline = time.monotonic() + timeout
    while True:
        rows = db.query(
            "MATCH (s:Session {session_id: $sid}) RETURN s.embedding_status AS status", params={"sid": session_id}
        )
        status = rows[0]["status"] if rows else None
        if status is not None or time.monotonic() > deadline:
            return status
        time.sleep(2)


def _check(graph: ActionsGraph, runtime: str, session_id: str, embed_timeout: float) -> list[str]:
    """Print the session's shape and return every expectation it misses."""
    db = graph._db
    expected = EXPECTED[runtime]
    session = graph.get_session(session_id)
    actions = graph.get_session_actions(session_id)
    kinds = [action.action_type.value for action in actions]
    results = [action for action in actions if isinstance(action, ToolResult)]
    linked = db.query(
        "MATCH (s:Session {session_id: $sid})-[:HAS_ACTION]->(c:ToolCall) "
        "OPTIONAL MATCH (c)-[:PARENT_OF]->(r:ToolResult) RETURN c.tool_name AS tool, r IS NOT NULL AS linked",
        params={"sid": session_id},
    )
    reconciliation = db.query(
        "MATCH (s:Session {session_id: $sid}) RETURN s.reconciliation_status AS status", params={"sid": session_id}
    )[0]["status"]
    embedding = _embedding_status(db, session_id, embed_timeout)

    print(f"\n== {runtime} session {session_id}")
    print(
        f"   status={session.status.value if session else None} ended={bool(session and session.ended_at)} "
        f"cwd={'set' if session and session.working_directory else 'MISSING'} "
        f"model={session.model if session else None}"
    )
    print(f"   reconciliation_status={reconciliation} embedding_status={embedding}")
    print(f"   actions: {kinds}")
    print(f"   tool calls linked to results: {[(row['tool'], row['linked']) for row in linked]}")
    for result in results:
        content = result.content if isinstance(result.content, str) else json.dumps(result.content)
        print(f"   result {result.tool_name}: {str(content)[:60]!r} error={result.is_error}")

    failures = []
    if session is None or not session.working_directory:
        failures.append("session has no working directory")
    if not linked:
        failures.append("no tool call recorded")
    elif not all(row["linked"] for row in linked):
        failures.append("a tool result is not linked to its call")
    if expected.result_text and not any(isinstance(result.content, str) and result.content for result in results):
        failures.append("no tool result text")
    if expected.prompt and "user_message" not in kinds:
        failures.append("no user prompt")
    if expected.reply and "assistant_message" not in kinds:
        failures.append("no assistant reply")
    if expected.ends_session != bool(session and session.ended_at):
        failures.append("session should have ended" if expected.ends_session else "session ended on a turn end")
    if reconciliation != "pending":
        failures.append(f"reconciliation_status is {reconciliation!r}, expected 'pending'")
    if embedding != "completed":
        failures.append(f"embedding_status is {embedding!r}, expected 'completed'")
    return failures


def main() -> int:
    """Report every listed runtime's sessions; with ``--check``, fail on unmet expectations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero when a session misses an expectation.")
    parser.add_argument("--embed-timeout", type=float, default=600, help="Seconds to wait for embedding.")
    parser.add_argument("runtimes", nargs="*", default=RUNTIMES)
    args = parser.parse_args()

    graph = ActionsGraph()
    failures: dict[str, list[str]] = {}
    for runtime in args.runtimes:
        session_ids = _session_ids(graph._db, runtime)
        if not session_ids:
            print(f"\n== {runtime}: NO SESSION")
            failures[runtime] = ["no session recorded"]
        for session_id in session_ids:
            missed = _check(graph, runtime, session_id, args.embed_timeout if args.check else 0)
            if missed:
                failures.setdefault(runtime, []).extend(missed)

    if args.check and failures:
        print("\nFAILED:")
        for runtime, missed in failures.items():
            print(f"  {runtime}: {'; '.join(missed)}")
        return 1
    if args.check:
        print("\nall expectations met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
