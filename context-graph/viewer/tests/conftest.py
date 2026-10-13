"""Fixtures for the viewer's tests, which run against a real Memgraph.

Sessions are captured through the real Claude Code hook pipeline (adapter, AgentLink and the actions-, sessions-
and resources-graph connectors), so the viewer reads exactly the shape capture writes. Extraction output (chunks,
entities, typed relations) is written by hand in the shape unstructured2graph leaves, as sessions-graph's recall
tests do: running a real extractor here would test extraction, not the viewer.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memgraph_toolbox.api.memgraph import Memgraph

USER = "ana"
OTHER_USER = "bob"
PROJECT = "/work/billing"
OTHER_PROJECT = "/work/crawler"


@pytest.fixture(autouse=True)
def config_file(monkeypatch, tmp_path):
    """Point the viewer and the hooks at a config naming the test Memgraph and user ``ana``.

    The capture-failure log follows the config file, so it lands in ``tmp_path`` too.
    """
    from memgraph_context_graph import spaces
    from memgraph_context_graph.adapters import _identity

    path = tmp_path / "config.toml"
    url = os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687")
    path.write_text(f'[identity]\nuser_id = "{USER}"\n\n[memgraph]\nurl = "{url}"\n', encoding="utf-8")
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(path))
    _identity._reset_cache()
    spaces._reset_cache()
    yield path
    _identity._reset_cache()
    spaces._reset_cache()


@pytest.fixture(autouse=True)
def _no_background_children(monkeypatch):
    """Turn ends spawn a detached ``sessions-graph embed``; it would write to the graph while a test reads it."""
    with contextlib.suppress(ImportError):
        import sessions_graph.connector

        monkeypatch.setattr(sessions_graph.connector, "_spawn_detached", lambda *args, **kwargs: None)


@pytest.fixture
def memgraph():
    """A real Memgraph client, wiped before and after each test; skips when none is reachable."""
    try:
        client = Memgraph(user_agent="context-graph-viewer-tests")
    except Exception as exc:
        pytest.skip(f"No live Memgraph reachable: {exc}")
    client.query("MATCH (n) DETACH DELETE n")
    yield client
    client.query("MATCH (n) DETACH DELETE n")
    client.close()


@pytest.fixture
def reader(memgraph):
    """The viewer's own reader, on the personal space the config file resolves to."""
    from context_graph_viewer.source import Reader, resolve_identity

    r = Reader(resolve_identity().spaces[0])
    yield r
    r.close()


class Capture:
    """Feeds Claude Code hook payloads through the real pipeline, one session at a time."""

    def __init__(self, config_path) -> None:
        self._config_path = config_path
        from memgraph_context_graph.adapters._identity import resolve_memgraph_env
        from memgraph_context_graph.adapters.claude_code import ClaudeCodeHooksAdapter
        from memgraph_context_graph.hooks.runner import create_link

        link = create_link(["actions-graph", "sessions-graph", "resources-graph"], memgraph_env=resolve_memgraph_env())
        self._adapter_class = ClaudeCodeHooksAdapter
        self._link = link

    def send(self, payload: dict[str, Any]) -> None:
        """Handle one hook payload, as ``memgraph-context-graph hook run claude-code`` would."""
        self._adapter_class(self._link).handle_payload(payload)

    @contextlib.contextmanager
    def _no_configured_user(self):
        """Capture as a machine with no ``identity.user_id``, which is how unattributed sessions arise."""
        from memgraph_context_graph import spaces
        from memgraph_context_graph.adapters import _identity

        original = self._config_path.read_text(encoding="utf-8")
        self._config_path.write_text("[memgraph]" + original.split("[memgraph]", 1)[1], encoding="utf-8")
        _identity._reset_cache()
        spaces._reset_cache()
        try:
            yield
        finally:
            self._config_path.write_text(original, encoding="utf-8")
            _identity._reset_cache()
            spaces._reset_cache()

    def session(
        self,
        session_id: str,
        *,
        user: str | None = USER,
        cwd: str = PROJECT,
        commands: tuple[str, ...] = ("ls",),
        failing: tuple[str, ...] = (),
        end: bool = True,
    ) -> str:
        """Capture a session: start, a prompt, one Bash call per command, a turn end, and optionally its end.

        Args:
            user: The payload's user; None captures with no configured user at all, leaving it unattributed.
        """
        if user is None:
            with self._no_configured_user():
                return self.session(session_id, user="", cwd=cwd, commands=commands, failing=failing, end=end)
        base = {"session_id": session_id, "cwd": cwd}
        start = {**base, "hook_event_name": "SessionStart", "source": "startup"}
        if user:
            start["user_id"] = user
        self.send(start)
        self.send({**base, "hook_event_name": "UserPromptSubmit", "prompt": "tidy the [billing] module\\n now"})
        for command in commands:
            tool_use_id = f"t-{uuid.uuid4().hex[:8]}"
            tool_input = {"command": command, "description": f"run {command.split()[0]}"}
            self.send({**base, "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": tool_input,
                       "tool_use_id": tool_use_id})  # fmt: skip
            event = "PostToolUseFailure" if command in failing else "PostToolUse"
            self.send({**base, "hook_event_name": event, "tool_name": "Bash", "tool_input": tool_input,
                       "tool_use_id": tool_use_id, "tool_response": {"stdout": "ok", "stderr": ""},
                       "error": "exit 1"})  # fmt: skip
        self.send({**base, "hook_event_name": "Stop"})
        if end:
            self.send({**base, "hook_event_name": "SessionEnd", "reason": "exit"})
        return session_id

    def subagent_session(self, session_id: str) -> str:
        """A session whose Task call spawns a subagent that runs one tool of its own."""
        base = {"session_id": session_id, "cwd": PROJECT}
        self.send({**base, "hook_event_name": "SessionStart", "source": "startup", "user_id": USER})
        task_input = {"description": "Explore the parser", "prompt": "find the parser", "subagent_type": "Explore"}
        self.send({**base, "hook_event_name": "PreToolUse", "tool_name": "Task", "tool_input": task_input,
                   "tool_use_id": "task-1"})  # fmt: skip
        self.send({**base, "hook_event_name": "SubagentStart", "agent_id": "agent-1", "agent_type": "Explore"})
        grep_input = {"pattern": "def parse", "path": "src"}
        self.send({**base, "hook_event_name": "PreToolUse", "tool_name": "Grep", "tool_input": grep_input,
                   "tool_use_id": "grep-1", "agent_id": "agent-1", "agent_type": "Explore"})  # fmt: skip
        self.send({**base, "hook_event_name": "PostToolUse", "tool_name": "Grep", "tool_input": grep_input,
                   "tool_use_id": "grep-1", "tool_response": {"matches": 1}, "agent_id": "agent-1",
                   "agent_type": "Explore"})  # fmt: skip
        self.send({**base, "hook_event_name": "SubagentStop", "agent_id": "agent-1", "agent_type": "Explore"})
        self.send({**base, "hook_event_name": "PostToolUse", "tool_name": "Task", "tool_input": task_input,
                   "tool_use_id": "task-1", "tool_response": {"result": "found it"}})  # fmt: skip
        return session_id


@pytest.fixture
def capture(memgraph, config_file):
    """Capture sessions through the real hooks pipeline."""
    return Capture(config_file)


def mark_reconciled(
    memgraph: Memgraph, session_id: str, *, status: str = "completed", at: datetime | None = None, **extra: Any
) -> None:
    """Set the reconciliation fields sessions-graph writes when it reconciles a session."""
    moment = (at or datetime.now(timezone.utc)).isoformat()
    memgraph.query(
        "MATCH (s:Session {session_id: $id}) SET s.reconciliation_status = $status, s.reconciled_at = $at, "
        "s.extraction_backend = 'GLiNER2Backend', s.ontology_version = 1, s += $extra",
        {"id": session_id, "status": status, "at": moment, "extra": extra},
    )


def gliner2_id(chunk: str, text: str) -> str:
    """A hash id, as GLiNER2 gives entities."""
    return hashlib.sha256(f"{chunk}|{text}".encode()).hexdigest()


def extract(
    memgraph: Memgraph,
    session_id: str,
    entities: list[tuple[str, str]],
    relations: list[tuple[str, str, str, float]] | None = None,
    *,
    chunk: str | None = None,
    nonconformant: tuple[str, ...] = (),
) -> dict[str, str]:
    """Write GLiNER2-shaped extraction for one session: a Chunk on its first action, entities, typed relations.

    Args:
        entities: ``(text, type)`` per entity; each gets a hash id unique to this chunk, as GLiNER2 does.
        relations: ``(head text, relation type, tail text, confidence)``; the tail ``"@user"`` is the User node.
        nonconformant: Relation types to mark ``ontology_conformant = false``.

    Returns:
        Entity text → its id in this chunk.
    """
    chunk = chunk or f"chunk-{session_id}"
    memgraph.query(
        "MATCH (s:Session {session_id: $id})-[:HAS_ACTION]->(a:Action) WITH a ORDER BY a.timestamp LIMIT 1 "
        "MERGE (c:Chunk {hash: $chunk}) MERGE (a)-[:HAS_CHUNK]->(c)",
        {"id": session_id, "chunk": chunk},
    )
    ids = {text: gliner2_id(chunk, text) for text, _ in entities}
    for text, entity_type in entities:
        # Labels can't be parameters; entity_type comes from the test's own literals.
        memgraph.query(
            f"MATCH (c:Chunk {{hash: $chunk}}) MERGE (e:gliner2:{entity_type} {{entity_id: $id}}) "
            "SET e.text = $text, e.entity_type = $type MERGE (e)-[:MENTIONED_IN {sources: [$chunk]}]->(c)",
            {"chunk": chunk, "id": ids[text], "text": text, "type": entity_type},
        )
    for head, rel_type, tail, confidence in relations or []:
        tail_match = "MATCH (t:User {user_id: $user})" if tail == "@user" else "MATCH (t {entity_id: $tail})"
        memgraph.query(
            f"MATCH (h {{entity_id: $head}}) {tail_match} "
            f"CREATE (h)-[:{rel_type} {{chunk: $chunk, confidence: $confidence, text: $text, role: 'user', "
            "valid_at: $valid_at, ontology_conformant: $conformant}]->(t)",
            {
                "head": ids[head],
                "tail": ids.get(tail),
                "user": USER,
                "chunk": chunk,
                "confidence": confidence,
                "text": f"{head} {rel_type} {tail} [see src\\parser.py]",
                "valid_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                "conformant": rel_type not in nonconformant,
            },
        )
    return ids
