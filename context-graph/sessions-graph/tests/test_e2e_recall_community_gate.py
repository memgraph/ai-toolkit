"""The Community gate: recall through the config's single space renders exactly what it rendered before spaces.

Map #501 decided that the Enterprise edition must not change Community: with
one space, fanning recall out across spaces and fusing their ranks has to give
back the single-space output untouched. This freezes that output on a fixed
two-user graph. Regenerate the expected file only for a change that is meant
to alter Community recall: ``RECALL_GATE_REGENERATE=1 pytest <this file>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("agent_context_graph")
pytest.importorskip("actions_graph")

from sessions_graph.embeddings import EmbeddingUnavailableError, check_available
from sessions_graph.tool import RecallTool

from actions_graph import MessageRole, Session
from agent_context_graph.adapters._identity import HookConfig

EXPECTED = Path(__file__).parent / "fixtures" / "recall_community_gate.json"

_TURNS = {
    "u1": [
        ("2026-09-01T10:00:00+00:00", "Our staging cluster is called kestrel and runs in eu-west-1."),
        ("2026-09-02T09:30:00+00:00", "I prefer pytest fixtures over setUp methods, keep tests flat."),
        ("2026-09-03T14:15:00+00:00", "We decided to move the payments service to Postgres 16 next sprint."),
        ("2026-09-04T16:45:00+00:00", "The flaky test in checkout was a timezone bug, fixed by pinning UTC."),
        ("2026-09-05T10:20:00+00:00", "Production deploys go out on Tuesdays and Thursdays only."),
        ("2026-09-08T13:00:00+00:00", "The payments migration is blocked until the read replica is ready."),
        ("2026-09-09T09:10:00+00:00", "My laptop is a 14 inch MacBook Pro with 36 GB of memory."),
        ("2026-09-10T17:30:00+00:00", "Rename the staging cluster to kestrel-2 after the upgrade."),
    ],
    "u2": [
        ("2026-09-01T11:00:00+00:00", "Our staging cluster is called osprey."),
        ("2026-09-03T08:00:00+00:00", "Payments stays on MySQL, we are not migrating."),
    ],
}
_MEMORY_FILES = {
    "/memories/feedback/testing.md": (
        "---\ndescription: how the user wants tests written\ntype: feedback\n---\n"
        "Use real Memgraph in tests, never mocks of Cypher strings.\n"
    ),
}
#: Narrower than the user's history, so which rows come back depends on ranking, not just on ownership.
WIDTHS = {"turns_k": "2", "text_k": "2", "memories_k": "1"}
QUESTIONS = (
    "What is the staging cluster called?",
    "What did we decide about the payments database?",
    "How should tests be written?",
)


@pytest.fixture
def community_graph(graph, memgraph, actions_graph):
    """Two users' sessions and one memory file, embedded the way session end does it."""
    try:
        check_available(memgraph)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"Memgraph can't embed (needs MAGE): {exc}")
    for user, turns in _TURNS.items():
        for index, (timestamp, text) in enumerate(turns):
            session = f"{user}-s{index}"
            actions_graph.ensure_session(Session(session_id=session, started_at=timestamp))
            memgraph.query(
                "MERGE (u:User {user_id: $u}) WITH u MATCH (s:Session {session_id: $s}) MERGE (u)-[:HAD_SESSION]->(s)",
                {"u": user, "s": session},
            )
            actions_graph.record_message(session_id=session, role=MessageRole.USER, content=text, timestamp=timestamp)
            graph.embed_session(session)
    store = graph.memory_store("u1")
    for path, text in _MEMORY_FILES.items():
        store.create(path, text)
    memgraph.query("MATCH (m:Memory) SET m.updated_at = '2026-09-06T12:00:00+00:00'")


def _community_config() -> HookConfig:
    return HookConfig(
        user_id="u1",
        memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        memgraph_user=os.environ.get("MEMGRAPH_USER", ""),
        memgraph_password=os.environ.get("MEMGRAPH_PASSWORD", ""),
        memgraph_database=os.environ.get("MEMGRAPH_DATABASE", "memgraph"),
        recall_settings=WIDTHS,
    )


def _without_today(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("Today is "))


def test_community_recall_is_unchanged(community_graph):
    tool = RecallTool()
    rendered = {
        question: _without_today(tool.call({"question": question}, _community_config()).text) for question in QUESTIONS
    }

    if os.environ.get("RECALL_GATE_REGENERATE"):
        EXPECTED.parent.mkdir(exist_ok=True)
        EXPECTED.write_text(json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert rendered == json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert "osprey" not in json.dumps(rendered)
