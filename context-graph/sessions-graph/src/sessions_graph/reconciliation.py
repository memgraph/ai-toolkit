"""Session-content reconciliation: turn Actions Graph / Memory text into entities.

Session content (Messages, ToolCalls, ToolResults, Memories) is mostly opaque
strings and JSON blobs today. This module extracts plain text worth
entity-extracting out of that content and hands it to unstructured2graph's
chunk + LightRAG pipeline, so knowledge embedded in past sessions becomes
queryable graph entities instead of unread properties.

Requires the ``sessions-graph[reconciliation]`` extra (actions-graph +
unstructured2graph).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actions_graph.models import Action

    from .models import Memory

logger = logging.getLogger(__name__)

# A single reconcilable text unit sent to unstructured2graph's LightRAG pipeline
# is capped at this length; anything longer is truncated with a logged
# warning rather than billed in full to an LLM.
MAX_RECONCILABLE_CHARS = 8000


def content_hash(text: str) -> str:
    """Same hashing convention unstructured2graph uses for Chunk.hash."""
    return hashlib.sha256(text.encode()).hexdigest()


def _content_to_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text") for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts) if parts else None
    return str(content)


def extract_reconcilable_text(action: Action) -> str | None:
    """Pull one plain-text unit worth entity-extracting out of an Action.

    Only Messages, ToolCalls, and ToolResults carry text worth running through
    LightRAG; other action types (errors, subagent events, structured output,
    permission requests, rate limits) return None. Returns None for
    empty/whitespace-only content.
    """
    from actions_graph.models import Message, ToolCall, ToolResult

    if isinstance(action, (Message, ToolResult)):
        text = _content_to_text(action.content)
    elif isinstance(action, ToolCall):
        try:
            text = json.dumps(action.tool_input, default=str)
        except (TypeError, ValueError):
            text = str(action.tool_input)
    else:
        return None

    if text is None or not text.strip():
        return None

    text = text.strip()
    if len(text) > MAX_RECONCILABLE_CHARS:
        logger.warning(
            "Truncating reconcilable text for action %s from %d to %d chars",
            action.action_id,
            len(text),
            MAX_RECONCILABLE_CHARS,
        )
        text = text[:MAX_RECONCILABLE_CHARS]

    return text


@dataclass(frozen=True)
class ReconciliationSource:
    """One piece of session content queued for entity extraction.

    ``kind`` identifies which node the resulting Chunk(s) get linked back to
    via HAS_CHUNK: "action" -> (:Action {action_id: node_id}), "memory" ->
    (:Memory {memory_id: node_id}).
    """

    kind: str
    node_id: str
    text: str


#: Maps an ReconciliationSource.kind to the (node label, id property) it links to.
NODE_LABELS: dict[str, tuple[str, str]] = {
    "action": ("Action", "action_id"),
    "memory": ("Memory", "memory_id"),
}


@dataclass(frozen=True)
class ReconciliationSummary:
    """Result of one ``SessionsGraph.reconcile_session`` call."""

    session_id: str
    status: str  # "completed" | "failed"
    texts_considered: int
    texts_deduped: int
    error: str | None = None


def build_reconciliation_sources(
    actions: list[Action],
    memories: list[Memory],
) -> list[ReconciliationSource]:
    """Build the ordered list of reconcilable text sources for a session.

    Order matters: it lines up 1:1 with the ``texts`` list passed to
    ``unstructured2graph.from_texts``, whose grouped return value is zipped
    back against this list by index to link Chunks to their source node.
    """
    sources: list[ReconciliationSource] = []
    for action in actions:
        text = extract_reconcilable_text(action)
        if text:
            sources.append(ReconciliationSource(kind="action", node_id=action.action_id, text=text))
    for memory in memories:
        if memory.content and memory.content.strip():
            sources.append(ReconciliationSource(kind="memory", node_id=memory.memory_id, text=memory.content.strip()))
    return sources
