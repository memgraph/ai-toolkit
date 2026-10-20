"""Session-content reconciliation: the text of a session's Actions and Memories, ready for extraction.

Pulls the plain text worth entity-extracting out of Messages, ToolCalls,
ToolResults and Memories. ``Reconciler.reconcile`` hands it to
unstructured2graph. Requires the ``sessions-graph[reconciliation]`` extra.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from actions_graph.models import Action

    from .models import Memory

logger = logging.getLogger(__name__)

#: One reconcilable text unit is truncated (with a warning) past this length.
MAX_RECONCILABLE_CHARS = 8000

#: A session's joined document stays one chunk up to this length: a safety
#: valve against a pathological session, sized well above the largest real one
#: seen (~68,500 chars). LightRAG re-splits whatever it receives by tokens anyway.
MAX_SESSION_BATCH_CHARS = 200_000


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

    Only Messages, ToolCalls and ToolResults carry such text; any other action
    type, a call to the ``memory`` tool, and empty or whitespace-only content
    return None.
    """
    from actions_graph.models import Message, ToolCall, ToolResult

    from .memory_store import is_memory_tool

    if isinstance(action, (ToolCall, ToolResult)) and is_memory_tool(action.tool_name):
        # The memory file itself is reconciled on its own (reconcile_memory);
        # the call that wrote it would only extract the same text twice.
        return None
    if isinstance(action, Message):
        # Prefixed with the speaker: who asserted something is part of what was
        # asserted. Without it the extractor can't tell a fact the user stated
        # from a suggestion the assistant made (#328), and the content-hash
        # dedupe would collapse the same sentence from each side into one.
        body = _content_to_text(action.content)
        text = f"{action.role.value}: {body}" if body and body.strip() else body
    elif isinstance(action, ToolResult):
        # No speaker: a tool result is output, not an utterance, and labelling
        # it as one would assert something untrue.
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

    ``kind`` names the node its Chunks link back to via HAS_CHUNK (see
    :data:`NODE_LABELS`). ``role`` is a Message's speaker, None for anything
    that is nobody's utterance; the GLiNER2 backend's mention resolver keys on
    it. ``valid_at`` (ISO-8601) becomes ``valid_at`` on every relationship
    extracted from it.
    """

    kind: str
    node_id: str
    text: str
    role: str | None = None
    valid_at: str | None = None


#: Maps a ReconciliationSource.kind to the (node label, id property) it links to.
NODE_LABELS: dict[str, tuple[str, str]] = {
    "action": ("Action", "action_id"),
    "memory": ("Memory", "memory_id"),
}


@dataclass(frozen=True)
class ReconciliationSummary:
    """Result of one ``Reconciler.reconcile`` call.

    ``nonconformant_entities`` and ``nonconformant_relations`` are the
    ontology integrity counts over this session's chunks
    (``unstructured2graph.ontology_report``), or None when the ontology was not
    enforced. An alarm, not a filter: GLiNER2 output is conformant by
    construction, so nonzero is a bug (#355).
    """

    session_id: str
    status: str  # "completed" | "failed"
    texts_considered: int
    texts_deduped: int
    error: str | None = None
    nonconformant_entities: int | None = None
    nonconformant_relations: int | None = None


@dataclass(frozen=True)
class MemoryReconciliation:
    """Result of one ``Reconciler.reconcile_memory`` call."""

    memory_id: str
    path: str | None
    status: str  # "completed" | "failed"
    chunks: int = 0
    error: str | None = None


def normalize_timestamp(value: str | None) -> str | None:
    """``value`` as ISO-8601 with an explicit offset (UTC assumed when it has none), or None if unparseable.

    Memgraph's ``datetime()`` rejects what it cannot parse, and one bad
    timestamp would fail the whole relationship write, so anything that isn't
    ISO-8601 is dropped here rather than there.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def build_reconciliation_sources(
    actions: Sequence[Action],
    memories: Sequence[Memory],
) -> list[ReconciliationSource]:
    """A session's reconcilable sources, actions first in recorded order.

    The order is the order they are joined into the session's one document,
    so the extractor reads the conversation as it happened.
    """
    from actions_graph.models import Message

    sources: list[ReconciliationSource] = []
    for action in actions:
        text = extract_reconcilable_text(action)
        if text:
            sources.append(
                ReconciliationSource(
                    kind="action",
                    node_id=action.action_id,
                    text=text,
                    role=action.role.value if isinstance(action, Message) else None,
                    valid_at=normalize_timestamp(action.timestamp),
                )
            )
    for memory in memories:
        if memory.content and memory.content.strip():
            sources.append(
                ReconciliationSource(
                    kind="memory",
                    node_id=memory.memory_id,
                    text=memory.content.strip(),
                    valid_at=normalize_timestamp(memory.created_at),
                )
            )
    return sources
