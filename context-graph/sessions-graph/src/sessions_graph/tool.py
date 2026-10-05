"""The ``recall`` tool: a harness's model asks its user's memory a question.

Registered under agent-context-graph's ``agent_context_graph.tools`` entry
point, so ``agent-context-graph mcp`` serves it and ``agent-context-graph
recall`` runs it. It returns context, not an answer: the harness's own model
answers from the rows, following the reading rules at their head (#394).

Whose memory and which Memgraph come from the config file, never from the
call, so a model can only read its own user's sessions. ``[recall]`` in the
same file may override recall's lanes and widths (``turns_k = 12``); the
defaults are what the benchmark measured.
"""

from __future__ import annotations

import threading
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from agent_context_graph.tools import ToolError, ToolResult

from .core import SessionsGraph
from .embeddings import DEFAULT_EMBEDDING_MODEL
from .recall import RecallConfig

if TYPE_CHECKING:
    from agent_context_graph.adapters._identity import HookConfig

_DESCRIPTION = """\
Search the user's memory: their own past conversations with you and other agents, \
across sessions and projects, including this one. Returns the matching messages \
and the facts extracted from them, each dated, oldest first. Call it whenever the \
user refers to something from before ("what did we decide about...", "the bug from \
last week"), asks about their own preferences, plans, history or possessions, or \
when earlier context would change your answer. Ask one specific question in the \
user's words; call again with another question to look for something else. Answer \
from the rows yourself, following the reading rules at the top of the result; if \
the rows don't contain the answer, say it isn't in memory."""


class RecallTool:
    """``recall(question)`` over the configured user's sessions; keeps one connection per config."""

    name = "recall"
    connector = "sessions-graph"
    description = _DESCRIPTION
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What to look for, as the user would ask it."},
        },
        "required": ["question"],
    }
    session_hint = (
        "This user's past sessions are in Context Graph memory. When earlier conversations, decisions or "
        "preferences could matter, call the `recall` tool with a question instead of asking the user to repeat them."
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._graph: SessionsGraph | None = None
        self._graph_key: tuple[str, ...] | None = None

    def call(self, arguments: dict[str, Any], config: HookConfig) -> ToolResult:
        """Recall what the configured user's sessions hold about ``arguments["question"]``.

        Raises:
            ToolError: no question, no ``identity.user_id`` configured, or invalid ``[recall]`` settings.
        """
        question = str(arguments.get("question") or "").strip()
        if not question:
            raise ToolError("recall needs a question.")
        if not config.user_id:
            raise ToolError(
                "No user is configured, so there is no memory to search. "
                "Set one with: agent-context-graph config set identity.user_id <name>"
            )
        try:
            settings = RecallConfig.from_mapping(config.recall_settings)
        except ValueError as exc:
            raise ToolError(f"[recall] in {_config_path()} is invalid: {exc}") from exc

        recalled = self._graph_for(config).recall(
            config.user_id, question, config=settings, model=config.embedding_model or DEFAULT_EMBEDDING_MODEL
        )
        return ToolResult(text=recalled.render(today=date.today().isoformat()), structured=recalled.to_json())

    def _graph_for(self, config: HookConfig) -> SessionsGraph:
        key = (config.memgraph_url, config.memgraph_user, config.memgraph_password, config.memgraph_database)
        with self._lock:
            if self._graph is None or self._graph_key != key:
                graph = SessionsGraph(
                    url=config.memgraph_url,
                    username=config.memgraph_user,
                    password=config.memgraph_password,
                    database=config.memgraph_database,
                )
                # Recall's text lane needs the message text index; setup is idempotent.
                graph.setup()
                self._graph, self._graph_key = graph, key
            return self._graph


def _config_path() -> str:
    from agent_context_graph.adapters._identity import config_file

    return str(config_file())


RECALL = RecallTool()
