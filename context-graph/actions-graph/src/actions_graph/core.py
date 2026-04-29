"""Core ActionsGraph class for storing and querying LLM actions in Memgraph.

This module provides the main interface for persisting and analyzing
LLM actions, tool calls, and sessions in a Memgraph graph database.

Graph Schema:
    Nodes:
        - (:Session) - LLM conversation sessions
        - (:Action) - Individual actions with labels for type (ToolCall, Message, etc.)
        - (:Tool) - Tool definitions
        - (:Tag) - Session/action tags

    Relationships:
        - (:Session)-[:HAS_ACTION]->(:Action)
        - (:Action)-[:FOLLOWED_BY]->(:Action) - Temporal sequence
        - (:Action)-[:PARENT_OF]->(:Action) - Nested actions (e.g., subagent)
        - (:Session)-[:FORKED_FROM]->(:Session)
        - (:Action)-[:USED_TOOL]->(:Tool)
        - (:Session)-[:HAS_TAG]->(:Tag)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from memgraph_toolbox.api.memgraph import Memgraph

from .models import (
    Action,
    ActionStatus,
    ActionType,
    ErrorEvent,
    Message,
    MessageRole,
    PermissionRequest,
    RateLimitEvent,
    Session,
    StructuredOutput,
    SubagentEvent,
    ToolCall,
    ToolResult,
)


class ActionsGraph:
    """Store and query LLM actions and sessions in Memgraph.

    Provides methods for:
    - Creating and managing sessions
    - Recording tool calls, messages, and other actions
    - Querying action history and analytics
    - Building action sequence graphs
    """

    def __init__(self, memgraph: Memgraph | None = None, **kwargs: Any):
        """Initialize ActionsGraph.

        Args:
            memgraph: An existing Memgraph client instance. If not provided,
                      a new one is created using kwargs / environment variables.
            **kwargs: Forwarded to Memgraph() when memgraph is None.
        """
        self._db = memgraph or Memgraph(**kwargs)
        self._last_action_id: dict[str, str] = {}  # session_id -> last action_id

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create constraints and indexes required for action storage."""
        # Session constraints and indexes
        self._db.query("CREATE CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;")
        self._db.query("CREATE INDEX ON :Session(session_id);")
        self._db.query("CREATE INDEX ON :Session(started_at);")
        self._db.query("CREATE INDEX ON :Session(status);")

        # Action constraints and indexes
        self._db.query("CREATE CONSTRAINT ON (a:Action) ASSERT a.action_id IS UNIQUE;")
        self._db.query("CREATE INDEX ON :Action(action_id);")
        self._db.query("CREATE INDEX ON :Action(session_id);")
        self._db.query("CREATE INDEX ON :Action(timestamp);")
        self._db.query("CREATE INDEX ON :Action(action_type);")

        # Tool indexes
        self._db.query("CREATE INDEX ON :Tool(name);")

        # Tag indexes
        self._db.query("CREATE INDEX ON :Tag(name);")

    def drop(self) -> None:
        """Remove all action-related constraints and indexes."""
        import contextlib

        with contextlib.suppress(Exception):
            self._db.query("DROP CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;")
        with contextlib.suppress(Exception):
            self._db.query("DROP CONSTRAINT ON (a:Action) ASSERT a.action_id IS UNIQUE;")
        # Indexes are dropped with constraints in most cases

    def clear(self) -> None:
        """Remove all session and action data from the graph."""
        self._db.query("MATCH (n) WHERE n:Session OR n:Action OR n:Tool DETACH DELETE n;")
        self._last_action_id.clear()

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    def create_session(self, session: Session) -> Session:
        """Create a new session in the graph.

        Args:
            session: Session object to persist

        Returns:
            The persisted session
        """
        self._db.query(
            """
            CREATE (s:Session {
                session_id: $session_id,
                started_at: $started_at,
                ended_at: $ended_at,
                status: $status,
                model: $model,
                total_cost_usd: $total_cost_usd,
                total_input_tokens: $total_input_tokens,
                total_output_tokens: $total_output_tokens,
                working_directory: $working_directory,
                git_branch: $git_branch,
                metadata: $metadata,
                parent_session_id: $parent_session_id
            })
            """,
            params={
                "session_id": session.session_id,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "status": session.status.value,
                "model": session.model,
                "total_cost_usd": session.total_cost_usd,
                "total_input_tokens": session.total_input_tokens,
                "total_output_tokens": session.total_output_tokens,
                "working_directory": session.working_directory,
                "git_branch": session.git_branch,
                "metadata": json.dumps(session.metadata),
                "parent_session_id": session.parent_session_id,
            },
        )

        # Handle forked sessions
        if session.parent_session_id:
            self._db.query(
                """
                MATCH (child:Session {session_id: $session_id})
                MATCH (parent:Session {session_id: $parent_session_id})
                MERGE (child)-[:FORKED_FROM]->(parent)
                """,
                params={
                    "session_id": session.session_id,
                    "parent_session_id": session.parent_session_id,
                },
            )

        # Handle tags
        if session.tags:
            self._db.query(
                """
                MATCH (s:Session {session_id: $session_id})
                UNWIND $tags AS tag_name
                MERGE (t:Tag {name: tag_name})
                MERGE (s)-[:HAS_TAG]->(t)
                """,
                params={"session_id": session.session_id, "tags": session.tags},
            )

        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Args:
            session_id: Unique session identifier

        Returns:
            Session object or None if not found
        """
        rows = self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.session_id AS session_id,
                   s.started_at AS started_at,
                   s.ended_at AS ended_at,
                   s.status AS status,
                   s.model AS model,
                   s.total_cost_usd AS total_cost_usd,
                   s.total_input_tokens AS total_input_tokens,
                   s.total_output_tokens AS total_output_tokens,
                   s.working_directory AS working_directory,
                   s.git_branch AS git_branch,
                   s.metadata AS metadata,
                   s.parent_session_id AS parent_session_id,
                   collect(t.name) AS tags
            """,
            params={"session_id": session_id},
        )

        if not rows:
            return None

        row = rows[0]
        return self._row_to_session(row)

    def end_session(
        self,
        session_id: str,
        *,
        status: ActionStatus = ActionStatus.COMPLETED,
        total_cost_usd: float | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
    ) -> Session | None:
        """Mark a session as ended.

        Args:
            session_id: Session to end
            status: Final status
            total_cost_usd: Total cost in USD
            total_input_tokens: Total input tokens
            total_output_tokens: Total output tokens

        Returns:
            Updated session or None if not found
        """
        ended_at = datetime.now(timezone.utc).isoformat()

        sets = ["s.ended_at = $ended_at", "s.status = $status"]
        params: dict[str, Any] = {
            "session_id": session_id,
            "ended_at": ended_at,
            "status": status.value,
        }

        if total_cost_usd is not None:
            sets.append("s.total_cost_usd = $total_cost_usd")
            params["total_cost_usd"] = total_cost_usd
        if total_input_tokens is not None:
            sets.append("s.total_input_tokens = $total_input_tokens")
            params["total_input_tokens"] = total_input_tokens
        if total_output_tokens is not None:
            sets.append("s.total_output_tokens = $total_output_tokens")
            params["total_output_tokens"] = total_output_tokens

        self._db.query(
            f"""
            MATCH (s:Session {{session_id: $session_id}})
            SET {", ".join(sets)}
            """,
            params=params,
        )

        # Clean up last action tracking
        self._last_action_id.pop(session_id, None)

        return self.get_session(session_id)

    def list_sessions(
        self,
        *,
        limit: int = 100,
        status: ActionStatus | None = None,
        tag: str | None = None,
    ) -> list[Session]:
        """List sessions with optional filtering.

        Args:
            limit: Maximum number of sessions to return
            status: Filter by status
            tag: Filter by tag

        Returns:
            List of sessions ordered by start time (newest first)
        """
        where_clauses = []
        params: dict[str, Any] = {"limit": limit}

        if status:
            where_clauses.append("s.status = $status")
            params["status"] = status.value

        if tag:
            where_clauses.append("EXISTS((s)-[:HAS_TAG]->(:Tag {name: $tag}))")
            params["tag"] = tag

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        rows = self._db.query(
            f"""
            MATCH (s:Session)
            {where_str}
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.session_id AS session_id,
                   s.started_at AS started_at,
                   s.ended_at AS ended_at,
                   s.status AS status,
                   s.model AS model,
                   s.total_cost_usd AS total_cost_usd,
                   s.total_input_tokens AS total_input_tokens,
                   s.total_output_tokens AS total_output_tokens,
                   s.working_directory AS working_directory,
                   s.git_branch AS git_branch,
                   s.metadata AS metadata,
                   s.parent_session_id AS parent_session_id,
                   collect(t.name) AS tags
            ORDER BY s.started_at DESC
            LIMIT $limit
            """,
            params=params,
        )

        return [self._row_to_session(row) for row in rows]

    # ------------------------------------------------------------------
    # Action operations
    # ------------------------------------------------------------------

    def record_action(self, action: Action) -> Action:
        """Record an action in the graph.

        Creates the action node, links it to the session, and creates
        temporal FOLLOWED_BY relationships with the previous action.

        Args:
            action: Action to record

        Returns:
            The recorded action
        """
        # Determine additional labels based on action type
        type_labels = self._get_type_labels(action)
        labels_str = ":Action" + "".join(f":{lbl}" for lbl in type_labels)

        # Build properties based on action type
        props = self._action_to_props(action)

        # Create the action node
        self._db.query(
            f"""
            CREATE (a{labels_str} {{
                action_id: $action_id,
                session_id: $session_id,
                action_type: $action_type,
                timestamp: $timestamp,
                status: $status,
                duration_ms: $duration_ms,
                parent_action_id: $parent_action_id,
                metadata: $metadata,
                properties: $properties
            }})
            """,
            params={
                "action_id": action.action_id,
                "session_id": action.session_id,
                "action_type": action.action_type.value,
                "timestamp": action.timestamp,
                "status": action.status.value,
                "duration_ms": action.duration_ms,
                "parent_action_id": action.parent_action_id,
                "metadata": json.dumps(action.metadata),
                "properties": json.dumps(props),
            },
        )

        # Link to session
        self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            MATCH (a:Action {action_id: $action_id})
            MERGE (s)-[:HAS_ACTION]->(a)
            """,
            params={"session_id": action.session_id, "action_id": action.action_id},
        )

        # Create temporal sequence
        last_action_id = self._last_action_id.get(action.session_id)
        if last_action_id:
            self._db.query(
                """
                MATCH (prev:Action {action_id: $prev_id})
                MATCH (curr:Action {action_id: $curr_id})
                MERGE (prev)-[:FOLLOWED_BY]->(curr)
                """,
                params={"prev_id": last_action_id, "curr_id": action.action_id},
            )

        self._last_action_id[action.session_id] = action.action_id

        # Handle parent action relationship
        if action.parent_action_id:
            self._db.query(
                """
                MATCH (parent:Action {action_id: $parent_id})
                MATCH (child:Action {action_id: $child_id})
                MERGE (parent)-[:PARENT_OF]->(child)
                """,
                params={"parent_id": action.parent_action_id, "child_id": action.action_id},
            )

        # Handle tool references
        if isinstance(action, ToolCall):
            self._db.query(
                """
                MATCH (a:Action {action_id: $action_id})
                MERGE (t:Tool {name: $tool_name})
                ON CREATE SET t.is_mcp = $is_mcp, t.mcp_server = $mcp_server
                MERGE (a)-[:USED_TOOL]->(t)
                """,
                params={
                    "action_id": action.action_id,
                    "tool_name": action.tool_name,
                    "is_mcp": action.is_mcp,
                    "mcp_server": action.mcp_server,
                },
            )

        return action

    def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str | None = None,
        **kwargs: Any,
    ) -> ToolCall:
        """Convenience method to record a tool call.

        Args:
            session_id: Session ID
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_use_id: Optional tool use ID for correlation
            **kwargs: Additional ToolCall fields

        Returns:
            The recorded ToolCall
        """
        action = ToolCall(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            **kwargs,
        )
        return self.record_action(action)  # type: ignore

    def record_tool_result(
        self,
        session_id: str,
        tool_use_id: str,
        tool_name: str,
        content: str | list[dict[str, Any]] | None = None,
        is_error: bool = False,
        error_message: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Convenience method to record a tool result.

        Args:
            session_id: Session ID
            tool_use_id: ID of the tool call
            tool_name: Name of the tool
            content: Result content
            is_error: Whether the execution failed
            error_message: Error message if failed
            **kwargs: Additional ToolResult fields

        Returns:
            The recorded ToolResult
        """
        action = ToolResult(
            session_id=session_id,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            content=content,
            is_error=is_error,
            error_message=error_message,
            **kwargs,
        )
        return self.record_action(action)  # type: ignore

    def record_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Message:
        """Convenience method to record a message.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional Message fields

        Returns:
            The recorded Message
        """
        action = Message(
            session_id=session_id,
            role=role,
            content=content,
            **kwargs,
        )
        return self.record_action(action)  # type: ignore

    def get_action(self, action_id: str) -> Action | None:
        """Retrieve an action by ID.

        Args:
            action_id: Unique action identifier

        Returns:
            Action object or None if not found
        """
        rows = self._db.query(
            """
            MATCH (a:Action {action_id: $action_id})
            RETURN a.action_id AS action_id,
                   a.session_id AS session_id,
                   a.action_type AS action_type,
                   a.timestamp AS timestamp,
                   a.status AS status,
                   a.duration_ms AS duration_ms,
                   a.parent_action_id AS parent_action_id,
                   a.metadata AS metadata,
                   a.properties AS properties,
                   labels(a) AS labels
            """,
            params={"action_id": action_id},
        )

        if not rows:
            return None

        return self._row_to_action(rows[0])

    def get_session_actions(
        self,
        session_id: str,
        *,
        action_type: ActionType | None = None,
        limit: int = 1000,
    ) -> list[Action]:
        """Get all actions for a session.

        Args:
            session_id: Session ID
            action_type: Filter by action type
            limit: Maximum number of actions

        Returns:
            List of actions ordered by timestamp
        """
        where_clauses = ["a.session_id = $session_id"]
        params: dict[str, Any] = {"session_id": session_id, "limit": limit}

        if action_type:
            where_clauses.append("a.action_type = $action_type")
            params["action_type"] = action_type.value

        where_str = f"WHERE {' AND '.join(where_clauses)}"

        rows = self._db.query(
            f"""
            MATCH (a:Action)
            {where_str}
            RETURN a.action_id AS action_id,
                   a.session_id AS session_id,
                   a.action_type AS action_type,
                   a.timestamp AS timestamp,
                   a.status AS status,
                   a.duration_ms AS duration_ms,
                   a.parent_action_id AS parent_action_id,
                   a.metadata AS metadata,
                   a.properties AS properties,
                   labels(a) AS labels
            ORDER BY a.timestamp
            LIMIT $limit
            """,
            params=params,
        )

        return [self._row_to_action(row) for row in rows]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_tool_usage_stats(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get tool usage statistics.

        Args:
            session_id: Optional session filter

        Returns:
            List of tool usage statistics
        """
        where_clause = ""
        params: dict[str, Any] = {}

        if session_id:
            where_clause = "WHERE a.session_id = $session_id"
            params["session_id"] = session_id

        rows = self._db.query(
            f"""
            MATCH (a:Action:ToolCall)-[:USED_TOOL]->(t:Tool)
            {where_clause}
            RETURN t.name AS tool_name,
                   t.is_mcp AS is_mcp,
                   t.mcp_server AS mcp_server,
                   count(a) AS call_count,
                   avg(a.duration_ms) AS avg_duration_ms,
                   sum(CASE WHEN a.status = 'failed' THEN 1 ELSE 0 END) AS error_count
            ORDER BY call_count DESC
            """,
            params=params,
        )

        return [dict(row) for row in rows]

    def get_action_sequence(
        self,
        session_id: str,
        *,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Get the sequence of actions in a session.

        Args:
            session_id: Session ID
            include_content: Whether to include full action content

        Returns:
            List of actions in sequence with relationships
        """
        rows = self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})-[:HAS_ACTION]->(a:Action)
            OPTIONAL MATCH (a)-[:FOLLOWED_BY]->(next:Action)
            OPTIONAL MATCH (a)-[:PARENT_OF]->(child:Action)
            RETURN a.action_id AS action_id,
                   a.action_type AS action_type,
                   a.timestamp AS timestamp,
                   a.status AS status,
                   a.properties AS properties,
                   next.action_id AS next_action_id,
                   collect(DISTINCT child.action_id) AS child_action_ids
            ORDER BY a.timestamp
            """,
            params={"session_id": session_id},
        )

        result = []
        for row in rows:
            item = {
                "action_id": row["action_id"],
                "action_type": row["action_type"],
                "timestamp": row["timestamp"],
                "status": row["status"],
                "next_action_id": row["next_action_id"],
                "child_action_ids": row["child_action_ids"],
            }
            if include_content:
                item["properties"] = json.loads(row["properties"]) if row["properties"] else {}
            result.append(item)

        return result

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of a session.

        Args:
            session_id: Session ID

        Returns:
            Summary statistics for the session
        """
        rows = self._db.query(
            """
            MATCH (s:Session {session_id: $session_id})
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(a:Action)
            RETURN s.session_id AS session_id,
                   s.started_at AS started_at,
                   s.ended_at AS ended_at,
                   s.status AS status,
                   s.model AS model,
                   s.total_cost_usd AS total_cost_usd,
                   s.total_input_tokens AS total_input_tokens,
                   s.total_output_tokens AS total_output_tokens,
                   count(a) AS action_count,
                   sum(CASE WHEN a.action_type = 'tool_call' THEN 1 ELSE 0 END) AS tool_call_count,
                   sum(CASE WHEN a.action_type = 'user_message' THEN 1 ELSE 0 END) AS user_message_count,
                   sum(CASE WHEN a.action_type = 'assistant_message' THEN 1 ELSE 0 END) AS assistant_message_count,
                   sum(CASE WHEN a.status = 'failed' THEN 1 ELSE 0 END) AS error_count
            """,
            params={"session_id": session_id},
        )

        if not rows:
            return {}

        row = rows[0]
        return {
            "session_id": row["session_id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "status": row["status"],
            "model": row["model"],
            "total_cost_usd": row["total_cost_usd"],
            "total_input_tokens": row["total_input_tokens"],
            "total_output_tokens": row["total_output_tokens"],
            "action_count": row["action_count"],
            "tool_call_count": row["tool_call_count"],
            "user_message_count": row["user_message_count"],
            "assistant_message_count": row["assistant_message_count"],
            "error_count": row["error_count"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_type_labels(self, action: Action) -> list[str]:
        """Get additional labels for an action based on its type."""
        labels = []
        if isinstance(action, ToolCall):
            labels.append("ToolCall")
        elif isinstance(action, ToolResult):
            labels.append("ToolResult")
        elif isinstance(action, Message):
            labels.append("Message")
            labels.append(action.role.value.title() + "Message")
        elif isinstance(action, StructuredOutput):
            labels.append("StructuredOutput")
        elif isinstance(action, SubagentEvent):
            labels.append("SubagentEvent")
        elif isinstance(action, PermissionRequest):
            labels.append("PermissionRequest")
        elif isinstance(action, ErrorEvent):
            labels.append("ErrorEvent")
        elif isinstance(action, RateLimitEvent):
            labels.append("RateLimitEvent")
        return labels

    def _action_to_props(self, action: Action) -> dict[str, Any]:
        """Extract type-specific properties from an action."""
        props: dict[str, Any] = {}

        if isinstance(action, ToolCall):
            props["tool_name"] = action.tool_name
            props["tool_input"] = action.tool_input
            props["tool_use_id"] = action.tool_use_id
            props["is_mcp"] = action.is_mcp
            props["mcp_server"] = action.mcp_server
        elif isinstance(action, ToolResult):
            props["tool_use_id"] = action.tool_use_id
            props["tool_name"] = action.tool_name
            props["content"] = action.content
            props["is_error"] = action.is_error
            props["error_message"] = action.error_message
        elif isinstance(action, Message):
            props["role"] = action.role.value
            props["content"] = action.content
            props["message_id"] = action.message_id
            props["model"] = action.model
            props["usage"] = action.usage
        elif isinstance(action, StructuredOutput):
            props["output_type"] = action.output_type
            props["output_data"] = action.output_data
            props["schema"] = action.schema
            props["validation_passed"] = action.validation_passed
        elif isinstance(action, SubagentEvent):
            props["agent_id"] = action.agent_id
            props["agent_type"] = action.agent_type
            props["description"] = action.description
            props["result"] = action.result
            props["usage"] = action.usage
        elif isinstance(action, PermissionRequest):
            props["tool_name"] = action.tool_name
            props["tool_input"] = action.tool_input
            props["decision"] = action.decision
            props["reason"] = action.reason
        elif isinstance(action, ErrorEvent):
            props["error_type"] = action.error_type
            props["error_message"] = action.error_message
            props["error_details"] = action.error_details
            props["recoverable"] = action.recoverable
        elif isinstance(action, RateLimitEvent):
            props["rate_limit_status"] = action.rate_limit_status
            props["rate_limit_type"] = action.rate_limit_type
            props["resets_at"] = action.resets_at
            props["utilization"] = action.utilization

        return props

    def _row_to_session(self, row: dict[str, Any]) -> Session:
        """Convert a database row to a Session object."""
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Session(
            session_id=row["session_id"],
            started_at=row["started_at"],
            ended_at=row.get("ended_at"),
            status=ActionStatus(row["status"]) if row.get("status") else ActionStatus.IN_PROGRESS,
            model=row.get("model"),
            total_cost_usd=row.get("total_cost_usd"),
            total_input_tokens=row.get("total_input_tokens", 0),
            total_output_tokens=row.get("total_output_tokens", 0),
            working_directory=row.get("working_directory"),
            git_branch=row.get("git_branch"),
            tags=row.get("tags", []),
            metadata=metadata or {},
            parent_session_id=row.get("parent_session_id"),
        )

    def _row_to_action(self, row: dict[str, Any]) -> Action:
        """Convert a database row to an Action object."""
        action_type = ActionType(row["action_type"])
        props = json.loads(row["properties"]) if row.get("properties") else {}
        metadata = json.loads(row["metadata"]) if row.get("metadata") else {}

        base_kwargs = {
            "action_id": row["action_id"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "status": ActionStatus(row["status"]) if row.get("status") else ActionStatus.COMPLETED,
            "duration_ms": row.get("duration_ms"),
            "parent_action_id": row.get("parent_action_id"),
            "metadata": metadata,
        }

        if action_type == ActionType.TOOL_CALL:
            return ToolCall(
                **base_kwargs,
                tool_name=props.get("tool_name", ""),
                tool_input=props.get("tool_input", {}),
                tool_use_id=props.get("tool_use_id"),
                is_mcp=props.get("is_mcp", False),
                mcp_server=props.get("mcp_server"),
            )
        elif action_type == ActionType.TOOL_RESULT:
            return ToolResult(
                **base_kwargs,
                tool_use_id=props.get("tool_use_id", ""),
                tool_name=props.get("tool_name", ""),
                content=props.get("content"),
                is_error=props.get("is_error", False),
                error_message=props.get("error_message"),
            )
        elif action_type in (
            ActionType.USER_MESSAGE,
            ActionType.ASSISTANT_MESSAGE,
            ActionType.SYSTEM_MESSAGE,
        ):
            role = MessageRole(props.get("role", "user"))
            return Message(
                **base_kwargs,
                role=role,
                content=props.get("content", ""),
                message_id=props.get("message_id"),
                model=props.get("model"),
                usage=props.get("usage"),
            )
        elif action_type == ActionType.STRUCTURED_OUTPUT:
            return StructuredOutput(
                **base_kwargs,
                output_type=props.get("output_type", ""),
                output_data=props.get("output_data"),
                schema=props.get("schema"),
                validation_passed=props.get("validation_passed", True),
            )
        elif action_type in (ActionType.SUBAGENT_START, ActionType.SUBAGENT_STOP):
            event = SubagentEvent(
                **base_kwargs,
                agent_id=props.get("agent_id", ""),
                agent_type=props.get("agent_type", ""),
                description=props.get("description", ""),
                result=props.get("result"),
                usage=props.get("usage"),
            )
            event.action_type = action_type
            return event
        elif action_type == ActionType.PERMISSION_REQUEST:
            return PermissionRequest(
                **base_kwargs,
                tool_name=props.get("tool_name", ""),
                tool_input=props.get("tool_input", {}),
                decision=props.get("decision"),
                reason=props.get("reason"),
            )
        elif action_type == ActionType.ERROR:
            return ErrorEvent(
                **base_kwargs,
                error_type=props.get("error_type", ""),
                error_message=props.get("error_message", ""),
                error_details=props.get("error_details"),
                recoverable=props.get("recoverable", True),
            )
        elif action_type == ActionType.RATE_LIMIT:
            return RateLimitEvent(
                **base_kwargs,
                rate_limit_status=props.get("rate_limit_status", ""),
                rate_limit_type=props.get("rate_limit_type"),
                resets_at=props.get("resets_at"),
                utilization=props.get("utilization"),
            )
        else:
            return Action(**base_kwargs, action_type=action_type)
