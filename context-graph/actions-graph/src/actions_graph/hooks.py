"""Integration hooks for Claude Agent SDK.

This module provides hooks that automatically track LLM actions
from the Claude Agent SDK and persist them to Memgraph.

Usage:
    from actions_graph import ActionsGraph
    from actions_graph.hooks import create_tracking_hooks

    graph = ActionsGraph()
    hooks = create_tracking_hooks(graph, session_id="my-session")

    # Use with Claude Agent SDK
    options = ClaudeAgentOptions(
        hooks=hooks,
        ...
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import (
    ActionStatus,
    ActionType,
    ErrorEvent,
    Message,
    MessageRole,
    PermissionRequest,
    RateLimitEvent,
    Session,
    SubagentEvent,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .core import ActionsGraph

    # Type hints for Claude Agent SDK (optional dependency)
    HookCallback = Callable[[dict[str, Any], str | None, dict[str, Any]], Awaitable[dict[str, Any]]]


class ActionTracker:
    """Tracks LLM actions and persists them to Memgraph.

    This class provides hook callbacks compatible with the Claude Agent SDK
    that automatically record all actions to an ActionsGraph instance.
    """

    def __init__(
        self,
        graph: ActionsGraph,
        session_id: str,
        *,
        track_tool_calls: bool = True,
        track_tool_results: bool = True,
        track_messages: bool = True,
        track_subagents: bool = True,
        track_permissions: bool = True,
        track_errors: bool = True,
        track_rate_limits: bool = True,
    ):
        """Initialize ActionTracker.

        Args:
            graph: ActionsGraph instance for persistence
            session_id: Session ID to associate actions with
            track_tool_calls: Whether to track PreToolUse events
            track_tool_results: Whether to track PostToolUse events
            track_messages: Whether to track UserPromptSubmit events
            track_subagents: Whether to track SubagentStart/Stop events
            track_permissions: Whether to track PermissionRequest events
            track_errors: Whether to track PostToolUseFailure events
            track_rate_limits: Whether to track rate limit events
        """
        self.graph = graph
        self.session_id = session_id
        self.track_tool_calls = track_tool_calls
        self.track_tool_results = track_tool_results
        self.track_messages = track_messages
        self.track_subagents = track_subagents
        self.track_permissions = track_permissions
        self.track_errors = track_errors
        self.track_rate_limits = track_rate_limits

        # Track tool use IDs to parent action IDs
        self._tool_use_to_action: dict[str, str] = {}

    async def pre_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for PreToolUse events.

        Records tool calls before they are executed.
        """
        if not self.track_tool_calls:
            return {}

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        actual_tool_use_id = input_data.get("tool_use_id") or tool_use_id

        # Get parent action from subagent context
        parent_action_id = None
        agent_id = input_data.get("agent_id")
        if agent_id:
            # This is a subagent tool call
            parent_action_id = self._tool_use_to_action.get(agent_id)

        action = ToolCall(
            session_id=self.session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=actual_tool_use_id,
            status=ActionStatus.IN_PROGRESS,
            parent_action_id=parent_action_id,
            metadata={
                "agent_id": agent_id,
                "agent_type": input_data.get("agent_type"),
                "cwd": input_data.get("cwd"),
            },
        )

        self.graph.record_action(action)

        # Track tool use ID to action ID mapping
        if actual_tool_use_id:
            self._tool_use_to_action[actual_tool_use_id] = action.action_id

        return {}

    async def post_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for PostToolUse events.

        Records tool results after execution.
        """
        if not self.track_tool_results:
            return {}

        tool_name = input_data.get("tool_name", "")
        actual_tool_use_id = input_data.get("tool_use_id") or tool_use_id
        tool_response = input_data.get("tool_response")

        # Determine if this is an error response
        is_error = False
        error_message = None
        content = tool_response

        if isinstance(tool_response, dict):
            is_error = tool_response.get("is_error", False)
            error_message = tool_response.get("error")
            content = tool_response.get("content", tool_response)

        # Find parent tool call action
        parent_action_id = None
        if actual_tool_use_id:
            parent_action_id = self._tool_use_to_action.get(actual_tool_use_id)

        action = ToolResult(
            session_id=self.session_id,
            tool_use_id=actual_tool_use_id or "",
            tool_name=tool_name,
            content=content if isinstance(content, (str, list)) else str(content),
            is_error=is_error,
            error_message=error_message,
            parent_action_id=parent_action_id,
            metadata={
                "agent_id": input_data.get("agent_id"),
                "agent_type": input_data.get("agent_type"),
            },
        )

        self.graph.record_action(action)
        return {}

    async def post_tool_use_failure(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for PostToolUseFailure events.

        Records tool execution failures.
        """
        if not self.track_errors:
            return {}

        tool_name = input_data.get("tool_name", "")
        actual_tool_use_id = input_data.get("tool_use_id") or tool_use_id
        error = input_data.get("error", "Unknown error")
        is_interrupt = input_data.get("is_interrupt", False)

        # Find parent tool call action
        parent_action_id = None
        if actual_tool_use_id:
            parent_action_id = self._tool_use_to_action.get(actual_tool_use_id)

        action = ErrorEvent(
            session_id=self.session_id,
            error_type="tool_failure",
            error_message=error,
            error_details={
                "tool_name": tool_name,
                "tool_use_id": actual_tool_use_id,
                "tool_input": input_data.get("tool_input", {}),
                "is_interrupt": is_interrupt,
            },
            recoverable=not is_interrupt,
            parent_action_id=parent_action_id,
        )

        self.graph.record_action(action)
        return {}

    async def user_prompt_submit(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for UserPromptSubmit events.

        Records user messages.
        """
        if not self.track_messages:
            return {}

        prompt = input_data.get("prompt", "")

        action = Message(
            session_id=self.session_id,
            role=MessageRole.USER,
            content=prompt,
            metadata={
                "cwd": input_data.get("cwd"),
                "permission_mode": input_data.get("permission_mode"),
            },
        )

        self.graph.record_action(action)
        return {}

    async def subagent_start(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for SubagentStart events.

        Records when subagents start.
        """
        if not self.track_subagents:
            return {}

        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")

        action = SubagentEvent(
            session_id=self.session_id,
            agent_id=agent_id,
            agent_type=agent_type,
            status=ActionStatus.IN_PROGRESS,
        )
        action.action_type = ActionType.SUBAGENT_START

        self.graph.record_action(action)

        # Track agent_id to action mapping for nested tool calls
        self._tool_use_to_action[agent_id] = action.action_id

        return {}

    async def subagent_stop(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for SubagentStop events.

        Records when subagents complete.
        """
        if not self.track_subagents:
            return {}

        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")
        stop_hook_active = input_data.get("stop_hook_active", False)

        # Find the start event
        parent_action_id = self._tool_use_to_action.get(agent_id)

        action = SubagentEvent(
            session_id=self.session_id,
            agent_id=agent_id,
            agent_type=agent_type,
            status=ActionStatus.COMPLETED,
            parent_action_id=parent_action_id,
            metadata={
                "stop_hook_active": stop_hook_active,
                "agent_transcript_path": input_data.get("agent_transcript_path"),
            },
        )
        action.action_type = ActionType.SUBAGENT_STOP

        self.graph.record_action(action)
        return {}

    async def permission_request(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for PermissionRequest events.

        Records permission requests.
        """
        if not self.track_permissions:
            return {}

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        action = PermissionRequest(
            session_id=self.session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            status=ActionStatus.PENDING,
            metadata={
                "permission_suggestions": input_data.get("permission_suggestions", []),
            },
        )

        self.graph.record_action(action)
        return {}

    async def notification(
        self,
        input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for Notification events.

        Records system notifications.
        """
        message = input_data.get("message", "")
        title = input_data.get("title")
        notification_type = input_data.get("notification_type", "")

        action = Message(
            session_id=self.session_id,
            role=MessageRole.SYSTEM,
            content=message,
            metadata={
                "title": title,
                "notification_type": notification_type,
            },
        )

        self.graph.record_action(action)
        return {}

    async def stop(
        self,
        _input_data: dict[str, Any],
        _tool_use_id: str | None,
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook callback for Stop events.

        Records when execution stops.
        """
        # Update session as ended
        self.graph.end_session(self.session_id, status=ActionStatus.COMPLETED)
        return {}


def create_tracking_hooks(
    graph: ActionsGraph,
    session_id: str,
    *,
    create_session: bool = True,
    session_kwargs: dict[str, Any] | None = None,
    **tracker_kwargs: Any,
) -> dict[str, list[Any]]:
    """Create Claude Agent SDK hooks for action tracking.

    This function creates a complete set of hooks that can be passed
    directly to ClaudeAgentOptions.hooks.

    Args:
        graph: ActionsGraph instance for persistence
        session_id: Session ID to associate actions with
        create_session: Whether to create the session in the graph
        session_kwargs: Additional kwargs for Session creation
        **tracker_kwargs: Additional kwargs for ActionTracker

    Returns:
        Dictionary of hook event names to hook matchers, ready for
        use with ClaudeAgentOptions.hooks

    Example:
        from actions_graph import ActionsGraph
        from actions_graph.hooks import create_tracking_hooks
        from claude_agent_sdk import query, ClaudeAgentOptions

        graph = ActionsGraph()
        graph.setup()

        hooks = create_tracking_hooks(graph, session_id="my-session-123")

        async for message in query(
            prompt="Hello!",
            options=ClaudeAgentOptions(
                hooks=hooks,
                allowed_tools=["Read", "Write"],
            ),
        ):
            print(message)
    """
    # Try to import HookMatcher from claude_agent_sdk
    try:
        from claude_agent_sdk import HookMatcher
    except ImportError:
        # Create a simple HookMatcher-like class if SDK not available
        class HookMatcher:  # type: ignore[no-redef]
            def __init__(
                self,
                matcher: str | None = None,
                hooks: list[Any] | None = None,
                timeout: float | None = None,
            ):
                self.matcher = matcher
                self.hooks = hooks or []
                self.timeout = timeout

    # Create session if requested
    if create_session:
        session_kwargs = session_kwargs or {}
        session = Session(session_id=session_id, **session_kwargs)
        graph.create_session(session)

    # Create tracker
    tracker = ActionTracker(graph, session_id, **tracker_kwargs)

    # Build hooks dictionary
    return {
        "PreToolUse": [HookMatcher(hooks=[tracker.pre_tool_use])],
        "PostToolUse": [HookMatcher(hooks=[tracker.post_tool_use])],
        "PostToolUseFailure": [HookMatcher(hooks=[tracker.post_tool_use_failure])],
        "UserPromptSubmit": [HookMatcher(hooks=[tracker.user_prompt_submit])],
        "SubagentStart": [HookMatcher(hooks=[tracker.subagent_start])],
        "SubagentStop": [HookMatcher(hooks=[tracker.subagent_stop])],
        "PermissionRequest": [HookMatcher(hooks=[tracker.permission_request])],
        "Notification": [HookMatcher(hooks=[tracker.notification])],
        "Stop": [HookMatcher(hooks=[tracker.stop])],
    }


def create_message_handler(
    graph: ActionsGraph,
    session_id: str,
) -> Callable[[Any], None]:
    """Create a message handler for processing Claude Agent SDK messages.

    This handler can be used to process messages from the Claude Agent SDK
    and record them to the ActionsGraph.

    Args:
        graph: ActionsGraph instance for persistence
        session_id: Session ID to associate actions with

    Returns:
        A callable that processes SDK messages

    Example:
        from actions_graph import ActionsGraph
        from actions_graph.hooks import create_message_handler
        from claude_agent_sdk import query

        graph = ActionsGraph()
        handler = create_message_handler(graph, "my-session")

        async for message in query(prompt="Hello!"):
            handler(message)
    """

    def handler(message: Any) -> None:
        """Process a Claude Agent SDK message and record to graph."""
        # Handle different message types
        message_type = getattr(message, "type", None) or type(message).__name__

        if message_type == "AssistantMessage" or hasattr(message, "content"):
            # Assistant message
            content_blocks = getattr(message, "content", [])
            model = getattr(message, "model", None)
            usage = getattr(message, "usage", None)
            message_id = getattr(message, "message_id", None)

            # Extract text content
            text_content = []
            for block in content_blocks:
                if hasattr(block, "text"):
                    text_content.append({"type": "text", "text": block.text})
                elif hasattr(block, "name"):
                    # Tool use block - already handled by hooks
                    pass
                elif hasattr(block, "thinking"):
                    text_content.append({"type": "thinking", "thinking": block.thinking})

            if text_content:
                action = Message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=text_content,
                    model=model,
                    usage=dict(usage) if usage else None,
                    message_id=message_id,
                )
                graph.record_action(action)

        elif message_type == "ResultMessage" or hasattr(message, "subtype"):
            # Result message - update session
            subtype = getattr(message, "subtype", "")
            total_cost = getattr(message, "total_cost_usd", None)
            usage = getattr(message, "usage", {})

            status = ActionStatus.COMPLETED
            if subtype.startswith("error"):
                status = ActionStatus.FAILED

            graph.end_session(
                session_id,
                status=status,
                total_cost_usd=total_cost,
                total_input_tokens=usage.get("input_tokens") if usage else None,
                total_output_tokens=usage.get("output_tokens") if usage else None,
            )

        elif message_type == "RateLimitEvent" or hasattr(message, "rate_limit_info"):
            # Rate limit event
            info = getattr(message, "rate_limit_info", message)
            action = RateLimitEvent(
                session_id=session_id,
                rate_limit_status=getattr(info, "status", ""),
                rate_limit_type=getattr(info, "rate_limit_type", None),
                resets_at=getattr(info, "resets_at", None),
                utilization=getattr(info, "utilization", None),
            )
            graph.record_action(action)

    return handler
