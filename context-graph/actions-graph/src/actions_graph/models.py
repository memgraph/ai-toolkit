"""Data models for tracking LLM actions and sessions in Memgraph.

This module defines the core data structures for representing:
- Sessions: LLM conversation sessions with metadata
- Actions: Base class for all trackable LLM actions
- ToolCalls: Invocations of tools by the LLM
- ToolResults: Results returned from tool executions
- Messages: User and assistant messages in the conversation
- StructuredOutputs: Validated structured outputs from the LLM

These models are designed to be compatible with the Claude Agent SDK
and can be extended for other LLM frameworks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ActionValidationError(ValueError):
    """Raised when an action field violates validation rules."""


# Validation patterns
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_ACTION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    """Generate a unique action ID."""
    return str(uuid4())


def validate_session_id(session_id: str) -> str:
    """Validate session ID format."""
    if not session_id or not _SESSION_ID_RE.match(session_id):
        raise ActionValidationError(f"session_id must match pattern {_SESSION_ID_RE.pattern}, got: {session_id!r}")
    return session_id


def validate_action_id(action_id: str) -> str:
    """Validate action ID format."""
    if not action_id or not _ACTION_ID_RE.match(action_id):
        raise ActionValidationError(f"action_id must match pattern {_ACTION_ID_RE.pattern}, got: {action_id!r}")
    return action_id


class ActionType(str, Enum):
    """Types of actions that can be tracked."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    STRUCTURED_OUTPUT = "structured_output"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"
    PERMISSION_REQUEST = "permission_request"
    RATE_LIMIT = "rate_limit"


class ActionStatus(str, Enum):
    """Status of an action."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    DENIED = "denied"


class MessageRole(str, Enum):
    """Role of a message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Session:
    """Represents an LLM conversation session.

    A session contains multiple actions and maintains state
    across the conversation.

    Attributes:
        session_id: Unique identifier for the session
        started_at: ISO timestamp when session started
        ended_at: ISO timestamp when session ended (if completed)
        status: Current status of the session
        model: LLM model used in the session
        total_cost_usd: Estimated total cost in USD
        total_input_tokens: Total input tokens consumed
        total_output_tokens: Total output tokens generated
        working_directory: Working directory for the session
        git_branch: Git branch at start of session
        tags: Optional tags for categorization
        metadata: Additional session metadata
        parent_session_id: ID of parent session if forked
    """

    session_id: str
    started_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    status: ActionStatus = ActionStatus.IN_PROGRESS
    model: str | None = None
    total_cost_usd: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    working_directory: str | None = None
    git_branch: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_session_id: str | None = None

    def __post_init__(self):
        validate_session_id(self.session_id)
        if self.parent_session_id:
            validate_session_id(self.parent_session_id)


@dataclass
class Action:
    """Base class for all trackable LLM actions.

    Actions represent discrete events in an LLM session, such as
    tool calls, messages, or structured outputs.

    Attributes:
        action_id: Unique identifier for the action
        session_id: ID of the session this action belongs to
        action_type: Type of the action
        timestamp: ISO timestamp when action occurred
        status: Current status of the action
        duration_ms: Duration in milliseconds (if applicable)
        parent_action_id: ID of parent action (for nested actions)
        metadata: Additional action metadata
    """

    action_id: str = field(default_factory=_generate_id)
    session_id: str = ""
    action_type: ActionType = ActionType.TOOL_CALL
    timestamp: str = field(default_factory=_utc_now)
    status: ActionStatus = ActionStatus.COMPLETED
    duration_ms: int | None = None
    parent_action_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_action_id(self.action_id)
        if self.session_id:
            validate_session_id(self.session_id)
        if self.parent_action_id:
            validate_action_id(self.parent_action_id)


@dataclass
class ToolCall(Action):
    """Represents a tool/function call made by the LLM.

    Attributes:
        tool_name: Name of the tool being called
        tool_input: Input parameters passed to the tool
        tool_use_id: Unique ID for this tool use (for correlating results)
        is_mcp: Whether this is an MCP (Model Context Protocol) tool
        mcp_server: MCP server name if applicable
    """

    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str | None = None
    is_mcp: bool = False
    mcp_server: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.TOOL_CALL
        if self.tool_name.startswith("mcp__"):
            self.is_mcp = True
            parts = self.tool_name.split("__")
            if len(parts) >= 2:
                self.mcp_server = parts[1]


@dataclass
class ToolResult(Action):
    """Represents the result of a tool execution.

    Attributes:
        tool_use_id: ID of the tool call this result corresponds to
        tool_name: Name of the tool that was executed
        content: Result content (text or structured)
        is_error: Whether the tool execution resulted in an error
        error_message: Error message if is_error is True
    """

    tool_use_id: str = ""
    tool_name: str = ""
    content: str | list[dict[str, Any]] | None = None
    is_error: bool = False
    error_message: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.TOOL_RESULT
        if self.is_error:
            self.status = ActionStatus.FAILED


@dataclass
class Message(Action):
    """Represents a message in the conversation.

    Attributes:
        role: Role of the message sender
        content: Message content (text or content blocks)
        message_id: API message ID (if available)
        model: Model that generated the message (for assistant messages)
        usage: Token usage for this message
    """

    role: MessageRole = MessageRole.USER
    content: str | list[dict[str, Any]] = ""
    message_id: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.role == MessageRole.USER:
            self.action_type = ActionType.USER_MESSAGE
        elif self.role == MessageRole.ASSISTANT:
            self.action_type = ActionType.ASSISTANT_MESSAGE
        else:
            self.action_type = ActionType.SYSTEM_MESSAGE


@dataclass
class StructuredOutput(Action):
    """Represents a validated structured output from the LLM.

    Attributes:
        output_type: Type/schema name of the structured output
        output_data: The structured output data
        schema: JSON schema used for validation (if any)
        validation_passed: Whether the output passed validation
    """

    output_type: str = ""
    output_data: Any = None
    schema: dict[str, Any] | None = None
    validation_passed: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.STRUCTURED_OUTPUT


@dataclass
@dataclass
class SubagentEvent(Action):
    """Represents a subagent lifecycle event.

    Attributes:
        agent_id: Unique identifier for the subagent
        agent_type: Type of the subagent
        description: Description of the subagent task
        result: Result from subagent (if completed)
        usage: Token usage for the subagent

    Note:
        The action_type should be set by the caller to either
        SUBAGENT_START or SUBAGENT_STOP after construction.
    """

    agent_id: str = ""
    agent_type: str = ""
    description: str = ""
    result: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class PermissionRequest(Action):
    """Represents a permission request event.

    Attributes:
        tool_name: Tool requesting permission
        tool_input: Input for the tool
        decision: Permission decision (allow/deny/ask)
        reason: Reason for the decision
    """

    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    reason: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.PERMISSION_REQUEST


@dataclass
class ErrorEvent(Action):
    """Represents an error that occurred during execution.

    Attributes:
        error_type: Type/classification of the error
        error_message: Human-readable error message
        error_details: Additional error details
        recoverable: Whether the error is recoverable
    """

    error_type: str = ""
    error_message: str = ""
    error_details: dict[str, Any] | None = None
    recoverable: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.ERROR
        self.status = ActionStatus.FAILED


@dataclass
class RateLimitEvent(Action):
    """Represents a rate limit event.

    Attributes:
        rate_limit_status: Status (allowed, allowed_warning, rejected)
        rate_limit_type: Type of rate limit
        resets_at: Unix timestamp when limit resets
        utilization: Fraction of limit consumed (0.0 to 1.0)
    """

    rate_limit_status: str = ""
    rate_limit_type: str | None = None
    resets_at: int | None = None
    utilization: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.action_type = ActionType.RATE_LIMIT
