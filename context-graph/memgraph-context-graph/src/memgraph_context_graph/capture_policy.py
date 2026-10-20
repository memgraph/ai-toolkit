"""Capture Policy: what a captured event may carry before any connector stores it.

Connectors persist whatever an event holds, including full tool input and
output, so anything sensitive must be removed here, once, before
:meth:`AgentLink.emit` hands the event on. The policy:

- strips ``<private>…</private>`` blocks (an unclosed ``<private>`` hides the rest
  of the text) and drops a message that was nothing but private;
- replaces known credential shapes, any ``[redact]`` pattern, and the secrets in
  the config file itself with ``[REDACTED:<name>]``;
- clips any one string to ``max_chars``;
- drops tool events for ``[capture] exclude_tools``.

``[capture] exclude_projects`` is not an event rule: most events carry no
working directory, so the hook runner checks the payload's and captures nothing
for an excluded project.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .events import MessageEvent, ToolEndEvent, ToolStartEvent

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .adapters._identity import HookConfig
    from .events import Event

#: Longest string a captured field keeps; tool output past this is rarely worth its storage.
DEFAULT_MAX_CHARS = 100_000

#: Applied in order: the Anthropic shape must replace before the broader OpenAI one sees ``sk-ant-``.
BUILTIN_PATTERNS: dict[str, str] = {
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    "aws_access_key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})",
    "anthropic_key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "openai_key": r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}",
    "slack_token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    "stripe_key": r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    "url_password": r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)",
}

#: Config-file secrets shorter than this are left alone: replacing "a" or "admin" everywhere would shred the text.
_MIN_SECRET_LENGTH = 6

_PRIVATE_BLOCK = re.compile(r"<private>.*?(?:</private>|\Z)", re.DOTALL | re.IGNORECASE)

#: Identifiers connectors join on; never rewritten, so redaction can't split one session or tool call into two.
_IDENTITY_FIELDS = frozenset(
    {"event_type", "session_id", "timestamp", "source_sdk", "tool_use_id", "tool_name", "user_id", "model"}
)


@dataclass(frozen=True)
class CapturePolicy:
    """Rewrites or drops an event before connectors see it.

    The default instance redacts the built-in patterns and clips at
    :data:`DEFAULT_MAX_CHARS`; :func:`policy_from_config` adds what the config
    file sets.
    """

    redact: bool = True
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = field(
        default_factory=lambda: tuple((name, re.compile(regex)) for name, regex in BUILTIN_PATTERNS.items())
    )
    #: Literal values (the config file's own credentials) replaced wherever they appear.
    secrets: tuple[str, ...] = ()
    max_chars: int | None = DEFAULT_MAX_CHARS
    exclude_tools: tuple[str, ...] = ()
    exclude_projects: tuple[str, ...] = ()

    def apply(self, event: Event) -> Event | None:
        """The event as connectors may store it, or None when it must not be captured at all."""
        if isinstance(event, (ToolStartEvent, ToolEndEvent)) and self.excludes_tool(event.tool_name):
            return None
        changes = {
            f.name: self._clean(value)
            for f in dataclasses.fields(event)
            if f.init and f.name not in _IDENTITY_FIELDS and (value := getattr(event, f.name)) is not None
        }
        cleaned = dataclasses.replace(event, **changes)
        if (
            isinstance(event, MessageEvent)
            and isinstance(cleaned, MessageEvent)
            and _was_only_private(event.content, cleaned.content)
        ):
            return None
        return cleaned

    def excludes_tool(self, tool_name: str) -> bool:
        """Whether ``[capture] exclude_tools`` names *tool_name*, exactly or by glob (``mcp__*``)."""
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.exclude_tools)

    def excludes_project(self, cwd: str | None) -> bool:
        """Whether *cwd* is, or is inside, a ``[capture] exclude_projects`` path or glob (``~`` expands)."""
        if not cwd:
            return False
        path = os.path.normpath(os.path.expanduser(cwd))
        for pattern in self.exclude_projects:
            root = os.path.normpath(os.path.expanduser(pattern))
            if fnmatch.fnmatchcase(path, root) or fnmatch.fnmatchcase(path, os.path.join(root, "*")):
                return True
        return False

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clean_text(value)
        if isinstance(value, dict):
            return {key: self._clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._clean(item) for item in value)
        return value

    def _clean_text(self, text: str) -> str:
        text = _PRIVATE_BLOCK.sub("", text)
        if self.redact:
            # Longest first, so a secret containing another is replaced whole.
            for secret in sorted(self.secrets, key=len, reverse=True):
                text = text.replace(secret, "[REDACTED:config_secret]")
            for name, pattern in self.patterns:
                text = pattern.sub(f"[REDACTED:{name}]", text)
        if self.max_chars is not None and len(text) > self.max_chars:
            text = f"{text[: self.max_chars]}…[truncated {len(text) - self.max_chars} chars]"
        return text


def _was_only_private(before: Any, after: Any) -> bool:
    return isinstance(before, str) and bool(_PRIVATE_BLOCK.search(before)) and not str(after).strip()


def policy_from_config(config: HookConfig) -> tuple[CapturePolicy, list[str]]:
    """The policy the config file's ``[capture]`` and ``[redact]`` sections describe, and what was wrong with them.

    A bad value never fails a hook: it falls back to the default and is
    reported, as :attr:`HookConfig.problems` is.
    """
    from .adapters._identity import FALSY_VALUES, _parse_list

    settings = config.capture_settings
    problems: list[str] = []
    redact = settings.get("redact", "").strip().lower() not in FALSY_VALUES
    patterns = list(CapturePolicy().patterns)
    for name, regex in config.redact_patterns.items():
        try:
            patterns.append((name, re.compile(regex)))
        except re.error as exc:
            problems.append(f"[redact] {name} is not a valid regular expression ({exc}) and was ignored")
    return (
        CapturePolicy(
            redact=redact,
            patterns=tuple(patterns),
            secrets=_config_secrets(config),
            max_chars=_max_chars(settings.get("max_chars"), problems),
            exclude_tools=_parse_list(settings.get("exclude_tools", "")),
            exclude_projects=_parse_list(settings.get("exclude_projects", "")),
        ),
        problems,
    )


def _max_chars(raw: str | None, problems: list[str]) -> int | None:
    """``[capture] max_chars``: a positive int, or 0 for no limit."""
    if raw is None or not raw.strip():
        return DEFAULT_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        problems.append(f"[capture] max_chars = {raw!r} is not a non-negative integer; using {DEFAULT_MAX_CHARS}")
        return DEFAULT_MAX_CHARS
    return value or None


def _config_secrets(config: HookConfig) -> tuple[str, ...]:
    values: Iterable[str | None] = (
        config.memgraph_password,
        config.openai_api_key,
        config.anthropic_api_key,
        config.github_token,
    )
    return tuple(value for value in values if value and len(value) >= _MIN_SECRET_LENGTH)
