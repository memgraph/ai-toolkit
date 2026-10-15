"""What colours the user's terminal can show, shared by setup's prompts and the viewer."""

from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Apple's Terminal gained 24-bit colour in macOS 26.
_APPLE_TERMINAL_TRUECOLOR_FROM = 26


def color_support(env: Mapping[str, str] | None = None, *, isatty: bool | None = None, macos: str | None = None) -> str:
    """The richest colour mode to use: ``"truecolor"``, ``"256"``, or ``""`` for none.

    ``COLORTERM`` is only a hint, and it travels: a shell profile that exports it reaches every terminal app, so
    Apple's Terminal, which before macOS 26 reads 24-bit escapes as separate codes (``38;2;251;110;0`` becomes
    red and green), is told it has truecolor. It is treated as 256-colour whatever ``COLORTERM`` says.

    Args:
        env: The environment to read; defaults to ``os.environ``.
        isatty: Whether output is a terminal; defaults to ``sys.stdout.isatty()``.
        macos: The macOS version, for Apple's Terminal; defaults to this machine's.
    """
    env = os.environ if env is None else env
    if not (sys.stdout.isatty() if isatty is None else isatty) or env.get("NO_COLOR") is not None:
        return ""
    if env.get("TERM_PROGRAM") == "Apple_Terminal":
        version = platform.mac_ver()[0] if macos is None else macos
        major = int(version.split(".")[0]) if version[:1].isdigit() else 0
        return "truecolor" if major >= _APPLE_TERMINAL_TRUECOLOR_FROM else "256"
    if env.get("COLORTERM") in ("truecolor", "24bit") or env.get("WT_SESSION"):
        return "truecolor"
    return "256"
