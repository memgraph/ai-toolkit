"""Writing a user's files without changing their line endings.

``Path.write_text`` turns ``\\n`` into the platform's line ending, so on Windows every file setup or
the memory switch rewrote came back CRLF, even when its content didn't change. A rewrite here keeps
the ending the file already had, and a new file gets ``\\n`` on every platform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def write_text_keeping_newlines(path: Path, text: str) -> None:
    """Write *text* to *path* as UTF-8 with the line ending *path* already uses.

    *text* may use ``\\n`` or ``\\r\\n``; it is written with ``\\r\\n`` only if the existing file holds one,
    else with ``\\n``.
    """
    newline = "\r\n" if path.is_file() and b"\r\n" in path.read_bytes() else "\n"
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline=newline)
