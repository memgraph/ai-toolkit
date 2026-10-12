"""Passages: the unit a message is embedded and shown in.

A message is one row in the graph, but a long one is many facts: an
assistant reply of 4,000 characters holds the fact a question needs anywhere
in it. Embedding it whole lets the embedder read only its opening (bge-small
stops at 512 tokens), and showing it whole costs every question that payload.
So a message is split into passages, each is embedded, and recall shows the
passages that matched rather than the message's opening.

Passages are exact consecutive substrings of the message, cut after a blank
line, a line end or a sentence end, packed up to :data:`PASSAGE_CHARS`. They
are recomputed from ``Action.text`` whenever needed rather than stored, so
the split can only drift from the vectors when this module changes -- and
:data:`PASSAGE_SCHEME` names the split, so a change re-embeds.
"""

from __future__ import annotations

import re

#: Upper bound on a passage's length: well inside bge-small's 512-token window,
#: and small enough that a matched passage is cheap to show.
PASSAGE_CHARS = 1200

#: Names the split, recorded with a session's vectors: change it whenever the
#: split changes, so vectors made for the old split are found stale.
PASSAGE_SCHEME = f"passages-{PASSAGE_CHARS}"

# Each unit ends after a blank line, a line end, or a sentence end and its spaces.
_UNIT = re.compile(r".*?(?:\n\s*\n|\n|[.!?](?:\s+|$)|$)", re.DOTALL)
_WORD = re.compile(r"\w+")


def split_passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """``text`` as consecutive passages of at most ``size`` characters; ``"".join`` gives ``text`` back.

    Raises:
        ValueError: ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f"passage size must be positive, got {size}")
    units: list[str] = []
    for match in _UNIT.finditer(text):
        unit = match.group(0)
        # A unit longer than a passage (code, a list without full stops) is cut hard.
        units.extend(unit[start : start + size] for start in range(0, len(unit), size))

    passages: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > size:
            passages.append(current)
            current = ""
        current += unit
    if current:
        passages.append(current)
    return passages or [text]


def words(text: str) -> set[str]:
    """``text``'s lowercased word tokens."""
    return {word.lower() for word in _WORD.findall(text)}


def best_passage(passages: list[str], hint: str, *, ignore: frozenset[str] = frozenset()) -> int:
    """The index of the passage sharing the most words with ``hint``; the first on a tie or no overlap.

    A passage containing ``hint`` verbatim (whitespace aside) wins outright:
    that is the sentence a fact was read from.
    """
    collapsed = " ".join(hint.split())
    if collapsed:
        for index, passage in enumerate(passages):
            if collapsed in " ".join(passage.split()):
                return index
    wanted = words(hint) - ignore
    overlaps = [len(wanted & words(passage)) for passage in passages]
    return max(range(len(passages)), key=lambda index: (overlaps[index], -index))


def excerpt(text: str, hits: list[int], chars: int) -> str:
    """What of ``text`` to show: all of it when it fits in ``chars``, else its hit passages.

    ``hits`` are passage indices, most relevant first; they are added while
    they fit and shown in text order, with ``…`` wherever text was left out.
    The first hit is always shown; it is cut to ``chars`` only when ``chars``
    is below :data:`PASSAGE_CHARS`, so keep the budget at least that wide.
    """
    if len(text) <= chars:
        return text
    passages = split_passages(text)
    chosen: list[int] = []
    used = 0
    for index in dict.fromkeys(hits or [0]):
        if not 0 <= index < len(passages):
            continue
        length = len(passages[index].strip())
        if chosen and used + length > chars:
            continue
        chosen.append(index)
        used += length
    if not chosen:
        chosen = [0]
    shown = []
    previous = -1
    for index in sorted(chosen):
        if index != previous + 1:
            shown.append("…")
        shown.append(passages[index].strip()[:chars])
        previous = index
    if previous != len(passages) - 1:
        shown.append("…")
    return " ".join(shown)
