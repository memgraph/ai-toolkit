"""Replay recorded GitHub GraphQL responses (``github.json``, written by ``record.py``)."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

RECORDED = Path(__file__).with_name("github.json")
PAGE_SIZE = 2


def fixture_key(query: str, variables: dict[str, Any]) -> str:
    """One recorded response per operation name and variables."""
    operation = re.search(r"query (\w+)", query)
    return f"{operation[1] if operation else '?'}:{json.dumps(variables, sort_keys=True)}"


class Replay:
    """A transport answering from ``github.json``; ``calls`` counts what was asked."""

    def __init__(self) -> None:
        self.responses: dict[str, dict[str, Any]] = json.loads(RECORDED.read_text(encoding="utf-8"))
        self.calls = 0

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        key = fixture_key(query, variables)
        if key not in self.responses:
            raise AssertionError(f"no recorded response for {key}; re-run tests/fixtures/record.py")
        return copy.deepcopy(self.responses[key])

    def edit(self, operation_prefix: str, change) -> None:
        """Apply ``change(body)`` to every recorded response whose key starts with ``operation_prefix``."""
        for key, body in self.responses.items():
            if key.startswith(operation_prefix):
                change(body)
