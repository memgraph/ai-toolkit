"""Re-record ``github.json``: real GraphQL responses the Sweep tests replay.

Run from ``context-graph/resources-graph`` with a token that can read public repos:

    GITHUB_TOKEN=$(gh auth token) uv run --package resources-graph python -m tests.fixtures.record

The page size is 2 so the recorded items exercise pagination.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path

from resources_graph.address import Address
from resources_graph.github import GitHubSource, UnresolvedError, http_transport

from .replay import PAGE_SIZE, RECORDED, fixture_key

ITEMS = [
    Address("item", "memgraph", "memgraph", 2000),  # an issue with 4 comments
    Address("item", "memgraph", "memgraph", 3600),  # a PR with 3 changed files
    Address("item", "memgraph", "memgraph", 4962),  # a PR with review threads
    Address("item", "memgraph", "memgraph", 999999),  # does not exist
]
REPOSITORIES = [Address("repo", "memgraph", "memgraph")]


def main() -> None:
    live = http_transport(os.environ["GITHUB_TOKEN"])
    recorded: dict[str, dict] = {}

    def recording(query: str, variables: dict) -> dict:
        body = live(query, variables)
        recorded[fixture_key(query, variables)] = copy.deepcopy(body)  # the source pages into it in place
        return body

    source = GitHubSource(recording, page_size=PAGE_SIZE)
    for address in ITEMS:
        with contextlib.suppress(UnresolvedError):  # the missing item is recorded as its not-found response
            source.fetch_item(address)
    for address in REPOSITORIES:
        source.fetch_repository(address)
    Path(RECORDED).write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {len(recorded)} responses to {RECORDED}")


if __name__ == "__main__":
    main()
