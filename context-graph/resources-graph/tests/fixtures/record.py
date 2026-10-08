"""Re-record ``github.json`` and ``github_rest.json``: real GraphQL and REST responses the Sweep tests replay.

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

from resources_graph.address import Address, parse_address
from resources_graph.github import GitHubSource, UnresolvedError, http_transport
from resources_graph.rest import RestSource, http_rest_transport

from .replay import PAGE_SIZE, RECORDED, RECORDED_REST, fixture_key

ITEMS = [
    Address("item", "memgraph", "memgraph", 2000),  # an issue with 4 comments
    Address("item", "memgraph", "memgraph", 3600),  # a PR with 3 changed files
    Address("item", "memgraph", "memgraph", 4962),  # a PR with review threads
    Address("item", "memgraph", "memgraph", 999999),  # does not exist
    Address("item", "memgraph", "memgraph", 4933),  # a PR that closes #4887
    Address("item", "memgraph", "memgraph", 4887),  # cross-referenced by #4933
]
REST_ITEMS = [
    Address("item", "memgraph", "memgraph", 2000),
    Address("item", "memgraph", "memgraph", 4962),
    Address("item", "memgraph", "memgraph", 999999),
]
REPOSITORIES = [Address("repo", "memgraph", "memgraph"), Address("repo", "memgraph", "gqlalchemy")]
# memgraph/gqlalchemy: small enough to record whole (36 open issues when recorded), still two pages of 25.
LISTINGS = [
    "gh issue list -R memgraph/gqlalchemy --limit 100",  # every open issue: fully expanded
    "gh issue list -R memgraph/gqlalchemy --limit 10",  # truncated
    'gh issue list -R memgraph/gqlalchemy --search "graph" --limit 5',  # free text
]


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
            source.item_freshness(address)
    for address in REPOSITORIES:
        source.fetch_repository(address)
    for command in LISTINGS:
        listing = parse_address(command)
        assert listing is not None, command
        source.fetch_listing(listing)
        source.listing_index(listing)
    Path(RECORDED).write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {len(recorded)} responses to {RECORDED}")

    # REST is recorded with the token only to spare the 60/hour budget; the bodies are the same without it.
    rest_live = http_rest_transport(os.environ["GITHUB_TOKEN"])
    rest_recorded: dict[str, list] = {}

    def rest_recording(path: str, accept: str) -> tuple[int, str]:
        status, body = rest_live(path, accept)
        rest_recorded[f"{path}|{accept}"] = [status, body]
        return status, body

    rest = RestSource(rest_recording)
    for address in REST_ITEMS:
        with contextlib.suppress(UnresolvedError):
            rest.fetch_item(address)
            rest.item_freshness(address)
    rest.fetch_repository(Address("repo", "memgraph", "memgraph"))
    Path(RECORDED_REST).write_text(json.dumps(rest_recorded, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {len(rest_recorded)} responses to {RECORDED_REST}")


if __name__ == "__main__":
    main()
