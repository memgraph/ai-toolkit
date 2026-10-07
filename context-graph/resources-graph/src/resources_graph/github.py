"""The Sweep's GitHub source: canonical, public-only content over GraphQL.

Only the Sweep calls this — never a hook, never the ``resource`` tool. The
token comes from the config file (``github.token``); GitHub's GraphQL API
needs one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .address import Address

GRAPHQL_URL = "https://api.github.com/graphql"

#: ``(query, variables) -> response body``; swapped for recorded responses in tests.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

_COMMENT = "id url body createdAt updatedAt author { login }"
_PAGE = "pageInfo { hasNextPage endCursor }"
_ITEM_FIELDS = f"""
  __typename id number url title body state createdAt updatedAt closedAt
  author {{ login }}
  repository {{ id nameWithOwner visibility }}
  labels(first: 100) {{ nodes {{ name }} }}
  assignees(first: 100) {{ nodes {{ login }} }}
  milestone {{ title }}
  comments(first: $pageSize) {{ totalCount {_PAGE} nodes {{ {_COMMENT} }} }}
"""
# Threads are paged; a single review thread's comments come in one page of 100.
_PULL_FIELDS = f"""
  merged baseRefName
  files(first: $pageSize) {{ totalCount {_PAGE} nodes {{ path }} }}
  reviewThreads(first: $pageSize) {{ {_PAGE} nodes {{ path comments(first: 100) {{ {_PAGE} nodes {{ {_COMMENT} }} }} }} }}
"""
# Resolved by number, never by URL: resource(url: ".../issues/N") reports a pull
# request as __typename Issue (verified while prototyping the graph model, #460).
ITEM_QUERY = f"""query Item($owner: String!, $repo: String!, $number: Int!, $pageSize: Int!) {{
  repository(owner: $owner, name: $repo) {{
    visibility
    issueOrPullRequest(number: $number) {{
      ... on Issue {{ {_ITEM_FIELDS} }}
      ... on PullRequest {{ {_ITEM_FIELDS} {_PULL_FIELDS} }}
    }}
  }}
}}"""
REPOSITORY_QUERY = """query Repository($owner: String!, $repo: String!) {
  repository(owner: $owner, name: $repo) {
    id nameWithOwner url description visibility updatedAt pushedAt stargazerCount
    readme: object(expression: "HEAD:README.md") { ... on Blob { text } }
  }
}"""
_MORE = {
    "comments": f"comments(first: $pageSize, after: $after) {{ {_PAGE} nodes {{ {_COMMENT} }} }}",
    "files": f"files(first: $pageSize, after: $after) {{ {_PAGE} nodes {{ path }} }}",
    "reviewThreads": (
        f"reviewThreads(first: $pageSize, after: $after) {{ {_PAGE} "
        f"nodes {{ path comments(first: 100) {{ {_PAGE} nodes {{ {_COMMENT} }} }} }} }}"
    ),
}


def more_query(connection: str) -> str:
    """The query for the next page of one of an item's connections."""
    on_issue = f"... on Issue {{ {_MORE[connection]} }}" if connection == "comments" else ""
    operation = "More" + connection[0].upper() + connection[1:]
    return f"""query {operation}($id: ID!, $after: String!, $pageSize: Int!) {{
  node(id: $id) {{ {on_issue} ... on PullRequest {{ {_MORE[connection]} }} }}
}}"""


class UnresolvedError(Exception):
    """An Address the Sweep couldn't turn into a Resource.

    ``reason`` is ``not_found``, ``not_public`` or ``rate_limited``.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class GitHubAuthError(Exception):
    """The configured token was rejected; no Touch can be resolved until it is fixed."""


def http_transport(token: str, *, timeout: float = 30.0) -> Transport:
    """A transport that POSTs to GitHub's GraphQL endpoint with ``token``."""

    def post(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            GRAPHQL_URL,
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Authorization": f"bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "memgraph-resources-graph",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise GitHubAuthError("GitHub rejected the configured github.token") from exc
            if exc.code in (403, 429):
                raise UnresolvedError("rate_limited", f"HTTP {exc.code}") from exc
            raise

    return post


class GitHubSource:
    """Fetches public Repositories, Issues and PullRequests at full capture depth."""

    def __init__(self, transport: Transport, *, page_size: int = 100) -> None:
        self._transport = transport
        self._page_size = page_size

    def fetch_item(self, address: Address) -> dict[str, Any]:
        """An Issue or PullRequest with every comment (and, for a PR, review comments and changed files).

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
            GitHubAuthError: the token was rejected.
        """
        data = self._call(
            ITEM_QUERY,
            {"owner": address.owner, "repo": address.repo, "number": address.number, "pageSize": self._page_size},
        )
        repository = data.get("repository")
        item = (repository or {}).get("issueOrPullRequest")
        if not item:
            raise UnresolvedError("not_found", address.key)
        _require_public(item["repository"])
        self._complete(item, "comments")
        if item["__typename"] == "PullRequest":
            self._complete(item, "files")
            self._complete(item, "reviewThreads")
        return item

    def fetch_repository(self, address: Address) -> dict[str, Any]:
        """A Repository with its README.

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
            GitHubAuthError: the token was rejected.
        """
        repository = self._call(REPOSITORY_QUERY, {"owner": address.owner, "repo": address.repo}).get("repository")
        if not repository:
            raise UnresolvedError("not_found", address.key)
        _require_public(repository)
        return repository

    def _complete(self, item: dict[str, Any], connection: str) -> None:
        """Page ``item[connection]`` to the end, so nothing is silently truncated."""
        page = item[connection]
        while page["pageInfo"]["hasNextPage"]:
            node = self._call(
                more_query(connection),
                {"id": item["id"], "after": page["pageInfo"]["endCursor"], "pageSize": self._page_size},
            )["node"]
            more = node[connection]
            page["nodes"].extend(more["nodes"])
            page["pageInfo"] = more["pageInfo"]

    def _call(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        body = self._transport(query, variables)
        errors = body.get("errors") or []
        kinds = {error.get("type") for error in errors}
        if "RATE_LIMITED" in kinds:
            raise UnresolvedError("rate_limited", errors[0].get("message", ""))
        if kinds & {"NOT_FOUND", "FORBIDDEN"}:
            raise UnresolvedError("not_found", errors[0].get("message", ""))
        if errors and not body.get("data"):
            raise RuntimeError(f"GitHub GraphQL error: {errors}")
        return body.get("data") or {}


def _require_public(repository: dict[str, Any]) -> None:
    """Only public content enters shared memory, whatever the token can see."""
    if repository.get("visibility") != "PUBLIC":
        raise UnresolvedError("not_public", repository.get("nameWithOwner", ""))
