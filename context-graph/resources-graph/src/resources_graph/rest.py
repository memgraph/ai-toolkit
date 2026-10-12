"""The Sweep's source when there are no credentials at all: GitHub's REST API, single items and repositories.

GitHub's GraphQL API needs a token even for public data; REST doesn't, but
allows 60 requests an hour per IP. An issue costs 3 or 4 requests here (the
issue, its comments, its timeline, its repository once), a pull request 6 or 7,
so this covers "someone pasted a link" with zero setup, not bulk reads.
Listings need credentials: they stay Unresolved (``rate_limited``) until
``gh auth login`` or ``github.token`` provides them.

Answers come back in the same shape :class:`~resources_graph.github.GitHubSource`
returns, so the store doesn't care which source fetched; REST ``node_id`` values
are GraphQL's ids.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .github import UnresolvedError

if TYPE_CHECKING:
    from .address import Address

API = "https://api.github.com"
_PER_PAGE = 100
_NEEDS_CREDENTIALS = "listings need credentials: log in with `gh auth login`, or set github.token"

#: ``(path, accept) -> (status, body text)``; swapped for recorded responses in tests.
RestTransport = Callable[[str, str], tuple[int, str]]


def http_rest_transport(token: str | None = None, *, timeout: float = 30.0) -> RestTransport:
    """GET from GitHub's REST API, unauthenticated unless ``token`` is given (recording fixtures uses one)."""

    def get(path: str, accept: str) -> tuple[int, str]:
        headers = {"Accept": accept, "User-Agent": "memgraph-resources-graph", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"bearer {token}"
        request = urllib.request.Request(API + path, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # follows a transferred issue's redirect
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")

    return get


class RestSource:
    """Single Issues, PullRequests and Repositories over REST, public only."""

    def __init__(self, transport: RestTransport) -> None:
        self._transport = transport
        self._repositories: dict[str, dict[str, Any]] = {}

    def fetch_item(self, address: Address) -> dict[str, Any]:
        """An Issue or PullRequest with every comment, cross-reference and (a PR) changed file and review comment.

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
        """
        issue, repository = self._issue(address)
        base = f"/repos/{repository['nameWithOwner']}"
        number = issue["number"]
        comments = [_comment(c) for c in self._pages(f"{base}/issues/{number}/comments")]
        sources = [
            {"source": {"id": event["source"]["issue"]["node_id"], "url": event["source"]["issue"]["html_url"]}}
            for event in self._pages(f"{base}/issues/{number}/timeline")
            if event.get("event") == "cross-referenced" and (event.get("source") or {}).get("issue")
        ]
        item: dict[str, Any] = {
            "__typename": "PullRequest" if issue.get("pull_request") else "Issue",
            "id": issue["node_id"],
            "number": number,
            "url": issue["html_url"],
            "title": issue["title"],
            "body": issue.get("body") or "",
            "state": issue["state"].upper(),
            "createdAt": issue["created_at"],
            "updatedAt": issue["updated_at"],
            "closedAt": issue.get("closed_at"),
            "author": {"login": (issue.get("user") or {}).get("login")},
            "repository": repository,
            "labels": {"nodes": [{"name": label["name"]} for label in issue.get("labels") or []]},
            "assignees": {"nodes": [{"login": user["login"]} for user in issue.get("assignees") or []]},
            "milestone": {"title": issue["milestone"]["title"]} if issue.get("milestone") else None,
            "comments": _connection(comments, total=len(comments)),
            "timelineItems": _connection(sources),
        }
        if item["__typename"] == "PullRequest":
            pull = self._json(f"{base}/pulls/{number}", address)
            files = [{"path": file["filename"]} for file in self._pages(f"{base}/pulls/{number}/files")]
            review = list(self._pages(f"{base}/pulls/{number}/comments"))
            item |= {
                "state": "MERGED" if pull.get("merged") else item["state"],
                "merged": bool(pull.get("merged")),
                "baseRefName": (pull.get("base") or {}).get("ref"),
                "closingIssuesReferences": {"nodes": []},  # not exposed by REST
                "files": _connection(files, total=len(files)),
                "reviewThreads": _connection(
                    [{"path": c["path"], "comments": _connection([_comment(c)])} for c in review]
                ),
            }
        return item

    def fetch_repository(self, address: Address) -> dict[str, Any]:
        """A Repository with its README.

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
        """
        repository = self._repository(f"{address.owner}/{address.repo}", address)
        status, readme = self._transport(
            f"/repos/{repository['nameWithOwner']}/readme", "application/vnd.github.raw+json"
        )
        _raise_for(status, address, allow_missing=True)
        return {**repository, "readme": {"text": readme} if status == 200 else None}

    def item_freshness(self, address: Address) -> dict[str, Any]:
        """An item's ``id``, ``number``, ``updatedAt`` and repository: one request, no content.

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
        """
        issue, repository = self._issue(address)
        return {
            "__typename": "PullRequest" if issue.get("pull_request") else "Issue",
            "id": issue["node_id"],
            "number": issue["number"],
            "updatedAt": issue["updated_at"],
            "repository": {"nameWithOwner": repository["nameWithOwner"], "visibility": repository["visibility"]},
        }

    def fetch_listing(self, address: Address) -> tuple[int, list[dict[str, Any]]]:
        """Not without credentials.

        Raises:
            UnresolvedError: always, ``rate_limited``.
        """
        raise UnresolvedError("rate_limited", _NEEDS_CREDENTIALS)

    def listing_index(self, address: Address) -> tuple[int, list[dict[str, Any]]]:
        """Not without credentials.

        Raises:
            UnresolvedError: always, ``rate_limited``.
        """
        raise UnresolvedError("rate_limited", _NEEDS_CREDENTIALS)

    def _issue(self, address: Address) -> tuple[dict[str, Any], dict[str, Any]]:
        issue = self._json(f"/repos/{address.owner}/{address.repo}/issues/{address.number}", address)
        # After a transfer the redirect lands elsewhere: the issue names its own repository.
        name_with_owner = issue["repository_url"].split("/repos/", 1)[1]
        return issue, self._repository(name_with_owner, address)

    def _repository(self, name_with_owner: str, address: Address) -> dict[str, Any]:
        key = name_with_owner.lower()
        if key not in self._repositories:
            repo = self._json(f"/repos/{name_with_owner}", address)
            if repo.get("private") or repo.get("visibility", "public") != "public":
                raise UnresolvedError("not_public", name_with_owner)
            self._repositories[key] = {
                "id": repo["node_id"],
                "nameWithOwner": repo["full_name"],
                "url": repo["html_url"],
                "description": repo.get("description"),
                "visibility": "PUBLIC",
                "updatedAt": repo.get("updated_at"),
                "pushedAt": repo.get("pushed_at"),
                "stargazerCount": repo.get("stargazers_count"),
            }
        return self._repositories[key]

    def _json(self, path: str, address: Address) -> Any:
        status, body = self._transport(path, "application/vnd.github+json")
        _raise_for(status, address)
        return json.loads(body)

    def _pages(self, path: str) -> Any:
        page = 1
        while True:
            status, body = self._transport(f"{path}?per_page={_PER_PAGE}&page={page}", "application/vnd.github+json")
            _raise_for(status, None)
            rows = json.loads(body)
            yield from rows
            if len(rows) < _PER_PAGE:
                return
            page += 1


def _raise_for(status: int, address: Address | None, *, allow_missing: bool = False) -> None:
    if status < 300 or (allow_missing and status == 404):
        return
    if status in (403, 429):  # the unauthenticated 60/hour budget is spent
        raise UnresolvedError("rate_limited", f"HTTP {status}")
    raise UnresolvedError("not_found", address.key if address else f"HTTP {status}")


def _connection(nodes: list[dict[str, Any]], *, total: int | None = None) -> dict[str, Any]:
    connection: dict[str, Any] = {"nodes": nodes, "pageInfo": {"hasNextPage": False, "endCursor": None}}
    if total is not None:
        connection["totalCount"] = total
    return connection


def _comment(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": comment["node_id"],
        "url": comment["html_url"],
        "body": comment.get("body") or "",
        "createdAt": comment["created_at"],
        "updatedAt": comment["updated_at"],
        "author": {"login": (comment.get("user") or {}).get("login")},
    }
