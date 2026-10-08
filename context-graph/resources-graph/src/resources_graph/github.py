"""The Sweep's GitHub source: canonical, public-only content over GraphQL.

Only the Sweep calls this — never a hook, never the ``resource`` tool.
GitHub's GraphQL API needs a token even for public data; see :func:`resolve_token`.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .address import Address

GRAPHQL_URL = "https://api.github.com/graphql"

#: ``(query, variables) -> response body``; swapped for recorded responses in tests.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

_COMMENT = "id url body createdAt updatedAt author { login }"
_PAGE = "pageInfo { hasNextPage endCursor }"
# Who mentioned this item: each cross-reference's source is the Issue or PR that referenced it.
_CROSS_REFERENCES = (
    "nodes { ... on CrossReferencedEvent { source { ... on Issue { id url } ... on PullRequest { id url } } } }"
)
_ITEM_FIELDS = f"""
  __typename id number url title body state createdAt updatedAt closedAt
  author {{ login }}
  repository {{ id nameWithOwner visibility }}
  labels(first: 100) {{ nodes {{ name }} }}
  assignees(first: 100) {{ nodes {{ login }} }}
  milestone {{ title }}
  comments(first: $pageSize) {{ totalCount {_PAGE} nodes {{ {_COMMENT} }} }}
  timelineItems(first: $pageSize, itemTypes: [CROSS_REFERENCED_EVENT]) {{ {_PAGE} {_CROSS_REFERENCES} }}
"""
# Threads are paged; a single review thread's comments come in one page of 100.
_PULL_FIELDS = f"""
  merged baseRefName
  closingIssuesReferences(first: 100) {{ nodes {{ id url }} }}
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
# Newest created first: the order gh issue|pr list shows, so a truncated
# Listing holds the same members the agent saw.
_NEWEST = "orderBy: {field: CREATED_AT, direction: DESC}"


def _listing_queries(name: str, issue_fields: str, pull_fields: str, page_size: str) -> tuple[str, str, str]:
    """The issues, pulls and search queries for one member selection, sharing filters and order.

    ``filterBy`` is a variable holding only the filters set: an explicit
    ``assignee: null`` means "unassigned issues only" to GitHub, not "any assignee".
    """
    issues = f"""query {name}Issues($owner: String!, $repo: String!, $first: Int!, $after: String{page_size},
                   $states: [IssueState!], $labels: [String!], $filterBy: IssueFilters) {{
  repository(owner: $owner, name: $repo) {{
    visibility nameWithOwner
    members: issues(first: $first, after: $after, states: $states, labels: $labels,
                    filterBy: $filterBy, {_NEWEST}) {{ totalCount {_PAGE} nodes {{ {issue_fields} }} }}
  }}
}}"""
    pulls = f"""query {name}Pulls($owner: String!, $repo: String!, $first: Int!, $after: String{page_size},
                  $states: [PullRequestState!], $labels: [String!]) {{
  repository(owner: $owner, name: $repo) {{
    visibility nameWithOwner
    members: pullRequests(first: $first, after: $after, states: $states, labels: $labels, {_NEWEST}) {{
      totalCount {_PAGE} nodes {{ {pull_fields} }} }}
  }}
}}"""
    search = f"""query {name}Search($q: String!, $first: Int!, $after: String{page_size}) {{
  members: search(query: $q, type: ISSUE, first: $first, after: $after) {{
    totalCount: issueCount {_PAGE} nodes {{ ... on Issue {{ {issue_fields} }} ... on PullRequest {{ {pull_fields} }} }}
  }}
}}"""
    return issues, pulls, search


ISSUES_QUERY, PULLS_QUERY, SEARCH_QUERY = _listing_queries(
    "", _ITEM_FIELDS, f"{_ITEM_FIELDS} {_PULL_FIELDS}", ", $pageSize: Int!"
)
# The light index a stored Listing is revalidated with: membership and updatedAt, no content.
_INDEX_FIELDS = "__typename id number updatedAt repository { nameWithOwner visibility }"
INDEX_QUERIES = dict(
    zip(("issues", "pulls", "search"), _listing_queries("Index", _INDEX_FIELDS, _INDEX_FIELDS, ""), strict=True)
)
FRESHNESS_QUERY = f"""query Freshness($owner: String!, $repo: String!, $number: Int!) {{
  repository(owner: $owner, name: $repo) {{
    issueOrPullRequest(number: $number) {{ ... on Issue {{ {_INDEX_FIELDS} }} ... on PullRequest {{ {_INDEX_FIELDS} }} }}
  }}
}}"""
_ISSUE_STATES = {"open": ["OPEN"], "closed": ["CLOSED"], "all": None}
_PULL_STATES = {"open": ["OPEN"], "closed": ["CLOSED", "MERGED"], "merged": ["MERGED"], "all": None}
# Members per page: pull requests carry files and review threads, so fewer fit a query's node budget.
_MEMBERS_PER_PAGE = {"issues": 25, "pulls": 10}

_MORE = {
    "comments": f"comments(first: $pageSize, after: $after) {{ {_PAGE} nodes {{ {_COMMENT} }} }}",
    "files": f"files(first: $pageSize, after: $after) {{ {_PAGE} nodes {{ path }} }}",
    "timelineItems": (
        f"timelineItems(first: $pageSize, after: $after, itemTypes: [CROSS_REFERENCED_EVENT]) {{ {_PAGE} {_CROSS_REFERENCES} }}"
    ),
    "reviewThreads": (
        f"reviewThreads(first: $pageSize, after: $after) {{ {_PAGE} "
        f"nodes {{ path comments(first: 100) {{ {_PAGE} nodes {{ {_COMMENT} }} }} }} }}"
    ),
}


def more_query(connection: str) -> str:
    """The query for the next page of one of an item's connections."""
    on_issue = f"... on Issue {{ {_MORE[connection]} }}" if connection in ("comments", "timelineItems") else ""
    operation = "More" + connection[0].upper() + connection[1:]
    return f"""query {operation}($id: ID!, $after: String!, $pageSize: Int!) {{
  node(id: $id) {{ {on_issue} ... on PullRequest {{ {_MORE[connection]} }} }}
}}"""


class Source(Protocol):
    """What the Sweep fetches with: GraphQL with credentials, or REST without (:mod:`resources_graph.rest`)."""

    def fetch_item(self, address: Address) -> dict[str, Any]: ...

    def fetch_repository(self, address: Address) -> dict[str, Any]: ...

    def fetch_listing(self, address: Address) -> tuple[int, list[dict[str, Any]]]: ...

    def item_freshness(self, address: Address) -> dict[str, Any]: ...

    def listing_index(self, address: Address) -> tuple[int, list[dict[str, Any]]]: ...


class UnresolvedError(Exception):
    """An Address the Sweep couldn't turn into a Resource.

    ``reason`` is ``not_found``, ``not_public`` or ``rate_limited``.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class GitHubAuthError(Exception):
    """The configured token was rejected; no Touch can be resolved until it is fixed."""


def resolve_token(configured: str | None) -> str | None:
    """The token the Sweep fetches with: ``github.token`` when set, else the user's own ``gh`` login.

    The ``gh`` login is the access the agent already reads GitHub with, so most
    users configure nothing. Asking ``gh`` is fine here and would not be in a
    hook: the Sweep is a command the user runs, never a hook subprocess.
    Returns None when neither is available.
    """
    if configured:
        return configured
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):  # gh not installed, or hung on a keyring prompt
        return None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


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
                raise GitHubAuthError("GitHub rejected the token (github.token, or the gh login)") from exc
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
        self._complete_item(item)
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

    def item_freshness(self, address: Address) -> dict[str, Any]:
        """An item's ``id``, ``number``, ``updatedAt`` and repository: one cheap call, no content.

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
            GitHubAuthError: the token was rejected.
        """
        data = self._call(FRESHNESS_QUERY, {"owner": address.owner, "repo": address.repo, "number": address.number})
        item = (data.get("repository") or {}).get("issueOrPullRequest")
        if not item:
            raise UnresolvedError("not_found", address.key)
        _require_public(item["repository"])
        return item

    def listing_index(self, address: Address) -> tuple[int, list[dict[str, Any]]]:
        """A Listing's total and its members' ``id``, ``number``, ``updatedAt`` and repository — no content.

        Same filters and order as :meth:`fetch_listing`, 100 members a call, so a
        stored Listing is revalidated cheaply and only changed members refetched.
        """
        return self._page_listing(address, INDEX_QUERIES, per_page=100, with_content=False)

    def fetch_listing(self, address: Address) -> tuple[int, list[dict[str, Any]]]:
        """A Listing's total size and its first ``address.limit`` members (all when None), newest first.

        Every member comes at full depth, like :meth:`fetch_item`. The repository's
        connections answer state and label filters; GitHub search answers free
        text, milestones and a pull request's author or assignee.

        Raises:
            UnresolvedError: missing, not public, or rate-limited.
            GitHubAuthError: the token was rejected.
        """
        queries = {"issues": ISSUES_QUERY, "pulls": PULLS_QUERY, "search": SEARCH_QUERY}
        per_page = _MEMBERS_PER_PAGE[address.item_kind or "issues"]
        return self._page_listing(address, queries, per_page=per_page, with_content=True)

    def _page_listing(
        self, address: Address, queries: dict[str, str], *, per_page: int, with_content: bool
    ) -> tuple[int, list[dict[str, Any]]]:
        filters = address.filter_map
        use_search = bool(
            address.query
            or filters.get("milestone")
            or (address.item_kind == "pulls" and (filters.get("author") or filters.get("assignee")))
        )
        if use_search:
            query = queries["search"]
            variables: dict[str, Any] = {"q": _search_text(address)}
        else:
            query, variables = self._connection(address, filters, queries)
        if with_content:
            variables["pageSize"] = self._page_size
        members: list[dict[str, Any]] = []
        total, after = 0, None
        while address.limit is None or len(members) < address.limit:
            wanted = per_page if address.limit is None else min(per_page, address.limit - len(members))
            data = self._call(query, {**variables, "first": wanted, "after": after})
            container = data if use_search else data.get("repository")
            if not container:
                raise UnresolvedError("not_found", address.key)
            if not use_search:
                _require_public(container)
            page = container["members"]
            total = page["totalCount"]
            for item in page["nodes"]:
                if not item:
                    continue
                _require_public(item["repository"])
                if with_content:
                    self._complete_item(item)
                members.append(item)
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = page["pageInfo"]["endCursor"]
        return total, members

    @staticmethod
    def _connection(address: Address, filters: dict[str, str], queries: dict[str, str]) -> tuple[str, dict[str, Any]]:
        labels = filters["labels"].split(",") if filters.get("labels") else None
        variables: dict[str, Any] = {"owner": address.owner, "repo": address.repo, "labels": labels}
        if address.item_kind == "pulls":
            return queries["pulls"], {**variables, "states": _PULL_STATES[filters["state"]]}
        filter_by = {
            name: filters[key] for name, key in (("createdBy", "author"), ("assignee", "assignee")) if filters.get(key)
        }
        return queries["issues"], {
            **variables,
            "states": _ISSUE_STATES[filters["state"]],
            "filterBy": filter_by or None,
        }

    def _complete_item(self, item: dict[str, Any]) -> None:
        self._complete(item, "comments")
        self._complete(item, "timelineItems")
        if item["__typename"] == "PullRequest":
            self._complete(item, "files")
            self._complete(item, "reviewThreads")

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


def _search_text(address: Address) -> str:
    """The GitHub search query for a Listing, newest created first like the connections."""
    filters = address.filter_map
    parts = [f"repo:{address.owner}/{address.repo}", "is:pr" if address.item_kind == "pulls" else "is:issue"]
    if filters["state"] != "all":
        parts.append(f"is:{filters['state']}")
    parts += [f'label:"{label}"' for label in filters["labels"].split(",")] if filters.get("labels") else []
    parts += [f'{name}:"{filters[name]}"' for name in ("author", "assignee", "milestone") if filters.get(name)]
    if address.query:
        parts.append(address.query)
    parts.append("sort:created-desc")
    return " ".join(parts)


def _require_public(repository: dict[str, Any]) -> None:
    """Only public content enters shared memory, whatever the token can see."""
    if repository.get("visibility") != "PUBLIC":
        raise UnresolvedError("not_public", repository.get("nameWithOwner", ""))
