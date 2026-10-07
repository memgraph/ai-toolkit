"""PROTOTYPE — throwaway. Sweep-side GitHub fetches via ``gh api graphql``.

The real Sweep would use the optional config-file token; the prototype borrows gh's.
Every fetch checks ``visibility == PUBLIC`` and reports an Unresolved reason otherwise.
"""

from __future__ import annotations

import json
import subprocess

COMMENT = "id url body createdAt updatedAt author { login }"
ITEM_COMMON = f"""
  id number url title body state createdAt updatedAt closedAt
  author {{ login }}
  repository {{ id nameWithOwner visibility }}
  labels(first: 30) {{ nodes {{ name }} }}
  assignees(first: 10) {{ nodes {{ login }} }}
  milestone {{ title }}
  comments(first: 100) {{ totalCount nodes {{ {COMMENT} }} }}
  timelineItems(first: 50, itemTypes: [CROSS_REFERENCED_EVENT]) {{
    nodes {{ ... on CrossReferencedEvent {{ source {{
      ... on Issue {{ id url }} ... on PullRequest {{ id url }} }} }} }}
  }}
"""
ISSUE = f"... on Issue {{ __typename {ITEM_COMMON} }}"
PULL = f"""... on PullRequest {{ __typename {ITEM_COMMON}
  merged baseRefName
  files(first: 100) {{ totalCount nodes {{ path }} }}
  closingIssuesReferences(first: 20) {{ nodes {{ id url }} }}
  reviewThreads(first: 50) {{ nodes {{ path comments(first: 50) {{ nodes {{ {COMMENT} }} }} }} }}
}}"""


class Unresolved(Exception):
    """Address could not become a Resource; ``reason`` is not_found | not_public | rate_limited."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def graphql(query: str, **variables) -> dict:
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    proc = subprocess.run(args, capture_output=True, text=True)
    body = json.loads(proc.stdout or "{}")
    errors = body.get("errors") or []
    if errors:
        kinds = {e.get("type") for e in errors}
        if "RATE_LIMITED" in kinds:
            raise Unresolved("rate_limited", errors[0].get("message", ""))
        if "NOT_FOUND" in kinds or "FORBIDDEN" in kinds:
            raise Unresolved("not_found", errors[0].get("message", ""))
        raise RuntimeError(errors)
    if proc.returncode and not body.get("data"):
        raise Unresolved("not_found", proc.stderr.strip())
    return body["data"]


def fetch_item(owner: str, name: str, number: int) -> dict:
    # Resolve by number, not URL: resource(url: .../issues/N) types a PR as Issue (verified for #460).
    q = f"""query($o: String!, $r: String!, $n: Int!) {{ repository(owner: $o, name: $r) {{
      visibility issueOrPullRequest(number: $n) {{ {ISSUE} {PULL} }} }} }}"""
    repo = graphql(q, o=owner, r=name, n=number).get("repository")
    node = (repo or {}).get("issueOrPullRequest")
    if not node:
        raise Unresolved("not_found", f"{owner}/{name}#{number}")
    _require_public(node["repository"])
    return node


def fetch_repo(owner: str, name: str) -> dict:
    q = """query($o: String!, $n: String!) { repository(owner: $o, name: $n) {
      id nameWithOwner url description visibility updatedAt pushedAt stargazerCount
      readme: object(expression: "HEAD:README.md") { ... on Blob { text } } } }"""
    node = graphql(q, o=owner, n=name).get("repository")
    if not node:
        raise Unresolved("not_found", f"{owner}/{name}")
    _require_public(node)
    return node


def fetch_listing(owner: str, name: str, item_kind: str, filters: dict, query: str | None, limit: int | None):
    """Yield (total_count, members) pages until ``limit`` members or the end."""
    page_size = 25
    fetched, cursor = 0, None
    while True:
        size = page_size if limit is None else min(page_size, limit - fetched)
        if size <= 0:
            return
        if query:
            q_text = f"repo:{owner}/{name} is:{'pr' if item_kind == 'pulls' else 'issue'} {query}"
            q = f"""query($q: String!, $n: Int!, $c: String) {{ search(query: $q, type: ISSUE, first: $n, after: $c) {{
              issueCount pageInfo {{ hasNextPage endCursor }} nodes {{ {ISSUE} {PULL} }} }} }}"""
            conn = graphql(q, q=q_text, n=size, c=cursor)["search"]
            total = conn["issueCount"]
        else:
            conn, total = _repo_connection(owner, name, item_kind, filters, size, cursor)
        nodes = [n for n in conn["nodes"] if n]
        for n in nodes:
            _require_public(n["repository"])
        fetched += len(nodes)
        yield total, nodes
        if not conn["pageInfo"]["hasNextPage"]:
            return
        cursor = conn["pageInfo"]["endCursor"]


def _repo_connection(owner, name, item_kind, filters, size, cursor):
    state = filters.get("state", "open")
    labels = filters["labels"].split(",") if filters.get("labels") else None
    if item_kind == "pulls":
        states = {"open": "[OPEN]", "closed": "[CLOSED, MERGED]", "merged": "[MERGED]", "all": "null"}[state]
        field = f"pullRequests(first: $n, after: $c, states: {states}, labels: $labels, orderBy: {{field: UPDATED_AT, direction: DESC}})"
        frag = PULL
    else:
        states = {"open": "[OPEN]", "closed": "[CLOSED]", "all": "null"}[state]
        fb = ", ".join(
            f'{k}: "{filters[v]}"' for k, v in (("createdBy", "author"), ("assignee", "assignee"), ("milestone", "milestone")) if filters.get(v)
        )
        field = f"issues(first: $n, after: $c, states: {states}, labels: $labels, filterBy: {{ {fb} }}, orderBy: {{field: UPDATED_AT, direction: DESC}})"
        frag = ISSUE
    q = f"""query($o: String!, $r: String!, $n: Int!, $c: String, $labels: [String!]) {{ repository(owner: $o, name: $r) {{
      visibility conn: {field} {{ totalCount pageInfo {{ hasNextPage endCursor }} nodes {{ {frag} }} }} }} }}"""
    args = ["gh", "api", "graphql", "-f", f"query={q}", "-f", f"o={owner}", "-f", f"r={name}", "-F", f"n={size}"]
    if cursor:
        args += ["-f", f"c={cursor}"]
    for label in labels or []:
        args += ["-f", f"labels[]={label}"]
    proc = subprocess.run(args, capture_output=True, text=True)
    body = json.loads(proc.stdout or "{}")
    if body.get("errors"):
        raise Unresolved("not_found", str(body["errors"][0].get("message")))
    repo = body["data"]["repository"]
    _require_public(repo)
    return repo["conn"], repo["conn"]["totalCount"]


def _require_public(repo: dict) -> None:
    if repo.get("visibility") != "PUBLIC":
        raise Unresolved("not_public", repo.get("nameWithOwner", ""))


def updated_at_probe(owner: str, name: str, n: int = 40) -> list[dict]:
    """Recent items with the latest reaction / label change / review comment beside their ``updatedAt``.

    An event newer than ``updatedAt`` proves that event kind does not bump ``updatedAt``.
    """
    q = """query($o: String!, $r: String!, $n: Int!) { repository(owner: $o, name: $r) {
      issues(first: $n, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { number updatedAt
        reactions(last: 1) { nodes { createdAt } }
        timelineItems(last: 1, itemTypes: [LABELED_EVENT, UNLABELED_EVENT]) { nodes { ... on LabeledEvent { createdAt } ... on UnlabeledEvent { createdAt } } }
        comments(last: 1) { nodes { createdAt reactions(last: 1) { nodes { createdAt } } } } } }
      pullRequests(first: $n, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { number updatedAt
        reactions(last: 1) { nodes { createdAt } }
        timelineItems(last: 1, itemTypes: [LABELED_EVENT, UNLABELED_EVENT]) { nodes { ... on LabeledEvent { createdAt } ... on UnlabeledEvent { createdAt } } }
        reviewThreads(last: 20) { nodes { comments(last: 1) { nodes { createdAt } } } } } } } }"""
    repo = graphql(q, o=owner, r=name, n=n)["repository"]
    rows = []
    for kind in ("issues", "pullRequests"):
        for node in repo[kind]["nodes"]:
            last = lambda conn: max((x["createdAt"] for x in conn["nodes"] if x), default=None)  # noqa: E731
            row = {"kind": kind, "number": node["number"], "updatedAt": node["updatedAt"]}
            row["reaction"] = last(node["reactions"])
            row["label"] = last(node["timelineItems"])
            if kind == "issues":
                row["comment_reaction"] = max(
                    (r["createdAt"] for c in node["comments"]["nodes"] for r in c["reactions"]["nodes"]), default=None
                )
            else:
                row["review_comment"] = max(
                    (c["createdAt"] for t in node["reviewThreads"]["nodes"] for c in t["comments"]["nodes"]), default=None
                )
            rows.append(row)
    return rows
