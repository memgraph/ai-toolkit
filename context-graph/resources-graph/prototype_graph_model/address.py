"""PROTOTYPE — throwaway. Pure address logic for the GitHub Resource graph model (#460).

Turns whatever the agent touched (URL, ``owner/repo#n``, ``gh`` argv, ``gh api`` path)
into one normalised Address, and decides listing subsumption. No I/O.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

STRUCTURED_FILTERS = ("state", "labels", "author", "assignee", "milestone")


@dataclass(frozen=True)
class Address:
    """A normalised Address. ``kind`` is repo | item | listing."""

    kind: str
    owner: str
    repo: str
    number: int | None = None
    item_kind: str | None = None  # issues | pulls (listing kind, or item hint)
    filters: tuple = field(default_factory=tuple)  # sorted (name, value) pairs
    query: str | None = None  # free-text search, exact-match only
    limit: int | None = None  # how many members the agent asked for; None = all

    @property
    def repo_key(self) -> str:
        return f"github:{self.owner}/{self.repo}".lower()

    @property
    def key(self) -> str:
        """Cache key. ``limit`` is deliberately not part of it — it decides expansion, not identity."""
        if self.kind == "repo":
            return self.repo_key
        if self.kind == "item":
            return f"{self.repo_key}#{self.number}"
        parts = [f"{k}={v}" for k, v in self.filters]
        if self.query:
            parts.append(f"q={self.query}")
        return f"{self.repo_key}/{self.item_kind}?{'&'.join(parts)}"

    @property
    def url(self) -> str:
        base = f"https://github.com/{self.owner}/{self.repo}"
        if self.kind == "item":
            return f"{base}/{'pull' if self.item_kind == 'pulls' else 'issues'}/{self.number}"
        return base

    def filter_dict(self) -> dict:
        return dict(self.filters)


def _norm_filters(raw: dict) -> tuple:
    out = {}
    state = (raw.get("state") or "open").lower()
    out["state"] = state
    labels = raw.get("labels") or raw.get("label")
    if labels:
        if isinstance(labels, str):
            labels = labels.split(",")
        out["labels"] = ",".join(sorted({label.strip().lower() for label in labels if label.strip()}))
    for name in ("author", "assignee", "milestone"):
        if raw.get(name):
            out[name] = str(raw[name]).lower()
    return tuple(sorted(out.items()))


def listing(owner, repo, item_kind, filters=None, query=None, limit=None) -> Address:
    return Address(
        "listing",
        owner,
        repo,
        item_kind=item_kind,
        filters=_norm_filters(filters or {}),
        query=query,
        limit=limit,
    )


SHORT_REF = re.compile(r"^([\w.-]+)/([\w.-]+)#(\d+)$")
GH_URL = re.compile(r"https?://github\.com/[^\s)>\]\"']+")


def parse(text: str) -> Address | None:
    """Parse one touched reference. Returns None when it isn't a GitHub Address."""
    text = text.strip()
    if m := SHORT_REF.match(text):
        return Address("item", m[1], m[2], number=int(m[3]))
    if text.startswith("gh "):
        return _parse_gh(shlex.split(text))
    if "github.com" in text:
        return _parse_url(text)
    return None


def _parse_url(url: str) -> Address | None:
    u = urlparse(url if "://" in url else "https://" + url)
    if u.netloc == "api.github.com":
        return _parse_api_path(u.path, parse_qs(u.query))
    parts = [p for p in u.path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if len(parts) == 2:
        return Address("repo", owner, repo)
    section = parts[2]
    if section in ("issues", "pull", "pulls") and len(parts) >= 4 and parts[3].isdigit():
        return Address("item", owner, repo, number=int(parts[3]), item_kind="pulls" if section != "issues" else None)
    if section in ("issues", "pulls"):
        q = parse_qs(u.query).get("q", [None])[0]
        return listing(owner, repo, section, query=q)
    return Address("repo", owner, repo)  # tree/blob/wiki paths: repo-level touch in v1


def _parse_api_path(path: str, qs: dict) -> Address | None:
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3 or parts[0] != "repos":
        return None
    owner, repo = parts[1], parts[2]
    if len(parts) == 3:
        return Address("repo", owner, repo)
    if parts[3] in ("issues", "pulls"):
        if len(parts) >= 5 and parts[4].isdigit():
            return Address("item", owner, repo, number=int(parts[4]))
        flat = {k: v[0] for k, v in qs.items()}
        flat.setdefault("state", "open")
        return listing(owner, repo, parts[3], flat, limit=int(flat["per_page"]) if "per_page" in flat else 30)
    return Address("repo", owner, repo)


def _parse_gh(argv: list[str]) -> Address | None:
    if len(argv) >= 3 and argv[1] == "api":
        path = argv[2].lstrip("/")
        p, _, q = path.partition("?")
        return _parse_api_path("/" + p, parse_qs(q))
    if len(argv) < 3 or argv[1] not in ("issue", "pr", "repo"):
        return None
    group, verb, rest = argv[1], argv[2], argv[3:]
    opts, positional = _opts(rest)
    owner_repo = opts.get("repo") or opts.get("R")
    if group == "repo" and verb == "view":
        target = positional[0] if positional else owner_repo
        owner, repo = target.split("/") if target else ("?", "?")
        return Address("repo", owner, repo)
    if not owner_repo:
        return None  # gh resolves the repo from the cwd's git remote; the hook would need cwd
    owner, repo = owner_repo.split("/")
    item_kind = "pulls" if group == "pr" else "issues"
    if verb == "view" and positional:
        ref = positional[0]
        if ref.startswith("http"):
            return _parse_url(ref)
        return Address("item", owner, repo, number=int(ref.lstrip("#")), item_kind=item_kind)
    if verb == "list":
        limit = int(opts.get("limit") or opts.get("L") or 30)
        labels = opts.get("label_list") or []
        filters = {
            "state": opts.get("state") or opts.get("s") or "open",
            "labels": labels,
            "author": opts.get("author") or opts.get("A"),
            "assignee": opts.get("assignee") or opts.get("a"),
            "milestone": opts.get("milestone") or opts.get("m"),
        }
        return listing(owner, repo, item_kind, filters, query=opts.get("search") or opts.get("S"), limit=limit)
    return None


def _opts(args: list[str]) -> tuple[dict, list]:
    opts: dict = {"label_list": []}
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            name = a.lstrip("-")
            if "=" in name:
                name, val = name.split("=", 1)
            else:
                val = args[i + 1] if i + 1 < len(args) else ""
                i += 1
            if name in ("label", "l"):
                opts["label_list"].extend(val.split(","))
            else:
                opts[name] = val
        else:
            positional.append(a)
        i += 1
    return opts, positional


def addresses_in_prompt(text: str) -> list[Address]:
    """PROMPTED Touches: every GitHub URL or short ref in free prompt text."""
    found = [parse(m) for m in GH_URL.findall(text)]
    found += [parse(m[0]) for m in re.finditer(r"\b[\w.-]+/[\w.-]+#\d+\b", text)]
    return [a for a in found if a]


def subsumes(cached: Address, wanted: Address) -> bool:
    """True when a *fully expanded* cached listing can answer ``wanted`` by local filtering.

    Free-text search never subsumes or is subsumed (exact-only). Every filter the cached
    listing applied must be applied identically by the wanted one; a narrower wanted
    filter is applied locally over the stored members.
    """
    if cached.kind != "listing" or wanted.kind != "listing":
        return False
    if (cached.repo_key, cached.item_kind) != (wanted.repo_key, wanted.item_kind):
        return False
    if cached.query or wanted.query:
        return False
    c, w = cached.filter_dict(), wanted.filter_dict()
    if c.get("state") != "all" and c.get("state") != w.get("state"):
        return False
    if c.get("labels") and not set(c["labels"].split(",")) <= set((w.get("labels") or "").split(",")):
        return False
    for name in ("author", "assignee", "milestone"):
        if c.get(name) and c.get(name) != w.get(name):
            return False
    return True


def member_matches(member: dict, wanted: Address) -> bool:
    """Local filter applied to a stored member when serving a subsumed listing."""
    w = wanted.filter_dict()
    if w.get("state") not in (None, "all") and member["state"].lower() != w["state"]:
        return False
    if w.get("labels") and not set(w["labels"].split(",")) <= {lb.lower() for lb in member["labels"]}:
        return False
    if w.get("author") and (member.get("author") or "").lower() != w["author"]:
        return False
    if w.get("assignee") and w["assignee"] not in {a.lower() for a in member.get("assignees") or []}:
        return False
    if w.get("milestone") and (member.get("milestone") or "").lower() != w["milestone"]:
        return False
    return True
