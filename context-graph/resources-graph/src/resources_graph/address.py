"""Addresses: how a touched resource was referred to, normalised to one cache key.

Pure: no I/O. A per-platform adapter turns what the agent did — a tool call or
prompt text — into :class:`Address` values; GitHub is the only platform so far.
An Address is never a Resource's identity (that is the platform's own id, e.g.
GitHub's ``node_id``): repositories get renamed and issues transferred.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

GITHUB = "github"

# First path segments on github.com that are site pages, not an owner.
_RESERVED_OWNERS = frozenset(
    {
        "about",
        "apps",
        "collections",
        "contact",
        "enterprise",
        "explore",
        "features",
        "issues",
        "login",
        "marketplace",
        "notifications",
        "orgs",
        "pricing",
        "pulls",
        "search",
        "security",
        "settings",
        "site",
        "sponsors",
        "topics",
        "users",
    }
)
_NAME = r"[A-Za-z0-9_.-]+"
_SHORT_REF = re.compile(rf"(?<![\w/.-])({_NAME})/({_NAME})#(\d+)\b")
_URL = re.compile(r"https?://(?:www\.)?(?:api\.)?github\.com/[^\s<>()\[\]\"'`]+")
_SHELLS = frozenset({"bash", "sh", "zsh"})


@dataclass(frozen=True)
class Address:
    """A normalised reference to one resource.

    ``kind`` is ``repo`` or ``item`` (an issue or pull request — GitHub numbers
    them in one sequence, so the key doesn't say which).
    """

    kind: str
    owner: str
    repo: str
    number: int | None = None
    platform: str = GITHUB

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", self.owner.lower())
        object.__setattr__(self, "repo", self.repo.lower())

    @property
    def key(self) -> str:
        """The cache key, e.g. ``github:memgraph/memgraph#123``."""
        base = f"{self.platform}:{self.owner}/{self.repo}"
        return f"{base}#{self.number}" if self.kind == "item" else base

    @property
    def repository(self) -> Address:
        """The Address of the repository this one belongs to."""
        return Address("repo", self.owner, self.repo, platform=self.platform)

    @classmethod
    def from_key(cls, key: str) -> Address:
        """Inverse of :attr:`key`.

        Raises:
            ValueError: ``key`` isn't a key this module produces.
        """
        platform, _, rest = key.partition(":")
        path, _, number = rest.partition("#")
        owner, _, repo = path.partition("/")
        if platform != GITHUB or not owner or not repo or (number and not number.isdigit()):
            raise ValueError(f"not a resource key: {key!r}")
        if number:
            return cls("item", owner, repo, int(number), platform)
        return cls("repo", owner, repo, platform=platform)


class PlatformAdapter(Protocol):
    """Finds one platform's Addresses in what the agent did."""

    def from_tool(self, tool_name: str, tool_input: Any) -> list[Address]:
        """Addresses a tool call fetches."""
        ...

    def from_text(self, text: str) -> list[Address]:
        """Addresses mentioned in free text, e.g. a prompt."""
        ...


class GitHubAdapter:
    """GitHub URLs, ``owner/repo#n`` refs, ``gh``/``curl`` commands and GitHub MCP tools."""

    def from_tool(self, tool_name: str, tool_input: Any) -> list[Address]:
        """Addresses a tool call fetches; empty when it doesn't read GitHub."""
        if not isinstance(tool_input, dict):
            return []
        name = tool_name.lower()
        if isinstance(tool_input.get("url"), str):
            return _unique(_from_url(tool_input["url"]))
        command = tool_input.get("command", tool_input.get("cmd"))
        if isinstance(command, (str, list)):
            return _unique(_from_command(command))
        if "github" in name:
            return _unique(_from_mcp_arguments(tool_input))
        return []

    def from_text(self, text: str) -> list[Address]:
        """Every issue, PR or repository URL and ``owner/repo#n`` ref in ``text``."""
        found = [address for url in _URL.findall(text) for address in _from_url(url)]
        found += [Address("item", m[1], m[2], int(m[3])) for m in _SHORT_REF.finditer(text) if _is_owner(m[1])]
        return _unique(found)


ADAPTERS: tuple[PlatformAdapter, ...] = (GitHubAdapter(),)


def addresses_from_tool(tool_name: str, tool_input: Any) -> list[Address]:
    """The Addresses a tool call fetches, across every platform adapter."""
    return _unique(a for adapter in ADAPTERS for a in adapter.from_tool(tool_name, tool_input))


def addresses_from_text(text: str) -> list[Address]:
    """The Addresses mentioned in free text, across every platform adapter."""
    return _unique(a for adapter in ADAPTERS for a in adapter.from_text(text))


def parse_address(text: str) -> Address | None:
    """One Address from what a model would pass the ``resource`` tool.

    Accepts a URL, ``owner/repo#n``, ``owner/repo``, ``gh``/``curl`` arguments
    or a ``gh api`` path. Returns None when ``text`` names no single resource.
    """
    text = text.strip()
    if text.startswith(("gh ", "curl ")):
        found = _from_command(text)
    elif text.startswith(("repos/", "/repos/")):
        found = _from_api_path(text)
    elif re.fullmatch(rf"{_NAME}/{_NAME}", text) and _is_owner(text.split("/")[0]):
        owner, repo = text.split("/")
        found = [Address("repo", owner, repo)]
    else:
        found = addresses_from_text(text) or _from_url(text if "://" in text else f"https://{text}")
    return found[0] if len(found) == 1 else None


def _from_url(url: str) -> list[Address]:
    parsed = urlparse(url.rstrip(".,;:!?"))
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "api.github.com":
        return _from_api_path(parsed.path)
    if host != "github.com":
        return []
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or not _is_owner(parts[0]):
        return []
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if len(parts) == 2:
        return [Address("repo", owner, repo)]
    if parts[2] in ("issues", "pull", "pulls") and len(parts) >= 4 and parts[3].isdigit():
        return [Address("item", owner, repo, int(parts[3]))]
    # Listings, code (tree/blob), wikis, actions: not single-resource Addresses yet.
    return []


def _from_api_path(path: str) -> list[Address]:
    parts = [part for part in path.split("?")[0].split("/") if part]
    if len(parts) < 3 or parts[0] != "repos":
        return []
    owner, repo = parts[1], parts[2]
    if len(parts) == 3:
        return [Address("repo", owner, repo)]
    if parts[3] in ("issues", "pulls") and len(parts) == 5 and parts[4].isdigit():
        return [Address("item", owner, repo, int(parts[4]))]
    return []


def _from_command(command: str | list[Any]) -> list[Address]:
    """Addresses read by ``gh``/``curl`` invocations anywhere in a shell command."""
    if isinstance(command, list):
        argv = [str(part) for part in command]
        if len(argv) >= 3 and argv[0].rsplit("/", 1)[-1] in _SHELLS and argv[1] in ("-c", "-lc"):
            return _from_command(argv[2])
        return _from_argv(argv)
    found: list[Address] = []
    for argv in _simple_commands(command):
        found += _from_argv(argv)
    return found


def _simple_commands(command: str) -> Iterator[list[str]]:
    """Split a shell command line into its simple commands (on ``|``, ``&&``, ``;`` ...)."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes: not a command we can read
        return
    current: list[str] = []
    for token in tokens:
        if token and set(token) <= set("|&;()<>"):
            if current:
                yield current
            current = []
        else:
            current.append(token)
    if current:
        yield current


def _from_argv(argv: list[str]) -> list[Address]:
    while argv and "=" in argv[0] and not argv[0].startswith("-"):  # leading VAR=value assignments
        argv = argv[1:]
    if not argv:
        return []
    program = argv[0].rsplit("/", 1)[-1]
    if program == "curl":
        return [address for arg in argv[1:] if "github.com" in arg for address in _from_url(arg)]
    if program != "gh" or len(argv) < 2:
        return []
    group, rest = argv[1], argv[2:]
    if group == "api":
        paths = [arg for arg in rest if not arg.startswith("-")]
        return _from_api_path(paths[0]) if paths else []
    options, positional = _gh_options(rest)
    if not positional:
        return []
    verb, targets = positional[0], positional[1:]
    if group == "repo" and verb == "view":
        target = targets[0] if targets else options.get("repo")
        return _repo_target(target) if target else []
    if group in ("issue", "pr") and verb == "view" and targets:
        target = targets[0]
        if "github.com" in target:
            return _from_url(target)
        repo_option = options.get("repo")
        number = target.lstrip("#")
        if repo_option and number.isdigit() and "/" in repo_option:
            owner, repo = repo_option.split("/", 1)
            return [Address("item", owner, repo, int(number))]
        # Without -R/--repo, gh resolves the repository from the working directory's remote.
    return []


def _repo_target(target: str) -> list[Address]:
    if "github.com" in target:
        return _from_url(target)
    owner, _, repo = target.partition("/")
    return [Address("repo", owner, repo)] if owner and repo and _is_owner(owner) else []


def _gh_options(args: list[str]) -> tuple[dict[str, str], list[str]]:
    """``gh`` flags with values (``-R``/``--repo`` normalised to ``repo``) and positional arguments."""
    options: dict[str, str] = {}
    positional: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith("-") and arg != "-":
            name, has_value, value = arg.lstrip("-").partition("=")
            if not has_value:
                if name in _GH_BOOLEAN_FLAGS or index + 1 >= len(args):
                    index += 1
                    continue
                value = args[index + 1]
                index += 1
            options["repo" if name in ("R", "repo") else name] = value
        else:
            positional.append(arg)
        index += 1
    return options, positional


_GH_BOOLEAN_FLAGS = frozenset({"web", "w", "comments", "c", "help", "h"})


def _from_mcp_arguments(arguments: dict[str, Any]) -> list[Address]:
    owner, repo = arguments.get("owner"), arguments.get("repo")
    if not isinstance(owner, str) or not isinstance(repo, str):
        return []
    for key in ("issue_number", "issueNumber", "pull_number", "pullNumber", "number"):
        number = arguments.get(key)
        if isinstance(number, int) or (isinstance(number, str) and number.isdigit()):
            return [Address("item", owner, repo, int(number))]
    return []


def _is_owner(name: str) -> bool:
    return name.lower() not in _RESERVED_OWNERS


def _unique(addresses: Iterable[Address]) -> list[Address]:
    return list(dict.fromkeys(addresses))
