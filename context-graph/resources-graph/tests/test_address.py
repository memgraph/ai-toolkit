"""Address parsing: which GitHub resources a tool call, prompt or ``resource`` argument names."""

from __future__ import annotations

import pytest

from resources_graph.address import Address, addresses_from_text, addresses_from_tool, parse_address


def keys(addresses):
    return [address.key for address in addresses]


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        ("WebFetch", {"url": "https://github.com/memgraph/memgraph/issues/123"}, ["github:memgraph/memgraph#123"]),
        ("WebFetch", {"url": "https://github.com/Memgraph/Memgraph/pull/7/files"}, ["github:memgraph/memgraph#7"]),
        ("WebFetch", {"url": "https://github.com/memgraph/memgraph"}, ["github:memgraph/memgraph"]),
        ("WebFetch", {"url": "https://api.github.com/repos/memgraph/mage/issues/5"}, ["github:memgraph/mage#5"]),
        ("WebFetch", {"url": "https://github.com/memgraph/memgraph/issues"}, []),  # a listing: not yet
        ("WebFetch", {"url": "https://github.com/memgraph/memgraph/blob/main/README.md"}, []),  # code: out of scope
        ("WebFetch", {"url": "https://github.com/orgs/memgraph/repositories"}, []),
        ("WebFetch", {"url": "https://example.com/memgraph/memgraph/issues/1"}, []),
        ("Bash", {"command": "gh issue view 12 -R memgraph/memgraph --comments"}, ["github:memgraph/memgraph#12"]),
        ("Bash", {"command": "gh issue view 12 --repo=memgraph/memgraph"}, ["github:memgraph/memgraph#12"]),
        (
            "Bash",
            {"command": "gh pr view '#34' --repo memgraph/memgraph --json title,body"},
            ["github:memgraph/memgraph#34"],
        ),
        ("Bash", {"command": "gh pr view https://github.com/a/b/pull/7"}, ["github:a/b#7"]),
        ("Bash", {"command": "gh issue view 12"}, []),  # repo comes from the cwd's remote: unknown here
        ("Bash", {"command": "gh repo view memgraph/mage"}, ["github:memgraph/mage"]),
        ("Bash", {"command": "gh api repos/memgraph/memgraph/issues/9 --jq .title"}, ["github:memgraph/memgraph#9"]),
        ("Bash", {"command": "gh api /repos/memgraph/memgraph"}, ["github:memgraph/memgraph"]),
        ("Bash", {"command": "gh api repos/memgraph/memgraph/issues/9/comments"}, []),
        ("Bash", {"command": "gh api graphql -f query='{ viewer { login } }'"}, []),
        (
            "Bash",
            {"command": "curl -s https://api.github.com/repos/memgraph/memgraph/pulls/3 | jq ."},
            ["github:memgraph/memgraph#3"],
        ),
        (
            "Bash",
            {"command": "cd /tmp && GH_PAGER= gh issue view 1 -R a/b; gh issue view 2 -R a/b"},
            ["github:a/b#1", "github:a/b#2"],
        ),
        ("Bash", {"command": "gh issue list -R memgraph/memgraph --limit 500"}, []),  # a listing: not yet
        ("Bash", {"command": "echo 'unbalanced"}, []),
        ("shell", {"command": ["bash", "-lc", "gh repo view memgraph/mage"]}, ["github:memgraph/mage"]),
        ("shell", {"command": ["gh", "issue", "view", "4", "-R", "a/b"]}, ["github:a/b#4"]),
        (
            "mcp__github__get_issue",
            {"owner": "memgraph", "repo": "memgraph", "issue_number": 5},
            ["github:memgraph/memgraph#5"],
        ),
        ("mcp__github__get_pull_request", {"owner": "a", "repo": "b", "pullNumber": "6"}, ["github:a/b#6"]),
        ("mcp__github__search_issues", {"q": "repo:a/b bug"}, []),
        ("Read", {"file_path": "/tmp/github.com/a/b/issues/1"}, []),
        ("Bash", "not a dict", []),
    ],
)
def test_addresses_from_tool(tool_name, tool_input, expected):
    assert keys(addresses_from_tool(tool_name, tool_input)) == expected


def test_addresses_from_text_finds_urls_and_short_refs_once():
    text = (
        "Look at https://github.com/memgraph/memgraph/pull/3600, then memgraph/mage#12 "
        "(and https://github.com/memgraph/memgraph/pull/3600 again), the repo https://github.com/memgraph/memgraph. "
        "Ignore https://github.com/orgs/memgraph and path/to/file#3."
    )
    assert keys(addresses_from_text(text)) == [
        "github:memgraph/memgraph#3600",
        "github:memgraph/memgraph",
        "github:memgraph/mage#12",  # path/to/file#3 is a path, not a ref
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("memgraph/memgraph#1", "github:memgraph/memgraph#1"),
        ("memgraph/memgraph", "github:memgraph/memgraph"),
        ("https://github.com/memgraph/memgraph/pull/9", "github:memgraph/memgraph#9"),
        ("github.com/a/b/issues/2", "github:a/b#2"),
        ("gh issue view 3 -R a/b", "github:a/b#3"),
        ("repos/a/b/pulls/4", "github:a/b#4"),
        ("https://github.com/memgraph/memgraph/issues", None),
        ("memgraph/memgraph#1 and memgraph/memgraph#2", None),
        ("nonsense", None),
        ("", None),
    ],
)
def test_parse_address(text, expected):
    address = parse_address(text)
    assert (address.key if address else None) == expected


def test_keys_round_trip_and_ignore_case():
    item = Address("item", "Memgraph", "MAGE", 12)
    assert item.key == "github:memgraph/mage#12"
    assert Address.from_key(item.key) == item
    assert Address.from_key("github:memgraph/mage") == item.repository


@pytest.mark.parametrize("key", ["memgraph/mage#1", "github:memgraph", "github:a/b#x", "web:a/b"])
def test_from_key_rejects_foreign_keys(key):
    with pytest.raises(ValueError):
        Address.from_key(key)
