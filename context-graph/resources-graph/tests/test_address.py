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
        (
            "WebFetch",
            {"url": "https://github.com/memgraph/memgraph/issues"},
            ["github:memgraph/memgraph/issues?state=open"],
        ),
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
        (
            "Bash",
            {"command": "gh issue list -R memgraph/memgraph --limit 500"},
            ["github:memgraph/memgraph/issues?state=open"],
        ),
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
        ("https://github.com/memgraph/memgraph/issues", "github:memgraph/memgraph/issues?state=open"),
        (
            "github:memgraph/memgraph/issues?labels=bug&state=open",
            "github:memgraph/memgraph/issues?labels=bug&state=open",
        ),
        ("github:nonsense", None),
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


@pytest.mark.parametrize(
    ("command", "key", "limit"),
    [
        ("gh issue list -R memgraph/memgraph", "github:memgraph/memgraph/issues?state=open", 30),
        ("gh issue list -R memgraph/memgraph --state all -L 500", "github:memgraph/memgraph/issues?state=all", 500),
        (
            "gh issue list -R memgraph/memgraph -l bug --label 'Priority/P1,docs'",
            "github:memgraph/memgraph/issues?labels=bug,docs,priority/p1&state=open",
            30,
        ),
        (
            "gh issue list --repo=memgraph/memgraph --author Alice -a bob -m v3.0 -s closed",
            "github:memgraph/memgraph/issues?assignee=bob&author=alice&milestone=v3.0&state=closed",
            30,
        ),
        ("gh pr list -R a/b -s merged --json number,title", "github:a/b/pulls?state=merged", 30),
        (
            'gh issue list -R a/b --search "replication  lag" -L 50',
            "github:a/b/issues?state=open&q=replication%20lag",
            50,
        ),
        (
            "https://github.com/memgraph/memgraph/issues?q=is%3Aissue+is%3Aclosed+label%3Abug+author%3Aalice",
            "github:memgraph/memgraph/issues?author=alice&labels=bug&state=closed",
            25,
        ),
        ("https://github.com/a/b/pulls?q=is:pr+memory+leak", "github:a/b/pulls?state=all&q=memory%20leak", 25),
    ],
)
def test_listings_normalise_to_one_key_with_the_asked_limit(command, key, limit):
    address = parse_address(command)
    assert address is not None
    assert (address.key, address.limit) == (key, limit)
    assert Address.from_key(address.key, limit=limit) == address


@pytest.mark.parametrize(
    "command",
    [
        "gh issue list",  # repository from the working directory
        "gh pr list -R a/b --draft",  # filters the key can't describe
        "gh pr list -R a/b --base main",
        "gh issue list -R a/b --mention me",
        "gh issue list -R a/b -s sideways",
        "gh pr list -R a/b -L many",
    ],
)
def test_listings_the_key_cannot_describe_are_not_touched(command):
    assert parse_address(command) is None


def test_the_limit_is_not_part_of_the_key():
    small = parse_address("gh issue list -R a/b -L 10")
    large = parse_address("gh issue list -R a/b -L 500")
    assert small is not None and large is not None
    assert small.key == large.key and small != large
