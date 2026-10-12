"""The GitHub source over recorded GraphQL responses: full depth, public only, failures as Unresolved reasons."""

from __future__ import annotations

import os

import pytest

from resources_graph.address import Address
from resources_graph.github import GitHubSource, UnresolvedError, resolve_token


def item(number: int) -> Address:
    return Address("item", "memgraph", "memgraph", number)


def test_issue_comments_are_paged_to_the_end(replay, page_size):
    issue = GitHubSource(replay, page_size=page_size).fetch_item(item(2000))

    assert issue["__typename"] == "Issue"
    assert issue["comments"]["totalCount"] == 4
    assert len(issue["comments"]["nodes"]) == 4  # two pages of two


def test_an_issues_url_number_that_is_a_pull_request_comes_back_as_one(replay, page_size):
    pull = GitHubSource(replay, page_size=page_size).fetch_item(item(3600))

    assert pull["__typename"] == "PullRequest"
    assert pull["id"].startswith("PR_")
    assert len(pull["files"]["nodes"]) == pull["files"]["totalCount"] == 3


def test_pull_request_review_threads_are_paged_with_their_paths(replay, page_size):
    pull = GitHubSource(replay, page_size=page_size).fetch_item(item(4962))

    threads = pull["reviewThreads"]["nodes"]
    assert len(threads) == 5
    assert all(thread["path"] for thread in threads)
    assert all(thread["comments"]["nodes"] for thread in threads)


def test_repository_comes_with_its_readme(replay):
    repository = GitHubSource(replay).fetch_repository(Address("repo", "memgraph", "memgraph"))

    assert repository["nameWithOwner"] == "memgraph/memgraph"
    assert repository["readme"]["text"]


def test_missing_item_is_not_found(replay, page_size):
    with pytest.raises(UnresolvedError) as raised:
        GitHubSource(replay, page_size=page_size).fetch_item(item(999999))
    assert raised.value.reason == "not_found"


def test_non_public_repository_is_refused_whatever_the_token_can_read(replay, page_size):
    # Recorded from a public repo, then marked private: GitHub can't be made to return this for a public one.
    replay.edit(
        "Item:",
        lambda body: (
            body["data"]["repository"]["issueOrPullRequest"]["repository"].update(visibility="PRIVATE")
            if body["data"]["repository"]["issueOrPullRequest"]
            else None
        ),
    )

    with pytest.raises(UnresolvedError) as raised:
        GitHubSource(replay, page_size=page_size).fetch_item(item(2000))
    assert raised.value.reason == "not_public"


def test_rate_limit_is_an_unresolved_reason():
    def limited(_query, _variables):
        return {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]}

    with pytest.raises(UnresolvedError) as raised:
        GitHubSource(limited).fetch_item(item(1))
    assert raised.value.reason == "rate_limited"


@pytest.fixture()
def gh_on_path(monkeypatch, tmp_path):
    """Put an executable ``gh`` that prints ``output`` and exits ``code`` first on PATH."""

    def install(output: str, code: int = 0) -> None:
        gh = tmp_path / "gh"
        gh.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\nexit {code}\n")
        gh.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    return install


def test_token_defaults_to_the_users_gh_login(gh_on_path):
    gh_on_path("gho_from_gh_login\n")
    assert resolve_token(None) == "gho_from_gh_login"


def test_configured_token_wins_over_the_gh_login(gh_on_path):
    gh_on_path("gho_from_gh_login")
    assert resolve_token("ghp_configured") == "ghp_configured"


def test_no_token_when_gh_is_logged_out(gh_on_path):
    gh_on_path("", code=1)
    assert resolve_token(None) is None


def test_no_token_without_gh(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_token(None) is None


def test_cross_references_and_closing_references_come_with_the_item(replay, page_size):
    source = GitHubSource(replay, page_size=page_size)
    pull = source.fetch_item(item(4933))
    issue = source.fetch_item(item(4887))

    assert [ref["id"] for ref in pull["closingIssuesReferences"]["nodes"]] == [issue["id"]]
    sources = [node["source"]["id"] for node in issue["timelineItems"]["nodes"] if node.get("source")]
    assert pull["id"] in sources
