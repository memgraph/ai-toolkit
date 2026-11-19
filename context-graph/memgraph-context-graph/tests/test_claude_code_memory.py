"""Tests for reading Claude Code's project-folder slugs back into checkouts.

Walking a slug against a real checkout is covered end to end in sessions-graph's
memory backend tests; a Windows drive can't be walked off Windows, so its root is
tested here.
"""

from pathlib import Path

import pytest

from memgraph_context_graph.adapters.claude_code_memory import _slug_root


@pytest.mark.parametrize(
    ("slug", "root"),
    [
        ("-Users-dev-repos-ai-toolkit", (Path("/"), ["Users", "dev", "repos", "ai", "toolkit"])),
        ("C--source-x", (Path("C:/"), ["source", "x"])),
        ("d--work-my-repo", (Path("d:/"), ["work", "my", "repo"])),
        # A dot-folder after the drive makes a third dash: C:\.config\x.
        ("C---config-x", (Path("C:/"), ["", "config", "x"])),
    ],
)
def test_a_slug_walks_from_the_root_its_path_started_at(slug, root):
    assert _slug_root(slug) == root


@pytest.mark.parametrize("slug", ["-", "", "C--", "source-x", "CD--x"])
def test_a_slug_with_no_root_has_no_checkout(slug):
    assert _slug_root(slug) is None
