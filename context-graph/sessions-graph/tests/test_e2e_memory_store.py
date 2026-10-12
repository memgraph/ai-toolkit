"""The memory tool's commands against a real Memgraph, asserted on results the model reads and on graph shape."""

from __future__ import annotations

import pytest
from sessions_graph.memory_store import MAX_FILE_BYTES, VIEW_CHAR_LIMIT, MemoryCommandError


@pytest.fixture
def store(graph):
    return graph.memory_store("alice", session_id="s-1")


def _error(call, *args) -> str:
    with pytest.raises(MemoryCommandError) as exc:
        call(*args)
    return str(exc.value)


class TestView:
    def test_empty_root_lists_itself(self, store):
        assert store.view("/memories") == (
            "Here're the files and directories up to 2 levels deep in /memories, "
            "excluding hidden items and node_modules:\n0B\t/memories"
        )

    def test_listing_goes_two_levels_deep_and_skips_hidden(self, store):
        store.create("/memories/prefs.md", "a" * 10)
        store.create("/memories/projects/repo/decisions.md", "b" * 20)
        store.create("/memories/projects/repo/deep/x.md", "c" * 5)
        store.create("/memories/.hidden/secret.md", "d")

        listing = store.view("/memories").split("\n")[1:]

        assert listing == [
            "35B\t/memories",
            "10B\t/memories/prefs.md",
            "25B\t/memories/projects/",
            "25B\t/memories/projects/repo/",
        ]

    def test_file_has_six_wide_line_numbers(self, store):
        store.create("/memories/notes.md", "Hello World\nline two")

        assert store.view("/memories/notes.md") == (
            "Here's the content of /memories/notes.md with line numbers:\n     1\tHello World\n     2\tline two"
        )

    def test_view_range_reads_a_slice_and_to_the_end(self, store):
        store.create("/memories/n.md", "one\ntwo\nthree\nfour")

        assert store.view("/memories/n.md", (2, 3)).split("\n")[1:] == ["     2\ttwo", "     3\tthree"]
        assert store.view("/memories/n.md", (3, -1)).split("\n")[1:] == ["     3\tthree", "     4\tfour"]
        assert "Invalid `view_range`" in _error(store.view, "/memories/n.md", (9, 10))

    def test_long_file_is_truncated_with_a_way_to_page(self, store):
        store.create("/memories/long.md", "\n".join("x" * 100 for _ in range(400)))

        shown = store.view("/memories/long.md")

        assert len(shown) <= VIEW_CHAR_LIMIT + 200
        assert "view the rest with view_range" in shown

    def test_missing_path(self, store):
        assert _error(store.view, "/memories/nope.md") == (
            "The path /memories/nope.md does not exist. Please provide a valid path."
        )


class TestWrites:
    def test_create_then_overwrite_keeps_one_node(self, store, memgraph):
        assert store.create("/memories/a.md", "first") == "File created successfully at: /memories/a.md"
        store.create("/memories/a.md", "second")

        rows = memgraph.query("MATCH (:User {user_id: 'alice'})-[:HAS_MEMORY]->(m:Memory) RETURN m.content AS c")
        assert [row["c"] for row in rows] == ["second"]

    def test_create_records_session_provenance(self, store, memgraph):
        store.create("/memories/a.md", "x")

        rows = memgraph.query("MATCH (:Session {session_id: 's-1'})-[:PRODUCED_MEMORY]->(m:Memory) RETURN m.path AS p")
        assert [row["p"] for row in rows] == ["/memories/a.md"]

    def test_project_files_link_to_their_project(self, store, memgraph):
        store.create("/memories/projects/github.com_memgraph_ai-toolkit/decisions.md", "x")
        store.create("/memories/projects/github.com_memgraph_ai-toolkit/deep/more.md", "y")
        store.create("/memories/prefs.md", "z")

        rows = memgraph.query("MATCH (m:Memory)-[:ABOUT]->(p:Project) RETURN p.key AS key, count(m) AS files")
        assert rows == [{"key": "github.com_memgraph_ai-toolkit", "files": 2}]

    def test_create_rejects_directories_files_as_parents_and_empty_or_huge_text(self, store):
        store.create("/memories/dir/a.md", "x")

        assert _error(store.create, "/memories/dir", "x") == "Error: /memories/dir is a directory"
        assert "is a file" in _error(store.create, "/memories/dir/a.md/b.md", "x")
        assert "would be empty" in _error(store.create, "/memories/e.md", "  \n")
        assert "over the 100K limit" in _error(store.create, "/memories/big.md", "x" * (MAX_FILE_BYTES + 1))

    def test_str_replace_edits_and_shows_a_snippet(self, store):
        store.create("/memories/p.md", "Favorite color: blue\nother")

        result = store.str_replace("/memories/p.md", "blue", "green")

        assert result.startswith("The memory file has been edited.")
        assert "     1\tFavorite color: green" in result
        assert store.files()[0].content == "Favorite color: green\nother"

    def test_str_replace_without_new_str_deletes(self, store):
        store.create("/memories/p.md", "keep drop")
        store.str_replace("/memories/p.md", " drop")
        assert store.files()[0].content == "keep"

    def test_str_replace_errors(self, store):
        store.create("/memories/p.md", "dup\ndup")

        assert _error(store.str_replace, "/memories/p.md", "zzz", "y") == (
            "No replacement was performed, old_str `zzz` did not appear verbatim in /memories/p.md."
        )
        assert _error(store.str_replace, "/memories/p.md", "dup", "y") == (
            "No replacement was performed. Multiple occurrences of old_str `dup` in lines: 1, 2. Please ensure it is unique"
        )
        assert "does not exist" in _error(store.str_replace, "/memories/none.md", "a", "b")

    def test_edit_from_a_stale_read_is_refused(self, store, memgraph, monkeypatch):
        store.create("/memories/p.md", "alpha")

        def read_then_lose_the_race(path):
            # Another session rewrites the file between this store's read and its write.
            memgraph.query("MATCH (m:Memory) SET m.content = 'alpha beta'")
            return "alpha"

        monkeypatch.setattr(store, "_content", read_then_lose_the_race)
        message = _error(store.str_replace, "/memories/p.md", "alpha", "gamma")

        monkeypatch.undo()
        assert "changed while it was being edited" in message
        assert store.files()[0].content == "alpha beta"

    def test_insert_after_a_line_and_at_the_top(self, store):
        store.create("/memories/t.md", "one\ntwo\n")

        assert store.insert("/memories/t.md", 1, "between\n") == "The file /memories/t.md has been edited."
        store.insert("/memories/t.md", 0, "top")

        assert store.files()[0].content == "top\none\nbetween\ntwo\n"
        assert _error(store.insert, "/memories/t.md", 9, "x") == (
            "Error: Invalid `insert_line` parameter: 9. It should be within the range of lines of the file: [0, 4]"
        )


class TestDeleteAndRename:
    def test_delete_file_and_directory(self, store):
        store.create("/memories/a.md", "x")
        store.create("/memories/dir/b.md", "x")
        store.create("/memories/dir/sub/c.md", "x")

        assert store.delete("/memories/a.md") == "Successfully deleted /memories/a.md"
        store.delete("/memories/dir")

        assert store.files() == []
        assert _error(store.delete, "/memories/a.md") == "Error: The path /memories/a.md does not exist"
        assert "Cannot delete the /memories directory itself" in _error(store.delete, "/memories")

    def test_rename_file_keeps_identity_and_relinks_project(self, store, memgraph):
        store.create("/memories/draft.md", "x")
        before = memgraph.query("MATCH (m:Memory) RETURN m.memory_id AS id")[0]["id"]

        assert store.rename("/memories/draft.md", "/memories/projects/repo/final.md") == (
            "Successfully renamed /memories/draft.md to /memories/projects/repo/final.md"
        )

        rows = memgraph.query(
            "MATCH (m:Memory)-[:ABOUT]->(p:Project) RETURN m.memory_id AS id, m.path AS path, p.key AS key"
        )
        assert rows == [{"id": before, "path": "/memories/projects/repo/final.md", "key": "repo"}]

    def test_rename_directory_moves_everything_under_it(self, store):
        store.create("/memories/old/a.md", "x")
        store.create("/memories/old/sub/b.md", "y")
        store.create("/memories/older.md", "z")

        store.rename("/memories/old", "/memories/new")

        assert [f.path for f in store.files()] == ["/memories/new/a.md", "/memories/new/sub/b.md", "/memories/older.md"]

    def test_rename_errors(self, store):
        store.create("/memories/a.md", "x")
        store.create("/memories/b.md", "y")
        store.create("/memories/dir/c.md", "z")

        assert _error(store.rename, "/memories/a.md", "/memories/b.md") == (
            "Error: The destination /memories/b.md already exists"
        )
        assert (
            _error(store.rename, "/memories/zz.md", "/memories/q.md")
            == "Error: The path /memories/zz.md does not exist"
        )
        assert "Cannot rename the /memories directory itself" in _error(store.rename, "/memories", "/memories/x")
        assert "into itself" in _error(store.rename, "/memories/dir", "/memories/dir/inner")


class TestIsolationAndPaths:
    def test_users_never_see_each_others_files(self, graph):
        graph.memory_store("alice").create("/memories/a.md", "alice's")
        bob = graph.memory_store("bob")

        assert bob.files() == []
        assert "does not exist" in _error(bob.view, "/memories/a.md")
        bob.create("/memories/a.md", "bob's")
        assert graph.memory_store("alice").files()[0].content == "alice's"

    @pytest.mark.parametrize(
        "path",
        [
            "/memories/../secrets.env",
            "/memories/a/../../etc",
            "/memories/%2e%2e/x",
            "/memories/a%2Fb",
            "/etc/passwd",
            "/memoriesX/a.md",
            "memories/a.md",
            "/memories/a\\b",
            "/memories/a\x00b",
        ],
    )
    def test_traversal_and_foreign_paths_are_rejected(self, store, path):
        assert _error(store.create, path, "x").startswith("Error: ")

    def test_paths_are_canonicalized(self, store):
        store.create("/memories//dir///a.md", "x")
        assert [f.path for f in store.files()] == ["/memories/dir/a.md"]
        assert store.view("/memories/dir/").startswith(
            "Here're the files and directories up to 2 levels deep in /memories/dir,"
        )


class TestExecute:
    def test_dispatches_the_tool_input_shape(self, store):
        store.execute({"command": "create", "path": "/memories/x.md", "file_text": "hi"})
        assert "1\thi" in store.execute({"command": "view", "path": "/memories/x.md", "view_range": [1, -1]})
        store.execute({"command": "rename", "old_path": "/memories/x.md", "new_path": "/memories/y.md"})
        assert store.execute({"command": "delete", "path": "/memories/y.md"}) == "Successfully deleted /memories/y.md"

    def test_bad_input_is_a_command_error(self, store):
        assert _error(store.execute, {"command": "explode"}) == "Error: unknown command explode"
        assert "`path` is required" in _error(store.execute, {"command": "view"})
        assert "`insert_line` is required" in _error(
            store.execute, {"command": "insert", "path": "/memories/a", "insert_text": "x"}
        )


class TestSaveMemoryPaths:
    def test_save_memory_files_under_notes_by_default(self, graph):
        memory = graph.save_memory("alice", "Prefers dark mode", memory_id="m-1")

        assert memory.path == "/memories/notes/m-1.md"
        assert graph.memory_store("alice").files()[0].path == "/memories/notes/m-1.md"
        assert graph.get_memories("alice")[0].path == "/memories/notes/m-1.md"

    def test_save_memory_at_a_project_path_links_the_project(self, graph, memgraph):
        graph.save_memory("alice", "Uses uv", path="/memories/projects/repo/tooling.md")

        rows = memgraph.query("MATCH (:Memory)-[:ABOUT]->(p:Project) RETURN p.key AS key")
        assert rows == [{"key": "repo"}]


class TestVersions:
    def test_every_discarding_write_keeps_what_it_replaced(self, store):
        store.create("/memories/a.md", "one")
        store.create("/memories/a.md", "two")
        store.str_replace("/memories/a.md", "two", "three")
        store.insert("/memories/a.md", 0, "zero")

        assert [v.content for v in store.versions("/memories/a.md")] == ["three", "two", "one"]
        assert all(v.session_id == "s-1" and not v.deleted for v in store.versions("/memories/a.md"))

    def test_rewriting_the_same_text_keeps_nothing(self, store):
        store.create("/memories/a.md", "same")
        store.create("/memories/a.md", "same")
        assert store.versions("/memories/a.md") == []

    def test_history_follows_a_rename_and_outlives_a_delete(self, store, memgraph):
        store.create("/memories/a.md", "v1")
        store.create("/memories/a.md", "v2")
        store.rename("/memories/a.md", "/memories/b.md")
        assert [v.content for v in store.versions("/memories/b.md")] == ["v1"]

        store.delete("/memories/b.md")

        history = store.versions("/memories/b.md")
        assert [(v.content, v.deleted) for v in history] == [("v2", True)]
        assert memgraph.query("MATCH (v:MemoryVersion) RETURN count(v) AS n")[0]["n"] == 2

    def test_deleting_a_directory_keeps_each_files_last_text(self, store):
        store.create("/memories/dir/a.md", "a")
        store.create("/memories/dir/b.md", "b")
        store.delete("/memories/dir")
        assert [v.content for v in store.versions("/memories/dir/a.md")] == ["a"]
        assert [v.content for v in store.versions("/memories/dir/b.md")] == ["b"]

    def test_only_the_newest_versions_are_kept(self, store):
        from sessions_graph.memory_store import MAX_VERSIONS

        for n in range(MAX_VERSIONS + 5):
            store.create("/memories/a.md", f"text {n}")

        kept = store.versions("/memories/a.md")
        assert len(kept) == MAX_VERSIONS
        assert kept[0].content == f"text {MAX_VERSIONS + 3}"

    def test_versions_are_per_user(self, graph):
        graph.memory_store("alice").create("/memories/a.md", "x")
        graph.memory_store("alice").create("/memories/a.md", "y")
        assert graph.memory_store("bob").versions("/memories/a.md") == []
