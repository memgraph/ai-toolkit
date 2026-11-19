"""Rewriting a user's file keeps its line endings on every platform."""

from memgraph_context_graph.adapters._text_file import write_text_keeping_newlines


def test_a_crlf_file_stays_crlf(tmp_path):
    path = tmp_path / "settings.json"
    path.write_bytes(b'{\r\n  "theme": "dark"\r\n}\r\n')

    write_text_keeping_newlines(path, '{\n  "theme": "light"\n}\n')

    assert path.read_bytes() == b'{\r\n  "theme": "light"\r\n}\r\n'


def test_an_lf_file_stays_lf(tmp_path):
    path = tmp_path / "config.toml"
    path.write_bytes(b"[a]\nb = 1\n")

    write_text_keeping_newlines(path, "[a]\nb = 2\n")

    assert path.read_bytes() == b"[a]\nb = 2\n"


def test_a_new_file_gets_lf(tmp_path):
    path = tmp_path / "MEMORY.md"

    write_text_keeping_newlines(path, "- [a](a.md)\n")

    assert path.read_bytes() == b"- [a](a.md)\n"


def test_text_already_in_crlf_is_not_doubled(tmp_path):
    path = tmp_path / "note.md"
    path.write_bytes(b"old\r\n")

    write_text_keeping_newlines(path, "new\r\nline\n")

    assert path.read_bytes() == b"new\r\nline\r\n"
