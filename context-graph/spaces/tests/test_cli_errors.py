"""What an admin is told when the connection fails, instead of a stack trace."""

from __future__ import annotations

from context_graph_spaces.cli import main

SPEC = 'members = ["ante"]\n'


def test_an_unreachable_memgraph_says_so(tmp_path, capsys):
    spec = tmp_path / "spaces.toml"
    spec.write_text(SPEC)

    assert main(["--url", "bolt://127.0.0.1:1", "plan", str(spec)]) == 1

    err = capsys.readouterr().err
    assert "Can't reach Memgraph at bolt://127.0.0.1:1" in err
    assert "Traceback" not in err


def test_a_refused_admin_login_says_how_to_pass_one(enterprise_url, tmp_path, capsys):
    spec = tmp_path / "spaces.toml"
    spec.write_text(SPEC)

    assert main(["--url", enterprise_url, "--user", "nobody", "--password", "wrong", "plan", str(spec)]) == 1

    err = capsys.readouterr().err
    assert "refused the admin login 'nobody'" in err
    assert "MEMGRAPH_USER" in err
