"""What an admin is told when the connection fails, instead of a stack trace."""

from __future__ import annotations

import re

import pytest

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


@pytest.fixture
def underprivileged(admin):
    """``underprivileged(*grants)``: a login holding only *grants*, dropped afterwards."""
    login, password = "halfadmin", "halfadmin-test-password"

    def make(*grants: str) -> tuple[str, str]:
        admin(f"CREATE USER {login} IDENTIFIED BY '{password}'", None, None)
        for grant in grants:
            admin(grant.format(login=login), None, None)
        return login, password

    yield make
    admin(f"DROP USER {login}", None, None)


def test_an_admin_without_privileges_is_told_how_to_get_them(enterprise_url, underprivileged, tmp_path, capsys):
    login, password = underprivileged()
    spec = tmp_path / "spaces.toml"
    spec.write_text(SPEC)

    assert main(["--url", enterprise_url, "--user", login, "--password", password, "plan", str(spec)]) == 1

    err = capsys.readouterr().err
    assert f"SET ROLE FOR {login} TO admin" in err
    assert "Traceback" not in err


def test_a_failed_apply_still_hands_out_the_logins_it_created(
    enterprise_url, underprivileged, connect, tmp_path, capsys
):
    # ALL PRIVILEGES carries no label permissions: apply creates carol, then can't write her catalog of spaces.
    login, password = underprivileged("GRANT ALL PRIVILEGES TO USER {login}")
    spec = tmp_path / "spaces.toml"
    spec.write_text('members = ["carol"]\n[spaces.payments]\nmembers = ["carol"]\n')

    assert main(["--url", enterprise_url, "--user", login, "--password", password, "apply", str(spec)]) == 1

    out, err = capsys.readouterr()
    carol = re.search(r"join \S+ carol --password='([^']+)'", out)
    assert carol, out
    assert connect("carol", carol.group(1), "RETURN 1 AS ok") == [{"ok": 1}]
    assert f"SET ROLE FOR {login} TO admin" in err
    assert "Traceback" not in err
