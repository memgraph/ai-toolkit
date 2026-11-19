"""``memgraph-context-graph join`` against a real Enterprise Memgraph: what a member sees, right or wrong."""

from __future__ import annotations

import pytest

from context_graph_spaces.provision import apply
from context_graph_spaces.spec import parse_spec
from memgraph_context_graph import cli, spaces
from memgraph_context_graph.adapters import _identity

from .conftest import ADMIN

SPEC = {"members": ["alice", "bob", "carol"], "spaces": {"payments": {"members": ["alice", "bob"]}}}


@pytest.fixture
def team(admin):
    return apply(parse_spec(SPEC), admin).passwords


def _join(url: str, login: str, *password: str) -> int:
    _identity._reset_cache()
    spaces._reset_cache()
    return cli.main(["join", url, login, *(f"--password={p}" for p in password)])


def test_join_asks_for_the_password_when_it_isnt_given(enterprise_url, team, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr("getpass.getpass", lambda prompt: asked.append(prompt) or team["alice"])

    assert _join(enterprise_url, "alice") == 0

    assert asked == ["Password for alice: "]
    assert "Personal space: u_alice." in capsys.readouterr().out


def test_a_member_without_shared_spaces_is_told_none_yet(enterprise_url, team, capsys):
    assert _join(enterprise_url, "carol", team["carol"]) == 0
    assert "Personal space: u_carol. Shared spaces: none yet." in capsys.readouterr().out


def test_a_wrong_password_says_the_login_doesnt_work(enterprise_url, team, capsys):
    assert _join(enterprise_url, "alice", "not-her-password") == 1

    err = capsys.readouterr().err
    assert "Memgraph refused the login 'alice'; check its password" in err
    assert "Traceback" not in err


def test_an_unreachable_memgraph_says_so(team, capsys):
    assert _join("bolt://127.0.0.1:1", "alice", team["alice"]) == 1

    err = capsys.readouterr().err
    assert "the login doesn't work yet" in err
    assert "Traceback" not in err


def test_a_login_spaces_never_provisioned_is_told_so(enterprise_url, team, capsys):
    assert _join(enterprise_url, *ADMIN) == 1

    err = capsys.readouterr().err
    assert f"The login {ADMIN[0]!r} has no personal space" in err
    assert f"add {ADMIN[0]} to the spaces spec's members" in err
    assert "Traceback" not in err


def test_joining_as_another_login_switches_who_you_are_and_keeps_the_rest(enterprise_url, team, capsys):
    assert _join(enterprise_url, "alice", team["alice"]) == 0
    _identity.write_config(memory_backend=_identity.MEMORY_BACKEND_GRAPH)
    capsys.readouterr()

    assert _join(enterprise_url, "bob", team["bob"]) == 0

    assert "Personal space: u_bob." in capsys.readouterr().out
    assert _identity.resolve_user_id({"user_id": "alice"}) == "bob"
    assert _identity.reload_config().memory_backend == _identity.MEMORY_BACKEND_GRAPH


def test_joining_again_shows_the_spaces_granted_since(enterprise_url, team, admin, capsys):
    assert _join(enterprise_url, "carol", team["carol"]) == 0
    capsys.readouterr()

    granted = {**SPEC, "spaces": {"payments": {"members": ["alice", "bob", "carol"]}}}
    apply(parse_spec(granted), admin)

    assert _join(enterprise_url, "carol", team["carol"]) == 0
    assert "Shared spaces: payments." in capsys.readouterr().out


def test_join_for_one_agent_moves_only_that_agent(enterprise_url, team, capsys):
    _identity.write_config(memgraph_url="bolt://localhost:7687", user_id="alice-local")

    _identity._reset_cache()
    spaces._reset_cache()
    assert cli.main(["join", enterprise_url, "alice", f"--password={team['alice']}", "--agent", "codex"]) == 0

    out = capsys.readouterr().out
    assert "in profile" in out and "for codex" in out
    assert "Next: memgraph-context-graph setup codex --memory-backend context-graph" in out
    _identity._reset_cache()
    _identity.select_agent("codex")
    codex = _identity.load_config()
    assert (codex.edition, codex.memgraph_user) == ("enterprise", "alice")
    assert spaces.spaces_for(codex).personal().target.url == enterprise_url
    _identity._reset_cache()
    _identity.select_agent("claude-code")
    claude = _identity.load_config()
    assert (claude.edition, claude.memgraph_url, claude.user_id) == (None, "bolt://localhost:7687", "alice-local")
    _identity._reset_cache()
    spaces._reset_cache()
