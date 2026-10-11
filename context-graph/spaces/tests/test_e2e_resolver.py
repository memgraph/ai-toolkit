"""The Enterprise resolver against a real Enterprise Memgraph: the login is the user, the main database the space."""

from __future__ import annotations

import pytest

from agent_context_graph import cli, spaces
from agent_context_graph.adapters import _identity
from agent_context_graph.spaces import SpaceError, spaces_for
from context_graph_spaces.provision import apply
from context_graph_spaces.resolver import EnterpriseSpaces
from context_graph_spaces.spec import parse_spec


@pytest.fixture(autouse=True)
def config_file(monkeypatch, tmp_path):
    """A private config file for this test; the resolver and config caches start empty."""
    path = tmp_path / "config.toml"
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(path))
    _identity._reset_cache()
    spaces._reset_cache()
    yield path
    _identity._reset_cache()
    spaces._reset_cache()


@pytest.fixture
def team(admin):
    spec = parse_spec({"members": ["alice", "bob"], "spaces": {"payments": {"members": ["alice"]}}})
    return apply(spec, admin).passwords


def _join(enterprise_url, login, password):
    assert cli.main(["join", enterprise_url, login, "--password", password]) == 0
    return spaces_for(_identity.reload_config())


def test_join_makes_the_login_the_user(enterprise_url, team, capsys):
    resolver = _join(enterprise_url, "alice", team["alice"])

    assert isinstance(resolver, EnterpriseSpaces)
    assert "Personal space: u_alice. Shared spaces: payments." in capsys.readouterr().out
    assert _identity.resolve_user_id({"user_id": "mallory"}) == "alice"
    assert _identity.resolve_memgraph_env()["MEMGRAPH_DATABASE"] == ""


def test_hooks_resolve_their_target_without_connecting(config_file):
    _identity.write_config(
        edition="enterprise",
        memgraph_url="bolt://unreachable.invalid:7687",
        memgraph_user="alice",
        memgraph_password="x",
    )

    assert _identity.resolve_user_id({}) == "alice"
    assert _identity.resolve_memgraph_env()["MEMGRAPH_URL"] == "bolt://unreachable.invalid:7687"


def test_each_login_writes_to_its_own_personal_space(enterprise_url, team, admin):
    from sessions_graph.tool import MemoryTool

    for login in ("alice", "bob"):
        _join(enterprise_url, login, team[login])
        MemoryTool().call(
            {"command": "create", "path": f"/memories/{login}.md", "file_text": f"{login} was here"},
            _identity.load_config(),
        )

    def paths(database):
        return [row["p"] for row in admin("MATCH (m:Memory) RETURN m.path AS p", database, None)]

    assert paths("u_alice") == ["/memories/alice.md"]
    assert paths("u_bob") == ["/memories/bob.md"]


def test_readable_spaces_come_from_the_catalog(enterprise_url, team, admin):
    resolver = _join(enterprise_url, "alice", team["alice"])

    assert [space.name for space in resolver.readable()] == ["personal", "payments"]
    assert resolver.shared_space("payments").target.database == "space_payments"
    with pytest.raises(SpaceError, match="yours: payments"):
        resolver.shared_space("sales")

    apply(parse_spec({"members": ["alice", "bob"]}), admin)
    assert [space.name for space in resolver.readable()] == ["personal", "payments"]  # cached until refreshed
    resolver.refresh()
    assert [space.name for space in resolver.readable()] == ["personal"]


@pytest.mark.parametrize(
    ("line", "key"),
    [('[identity]\nuser_id = "mallory"', "identity.user_id"), ('[memgraph]\ndatabase = "u_bob"', "memgraph.database")],
)
def test_keys_the_login_decides_are_refused(config_file, line, key, capsys):
    config_file.write_text(f'edition = "enterprise"\n{line}\n[memgraph]\nuser = "alice"\n')

    with pytest.raises(SpaceError, match=f"Remove {key}"):
        spaces_for(_identity.reload_config())
    assert cli.main(["doctor", "--runtime", "claude-code"]) == 1
    assert f"Remove {key}" in capsys.readouterr().out


def test_a_real_hook_records_the_session_under_the_login(enterprise_url, team, admin, monkeypatch):
    import io
    import json

    from agent_context_graph.hooks.runner import run_hook
    from agent_context_graph.hooks.runtime_plugin import load_runtime_plugins

    _join(enterprise_url, "alice", team["alice"])
    payload = {"hook_event_name": "SessionStart", "session_id": "alice-s1", "cwd": "/tmp", "user_id": "mallory"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    run_hook(load_runtime_plugins()["claude-code"], ["--connector", "sessions-graph", "--strict"])

    query = "MATCH (u:User)-[:HAD_SESSION]->(s:Session) RETURN u.user_id AS u, s.session_id AS s"
    assert admin(query, "u_alice", None) == [{"u": "alice", "s": "alice-s1"}]
    assert admin(query, "u_bob", None) == []


def test_a_hook_with_a_refused_config_skips_quietly(config_file, monkeypatch, capsys):
    import io

    from agent_context_graph.hooks.runner import run_hook
    from agent_context_graph.hooks.runtime_plugin import load_runtime_plugins

    config_file.write_text('edition = "enterprise"\n[identity]\nuser_id = "mallory"\n[memgraph]\nuser = "alice"\n')
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name": "SessionStart", "session_id": "s"}'))

    assert run_hook(load_runtime_plugins()["claude-code"], ["--connector", "sessions-graph"]) == 0
