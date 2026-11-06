"""``init`` and ``compose``: getting a spaces server running and ready, before any spec is applied."""

from __future__ import annotations

import pytest

from context_graph_spaces.cli import main
from context_graph_spaces.instance import ADMIN_PRIVILEGES, _embeddings, create_first_admin

from .conftest import ADMIN


def test_init_checks_an_instance_that_already_has_its_admin(enterprise_url, admin, capsys):
    assert main(["--url", enterprise_url, "--user", ADMIN[0], "--password", ADMIN[1], "init"]) == 0

    out = capsys.readouterr().out
    assert "Created" not in out
    for check in ("Enterprise licence", "Admin privileges", "Embeddings", "Memory limit"):
        assert f"ok   {check}: " in out
    assert "BAAI/bge-small-en-v1.5 loads (384 dimensions)" in out
    assert "Ready for spaces" in out


def test_init_accepts_an_admin_with_the_privileges_granted_directly_and_no_role(enterprise_url, admin, capsys):
    login, password = "directadmin", "directadmin-test-password"
    admin(f"CREATE USER {login} IDENTIFIED BY '{password}'", None, None)
    admin(f"GRANT {', '.join(ADMIN_PRIVILEGES)} TO {login}", None, None)
    admin(f"GRANT DATABASE * TO {login}", None, None)
    try:
        assert main(["--url", enterprise_url, "--user", login, "--password", password, "init"]) == 0
    finally:
        admin(f"DROP USER {login}", None, None)

    out = capsys.readouterr().out
    assert f"ok   Admin privileges: {login} holds every privilege init and apply use, on every database" in out


def test_init_creates_no_second_admin_and_asks_for_the_existing_one(enterprise_url, admin, capsys):
    assert create_first_admin(enterprise_url, "cgadmin") is None
    assert "cgadmin" not in {row["user"] for row in admin("SHOW USERS", None, None)}

    assert main(["--url", enterprise_url, "init"]) == 1
    assert "already has logins, so init created none" in capsys.readouterr().err


def test_init_tells_an_admin_without_the_admin_role_how_to_get_it(enterprise_url, admin, capsys):
    login, password = "halfadmin", "halfadmin-test-password"
    admin(f"CREATE USER {login} IDENTIFIED BY '{password}'", None, None)
    try:
        assert main(["--url", enterprise_url, "--user", login, "--password", password, "init"]) == 1
    finally:
        admin(f"DROP USER {login}", None, None)

    out, err = capsys.readouterr()
    assert f"SET ROLE FOR {login} TO admin" in out + err
    assert f"GRANT DATABASE * TO {login}" in out + err
    assert "Traceback" not in err


def test_init_says_why_the_model_wont_load_when_the_procedure_is_there():
    import neo4j

    def query(statement, database, params):
        if statement.startswith("CALL mg.procedures()"):
            return [{"name": "embeddings.text"}]
        if statement.startswith("CALL embeddings.text"):
            return [{"success": False, "embeddings": None}]
        # embeddings.model_info re-raises what the download hit, with its traceback.
        raise neo4j.exceptions.ClientError(
            "Traceback (most recent call last):\n  File ...\n"
            "OSError: [Errno 30] Read-only file system: '/home/memgraph/.cache'\n}"
        )

    check = _embeddings(query)

    assert not check.ok
    assert "can't load BAAI/bge-small-en-v1.5: OSError: [Errno 30] Read-only file system" in check.detail
    assert "HF_HOME=/var/lib/memgraph/huggingface" in check.detail


@pytest.mark.parametrize("name", ["admin", "cg_admin", "Admin"])
def test_init_refuses_an_admin_name_that_would_clash(name, capsys):
    assert main(["--url", "bolt://127.0.0.1:1", "init", "--admin", name]) == 2
    assert "--admin" in capsys.readouterr().err


def test_compose_sizes_memgraph_below_its_container_and_holds_the_worker_back(capsys):
    from memgraph_context_graph.setup.memgraph import IMAGE

    args = ["compose", "--bind", "10.0.0.5", "--port", "7688", "--memory-gib", "4", "--spec", "/srv/spaces.toml"]
    assert main(args) == 0

    out = capsys.readouterr().out
    assert f"image: {IMAGE}" in out
    assert '"10.0.0.5:7688:7687"' in out
    assert "mem_limit: 4g" in out and "--memory-limit=3584" in out
    assert 'profiles: ["worker"]' in out
    assert "/srv/spaces.toml:/spec/spaces.toml:ro" in out
    assert '"reconcile", "/spec/spaces.toml", "--watch"' in out


def test_compose_refuses_a_container_too_small_for_recall(capsys):
    assert main(["compose", "--memory-gib", "1"]) == 2
    assert "at least 2" in capsys.readouterr().err
