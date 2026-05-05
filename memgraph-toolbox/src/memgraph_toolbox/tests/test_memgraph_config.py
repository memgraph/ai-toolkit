from memgraph_toolbox.api.memgraph import MEMGRAPH_ENV_DEFAULTS, MEMGRAPH_ENV_KEYS, memgraph_env


def test_memgraph_env_uses_defaults_without_environment():
    assert memgraph_env(environ={}) == MEMGRAPH_ENV_DEFAULTS
    assert tuple(MEMGRAPH_ENV_DEFAULTS) == MEMGRAPH_ENV_KEYS


def test_memgraph_env_prefers_explicit_values_over_environment():
    env = {
        "MEMGRAPH_URL": "bolt://env:7687",
        "MEMGRAPH_USER": "env-user",
        "MEMGRAPH_PASSWORD": "env-password",
        "MEMGRAPH_DATABASE": "env-db",
    }

    assert memgraph_env(
        url="bolt://arg:7687",
        username="arg-user",
        password="arg-password",
        database="arg-db",
        environ=env,
    ) == {
        "MEMGRAPH_URL": "bolt://arg:7687",
        "MEMGRAPH_USER": "arg-user",
        "MEMGRAPH_PASSWORD": "arg-password",
        "MEMGRAPH_DATABASE": "arg-db",
    }


def test_memgraph_env_allows_empty_explicit_auth_values():
    env = {
        "MEMGRAPH_USER": "env-user",
        "MEMGRAPH_PASSWORD": "env-password",
    }

    values = memgraph_env(username="", password="", environ=env)

    assert values["MEMGRAPH_USER"] == ""
    assert values["MEMGRAPH_PASSWORD"] == ""
