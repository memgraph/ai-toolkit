from memgraph_toolbox.api.memgraph import (
    MEMGRAPH_ENV_DEFAULTS,
    MEMGRAPH_ENV_KEYS,
    memgraph_env,
    to_routing_url,
)


def test_memgraph_env_uses_defaults_without_environment():
    assert memgraph_env(environ={}) == MEMGRAPH_ENV_DEFAULTS
    assert tuple(MEMGRAPH_ENV_DEFAULTS) == MEMGRAPH_ENV_KEYS


def test_memgraph_env_prefers_explicit_values_over_environment():
    env = {
        "MEMGRAPH_URL": "bolt://env:7687",
        "MEMGRAPH_USER": "env-user",
        "MEMGRAPH_PASSWORD": "env-password",
        "MEMGRAPH_DATABASE": "env-db",
        "MEMGRAPH_HA_CLUSTER": "false",
    }

    assert memgraph_env(
        url="bolt://arg:7687",
        username="arg-user",
        password="arg-password",
        database="arg-db",
        ha_cluster=True,
        environ=env,
    ) == {
        "MEMGRAPH_URL": "bolt://arg:7687",
        "MEMGRAPH_USER": "arg-user",
        "MEMGRAPH_PASSWORD": "arg-password",
        "MEMGRAPH_DATABASE": "arg-db",
        "MEMGRAPH_HA_CLUSTER": "true",
    }


def test_memgraph_env_allows_empty_explicit_auth_values():
    env = {
        "MEMGRAPH_USER": "env-user",
        "MEMGRAPH_PASSWORD": "env-password",
    }

    values = memgraph_env(username="", password="", environ=env)

    assert values["MEMGRAPH_USER"] == ""
    assert values["MEMGRAPH_PASSWORD"] == ""


def test_memgraph_env_reads_ha_cluster_flag_from_environment():
    assert memgraph_env(environ={"MEMGRAPH_HA_CLUSTER": "true"})["MEMGRAPH_HA_CLUSTER"] == "true"
    assert memgraph_env(environ={})["MEMGRAPH_HA_CLUSTER"] == "false"


def test_memgraph_env_explicit_ha_cluster_wins_over_environment():
    env = {"MEMGRAPH_HA_CLUSTER": "true"}
    assert memgraph_env(ha_cluster=False, environ=env)["MEMGRAPH_HA_CLUSTER"] == "false"


def test_to_routing_url_upgrades_bolt_to_neo4j():
    assert to_routing_url("bolt://mg-coordinators:7687") == "neo4j://mg-coordinators:7687"
    assert to_routing_url("bolt+s://mg-coordinators:7687") == "neo4j+s://mg-coordinators:7687"


def test_to_routing_url_leaves_routing_scheme_unchanged():
    assert to_routing_url("neo4j://mg-coordinators:7687") == "neo4j://mg-coordinators:7687"
    assert to_routing_url("neo4j+s://mg-coordinators:7687") == "neo4j+s://mg-coordinators:7687"
