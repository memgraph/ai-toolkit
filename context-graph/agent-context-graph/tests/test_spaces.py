"""The space resolver seam: which edition a config selects, and what Community resolves to."""

import pytest

from agent_context_graph import spaces
from agent_context_graph.adapters import _identity
from agent_context_graph.spaces import (
    PERSONAL,
    CommunitySpaces,
    MemgraphTarget,
    SpaceError,
    SpaceResolver,
    spaces_for,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    _identity._reset_cache()
    spaces._reset_cache()
    yield
    _identity._reset_cache()
    spaces._reset_cache()


@pytest.fixture()
def config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "context-graph"
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config_dir / "config.toml"))
    return config_dir


def _config(**overrides):
    return _identity.HookConfig(
        user_id="ante",
        memgraph_url="bolt://db:7687",
        memgraph_user="mg",
        memgraph_password="pw",
        memgraph_database="mine",
        **overrides,
    )


def test_a_config_without_an_edition_is_community():
    resolver = spaces_for(_config())

    assert isinstance(resolver, CommunitySpaces)
    assert isinstance(resolver, SpaceResolver)
    assert resolver.edition == "community"


def test_community_is_exactly_the_configured_database():
    resolver = spaces_for(_config(edition="community"))

    personal = resolver.personal()
    assert personal.name == PERSONAL
    assert not personal.shared
    assert personal.target == MemgraphTarget(url="bolt://db:7687", user="mg", password="pw", database="mine")
    assert resolver.readable() == [personal]
    assert resolver.user_id() == "ante"


def test_community_honours_a_payload_user_id():
    assert spaces_for(_config()).user_id({"user_id": "from-payload"}) == "from-payload"
    assert spaces_for(_config()).user_id({"user_id": ""}) == "ante"


def test_community_has_no_shared_spaces():
    with pytest.raises(SpaceError, match="needs the Enterprise edition"):
        spaces_for(_config()).shared_space("payments")


def test_enterprise_without_its_package_says_what_to_install(monkeypatch):
    monkeypatch.setattr(spaces, "_edition_factory", lambda edition: None)

    with pytest.raises(SpaceError, match="context-graph-spaces"):
        spaces_for(_config(edition="enterprise"))


def test_an_unknown_edition_is_an_error():
    with pytest.raises(SpaceError, match="Unknown edition 'team'"):
        spaces_for(_config(edition="team"))


def test_the_resolver_is_kept_for_the_same_config_and_rebuilt_for_a_new_one():
    config = _config()

    assert spaces_for(config) is spaces_for(config)
    assert spaces_for(_config()) is not spaces_for(config)


def test_the_target_speaks_memgraph_toolboxs_env_contract():
    target = MemgraphTarget(url="bolt://db:7687", user="mg", password="pw", database=None)

    assert target.env() == {
        "MEMGRAPH_URL": "bolt://db:7687",
        "MEMGRAPH_USER": "mg",
        "MEMGRAPH_PASSWORD": "pw",
        "MEMGRAPH_DATABASE": "",
    }
    assert target.client_kwargs() == {"url": "bolt://db:7687", "username": "mg", "password": "pw", "database": None}


# --- edition in the config file ---


def test_edition_is_a_top_level_key(config_dir):
    _identity.write_config(edition="enterprise", user_id="ante")

    text = (config_dir / "config.toml").read_text()
    assert text.index('edition = "enterprise"') < text.index("[identity]")
    assert _identity.load_config().edition == "enterprise"


def test_a_config_without_edition_writes_none(config_dir):
    _identity.write_config(user_id="ante")

    lines = (config_dir / "config.toml").read_text().splitlines()
    assert not any(line.startswith("edition") for line in lines)
    assert _identity.load_config().edition is None


def test_bootstrap_keeps_the_edition(config_dir):
    _identity.write_config(edition="enterprise")
    _identity.write_full_config(user_id="ante")

    assert _identity.load_config().edition == "enterprise"


def test_hooks_resolve_through_the_community_resolver_unchanged(config_dir):
    _identity.write_full_config(
        user_id="ante", memgraph_url="bolt://db:7687", memgraph_user="mg", memgraph_password="pw"
    )

    assert _identity.resolve_user_id({}) == "ante"
    assert _identity.resolve_memgraph_env() == {
        "MEMGRAPH_URL": "bolt://db:7687",
        "MEMGRAPH_USER": "mg",
        "MEMGRAPH_PASSWORD": "pw",
        "MEMGRAPH_DATABASE": "memgraph",
    }
    assert _identity.resolve_memgraph_env(database="other")["MEMGRAPH_DATABASE"] == "other"
