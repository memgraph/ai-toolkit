"""Tests for agent_context_graph.adapters._identity (Hook Configuration)."""

import pytest

from agent_context_graph.adapters import _identity


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset module-level cache before each test."""
    _identity._reset_cache()
    yield
    _identity._reset_cache()


@pytest.fixture()
def config_dir(monkeypatch, tmp_path):
    """Point config at a temp directory."""
    config_dir = tmp_path / "context-graph"
    config_file = config_dir / "config.toml"
    monkeypatch.setattr(_identity, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(_identity, "_CONFIG_FILE", config_file)
    return config_dir


# --- resolve_user_id ---


def test_payload_user_id_takes_priority(config_dir):
    _identity.write_config(user_id="config-user")
    result = _identity.resolve_user_id({"user_id": "payload-user"})
    assert result == "payload-user"


def test_config_file_fallback(config_dir):
    _identity.write_config(user_id="config-user")
    result = _identity.resolve_user_id({})
    assert result == "config-user"


def test_returns_none_when_no_config(config_dir):
    result = _identity.resolve_user_id({})
    assert result is None


# --- resolve_memgraph_env ---


def test_memgraph_defaults_when_no_config(config_dir):
    env = _identity.resolve_memgraph_env()
    assert env["MEMGRAPH_URL"] == "bolt://localhost:7687"
    assert env["MEMGRAPH_USER"] == ""
    assert env["MEMGRAPH_PASSWORD"] == ""
    assert env["MEMGRAPH_DATABASE"] == "memgraph"


def test_memgraph_from_config_file(config_dir):
    _identity.write_full_config(
        memgraph_url="bolt://remote:7687",
        memgraph_user="admin",
        memgraph_password="secret",
        memgraph_database="mydb",
    )
    _identity._reset_cache()
    env = _identity.resolve_memgraph_env()
    assert env["MEMGRAPH_URL"] == "bolt://remote:7687"
    assert env["MEMGRAPH_USER"] == "admin"
    assert env["MEMGRAPH_PASSWORD"] == "secret"
    assert env["MEMGRAPH_DATABASE"] == "mydb"


def test_cli_flag_overrides_config(config_dir):
    _identity.write_full_config(memgraph_url="bolt://config:7687")
    _identity._reset_cache()
    env = _identity.resolve_memgraph_env(url="bolt://flag:9999")
    assert env["MEMGRAPH_URL"] == "bolt://flag:9999"


# --- write_config ---


def test_write_config_creates_file_with_0600(config_dir):
    path = _identity.write_config(user_id="testuser")
    assert path.is_file()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_write_config_preserves_existing_values(config_dir):
    _identity.write_full_config(user_id="original", memgraph_url="bolt://first:7687")
    _identity._reset_cache()
    _identity.write_config(user_id="updated")
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.user_id == "updated"
    assert config.memgraph_url == "bolt://first:7687"


def test_write_full_config_writes_all_defaults(config_dir):
    _identity.write_full_config()
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.user_id is None  # empty string -> None
    assert config.memgraph_url == "bolt://localhost:7687"
    assert config.memgraph_database == "memgraph"


# --- load_config ---


def test_load_config_caches(config_dir):
    _identity.write_config(user_id="cached")
    config1 = _identity.load_config()
    # Modify file behind the cache's back.
    _identity._CONFIG_FILE.write_text('[identity]\nuser_id = "changed"\n')
    config2 = _identity.load_config()
    # Should still return cached value.
    assert config1.user_id == config2.user_id == "cached"


# --- _parse_toml ---


def test_parse_toml_handles_comments(config_dir, tmp_path):
    f = tmp_path / "test.toml"
    f.write_text('# comment\n[identity]\n# another\nuser_id = "hello"\n')
    sections = _identity._parse_toml(f)
    assert sections["identity"]["user_id"] == "hello"


def test_parse_toml_handles_bare_values(config_dir, tmp_path):
    f = tmp_path / "test.toml"
    f.write_text("[memgraph]\nurl = bolt://localhost:7687\n")
    sections = _identity._parse_toml(f)
    assert sections["memgraph"]["url"] == "bolt://localhost:7687"
