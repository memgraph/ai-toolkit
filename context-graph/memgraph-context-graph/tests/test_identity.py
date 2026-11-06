"""Tests for memgraph_context_graph.adapters._identity (Hook Configuration)."""

import pytest

from memgraph_context_graph.adapters import _identity


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
    # Uses the supported override (ADR 0003) rather than monkeypatching module
    # privates, so these tests exercise the same path a real isolated session
    # takes instead of a shape only tests can produce.
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config_dir / "config.toml"))
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
    _identity.write_config(
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
    _identity.write_config(memgraph_url="bolt://config:7687")
    _identity._reset_cache()
    env = _identity.resolve_memgraph_env(url="bolt://flag:9999")
    assert env["MEMGRAPH_URL"] == "bolt://flag:9999"


# --- resolve_llm_env ---


def test_llm_defaults_when_no_config(config_dir):
    env = _identity.resolve_llm_env()
    assert env["OPENAI_API_KEY"] == ""
    assert env["ANTHROPIC_API_KEY"] == ""


def test_llm_from_config_file(config_dir):
    _identity.write_config(openai_api_key="sk-openai-secret", anthropic_api_key="sk-anthropic-secret")
    _identity._reset_cache()
    env = _identity.resolve_llm_env()
    assert env["OPENAI_API_KEY"] == "sk-openai-secret"
    assert env["ANTHROPIC_API_KEY"] == "sk-anthropic-secret"


def test_write_config_preserves_llm_keys(config_dir):
    _identity.write_config(openai_api_key="sk-openai-secret")
    _identity._reset_cache()
    _identity.write_config(user_id="someone")
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.openai_api_key == "sk-openai-secret"
    assert config.user_id == "someone"


# --- [reconcile] auto_reconcile ---


def test_auto_reconcile_is_off_when_never_configured(config_dir):
    """Off when never configured -- ADR 0002 leaves no ambient env var to fall back to."""
    assert not _identity.load_config().auto_reconcile


def test_auto_reconcile_from_config_file(config_dir):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()
    assert _identity.load_config().auto_reconcile is True


def test_auto_reconcile_explicit_false_is_not_collapsed_to_none(config_dir):
    _identity.write_config(auto_reconcile=False)
    _identity._reset_cache()
    assert _identity.load_config().auto_reconcile is False


def test_write_config_preserves_auto_reconcile(config_dir):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()
    _identity.write_config(user_id="someone")
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.auto_reconcile is True
    assert config.user_id == "someone"


def test_write_config_can_toggle_auto_reconcile_off(config_dir):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()
    _identity.write_config(auto_reconcile=False)
    _identity._reset_cache()
    assert _identity.load_config().auto_reconcile is False


def test_write_full_config_preserves_auto_reconcile_when_not_given(config_dir):
    """Simulates re-running `bootstrap`: it must not silently revert
    reconcile.auto_reconcile to off just because it doesn't pass it (finding 2)."""
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()
    _identity.write_config(user_id="rebootstrapped")
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.auto_reconcile is True
    assert config.user_id == "rebootstrapped"


def test_write_full_config_can_explicitly_set_auto_reconcile(config_dir):
    _identity.write_config(auto_reconcile=True)
    _identity._reset_cache()
    _identity.write_config(auto_reconcile=False)
    _identity._reset_cache()
    assert _identity.load_config().auto_reconcile is False


# --- parse_bool_flag ---


def test_parse_bool_flag_truthy_values():
    for value in ("1", "true", "True", "yes", "YES", "on"):
        assert _identity.parse_bool_flag(value) is True


def test_parse_bool_flag_falsy_values():
    for value in ("0", "false", "no", "off", "", "garbage"):
        assert _identity.parse_bool_flag(value) is False


def test_parse_bool_flag_passes_real_bool_through(config_dir):
    assert _identity.parse_bool_flag(True) is True
    assert _identity.parse_bool_flag(False) is False


# --- _parse_toml inline comments ---


def test_parse_toml_strips_inline_comment_on_bare_value(config_dir, tmp_path):
    f = tmp_path / "test.toml"
    f.write_text("[reconcile]\nauto_reconcile = true  # off for now\n")
    sections = _identity._parse_toml(f)
    assert sections["reconcile"]["auto_reconcile"] == "true"


def test_parse_toml_leaves_quoted_value_with_hash_alone(config_dir, tmp_path):
    f = tmp_path / "test.toml"
    f.write_text('[identity]\nuser_id = "user#123"\n')
    sections = _identity._parse_toml(f)
    assert sections["identity"]["user_id"] == "user#123"


# --- write_config ---


def test_write_config_creates_file_with_0600(config_dir):
    path = _identity.write_config(user_id="testuser")
    assert path.is_file()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_write_config_preserves_existing_values(config_dir):
    _identity.write_config(user_id="original", memgraph_url="bolt://first:7687")
    _identity._reset_cache()
    _identity.write_config(user_id="updated")
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.user_id == "updated"
    assert config.memgraph_url == "bolt://first:7687"


def test_write_full_config_writes_all_defaults(config_dir):
    _identity.write_config()
    _identity._reset_cache()
    config = _identity.load_config()
    assert config.user_id is None  # empty string -> None
    assert config.memgraph_url == "bolt://localhost:7687"
    assert config.memgraph_database == "memgraph"
    assert config.auto_reconcile is None


# --- load_config ---


def test_load_config_caches(config_dir):
    _identity.write_config(user_id="cached")
    config1 = _identity.load_config()
    # Modify file behind the cache's back.
    _identity.config_file().write_text('[identity]\nuser_id = "changed"\n')
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


# --- [capture], [extraction], [llm] derive_model ---


def test_capture_connectors_unset_is_none(config_dir):
    _identity.write_config(user_id="u")
    assert _identity.load_config().connectors is None


def test_capture_connectors_round_trip_and_survive_other_writes(config_dir):
    _identity.write_config(connectors=("actions-graph", "sessions-graph"))
    _identity.write_config(user_id="u")
    _identity.write_config(user_id="v")
    assert _identity.load_config().connectors == ("actions-graph", "sessions-graph")


def test_capture_connectors_empty_means_none_enabled(config_dir):
    _identity.write_config(connectors=())
    assert _identity.load_config().connectors == ()


def test_extraction_and_derive_model_survive_rewrites(config_dir):
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        '[llm]\nderive_model = "claude-opus-5-5"\n\n[extraction]\nmodel = "my/checkpoint"\nentity_threshold = "0.4"\n'
    )
    _identity.write_config(user_id="u")
    _identity.write_config(user_id="v")

    assert _identity.resolve_derive_model() == "claude-opus-5-5"
    assert _identity.resolve_extraction() == {"model": "my/checkpoint", "entity_threshold": "0.4"}


# --- profiles: one agent's memory elsewhere ---

_PROFILED = """\
[identity]
user_id = "ana"

[memgraph]
url = "bolt://localhost:7687"
database = "memgraph"

[capture]
redact = "true"

[profiles.team]
edition = "enterprise"

[profiles.team.memgraph]
url = "neo4j://memgraph.corp:7687"
user = "ana"
password = "secret"

[profiles.team.memory]
backend = "context-graph"

[agents]
codex = "team"
"""


@pytest.fixture()
def profiled(config_dir):
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.toml"
    path.write_text(_PROFILED)
    return path


def test_a_file_without_profiles_resolves_the_top_level_for_every_agent(config_dir):
    _identity.write_config(memgraph_url="bolt://remote:7687")
    _identity.select_agent("codex")

    config = _identity.load_config()
    assert config.memgraph_url == "bolt://remote:7687"
    assert config.profile is None
    assert "[agents]" not in (config_dir / "config.toml").read_text()


def test_an_agent_mapped_to_a_profile_resolves_its_destination(profiled):
    _identity.select_agent("codex")

    config = _identity.load_config()
    assert config.profile == "team"
    assert config.edition == "enterprise"
    assert (config.memgraph_url, config.memgraph_user) == ("neo4j://memgraph.corp:7687", "ana")
    assert config.graph_memory
    assert config.problems == ()


def test_an_agent_not_mapped_keeps_the_top_level(profiled):
    _identity.select_agent("claude-code")

    config = _identity.load_config()
    assert config.profile is None
    assert (config.edition, config.memgraph_url, config.memory_backend) == (None, "bolt://localhost:7687", None)
    assert config.user_id == "ana"


def test_a_profile_is_a_whole_destination_and_privacy_stays_shared(profiled):
    _identity.select_agent("codex")

    config = _identity.load_config()
    # The top level's database and identity would override what the Enterprise login decides.
    assert "memgraph.database" not in config.present
    assert "identity.user_id" not in config.present
    assert config.capture_settings == {"redact": "true"}


def test_writes_while_a_profile_applies_go_to_it_and_keep_the_rest(profiled):
    _identity.select_agent("codex")

    _identity.write_config(memory_backend="native", connectors=("sessions-graph",))

    _identity.select_agent("claude-code")
    top = _identity.load_config()
    assert (top.memory_backend, top.connectors) == (None, ("sessions-graph",))
    _identity.select_agent("codex")
    team = _identity.load_config()
    assert (team.memory_backend, team.memgraph_password, team.connectors) == ("native", "secret", ("sessions-graph",))
    assert team.agents == {"codex": "team"}


def test_a_rewrite_keeps_profiles_and_agents_as_written(profiled):
    _identity.write_config(auto_reconcile=True)

    _identity.select_agent("codex")
    config = _identity.load_config()
    assert config.memgraph_url == "neo4j://memgraph.corp:7687"
    assert config.profiles["team"]["memory.backend"] == "context-graph"


def test_select_profile_creates_the_profile_on_first_write(config_dir):
    _identity.write_config(memgraph_url="bolt://localhost:7687")
    _identity.select_profile("second")

    _identity.write_config(memgraph_url="bolt://localhost:7688")
    _identity.map_agents(["cursor"], "second")

    _identity.select_profile(None)
    _identity.select_agent("cursor")
    assert _identity.load_config().memgraph_url == "bolt://localhost:7688"
    _identity.select_agent("codex")
    assert _identity.load_config().memgraph_url == "bolt://localhost:7687"


def test_mapping_an_agent_to_default_moves_it_back_to_the_top_level(profiled):
    _identity.map_agents(["codex"], _identity.DEFAULT_PROFILE)
    _identity.select_agent("codex")

    assert _identity.load_config().memgraph_url == "bolt://localhost:7687"
    assert "team" in _identity.load_config().profiles


def test_an_agent_mapped_to_a_missing_profile_is_a_problem(profiled):
    profiled.write_text(_PROFILED.replace('codex = "team"', 'codex = "tema"'))
    _identity.select_agent("codex")

    config = _identity.load_config()
    assert config.memgraph_url == "bolt://localhost:7687"
    assert config.problems == (
        "[agents] codex names profile 'tema', which the file doesn't define; using the top level",
    )


def test_a_profile_cannot_change_how_capture_behaves(profiled):
    profiled.write_text(_PROFILED + '\n[profiles.team.capture]\nredact = "false"\n')
    _identity.select_agent("codex")

    config = _identity.load_config()
    assert config.capture_settings == {"redact": "true"}
    assert config.problems == ("[profiles.team.capture] redact can't differ per profile and was ignored",)


def test_a_spawned_process_resolves_the_profile_its_parent_used(profiled, monkeypatch):
    _identity.select_agent("codex")
    env = _identity.spawn_env()
    assert env[_identity.PROFILE_ENV] == "team"

    _identity._reset_cache()
    monkeypatch.setenv(_identity.PROFILE_ENV, env[_identity.PROFILE_ENV])
    assert _identity.load_config().memgraph_url == "neo4j://memgraph.corp:7687"


def test_the_top_level_is_handed_down_by_name_over_an_ambient_profile(profiled, monkeypatch):
    monkeypatch.setenv(_identity.PROFILE_ENV, "team")
    _identity.select_profile(_identity.DEFAULT_PROFILE)

    assert _identity.load_config().profile is None
    assert _identity.spawn_env()[_identity.PROFILE_ENV] == _identity.DEFAULT_PROFILE


@pytest.mark.parametrize("name", ["default", "a.b", "with space", ""])
def test_a_profile_name_is_checked(name):
    with pytest.raises(ValueError):
        _identity.check_profile_name(name)
