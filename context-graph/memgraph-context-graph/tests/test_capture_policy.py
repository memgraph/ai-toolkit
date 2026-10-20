"""The capture policy's rewriting rules; what reaches a real graph is in actions-graph's test_e2e_capture_policy."""

import pytest

from memgraph_context_graph import AgentLink
from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.capture_policy import DEFAULT_MAX_CHARS, CapturePolicy, policy_from_config
from memgraph_context_graph.events import MessageEvent, SessionStartEvent, ToolEndEvent, ToolStartEvent
from memgraph_context_graph.protocols import GraphConnector

# Assembled, so this file holds no string a secret scanner would flag.
_GITHUB = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
_ANTHROPIC = "sk-ant-" + "api03-abcdefghijklmnopqrstuv"
_AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


@pytest.mark.parametrize(
    ("secret", "name"),
    [
        (_GITHUB, "github_token"),
        (_ANTHROPIC, "anthropic_key"),
        ("sk-proj-" + "abcdefghijklmnopqrstuvwx", "openai_key"),
        (_AWS, "aws_access_key"),
        ("xoxb-" + "123456789012-abcdef", "slack_token"),
        ("sk_live_" + "abcdefghijklmnop1234", "stripe_key"),
        ("AIza" + "B" * 35, "google_api_key"),
        ("eyJhbGciOiJI.eyJzdWIiOiIx.c2lnbmF0dXJl", "jwt"),
    ],
)
def test_builtin_patterns_redact_known_credential_shapes(secret, name):
    event = CapturePolicy().apply(ToolStartEvent(session_id="s", tool_name="Bash", tool_input={"command": secret}))

    assert event.tool_input == {"command": f"[REDACTED:{name}]"}


def test_private_keys_and_url_passwords_are_redacted():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    text = f"key:\n{pem}\nurl: postgres://admin:hunter22@db:5432/x"

    event = CapturePolicy().apply(ToolEndEvent(session_id="s", tool_name="Read", result=text))

    assert event.result == "key:\n[REDACTED:private_key]\nurl: postgres://[REDACTED:url_password]@db:5432/x"


def test_nested_values_are_redacted_and_identifiers_kept():
    event = ToolEndEvent(
        session_id="s-1",
        tool_name="Bash",
        tool_use_id="tu-1",
        result=[{"text": f"token {_GITHUB}"}],
        metadata={"env": {"GH": _GITHUB}},
    )

    cleaned = CapturePolicy().apply(event)

    assert cleaned.result == [{"text": "token [REDACTED:github_token]"}]
    assert cleaned.metadata == {"env": {"GH": "[REDACTED:github_token]"}}
    assert (cleaned.session_id, cleaned.tool_use_id, cleaned.tool_name) == ("s-1", "tu-1", "Bash")
    assert event.result == [{"text": f"token {_GITHUB}"}], "the emitted event itself is left alone"


def test_private_blocks_are_stripped_and_an_unclosed_one_hides_the_rest():
    policy = CapturePolicy()

    closed = policy.apply(MessageEvent(session_id="s", role="user", content="fix <private>my salary</private> bug"))
    unclosed = policy.apply(MessageEvent(session_id="s", role="user", content="deploy <PRIVATE>and the rest"))

    assert closed.content == "fix  bug"
    assert unclosed.content == "deploy "


def test_a_message_that_was_only_private_is_dropped():
    policy = CapturePolicy()

    assert policy.apply(MessageEvent(session_id="s", role="user", content=" <private>diary</private> ")) is None
    assert policy.apply(MessageEvent(session_id="s", role="user", content="")) is not None


def test_long_strings_are_clipped():
    event = CapturePolicy(max_chars=10).apply(ToolEndEvent(session_id="s", tool_name="Read", result="x" * 25))

    assert event.result == "x" * 10 + "…[truncated 15 chars]"


def test_excluded_tools_are_dropped_by_name_or_glob():
    policy = CapturePolicy(exclude_tools=("TodoWrite", "mcp__secrets__*"))

    assert policy.apply(ToolStartEvent(session_id="s", tool_name="TodoWrite")) is None
    assert policy.apply(ToolEndEvent(session_id="s", tool_name="mcp__secrets__get")) is None
    assert policy.apply(ToolStartEvent(session_id="s", tool_name="Bash")) is not None


def test_excluded_projects_cover_the_path_and_everything_under_it(tmp_path):
    policy = CapturePolicy(exclude_projects=(str(tmp_path / "secret"), "~/clients/*"))

    assert policy.excludes_project(str(tmp_path / "secret"))
    assert policy.excludes_project(str(tmp_path / "secret" / "src"))
    assert not policy.excludes_project(str(tmp_path / "secretive"))
    assert policy.excludes_project("~/clients/acme/app")
    assert not policy.excludes_project(None)


def test_link_hands_connectors_the_policed_event_and_nothing_it_drops():
    seen = []

    class Recording(GraphConnector):
        def on_event(self, event):
            seen.append(event)

    link = AgentLink(policy=CapturePolicy(exclude_tools=("TodoWrite",)))
    link.add_connector(Recording())
    link.emit(SessionStartEvent(session_id="s", working_directory="/repo"))
    link.emit(ToolStartEvent(session_id="s", tool_name="TodoWrite"))
    link.emit(ToolStartEvent(session_id="s", tool_name="Bash", tool_input={"command": f"echo {_AWS}"}))

    assert [type(event).__name__ for event in seen] == ["SessionStartEvent", "ToolStartEvent"]
    assert seen[1].tool_input == {"command": "echo [REDACTED:aws_access_key]"}


@pytest.fixture
def config_file(monkeypatch, tmp_path):
    """Writes the given text as this test's config file and returns it parsed."""

    def write(text):
        path = tmp_path / "policy.toml"
        path.write_text(text)
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(path))
        _identity._reset_cache()
        return _identity.load_config()

    return write


def test_config_sets_the_policy_and_keeps_it_across_rewrites(config_file):
    config = config_file(
        "[llm]\n"
        'anthropic_api_key = "configured-key-123"\n'
        "[capture]\n"
        'connectors = "actions-graph"\n'
        'max_chars = "50"\n'
        'exclude_tools = "TodoWrite, Skill"\n'
        'exclude_projects = "~/private"\n'
        "[redact]\n"
        "ticket = 'TICKET-[0-9]+'\n",
    )

    policy, problems = policy_from_config(config)
    event = policy.apply(MessageEvent(session_id="s", role="user", content="TICKET-42 uses configured-key-123"))

    assert problems == []
    assert event.content == "[REDACTED:ticket] uses [REDACTED:config_secret]"
    assert (policy.max_chars, policy.exclude_tools, policy.exclude_projects) == (
        50,
        ("TodoWrite", "Skill"),
        ("~/private",),
    )

    _identity.write_config(user_id="someone")
    reread = _identity.load_config()
    assert reread.capture_settings == config.capture_settings
    assert reread.redact_patterns == {"ticket": "TICKET-[0-9]+"}
    assert reread.connectors == ("actions-graph",)


def test_redaction_can_be_switched_off_but_private_blocks_still_go(config_file):
    policy, _ = policy_from_config(config_file('[capture]\nredact = "false"\nmax_chars = "0"\n'))

    event = policy.apply(MessageEvent(session_id="s", role="user", content=f"{_AWS} <private>x</private>" + "y" * 200))

    assert event.content == f"{_AWS} " + "y" * 200
    assert policy.max_chars is None


def test_bad_config_values_fall_back_and_are_reported(config_file):
    policy, problems = policy_from_config(config_file("[capture]\nmax_chars = \"lots\"\n[redact]\nbad = '(['\n"))

    assert policy.max_chars == DEFAULT_MAX_CHARS
    assert [name for name, _ in policy.patterns if name == "bad"] == []
    assert len(problems) == 2
