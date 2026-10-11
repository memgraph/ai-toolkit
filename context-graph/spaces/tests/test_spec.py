"""Spec parsing and validation: pure, no Memgraph."""

import pytest

from context_graph_spaces.spec import DEFAULT_SHARED_LABELS, SpecError, load_spec, parse_spec


def test_a_full_spec(tmp_path):
    path = tmp_path / "spaces.toml"
    path.write_text(
        'members = ["ante", "bob"]\n'
        "personal_memory_limit_mb = 512\n"
        "[spaces.payments]\n"
        'members = ["ante"]\n'
        "memory_limit_mb = 1024\n"
        'labels = ["Memory"]\n'
    )

    spec = load_spec(path)

    assert spec.members == ("ante", "bob")
    assert spec.personal_memory_limit_mb == 512
    (payments,) = spec.spaces
    assert (payments.name, payments.members, payments.labels, payments.memory_limit_mb) == (
        "payments",
        ("ante",),
        ("Memory",),
        1024,
    )
    assert spec.spaces_of("ante") == [payments]
    assert spec.spaces_of("bob") == []


def test_a_space_defaults_to_curated_memory():
    spec = parse_spec({"members": ["ante"], "spaces": {"team": {"members": ["ante"]}}})

    assert spec.spaces[0].labels == DEFAULT_SHARED_LABELS


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"members": ["Ante"]}, "lowercase"),
        ({"members": ["ante; DROP"]}, "lowercase"),
        ({"members": ["cg_bot"]}, "reserved prefix"),
        ({"members": ["ante", "ante"]}, "twice"),
        ({"members": ["ante"], "spaces": {"team": {"members": ["bob"]}}}, "aren't in the top-level members"),
        ({"members": ["ante"], "spaces": {"Team": {"members": []}}}, "lowercase"),
        ({"members": ["ante"], "spaces": {"team": {"labels": ["Memory) DELETE"]}}}, "label or edge-type names"),
        ({"members": ["ante"], "spaces": {"team": {"memory_limit_mb": 0}}}, "positive"),
        ({"members": ["ante"], "admins": []}, "Unknown key"),
    ],
)
def test_invalid_specs_are_rejected(data, message):
    with pytest.raises(SpecError, match=message):
        parse_spec(data)


def test_a_broken_file_names_itself(tmp_path):
    path = tmp_path / "spaces.toml"
    path.write_text("members = [")

    with pytest.raises(SpecError, match=r"spaces\.toml"):
        load_spec(path)
