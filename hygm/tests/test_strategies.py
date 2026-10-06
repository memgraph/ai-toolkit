import textwrap

import pytest

from hygm import ManualStrategy, NodeType, RelationType


def _write(tmp_path, body):
    path = tmp_path / "ontology.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_manual_strategy_parses_identity_and_endpoints(tmp_path):
    path = _write(
        tmp_path,
        """
        entity_types:
          - {label: User, description: the user, identity: global}
          - {label: Person, description: someone else, identity: global}
          - {label: Duration, description: a length of time, identity: span}
          - {label: Topic, description: a subject}
        relation_types:
          - {label: personal_best, description: best time, start_labels: [User, Person], end_labels: [Duration]}
          - {label: mentions, description: anything}
        """,
    )
    model = ManualStrategy().create_model(path)
    assert model.node_types == (
        NodeType("User", "the user", "global"),
        NodeType("Person", "someone else", "global"),
        NodeType("Duration", "a length of time", "span"),
        NodeType("Topic", "a subject", "chunk"),
    )
    assert model.relation_types == (
        RelationType("personal_best", "best time", ("User", "Person"), ("Duration",)),
        RelationType("mentions", "anything"),
    )


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "entity_types: [{label: A, description: a}]\nrelation_types: [{label: r, description: r, end_labels: [B]}]",
            "undeclared",
        ),
        ("entity_types: [{label: A, description: a, identity: session}]", "identity"),
        (
            "entity_types: [{label: A, description: a}]\nrelation_types: [{label: r, description: r, end_labels: A}]",
            "list of labels",
        ),
        ("entity_types: [{label: '1A', description: a}]", "identifier"),
        ("entity_types: [{label: A}]", "label.*description"),
        ("entity_types: {A: a}", "entity_types"),
        ("relation_types: []", "entity_types"),
        ("entity_types: [{label: A, description: a}]\nrelation_types: {r: r}", "relation_types"),
        ("entity_types: [\n", "Invalid YAML"),
    ],
)
def test_manual_strategy_rejects_malformed_models(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        ManualStrategy().create_model(_write(tmp_path, body))


def test_manual_strategy_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        ManualStrategy().create_model(tmp_path / "missing.yaml")


def test_a_model_round_trips_through_its_mapping():
    from hygm import default_model, model_from_mapping, model_to_mapping

    model = default_model()

    assert model_from_mapping(model_to_mapping(model), "version 1") == model


def test_a_mapping_error_names_its_source():
    from hygm import model_from_mapping

    with pytest.raises(ValueError, match="Ontology version 3 must be a mapping"):
        model_from_mapping([], "version 3")


def test_a_store_of_types_reads_back_without_the_model_gate():
    """A pool of retired relations may name types it doesn't hold; it is never extracted against."""
    from hygm import model_from_mapping

    pool = {"entity_types": [], "relation_types": [{"label": "paid", "description": "", "start_labels": ["User"]}]}

    with pytest.raises(ValueError, match="undeclared"):
        model_from_mapping(pool, "pool")
    assert model_from_mapping(pool, "pool", validate=False).relation_labels() == ("paid",)
