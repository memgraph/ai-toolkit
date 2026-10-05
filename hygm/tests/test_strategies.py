import textwrap

import pytest

from hygm import LlmRecommendationStrategy, ManualStrategy, NodeType, RelationType


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


def test_llm_strategy_is_an_interface_until_its_evidence_run_passes():
    strategy = LlmRecommendationStrategy(llm=object(), observer=object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(NotImplementedError, match="372"):
        strategy.create_model(["some text"])
