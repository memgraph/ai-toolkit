from hygm import (
    HygmModel,
    NodeType,
    RelationType,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_model,
)


def _model(*relations, labels=("User", "Person", "Location")):
    return HygmModel(node_types=tuple(NodeType(label) for label in labels), relation_types=relations)


def test_a_conforming_model_passes():
    result = validate_model(_model(RelationType("visited", start_labels=("User", "Person"), end_labels=("Location",))))
    assert result.success
    assert result.issues == []


def test_undeclared_endpoint_label_is_critical():
    result = validate_model(_model(RelationType("visited", end_labels=("Museum",))))
    assert not result.success
    assert "undeclared node type 'Museum'" in result.critical_issues[0].message
    assert result.critical_issues[0].category is ValidationCategory.STRUCTURE


def test_user_endpoint_without_person_is_critical():
    result = validate_model(_model(RelationType("visited", start_labels=("User",), end_labels=("Location",))))
    assert not result.success
    assert "'User' but not 'Person'" in result.critical_issues[0].message


def test_unconstrained_endpoint_counts_as_every_declared_label_for_the_user_rule():
    assert validate_model(_model(RelationType("mentions"))).success
    result = validate_model(_model(RelationType("mentions"), labels=("User", "Location")))
    assert len(result.critical_issues) == 2  # both sides resolve to {User, Location}


def test_duplicate_labels_are_critical():
    model = HygmModel(
        node_types=(NodeType("Person"), NodeType("Person")),
        relation_types=(RelationType("knows"), RelationType("knows")),
    )
    messages = [issue.message for issue in validate_model(model).critical_issues]
    assert "Duplicate node label 'Person'" in messages
    assert "Duplicate relation type 'knows'" in messages


def test_add_issue_only_fails_on_critical():
    result = ValidationResult("check")
    result.add_issue(ValidationIssue(ValidationSeverity.WARNING, ValidationCategory.COVERAGE, "w"))
    assert result.success
    assert result.summary() == "check: 0 critical, 1 warnings"
    result.add_issue(ValidationIssue(ValidationSeverity.CRITICAL, ValidationCategory.STRUCTURE, "c"))
    assert not result.success
