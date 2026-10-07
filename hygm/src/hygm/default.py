"""The generic default model and the labels every learned model is built around (#436)."""

from importlib.resources import as_file, files

from .strategies import ManualStrategy
from .types import HygmModel
from .validation import PERSON_LABEL, USER_LABEL

#: Value types: one node per mention, since every "3" in a session is a different fact.
VALUE_LABELS: tuple[str, ...] = ("Duration", "Quantity", "Money", "Date", "TimeWindow")

#: The fixed core (#353). Derivation never merges or retires these, and they
#: sit outside the active-type cap.
CORE_LABELS: tuple[str, ...] = (USER_LABEL, PERSON_LABEL, *VALUE_LABELS)

#: Types that collect what the model has no better type for. The adoption gate
#: measures the share of mentions landing here, so these stay catch-alls in
#: every version, whatever a learned run merges into them.
CATCH_ALL_LABELS: tuple[str, ...] = ("Topic", "Artifact")


def default_model() -> HygmModel:
    """The generic default: the fixed core, Organization/Location/Event, the catch-alls, and generic relations."""
    with as_file(files("hygm") / "default_ontology.yaml") as path:
        return ManualStrategy().create_model(path)


def with_core(model: HygmModel) -> HygmModel:
    """`model` with any core node type it lacks added from the default, after its own types.

    A core type `model` already declares is kept as `model` declares it.
    """
    declared = set(model.node_labels())
    missing = tuple(t for t in default_model().node_types if t.label in CORE_LABELS and t.label not in declared)
    return HygmModel(node_types=model.node_types + missing, relation_types=model.relation_types)
