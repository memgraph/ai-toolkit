"""The generic default model learned models start from (#436)."""

from importlib.resources import as_file, files

from .labels import CORE_LABELS
from .strategies import ManualStrategy
from .types import HygmModel


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
